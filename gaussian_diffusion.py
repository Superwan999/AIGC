import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from tqdm.auto import tqdm
from functools import partial
from collections import namedtuple
from einops import reduce

from utils import default, extract, identity


def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    :param timesteps: how many steps you're going to complete the diffusion procedure
    :return: betas for each step
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule, proposed in Improved DDPM (https://arxiv.org/pdf/2102.09672.pdf)
    :param timesteps:
    :param s:
    :return:
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule, could have better FID score,
    proposed in (Diffusion model distillation) https://arxiv.org/abs/2212.11972 - Figure B.1
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 args,
                 device,
                 network,
                 min_snr_loss_weight=True):
        super().__init__()

        self.args = args
        self.device = device

        self.objective = self.args.objective
        assert self.objective in {'pred_noise', 'pred_x0', 'pred_v'}

        # network
        self.network = network.to(device)

        # helper function to register buffer from float64 to float32
        # register buffer allows one add tensors that are not trainable parameters to the module
        # for the purpose os GPU acceleration or state-saving
        # register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # define betas: beta1, beta2, ..., beta_n
        beta_schedule = self.args.beta_schedule_type
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
        # tensor([0.0001, 0.0001, 0.0002, 0.0002, 0.0003, 0.0003, 0.0003, 0.0004, 0.0004, ... 0.2, 0.2], dtype=float64)
        betas = beta_schedule_fn(self.args.timesteps)

        # number of time steps: 500
        self.num_timesteps = int(betas.shape[0])

        # ddim
        if self.args.sampling_timesteps != self.num_timesteps:
            self.is_ddim_sampling = self.args.sampling_timesteps < self.num_timesteps
            self.sampling_timesteps = self.args.sampling_timesteps
            self.ddim_sampling_eta = self.args.ddim_sampling_eta
        else:
            self.is_ddim_sampling = False

        # define alphas
        # get a1, a2, ..., an
        # tensor([0.9999, 0.9999, 0.9998, ..., 0.9800, 0.9800], dtype=torch.float64)
        alphas = 1. - betas

        # a1, a1a2, a1a2a3, ..., PI(a1, a2, ..., an)
        # tensor([0.9999, 0.9998, 0.9996, ..., 0.0065, 0.0064], dtype=torch.float64)
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        # 1, a1, a1a2, ..., PI(1, a1, a2, ..., a_n-1)
        # tensor([1.0000, 0.9999, 0.9998, 0.9996, ..., 0.0066, 0.0065], dtype=torch.float64)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # sqrt(1/a1), sqrt(1/a2), ..., sqrt(1/an)
        # tensor([1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0001, 1.0002, ..., 1.0101, 1.0102], dtype=torch.float64)
        sqrt_recip_alphas = 1.0 / torch.sqrt(alphas)

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # xt = sqrt(alphas_cumprod)x0 + sqrt(1 - alphas_cumprod) * noise
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # beta_t^bar = posterior_ariance
        # tensor([0.0000e+00, 5.8316e-05, 1.0276e-04, ..., 1.9957e-02, 1.9997e-02], dtype=torch.float64)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)

        # attention: below is aiming for stable sampling
        # x0 = A·xt - B·epsilon(generated)
        # tensor([ 1.0000,  1.0001,  1.0002, ..., 12.2958, 12.4203, 12.5464])
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))           # A
        # tensor([1.0000e-02, 1.5489e-02, ..., 1.2255e+01, 1.2380e+01, 1.2507e+01])
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))     # B

        # mu_(t-1) = mean_coef1 * clip(x0) + mean_coef2 * xt
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # tensor([-46.0517, -9.7496, -9.1831, ..., -3.9162,  -3.9142,  -3.9122])
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        # tensor([1.0000, 0.5831, 0.4284, ..., 0.0016, 0.0016, 0.0016])
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        # tensor([0.0000, 0.4169, 0.5716, ..., 0.9899, 0.9898, 0.9898])
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss weight
        # Efficient Diffusion Training via Min-SNR Weighting Strategy: https://arxiv.org/abs/2303.09556
        # snr: signal noise ratio
        # tensor([9.9990e+03, 4.1680e+03, 2.3823e+03, ..., 6.6584e-03, 6.5247e-03, 6.3933e-03], dtype=torch.float64)
        snr = alphas_cumprod / (1. - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=self.args.min_snr_gamma)

        if self.objective == 'pred_noise':
            register_buffer('loss_weight', maybe_clipped_snr / snr)     # tensor([1., 1., 1., ...)
        elif self.objective == 'pred_x0':
            register_buffer('loss_weight', maybe_clipped_snr)
        elif self.objective == 'pred_v':
            register_buffer('loss_weight', maybe_clipped_snr / (snr + 1))

        self.ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

    # forward diffusion (using the nice property)
    # @autocast(enabled=False)      # in semi-official code, this means it prevents the mixed-precision training
    #                                 in cuda, it automatically mixes precision when training for efficiency
    def q_sample(self, x_start, t, noise=None):
        # noise = default(noise, lambda: torch.randn_like(x_start))

        # [b, 1, 1, 1]
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        # x_t = sqrt(alpha_cumprod) * x_0 + sqrt(1 - alpha_cumprod) * noise
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # in semi-official repo
        # offset noise - https://www.crosslabs.org/blog/diffusion-with-offset-noise

        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)


        # in semi-official repo
        # self-conditioning

        # for DDPM training, network_out = predicted_noise
        network_out = self.network(x_t, t)

        # in Improved DDPM, we have different target
        if self.args.objective == 'pred_noise':
            target = noise
        elif self.args.objective == 'pred_v':
            target = self.predict_v(x_start, t, noise)
        elif self.args.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.args.objective}')

        if self.args.loss_type == 'huber':
            loss = F.smooth_l1_loss(network_out, target, reduction='none')
        elif self.args.loss_type == 'l1':
            loss = F.l1_loss(network_out, target, reduction='none')
        elif self.args.loss_type == 'l2':
            loss = F.mse_loss(network_out, target, reduction='none')
        else:
            raise NotImplementedError()

        # loss could have different components with different weight
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    # reverse process
    def predict_start_from_noise(self, x_t, t, noise):
        # get the practical x_0
        # x0 = 1/sqrt(alpha^bar)·xt - sqrt((1 - alpha^bar)/alpha^bar) * noise[generated]
        # x0 = A·xt - B·epsilon
        return extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - \
               extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise

    def predict_noise_from_start(self, x_t, t, x0):
        # x0 = A·xt - B·epsilon
        # epsilon = (A·xt - x0) / B
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def model_predictions(self, x, t, clip_x_start=False, rederive_pred_noise=False):
        network_output = self.network(x, t)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity

        if self.args.objective == 'pred_noise':
            pred_noise = network_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = network_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = network_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return self.ModelPrediction(pred_noise, x_start)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True):
        preds = self.model_predictions(x, t)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.inference_mode()
    def p_sample(self, x, t, t_index):
        if not self.args.stable_sampling:
            # original sampling method
            betas_t = extract(self.betas, t, x.shape)
            sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
            sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

            x_mean = sqrt_recip_alphas_t * (
                    x - betas_t * self.network(x, t) / sqrt_one_minus_alphas_cumprod_t
            )

            if t_index == 0:
                return x_mean

            else:
                # posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
                posterior_variance_t = extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)

                return x_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            # stabilized sampling
            model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t)
            noise = torch.randn_like(x) if t_index > 0 else 0.
            pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
            return pred_img

    @torch.inference_mode()
    def ddim_sample(self, shape, return_all_timesteps=False):
        batch_size = self.args.sample_batch_size

        # if sampling_timesteps == total_timesteps, [-1, 0, 1, 2, ..., T-1]
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))   # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        image = torch.randn(shape, device=self.device)
        return_images = [image.cpu().numpy()]

        for time, time_next in tqdm(time_pairs,
                                    desc='sampling loop time step'):
            time_cond = torch.full((batch_size,),
                                   time,
                                   device=self.device,
                                   dtype=torch.long)
            pred_noise, x_start = self.model_predictions(image,
                                                         time_cond,
                                                         clip_x_start=True,
                                                         rederive_pred_noise=True)

            if time_next < 0:
                image = x_start
                return_images.append(image.cpu().numpy())
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            noise = torch.randn_like(image)

            image = alpha_next.sqrt() * x_start + \
                    c * pred_noise + \
                    sigma * noise

            if return_all_timesteps:
                return_images.append(image.cpu().numpy())

        return return_images

    @torch.inference_mode()
    def p_sample_loop(self, shape, return_all_timesteps=False):
        batch_size = self.args.sample_batch_size
        image = torch.randn(shape, device=self.device)
        return_images = [image.cpu().numpy()]

        for i in tqdm(reversed(range(0, self.args.timesteps)),
                      desc='sampling loop time step',
                      total=self.args.timesteps):
            #                            torch.full(shape, value, device, dtype)
            image = self.p_sample(image,                                                    # image
                                  torch.full((batch_size,),                                 # time
                                             i, device=self.device, dtype=torch.long),
                                  i)                                                        # time index
            if return_all_timesteps:
                return_images.append(image.cpu().numpy())
            else:
                if i == 0:
                    return_images.append(image.cpu().numpy())
        # return_images = torch.sta
        return return_images

    @torch.inference_mode()
    def sample(self, sample_batch_size, channels, img_height, img_width, return_all_timesteps=False):
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        return sample_fn(shape=(sample_batch_size,
                                channels,
                                img_height,
                                img_width),
                         return_all_timesteps=return_all_timesteps)

    # training
    def forward(self, img):
        b, _, _, _ = img.shape
        # tensor([143, 409, 298, 250, ..., 203, 172], device='cuda:0)
        t = torch.randint(0, self.num_timesteps, (b,), device=self.device).long()
        return self.p_losses(img, t)







