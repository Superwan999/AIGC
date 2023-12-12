import math
from functools import partial
# from inspect import isfunction

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import torch
from torch import nn, einsum
import torch.nn.functional as F

from utils import exists, default


''' Function helpers '''
def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)

    return arr


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), kernel_size=3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        # convert tensor shape from [b, c, h, w] to [b, c * p1 * p2,
        #                                               h / p1,
        #                                               w / p2]
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), kernel_size=1)
    )


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


''' Group Normalization '''
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * (x.shape[1] ** 0.5)


''' Position Embedding '''
class SinusoidalPositionEmbeddings(nn.Module):
    """
    Takes a tensor of shape (batch_size, 1) as input (i.e. the noise levels of several noisy images in a batch),
    and turns this into a tensor of shape (batch_size, dim) where dim is the dimensionality of the position embeddings.
    Then the tensor will be added to each residual block
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device

        # half sin and half cos
        half_dim = self.dim // 2

        # -1 ensures a wider range of frequencies.
        # e.g, half_dim = 4,
        # emb=[e^(-0), e^(-3.07), e^(-6.14), e^(-9.21)] = [1, 0.046, 0.0021, 0.0001]
        # if no -1:
        # emb = [e^(-0), e^(-2.3), e^(-4.6), e^(-6.9)] = [1, 0.1, 0.01, 0.001]
        # that would be much narrower
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # shape(time[:, None]) = (batch_size, )
        # shape(emb[None, :]) = (1, half_dim)
        # emb.shape = [batch_size, half_dim]
        emb = time[:, None] * emb[None, :]

        # emb.shape = [batch_size, dim]
        # sin = scale / cos = shift
        # x = scale * x + shift
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


''' Res Block:
In DDPM, people use Wide ResNet Block, link: https://arxiv.org/pdf/1605.07146.pdf
    the core idea is to increase the number of channel in resblocks.
    
Here the implementation is replaced to weight standardized convolution, link: https://arxiv.org/pdf/1903.10520.pdf,
    the core idea is to normalize the weight of convolution with zero-mean and one-variance 
    which is purported working synergistically with group normalization
    [better for micro-batch and group normalization]
'''
class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        # print('weight x dtype: ', x.dtype)
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        # shape(self.weight) = [out_dim, in_dim, kh, kw]
        weight = self.weight
        # get the mean for each "inner" tensor
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        # unbiased=True, variance is unbiased, /N
        # unbiased=False, variance is biased, /(N - 1)
        # partial: set unbiased=False to be a default number to torch.var() function
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()   # .rsqrt = reciporcal of sqrt = 1/sqrt(var)

        return F.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, kernel_size=3, padding=1)
        # self.proj = nn.Conv2d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            # scale and shift is coming from the embedded time tensor
            scale, shift = scale_shift
            # +1:
            # if scale -> 0 (at initializing stage, scale might be 0), x remain roughly the same
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    # The usage of "*":
    # after the sign of "*", passed arguments (time_emb_dim and groups) must be provided as keyword arguments
    # in order to increase the readability and clarity
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, kernel_size=1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")    # here the c has been doubled
            scale_shift = time_emb.chunk(2, dim=1)  # scale: [b, ceil(c/2), 1, 1], shift: [b, c/2, 1, 1]

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


''' Attention:
Here implemented 2 types of Attention: Original one and linear attention module
The Linear Attention version is proposed here called Efficient Attention: https://arxiv.org/pdf/1812.01243.pdf
In original attention, the math here is: attn = softmax(Q·K^T/sqrt(dk)) * V, where
                       Q = n x dk, K^T = dk x n => softmax(·) = n x n, 
                       V = n x dv,
                       => attn = n x dv 
In linear attention, the math here is: attn = pq(Q) * (pk(K)^T * V), where
                       pk(K)^T = dk x n, V = n x dv => (·) = dk x dv
                       pq(Q) = n x dk
                       => linear_attn = n x dv == attn, but there's only O(n) while original attn is O(n^2)
                     and pq = softmax(Q's row)
                         pk = softmax(K's col)
'''
class Attention(nn.Module):
    # strictly matching the formula of traditional attention module described in transformer paper
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)

        q = q * self.scale

        # <=> Q·K^T, get multiplication result along d-axis
        #     n x dk * dk x n => n x n , get multiplication result along dk-axis
        # so here i & j corresponds to n & n, dk is d
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()     # for stability
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** (-0.5)
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, kernel_size=1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, kernel_size=1),
                                    RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)   # softmax along with d-dimension
        k = k.softmax(dim=-1)   # softmax along with n-dimension

        q = q * self.scale
        # pk(K)^T = dk x n, V = n x dv => (·) = dk x dv
        # remove n-dimension, keep dk (here d) and dv (here e)
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        # pq(Q) = n x dk * (dk x dv) = n dv, remove dk-dimension, keep n and dv(here e)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


''' Conditional UNet:
Input: noisy images of shape   (batch_size, num_channels, height, width) + 
       a batch of noise levels (batch_size, 1)
Output: (batch_size, num_channels, height, width)
'''
class Unet(nn.Module):
    # parameters are more / larger in semi-official repo:
    # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/guided_diffusion.py#L764
    def __init__(
            self,
            dim,
            init_dim=None,
            out_dim=None,
            dim_mults=(1, 2, 4, 8),
            channels=3,
            self_condition=False,
            resnet_block_groups=4,      # semi-official, 8
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        #                                        originally, k=7,           p=3
        self.init_conv = nn.Conv2d(input_channels, init_dim, kernel_size=7, padding=3)

        # assume dim == init_dim == 64
        # dims = [64, 64, 128, 256, 512]
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        # [(64,64),(64,128),(128,256),(256,512)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        # we have improved time embedding methods in semi-official repo
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # layers: encoder - downsample
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, kernel_size=3, padding=1)
                    ]
                )
            )

        # layers: bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # layers: decoder - upsample
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
               nn.ModuleList(
                   [
                       block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                       block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                       Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                       Upsample(dim_out, dim_in)
                       if not is_last
                       else nn.Conv2d(dim_out, dim_in, kernel_size=3, padding=1)
                   ]
               )
            )

        # output
        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, kernel_size=1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()   # for the final output stage

        t = self.time_mlp(time)     # [b, dim * 4]

        h = []

        # encode: downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # decode: upsmaple
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
















