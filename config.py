import argparse


def load_config():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--image_size',
                        type=int,
                        default=64,
                        help='height & width of a processed image')
    parser.add_argument('--data_path',
                        type=str,
                        default='train_data/flowers',
                        help='height & width of a processed image')

    # network
    parser.add_argument('--dim',
                        type=int,
                        default=64,
                        help='basic dim before multiply dim_mults')
    parser.add_argument('--channels',
                        type=int,
                        default=3,
                        help='input & out channel for input & generated image')
    parser.add_argument('--dim_mults',
                        default=(1, 2, 4, 8),
                        help='input & out channel for input & generated image')

    # diffusion
    parser.add_argument('--timesteps',
                        type=int,
                        default=1000,
                        help='number of steps for diffusion procedure')
    parser.add_argument('--beta_schedule_type',
                        type=str,
                        default='linear',
                        help='choose from linear, cosine or sigmoid')
    parser.add_argument('--objective',
                        type=str,
                        default='pred_noise',
                        help='choose from pred_noise, pred_v or pred_x0 based on different method. \
                              pred_noise is the original DDPM method')

    # training
    parser.add_argument('-l',
                        '--lr',
                        type=float,
                        default=0.0001,
                        help='initial learning rate')
    parser.add_argument('--loss_type',
                        type=str,
                        default='huber',
                        help='choose from l1, l2 or huber')
    parser.add_argument('--batch_size',
                        type=int,
                        default=16)
    parser.add_argument('--epochs',
                        type=int,
                        default=10)
    parser.add_argument('--cuda',
                        type=bool,
                        default=True)

    # sampling
    parser.add_argument('--sample_batch_size',
                        type=int,
                        default=2,
                        help='choose from l1, l2 or huber')
    parser.add_argument('--per_save_epoch_cnt',
                        type=int,
                        default=1,
                        help='save sampling result per epoch')
    parser.add_argument('--results_folder',
                        type=str,
                        default='generated_ddpm_samples_pred_noise',
                        help='generated samples are saved here by default')
    parser.add_argument('--stable_sampling',
                        type=bool,
                        default=True,
                        help='numerical stabilization during sampling')
    parser.add_argument('--sampling_timesteps',
                        default=1000,
                        help='number of sampling steps for diffusion procedure')

    # DDIM
    parser.add_argument('--ddim_sampling_eta',
                        type=float,
                        default=1.0,
                        help='1.: DDPM, 0.: DDIM')

    # loss weight
    parser.add_argument('--min_snr_gamma',
                        type=int,
                        default=5,
                        help='gamma in min-snr-gamma')

    # verbose
    parser.add_argument('--showing_steps_cnt',
                        type=int,
                        default=10,
                        help='show training loss per steps')

    args = parser.parse_args()

    return args


