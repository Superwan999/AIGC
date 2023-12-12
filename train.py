import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.utils import save_image

from dataset import CustomDataset
from network import Unet
from gaussian_diffusion import GaussianDiffusion
from config import load_config

import gc
import os

import numpy as np
from pathlib import Path
from PIL import Image

torch.manual_seed(42)
class Trainer:
    def __init__(self,
                 args,
                 diffusion_model,
                 device):
        self.args = args
        self.device = device

        # data
        self.dataset = CustomDataset(folder=args.data_path,
                                     image_size=args.image_size)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.args.batch_size,
                                     shuffle=True)
                                     # shuffle=True)

        # model
        self.model = diffusion_model

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    def train(self):
        os.makedirs(self.args.results_folder, exist_ok=True)

        for epoch_id in range(1, self.args.epochs + 1):

            epoch_loss = 0
            for index, batch in enumerate(self.dataloader):

                self.optimizer.zero_grad()

                # tensor([[[[ 0.1922,  0.3412,  0.3176,  ..., -0.6471, -0.6784, -0.7569],
                #           [ 0.2549,  0.4510,  0.4118,  ..., -0.6627, -0.6941, -0.7725],
                #           [ 0.3333,  0.4431,  0.3255,  ..., -0.6784, -0.7176, -0.7961],
                #           ...,
                #           [-0.0824, -0.0824, -0.0745,  ..., -0.1373, -0.3176, -0.5529],
                #           [-0.0745, -0.0667, -0.0588,  ..., -0.0510, -0.1294, -0.2706],
                #           [-0.0667, -0.0588, -0.0588,  ..., -0.0353, -0.0667, -0.1216]]]],
                #        device='cuda:0')
                image = batch.to(self.device)
                loss = self.model(image)
                epoch_loss += loss.item()

                # backward propagation
                loss.backward()

                # show training result of Discriminator
                if index % self.args.showing_steps_cnt == 0:
                    showing_str = '    Epoch[{}]({}/{}): ' \
                                  '    Loss: ({:.6f}), '.format(epoch_id, index + 1, len(self.dataloader),
                                                                loss.item())
                    print(showing_str)

                # update weights
                self.optimizer.step()
            print(f'Epoch{epoch_id} loss: {epoch_loss / len(self.dataloader)}')

            if epoch_id % self.args.per_save_epoch_cnt == 0:
                # save images
                all_images = self.model.sample(sample_batch_size=self.args.sample_batch_size,
                                               channels=self.args.channels,
                                               img_height=self.args.image_size,
                                               img_width=self.args.image_size)
                all_images = [(img + 1) * 0.5 for img in all_images]
                all_images_rgb = [np.clip(img * 255, 0, 255).astype(np.uint8) for img in all_images]

                save_path = f'{self.args.results_folder}/sample-{epoch_id}'
                os.makedirs(save_path, exist_ok=True)
                for time_id, images in enumerate(all_images_rgb):
                    for image_id, img in enumerate(images):
                        img_rgb = np.transpose(img, (1, 2, 0))
                        image = Image.fromarray(img_rgb)
                        image.save(os.path.join(save_path, f"image_time_{time_id}_image_{image_id}.png"))


if __name__ == '__main__':
    gc.collect()
    args = load_config()
    if torch.cuda.is_available() and args.cuda is True:
        device = 'cuda'
        args.cuda = True
    else:
        device = 'cpu'
        args.cuda = False

    # model
    # dim,
    # init_dim = None, not the initial channel
    # out_dim = None,
    # dim_mults = (1, 2, 4, 8),
    # channels = 3,
    # self_condition = False,
    # resnet_block_groups = 4,  # semi-official, 8
    network = Unet(
        dim=args.dim,
        channels=args.channels,
        dim_mults=args.dim_mults
    )

    diffusion = GaussianDiffusion(
        args=args,
        device=device,
        network=network,
    )

    trainer = Trainer(
        args=args,
        diffusion_model=diffusion,
        device=device
    )

    trainer.train()

