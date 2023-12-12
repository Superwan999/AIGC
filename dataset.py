from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
import torch.nn as nn

from pathlib import Path
from PIL import Image
from functools import partial

from utils import exists

''' If you just want to make a simple demo '''
# one can download mnist/fashion-mnist etc dataset easily by using datasets provided by Kaggle
# 1. install datasets: pip install datasets
# 2. import dataset
#    from datasets import load_dataset
# 3. get dataset from load_dataset
#    dataset = load_dataset("mnist")


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


class CustomDataset(Dataset):
    def __init__(self, folder, image_size, exts=None, augment_horizontal_flip=True, convert_image_to=None):
        if exts is None:
            exts = ['jpg', 'jpeg', 'png', 'tiff']
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [
            p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')
        ]

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size),
            T.RandomHorizontalFlip() if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Lambda(lambda t: (t * 2) - 1)     # convert image to [-1, 1]
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        # print('index: ', index, '  path: ', path)
        img = Image.open(path)
        return self.transform(img)




