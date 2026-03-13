import os
import random

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class GameIRSuperResolutionDataset(Dataset):
    def __init__(
        self,
        lr_dir,
        hr_dir,
        hr_patch_size=256,
        scale=2,
        augment=True,
        return_name=False,
        patch_per_image=1,
    ):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.scale = scale
        self.augment = augment
        self.return_name = return_name
        self.patch_per_image = patch_per_image
        self.image_names = sorted(
            f
            for f in os.listdir(hr_dir)
            if os.path.isfile(os.path.join(hr_dir, f))
            and os.path.isfile(os.path.join(lr_dir, f))
        )

    def __len__(self):
        return len(self.image_names) * self.patch_per_image

    def __getitem__(self, index):
        image_index = index % len(self.image_names)
        img_name = self.image_names[image_index]
        lr_image = Image.open(os.path.join(self.lr_dir, img_name)).convert('RGB')
        hr_image = Image.open(os.path.join(self.hr_dir, img_name)).convert('RGB')

        hr_w, hr_h = hr_image.size
        if hr_h < self.hr_patch_size or hr_w < self.hr_patch_size:
            raise ValueError(
                f'Image {img_name} is smaller than requested HR patch size {self.hr_patch_size}'
            )

        top = random.randint(0, hr_h - self.hr_patch_size)
        left = random.randint(0, hr_w - self.hr_patch_size)

        hr_patch = TF.crop(hr_image, top, left, self.hr_patch_size, self.hr_patch_size)
        lr_patch = TF.crop(
            lr_image,
            top // self.scale,
            left // self.scale,
            self.lr_patch_size,
            self.lr_patch_size,
        )

        if self.augment and random.random() > 0.5:
            hr_patch = TF.hflip(hr_patch)
            lr_patch = TF.hflip(lr_patch)

        hr_patch = TF.normalize(
            TF.to_tensor(hr_patch), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )
        lr_patch = TF.normalize(
            TF.to_tensor(lr_patch), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        )

        if self.return_name:
            return lr_patch, hr_patch, img_name

        return lr_patch, hr_patch
