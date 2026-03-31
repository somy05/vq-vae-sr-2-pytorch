import os
import random

import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset


class GameIRSuperResolutionDataset(Dataset):
    """Dataset for paired LR/HR super-resolution training.

    Supports two modes:

    1. **Flat directory mode** (original):
       Pass ``lr_dir`` and ``hr_dir`` pointing to two flat folders that
       contain images with matching filenames.

    2. **GameIR nested mode**:
       Pass ``root`` pointing to the GameIR dataset root.  The loader
       will recursively discover all ``<scene>/<sequence>/<lr_res>/``
       and ``<scene>/<sequence>/<hr_res>/`` pairs and collect every
       file whose name ends with ``suffix`` (default ``_rgb.png``).

       Example layout::

           GameIR-SR_sample/
             static_town08/
               00/
                 720p/
                   00000_rgb.png
                 1440p/
                   00000_rgb.png

    Args:
        lr_dir:         Path to flat LR image directory (mode 1).
        hr_dir:         Path to flat HR image directory (mode 1).
        root:           Path to GameIR dataset root    (mode 2).
        lr_res:         LR resolution folder name, e.g. ``'720p'``  (mode 2).
        hr_res:         HR resolution folder name, e.g. ``'1440p'`` (mode 2).
        suffix:         Only include files whose name ends with this
                        string.  Set to ``None`` to include all files.
        hr_patch_size:  Crop size on the HR image (default 256).
        scale:          Down-scale factor between HR and LR (default 2).
        augment:        Apply random horizontal flip (default True).
        return_name:    If True, also return the image identifier.
        patch_per_image: Virtual dataset multiplier (default 1).
    """

    def __init__(
        self,
        lr_dir=None,
        hr_dir=None,
        *,
        root=None,
        lr_res='720p',
        hr_res='1440p',
        suffix='_rgb.png',
        hr_patch_size=256,
        scale=2,
        augment=True,
        return_name=False,
        patch_per_image=1,
    ):
        self.hr_patch_size = hr_patch_size
        self.lr_patch_size = hr_patch_size // scale
        self.scale = scale
        self.augment = augment
        self.return_name = return_name
        self.patch_per_image = patch_per_image

        # Build list of (lr_path, hr_path, name) tuples
        if root is not None:
            self.pairs = self._discover_gameir(root, lr_res, hr_res, suffix)
        elif lr_dir is not None and hr_dir is not None:
            self.pairs = self._discover_flat(lr_dir, hr_dir, suffix)
        else:
            raise ValueError(
                'Provide either (lr_dir, hr_dir) for flat mode '
                'or (root, lr_res, hr_res) for GameIR nested mode.'
            )

        if len(self.pairs) == 0:
            raise RuntimeError(
                'No image pairs found. Check your paths and suffix filter.'
            )

    # ----- discovery helpers ------------------------------------------------

    @staticmethod
    def _discover_flat(lr_dir, hr_dir, suffix):
        """Flat directory mode: two folders with matching filenames."""
        pairs = []
        for f in sorted(os.listdir(hr_dir)):
            hr_path = os.path.join(hr_dir, f)
            lr_path = os.path.join(lr_dir, f)
            if not os.path.isfile(hr_path) or not os.path.isfile(lr_path):
                continue
            if suffix and not f.endswith(suffix):
                continue
            pairs.append((lr_path, hr_path, f))
        return pairs

    @staticmethod
    def _discover_gameir(root, lr_res, hr_res, suffix):
        """GameIR nested mode: root/scene/sequence/resolution/*.png"""
        pairs = []
        for dirpath, dirnames, filenames in os.walk(root):
            # Look for directories that contain the lr_res subfolder
            if lr_res not in dirnames or hr_res not in dirnames:
                continue
            lr_dir = os.path.join(dirpath, lr_res)
            hr_dir = os.path.join(dirpath, hr_res)
            # Relative id for this scene/sequence (e.g. "static_town08/00")
            rel = os.path.relpath(dirpath, root)

            for f in sorted(os.listdir(hr_dir)):
                if suffix and not f.endswith(suffix):
                    continue
                hr_path = os.path.join(hr_dir, f)
                lr_path = os.path.join(lr_dir, f)
                if os.path.isfile(hr_path) and os.path.isfile(lr_path):
                    name = os.path.join(rel, f)
                    pairs.append((lr_path, hr_path, name))
        return pairs

    # ----- Dataset interface ------------------------------------------------

    def __len__(self):
        return len(self.pairs) * self.patch_per_image

    def __getitem__(self, index):
        image_index = index % len(self.pairs)
        lr_path, hr_path, name = self.pairs[image_index]

        lr_image = Image.open(lr_path).convert('RGB')
        hr_image = Image.open(hr_path).convert('RGB')

        hr_w, hr_h = hr_image.size
        if hr_h < self.hr_patch_size or hr_w < self.hr_patch_size:
            raise ValueError(
                f'Image {name} ({hr_w}x{hr_h}) is smaller than '
                f'requested HR patch size {self.hr_patch_size}'
            )

        top = random.randint(0, (hr_h - self.hr_patch_size) // self.scale) * self.scale
        left = random.randint(0, (hr_w - self.hr_patch_size) // self.scale) * self.scale

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
            return lr_patch, hr_patch, name

        return lr_patch, hr_patch
