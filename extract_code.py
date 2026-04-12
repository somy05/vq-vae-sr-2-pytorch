"""
Pre-compute VQ-VAE latent codes for all images in the dataset.
Saves everything into a single .pt file for fast prior training.

Usage:
    python extract_code.py \
        --ckpt checkpoint/vqvae_295.pt \
        --root /path/to/GameIR-SR \
        --suffix .rgb.png \
        --out codes.pt
"""

import argparse

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import GameIRSuperResolutionDataset
from vqvae import VQVAE


@torch.no_grad()
def extract_codes(loader, model, device):
    all_top = []
    all_bottom = []
    all_lr_top = []

    for lr_img, hr_img in tqdm(loader, desc='Extracting codes'):
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        _, _, _, hr_top, hr_bottom = model.encode(hr_img)
        _, _, _, lr_top, _ = model.encode(lr_img)

        all_top.append(hr_top.cpu())
        all_bottom.append(hr_bottom.cpu())
        all_lr_top.append(lr_top.cpu())

    return {
        'top': torch.cat(all_top, 0),        # [N, 32, 32]
        'bottom': torch.cat(all_bottom, 0),   # [N, 64, 64]
        'lr_top': torch.cat(all_lr_top, 0),   # [N, 16, 16]
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--lr_path', type=str, default=None)
    parser.add_argument('--hr_path', type=str, default=None)
    parser.add_argument('--root', type=str, default=None, help='GameIR dataset root (nested mode)')
    parser.add_argument('--lr_res', type=str, default='720p', help='LR resolution folder name')
    parser.add_argument('--hr_res', type=str, default='1440p', help='HR resolution folder name')
    parser.add_argument('--suffix', type=str, default='_rgb.png', help='Image filename suffix filter')
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = GameIRSuperResolutionDataset(
        lr_dir=args.lr_path,
        hr_dir=args.hr_path,
        root=args.root,
        lr_res=args.lr_res,
        hr_res=args.hr_res,
        suffix=args.suffix,
        hr_patch_size=args.size,
        scale=args.scale,
        augment=False,       # No augmentation for extraction
        patch_per_image=1,   # One patch per image
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f'Dataset: {len(dataset)} images')
    print(f'Device: {device}')

    model = VQVAE()
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model = model.to(device)
    model.eval()

    codes = extract_codes(loader, model, device)

    print(f'\nExtracted shapes:')
    print(f'  top:    {codes["top"].shape}')
    print(f'  bottom: {codes["bottom"].shape}')
    print(f'  lr_top: {codes["lr_top"].shape}')

    torch.save(codes, args.out)
    print(f'\nSaved to {args.out}')
