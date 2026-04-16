"""
Training script for the Direct SR network.

Usage:
    python train_sr.py \
        --root /path/to/GameIR-SR \
        --suffix .rgb.png \
        --epoch 200 \
        --batch 16 \
        --lr 1e-3
"""

import argparse
import os
import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from dataset import GameIRSuperResolutionDataset
from sr_model import DirectSRNet
from perceptual_loss import PerceptualLoss


def calc_psnr(pred, target):
    """Calculate PSNR for images in [-1, 1] range."""
    # Convert to [0, 1]
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = F.mse_loss(pred_01, target_01)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse.item())


def train_epoch(model, loader, optimizer, scheduler, device, epoch,
                perceptual_fn=None, lambda_perceptual=0.0):
    model.train()
    pbar = tqdm(loader)
    loss_fn = nn.L1Loss()

    loss_sum = 0.0
    psnr_sum = 0.0
    batch_count = 0

    for lr_img, hr_img in pbar:
        lr_img = lr_img.to(device)
        hr_img = hr_img.to(device)

        pred = model(lr_img)
        loss = loss_fn(pred, hr_img)

        if perceptual_fn is not None and lambda_perceptual > 0:
            p_loss = perceptual_fn(pred, hr_img)
            loss = loss + lambda_perceptual * p_loss

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        with torch.no_grad():
            psnr = calc_psnr(pred, hr_img)

        loss_sum += loss.item()
        psnr_sum += psnr
        batch_count += 1

        pbar.set_description(
            f'Epoch {epoch + 1} | L1: {loss.item():.5f} | '
            f'PSNR: {psnr:.2f} dB | Avg L1: {loss_sum / batch_count:.5f} | '
            f'Avg PSNR: {psnr_sum / batch_count:.2f} dB | '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )

    return loss_sum / batch_count, psnr_sum / batch_count


def save_samples(model, dataset, device, epoch, save_dir):
    """Save a visual comparison: LR bicubic | SR | GT."""
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        # Grab first 4 images
        images = []
        for idx in range(min(4, len(dataset))):
            lr_img, hr_img = dataset[idx]
            lr_img = lr_img.unsqueeze(0).to(device)
            hr_img = hr_img.unsqueeze(0).to(device)

            pred = model(lr_img)
            bicubic = F.interpolate(
                lr_img, scale_factor=2, mode='bicubic', align_corners=False
            )

            # Stack: bicubic | prediction | ground truth
            images.extend([bicubic.cpu(), pred.cpu(), hr_img.cpu()])

        grid = torch.cat(images, dim=0)
        save_image(
            grid, f'{save_dir}/sr_samples_epoch_{epoch + 1:03d}.png',
            nrow=3, normalize=True, value_range=(-1, 1)
        )

    model.train()


def main():
    parser = argparse.ArgumentParser(description='Train Direct SR Network')
    parser.add_argument('--root', type=str, required=True,
                        help='GameIR dataset root')
    parser.add_argument('--suffix', type=str, default='_rgb.png')
    parser.add_argument('--lr_res', type=str, default='720p')
    parser.add_argument('--hr_res', type=str, default='1440p')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--size', type=int, default=256,
                        help='HR patch size for training')
    parser.add_argument('--n_channels', type=int, default=64)
    parser.add_argument('--n_blocks', type=int, default=16)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--save_dir', type=str, default='checkpoint')
    parser.add_argument('--sample_dir', type=str, default='sr_samples')
    parser.add_argument('--lambda_perceptual', type=float, default=0.0,
                        help='Weight for perceptual loss (0 = L1 only)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # Dataset
    dataset = GameIRSuperResolutionDataset(
        root=args.root,
        lr_res=args.lr_res,
        hr_res=args.hr_res,
        suffix=args.suffix,
        hr_patch_size=args.size,
        scale=args.scale,
        augment=True,
        patch_per_image=4,
    )
    print(f'Dataset: {len(dataset)} patches')

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Model
    model = DirectSRNet(
        scale=args.scale,
        n_channels=args.n_channels,
        n_blocks=args.n_blocks,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f'Model: {args.n_channels}ch × {args.n_blocks} blocks = {n_params:,} params')

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    model = model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=len(loader) * args.epoch,
        pct_start=0.05,
        anneal_strategy='cos',
    )

    # Resume
    start_epoch = 0
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device)
        state = ckpt['model']
        # Handle DataParallel state dict
        if list(state.keys())[0].startswith('module.') and not isinstance(model, nn.DataParallel):
            state = {k.replace('module.', ''): v for k, v in state.items()}
        elif not list(state.keys())[0].startswith('module.') and isinstance(model, nn.DataParallel):
            state = {f'module.{k}': v for k, v in state.items()}
        model.load_state_dict(state)
        start_epoch = ckpt.get('epoch', 0)
        print(f'Resumed from epoch {start_epoch}')

    os.makedirs(args.save_dir, exist_ok=True)

    # Perceptual loss (optional)
    perceptual_fn = None
    if args.lambda_perceptual > 0:
        perceptual_fn = PerceptualLoss().to(device)
        print(f'Using perceptual loss (λ={args.lambda_perceptual})')
    else:
        print('Using L1 loss only')

    # Training loop
    for epoch in range(start_epoch, args.epoch):
        avg_loss, avg_psnr = train_epoch(
            model, loader, optimizer, scheduler, device, epoch,
            perceptual_fn=perceptual_fn,
            lambda_perceptual=args.lambda_perceptual,
        )

        # Save samples every 10 epochs
        if (epoch + 1) % 10 == 0:
            m = model.module if isinstance(model, nn.DataParallel) else model
            save_samples(m, dataset, device, epoch, args.sample_dir)

        # Save checkpoint every 25 epochs and at the end
        if (epoch + 1) % 25 == 0 or epoch == args.epoch - 1:
            m = model.module if isinstance(model, nn.DataParallel) else model
            torch.save({
                'model': m.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch + 1,
                'args': vars(args),
                'avg_loss': avg_loss,
                'avg_psnr': avg_psnr,
            }, f'{args.save_dir}/sr_direct_{epoch + 1:03d}.pt')
            print(f'  → Saved checkpoint (PSNR: {avg_psnr:.2f} dB)')


if __name__ == '__main__':
    main()
