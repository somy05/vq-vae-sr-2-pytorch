import argparse
import math
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.utils import save_image

from dataset import GameIRSuperResolutionDataset
from sr_model import DirectSRNet, inject_lora, extract_lora_state_dict


def calc_psnr(pred, target):
    pred_01 = (pred + 1) / 2
    target_01 = (target + 1) / 2
    mse = F.mse_loss(pred_01, target_01)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(1.0 / mse.item())


def train_epoch(model, loader, optimizer, scheduler, device, epoch):
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
            f'LoRA Epoch {epoch + 1} | L1: {loss.item():.5f} | '
            f'PSNR: {psnr:.2f} dB | Avg: {psnr_sum / batch_count:.2f} dB | '
            f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
        )

    return loss_sum / batch_count, psnr_sum / batch_count


def save_samples(model, dataset, device, epoch, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        images = []
        for idx in range(min(4, len(dataset))):
            lr_img, hr_img = dataset[idx]
            lr_img = lr_img.unsqueeze(0).to(device)
            hr_img = hr_img.unsqueeze(0).to(device)

            pred = model(lr_img)
            bicubic = F.interpolate(
                lr_img, scale_factor=2, mode='bicubic', align_corners=False
            )
            images.extend([bicubic.cpu(), pred.cpu(), hr_img.cpu()])

        grid = torch.cat(images, dim=0)
        save_image(
            grid, f'{save_dir}/lora_samples_epoch_{epoch + 1:03d}.png',
            nrow=3, normalize=True, value_range=(-1, 1)
        )
    model.train()


def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for SR')
    parser.add_argument('--base_ckpt', type=str, required=True,
                        help='Path to pre-trained base model checkpoint')

    # Data source: GameIR nested structure
    parser.add_argument('--root', type=str, default=None,
                        help='GameIR-style dataset root')
    parser.add_argument('--suffix', type=str, default='.rgb.png')
    parser.add_argument('--lr_res', type=str, default='720p')
    parser.add_argument('--hr_res', type=str, default='1440p')

    # Data source: flat directories
    parser.add_argument('--lr_dir', type=str, default=None,
                        help='Flat LR image directory')
    parser.add_argument('--hr_dir', type=str, default=None,
                        help='Flat HR image directory')

    # LoRA config
    parser.add_argument('--rank', type=int, default=4,
                        help='LoRA rank (default 4)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='LoRA scaling factor')

    # Training config
    parser.add_argument('--size', type=int, default=256,
                        help='HR patch size')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-3,
                        help='Learning rate (higher than base training)')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--patch_per_image', type=int, default=4,
                        help='Random crops per image')

    # Output
    parser.add_argument('--save_dir', type=str, default='lora_weights')
    parser.add_argument('--sample_dir', type=str, default='lora_samples')
    parser.add_argument('--name', type=str, default='game_lora',
                        help='Name for the LoRA weights file')

    args = parser.parse_args()

    if not args.root and not (args.lr_dir and args.hr_dir):
        parser.error('Provide either --root (GameIR) or --lr_dir + --hr_dir (flat)')

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f'Device: {device}')

    # ── Load base model ─────────────────────────────────────────────
    print(f'\nLoading base model: {args.base_ckpt}')
    ckpt = torch.load(args.base_ckpt, map_location='cpu')
    ckpt_args = ckpt.get('args', {})

    model = DirectSRNet(
        scale=ckpt_args.get('scale', 2),
        n_channels=ckpt_args.get('n_channels', 64),
        n_blocks=ckpt_args.get('n_blocks', 16),
        fast_tail=ckpt_args.get('fast_tail', False),
    )

    state = ckpt['model']
    # Handle DataParallel state dict
    if list(state.keys())[0].startswith('module.'):
        state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)

    # ── Inject LoRA ─────────────────────────────────────────────────
    print(f'\nInjecting LoRA (rank={args.rank}, alpha={args.alpha})')
    model = inject_lora(model, rank=args.rank, alpha=args.alpha)
    model = model.to(device)

    # ── Dataset ─────────────────────────────────────────────────────
    scale = ckpt_args.get('scale', 2)

    if args.root:
        dataset = GameIRSuperResolutionDataset(
            root=args.root,
            lr_res=args.lr_res,
            hr_res=args.hr_res,
            suffix=args.suffix,
            hr_patch_size=args.size,
            scale=scale,
            augment=True,
            patch_per_image=args.patch_per_image,
        )
    else:
        dataset = GameIRSuperResolutionDataset(
            lr_dir=args.lr_dir,
            hr_dir=args.hr_dir,
            suffix=None,
            hr_patch_size=args.size,
            scale=scale,
            augment=True,
            patch_per_image=args.patch_per_image,
        )

    print(f'Dataset: {len(dataset)} patches '
          f'({len(dataset) // args.patch_per_image} images × '
          f'{args.patch_per_image} patches)')

    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Optimizer (only LoRA params) ────────────────────────────────
    lora_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(lora_params, lr=args.lr)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=len(loader) * args.epoch,
        pct_start=0.1,
        anneal_strategy='cos',
    )

    os.makedirs(args.save_dir, exist_ok=True)

    # ── Training loop ───────────────────────────────────────────────
    print(f'\nTraining LoRA for {args.epoch} epochs...\n')

    for epoch in range(args.epoch):
        avg_loss, avg_psnr = train_epoch(
            model, loader, optimizer, scheduler, device, epoch
        )

        # Save samples every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            save_samples(model, dataset, device, epoch, args.sample_dir)

        # Save LoRA weights every 5 epochs and at the end
        if (epoch + 1) % 5 == 0 or epoch == args.epoch - 1:
            lora_state = extract_lora_state_dict(model)
            save_path = os.path.join(
                args.save_dir, f'{args.name}_epoch{epoch + 1:03d}.pt'
            )
            torch.save({
                'lora_state': lora_state,
                'rank': args.rank,
                'alpha': args.alpha,
                'base_ckpt': args.base_ckpt,
                'epoch': epoch + 1,
                'avg_loss': avg_loss,
                'avg_psnr': avg_psnr,
            }, save_path)

            # File size
            size_kb = os.path.getsize(save_path) / 1024
            print(f'  → Saved LoRA: {save_path} ({size_kb:.1f} KB) | '
                  f'PSNR: {avg_psnr:.2f} dB')

    # Save final weights with clean name
    final_path = os.path.join(args.save_dir, f'{args.name}.pt')
    lora_state = extract_lora_state_dict(model)
    torch.save({
        'lora_state': lora_state,
        'rank': args.rank,
        'alpha': args.alpha,
        'base_ckpt': args.base_ckpt,
        'epoch': args.epoch,
        'avg_loss': avg_loss,
        'avg_psnr': avg_psnr,
    }, final_path)
    size_kb = os.path.getsize(final_path) / 1024
    print(f'\n  ✓ Final LoRA saved: {final_path} ({size_kb:.1f} KB)')


if __name__ == '__main__':
    main()
