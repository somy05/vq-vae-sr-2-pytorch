"""
Evaluate specific images to get their exact PSNR and SSIM.

Usage:
    python eval_specific.py \
        --ckpt checkpoint_fast/sr_direct_030.pt \
        --lr path/to/720p/00000050.rgb.png path/to/720p/00000030.rgb.png
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from sr_model import DirectSRNet, inject_lora, load_lora_weights
from eval_sr import calc_psnr, calc_ssim, load_and_prepare


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Evaluate specific images')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--lr', nargs='+', required=True,
                        help='Full paths to LR (720p) images. HR found by replacing 720p with 1440p.')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the stitched comparison images')
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to LoRA weights file')
    args = parser.parse_args()

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    print(f'Loading model: {args.ckpt}')
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get('args', {})

    scale = ckpt_args.get('scale', 2)
    model = DirectSRNet(
        scale=scale,
        n_channels=ckpt_args.get('n_channels', 64),
        n_blocks=ckpt_args.get('n_blocks', 16),
        fast_tail=ckpt_args.get('fast_tail', False),
    )
    model.load_state_dict(ckpt['model'])

    # Inject LoRA if specified
    if args.lora:
        print(f'Loading LoRA: {args.lora}')
        inject_lora(model)
        load_lora_weights(model, args.lora)

    model = model.to(device)
    model.eval()

    print('\nImage | PSNR (Bic) | PSNR (SR) | Δ PSNR | SSIM (Bic) | SSIM (SR) | Δ SSIM')
    print('-' * 95)

    for lr_path in args.lr:
        # Find the matching HR by swapping 720p → 1440p
        hr_path = lr_path.replace('720p', '1440p')

        if not os.path.exists(lr_path):
            print(f'LR not found: {lr_path}')
            continue
        if not os.path.exists(hr_path):
            print(f'HR not found: {hr_path}')
            continue

        lr_tensor, (lr_w, lr_h) = load_and_prepare(lr_path)
        hr_tensor, (hr_w, hr_h) = load_and_prepare(hr_path)

        expected_h, expected_w = lr_h * scale, lr_w * scale
        hr_tensor = TF.resize(hr_tensor.squeeze(0), [expected_h, expected_w],
                              antialias=True).unsqueeze(0)

        bicubic = F.interpolate(
            lr_tensor, size=(expected_h, expected_w),
            mode='bicubic', align_corners=False
        )

        sr_output = model(lr_tensor.to(device)).cpu()

        psnr_bic = calc_psnr(bicubic, hr_tensor)
        psnr_sr = calc_psnr(sr_output, hr_tensor)
        ssim_bic = calc_ssim(bicubic, hr_tensor)
        ssim_sr = calc_ssim(sr_output, hr_tensor)

        delta_psnr = psnr_sr - psnr_bic
        delta_ssim = ssim_sr - ssim_bic

        name = os.path.basename(lr_path)
        # Include parent dirs for context
        parts = lr_path.split(os.sep)
        short = os.sep.join(parts[-4:]) if len(parts) >= 4 else lr_path

        print(f'{short:40s} | {psnr_bic:10.2f} | {psnr_sr:9.2f} | {delta_psnr:+6.2f} | '
              f'{ssim_bic:10.4f} | {ssim_sr:8.4f} | {delta_ssim:+7.4f}')

        if args.output_dir:
            comparison = [bicubic.squeeze(0), sr_output.squeeze(0), hr_tensor.squeeze(0)]
            safe_name = short.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(args.output_dir, f'delta{delta_psnr:+.2f}_{safe_name}')
            save_image(comparison, save_path, nrow=3, padding=0,
                       normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    main()
