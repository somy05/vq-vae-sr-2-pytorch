"""
Evaluate specific images quickly to get their exact PSNR and SSIM.
Useful if you lost the terminal output but still know which images you want to test.

Usage:
    python eval_specific.py \
        --ckpt checkpoint_fast/sr_direct_030.pt \
        --test_root ../mini_dataset/mini_dataset/test/GameIR-SR/GameIR-SR \
        --images "static_town08/00/00000030.rgb.png" "dynamic_town05/19/00000050.rgb.png"
"""

import argparse
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from sr_model import DirectSRNet
from eval_sr import calc_psnr, calc_ssim, load_and_prepare


@torch.no_grad()
def sr_image(model, lr_tensor, device):
    """Run model on LR tensor."""
    lr_tensor = lr_tensor.to(device)
    output = model(lr_tensor)
    return output


def main():
    parser = argparse.ArgumentParser(description='Evaluate specific images')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--test_root', type=str, required=True,
                        help='GameIR test dataset root')
    parser.add_argument('--images', nargs='+', required=True,
                        help='List of relative image paths (e.g. static_town08/00/00000030.rgb.png)')
    args = parser.parse_args()

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
    model = model.to(device)
    model.eval()

    print('\nImage | PSNR (Bic) | PSNR (SR) | Δ PSNR | SSIM (Bic) | SSIM (SR) | Δ SSIM')
    print('-' * 85)

    for name in args.images:
        # Reconstruct paths
        # If the user copied the safe name with underscores (e.g. static_town08_00_00000030.rgb.png),
        # try to guess the slashes. Usually it's scene/sequence/file.
        if '_' in name and '/' not in name:
            parts = name.split('_')
            if len(parts) >= 4:
                # e.g. static_town08, 00, 00000030.rgb.png
                name = f"{parts[0]}_{parts[1]}/{parts[2]}/{'_'.join(parts[3:])}"

        # The HR image is in the 1440p folder, LR in 720p
        # Assumes name is like: static_town08/00/00000030.rgb.png
        parts = name.split('/')
        if len(parts) != 3:
            print(f'Skipping {name} (must be scene/seq/filename)')
            continue

        scene, seq, filename = parts
        hr_path = os.path.join(args.test_root, scene, seq, '1440p', filename)
        lr_path = os.path.join(args.test_root, scene, seq, '720p', filename)

        if not os.path.exists(hr_path) or not os.path.exists(lr_path):
            print(f'File not found for {name}')
            continue

        # Load and prep
        lr_tensor, (lr_w, lr_h) = load_and_prepare(lr_path)
        hr_tensor, (hr_w, hr_h) = load_and_prepare(hr_path)

        expected_h, expected_w = lr_h * scale, lr_w * scale
        hr_tensor = TF.resize(hr_tensor.squeeze(0), [expected_h, expected_w],
                              antialias=True).unsqueeze(0)

        # Baseline
        bicubic = F.interpolate(
            lr_tensor, size=(expected_h, expected_w),
            mode='bicubic', align_corners=False
        )

        # Predict
        sr_output = sr_image(model, lr_tensor, device).cpu()

        # Metrics
        psnr_bic = calc_psnr(bicubic, hr_tensor)
        psnr_sr = calc_psnr(sr_output, hr_tensor)
        ssim_bic = calc_ssim(bicubic, hr_tensor)
        ssim_sr = calc_ssim(sr_output, hr_tensor)
        
        delta_psnr = psnr_sr - psnr_bic
        delta_ssim = ssim_sr - ssim_bic

        # Print row
        safe_name = name.replace('/', '_')
        if len(safe_name) > 35:
            safe_name = safe_name[-35:]
            
        print(f'{safe_name:35s} | {psnr_bic:10.2f} | {psnr_sr:9.2f} | {delta_psnr:+6.2f} | '
              f'{ssim_bic:10.4f} | {ssim_sr:8.4f} | {delta_ssim:+7.4f}')


if __name__ == '__main__':
    main()
