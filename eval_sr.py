"""
Evaluation and benchmarking for Direct SR.

Computes PSNR, runs FPS benchmark, and saves visual comparisons.

Usage:
    python eval_sr.py \
        --ckpt checkpoint/sr_direct_200.pt \
        --lr_image path/to/720p.png \
        --hr_image path/to/1440p.png \
        --benchmark
"""

import argparse
import math
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

from sr_model import DirectSRNet


def calc_psnr(pred, target):
    """PSNR for images in [-1, 1] range, reported on [0, 255] scale."""
    pred_255 = ((pred + 1) / 2 * 255).clamp(0, 255)
    target_255 = ((target + 1) / 2 * 255).clamp(0, 255)
    mse = F.mse_loss(pred_255, target_255)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255 ** 2 / mse.item())


@torch.no_grad()
def sr_image(model, lr_tensor, device):
    """Run SR on a single image."""
    lr_tensor = lr_tensor.to(device)
    with torch.cuda.amp.autocast():
        sr = model(lr_tensor)
    return sr


def benchmark(model, lr_tensor, device, warmup=5, runs=50):
    """FPS benchmark with warmup."""
    print(f'\nBenchmarking ({warmup} warmup + {runs} timed runs)...')

    for _ in range(warmup):
        _ = sr_image(model, lr_tensor, device)
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()
        _ = sr_image(model, lr_tensor, device)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    fps = 1000.0 / avg_ms

    return avg_ms, min_ms, max_ms, fps


def main():
    parser = argparse.ArgumentParser(description='Evaluate Direct SR')
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--lr_image', type=str, required=True)
    parser.add_argument('--hr_image', type=str, default=None)
    parser.add_argument('--output', type=str, default='sr_direct_result.png')
    parser.add_argument('--do_benchmark', action='store_true')
    parser.add_argument('--benchmark_runs', type=int, default=50)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get('args', {})

    model = DirectSRNet(
        scale=ckpt_args.get('scale', 2),
        n_channels=ckpt_args.get('n_channels', 64),
        n_blocks=ckpt_args.get('n_blocks', 16),
    )
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    scale = ckpt_args.get('scale', 2)
    print(f'Model: {n_params:,} params | Scale: {scale}×')

    # Load LR image
    lr_img = Image.open(args.lr_image).convert('RGB')
    lr_w, lr_h = lr_img.size
    hr_w, hr_h = lr_w * scale, lr_h * scale

    lr_tensor = TF.normalize(TF.to_tensor(lr_img), [0.5] * 3, [0.5] * 3)
    lr_tensor = lr_tensor.unsqueeze(0)

    print(f'LR: {lr_w}×{lr_h} → HR: {hr_w}×{hr_h}')

    # Run SR
    start = time.perf_counter()
    sr_output = sr_image(model, lr_tensor, device)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    print(f'SR done in {elapsed:.1f} ms')

    # Build comparison
    images = []

    # Bicubic
    bicubic = F.interpolate(
        lr_tensor, size=(hr_h, hr_w), mode='bicubic', align_corners=False
    )
    images.append(bicubic.squeeze(0))
    images.append(sr_output.cpu().squeeze(0))

    # PSNR vs bicubic
    if args.hr_image:
        hr_gt = Image.open(args.hr_image).convert('RGB')
        hr_gt = TF.center_crop(hr_gt, [hr_h, hr_w])
        hr_gt_tensor = TF.normalize(TF.to_tensor(hr_gt), [0.5] * 3, [0.5] * 3)
        hr_gt_tensor = hr_gt_tensor.unsqueeze(0)

        psnr_bicubic = calc_psnr(bicubic, hr_gt_tensor)
        psnr_sr = calc_psnr(sr_output.cpu(), hr_gt_tensor)

        print(f'\nPSNR:')
        print(f'  Bicubic:   {psnr_bicubic:.2f} dB')
        print(f'  SR (ours): {psnr_sr:.2f} dB')
        print(f'  Δ:         {psnr_sr - psnr_bicubic:+.2f} dB')

        images.append(hr_gt_tensor.squeeze(0))

    # Save
    save_image(images, args.output, nrow=len(images),
               normalize=True, value_range=(-1, 1))

    sr_only = args.output.replace('.png', '_only.png')
    save_image(sr_output, sr_only, normalize=True, value_range=(-1, 1))

    labels = ['Bicubic', 'SR (ours)']
    if args.hr_image:
        labels.append('Ground Truth')
    print(f'\nSaved: {args.output} ({" | ".join(labels)})')

    # Benchmark
    if args.do_benchmark:
        avg_ms, min_ms, max_ms, fps = benchmark(
            model, lr_tensor, device, runs=args.benchmark_runs
        )
        print(f'\n{"=" * 50}')
        print(f'BENCHMARK ({lr_w}×{lr_h} → {hr_w}×{hr_h})')
        print(f'{"=" * 50}')
        print(f'Average: {avg_ms:.1f} ms  ({fps:.1f} FPS)')
        print(f'Min:     {min_ms:.1f} ms  ({1000 / min_ms:.1f} FPS)')
        print(f'Max:     {max_ms:.1f} ms  ({1000 / max_ms:.1f} FPS)')
        print(f'{"=" * 50}')


if __name__ == '__main__':
    main()
