"""
Evaluation and benchmarking for Direct SR.

Supports two modes:
  1. Single-image: evaluate on one LR/HR pair with visual comparison.
  2. Batch dataset: evaluate over an entire GameIR test directory and
     compute average PSNR across all images.

Usage (single image):
    python eval_sr.py \
        --ckpt checkpoint/sr_direct_030.pt \
        --lr_image path/to/720p.png \
        --hr_image path/to/1440p.png \
        --do_benchmark

Usage (batch dataset):
    python eval_sr.py \
        --ckpt checkpoint/sr_direct_030.pt \
        --test_root path/to/GameIR-SR-mini-test/ \
        --do_benchmark
"""

import argparse
import math
import os
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


def calc_ssim(pred, target, window_size=11):
    """SSIM for images in [-1, 1] range.

    Uses a Gaussian-weighted sliding window approach.
    Returns a scalar SSIM value in [0, 1].
    """
    # Convert to [0, 1]
    pred = ((pred + 1) / 2).clamp(0, 1)
    target = ((target + 1) / 2).clamp(0, 1)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = g.unsqueeze(1) * g.unsqueeze(0)  # 2D Gaussian
    window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    window = window.expand(pred.size(1), 1, -1, -1).to(pred.device)

    pad = window_size // 2
    mu1 = F.conv2d(pred, window, padding=pad, groups=pred.size(1))
    mu2 = F.conv2d(target, window, padding=pad, groups=target.size(1))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=pad, groups=pred.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=pad, groups=target.size(1)) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=pad, groups=pred.size(1)) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean().item()


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


def discover_pairs(root, lr_res='720p', hr_res='1440p', suffix='_rgb.png'):
    """Discover all LR/HR image pairs in a GameIR directory tree."""
    pairs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if lr_res not in dirnames or hr_res not in dirnames:
            continue
        lr_dir = os.path.join(dirpath, lr_res)
        hr_dir = os.path.join(dirpath, hr_res)
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


def load_and_prepare(image_path, normalize=True):
    """Load an image and convert to tensor."""
    img = Image.open(image_path).convert('RGB')
    tensor = TF.to_tensor(img)
    if normalize:
        tensor = TF.normalize(tensor, [0.5] * 3, [0.5] * 3)
    return tensor.unsqueeze(0), img.size  # (w, h)


@torch.no_grad()
def eval_batch(model, pairs, device, scale, output_dir=None, save_samples=5):
    """Evaluate model on all image pairs, returning average PSNR and SSIM."""
    psnr_bicubic_list = []
    psnr_sr_list = []
    ssim_bicubic_list = []
    ssim_sr_list = []
    saved = 0

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    for i, (lr_path, hr_path, name) in enumerate(pairs):
        lr_tensor, (lr_w, lr_h) = load_and_prepare(lr_path)
        hr_tensor, (hr_w, hr_h) = load_and_prepare(hr_path)

        # Ensure HR matches expected size
        expected_h, expected_w = lr_h * scale, lr_w * scale
        hr_tensor = TF.resize(hr_tensor.squeeze(0), [expected_h, expected_w],
                              antialias=True).unsqueeze(0)

        # Bicubic baseline
        bicubic = F.interpolate(
            lr_tensor, size=(expected_h, expected_w),
            mode='bicubic', align_corners=False
        )

        # SR output
        sr_output = sr_image(model, lr_tensor, device).cpu()

        # Compute PSNR
        psnr_bic = calc_psnr(bicubic, hr_tensor)
        psnr_sr = calc_psnr(sr_output, hr_tensor)
        psnr_bicubic_list.append(psnr_bic)
        psnr_sr_list.append(psnr_sr)

        # Compute SSIM
        ssim_bic = calc_ssim(bicubic, hr_tensor)
        ssim_sr = calc_ssim(sr_output, hr_tensor)
        ssim_bicubic_list.append(ssim_bic)
        ssim_sr_list.append(ssim_sr)

        # Progress
        delta = psnr_sr - psnr_bic
        print(f'  [{i+1:4d}/{len(pairs)}] {name:50s}  '
              f'PSNR: {psnr_sr:.2f} dB (Δ{delta:+.2f})  '
              f'SSIM: {ssim_sr:.4f}')

        # Save a few sample comparisons
        if output_dir and saved < save_samples:
            comparison = [bicubic.squeeze(0), sr_output.squeeze(0),
                          hr_tensor.squeeze(0)]
            safe_name = name.replace('/', '_').replace('\\', '_')
            save_path = os.path.join(output_dir, f'sample_{safe_name}')
            save_image(comparison, save_path, nrow=3,
                       normalize=True, value_range=(-1, 1))
            saved += 1

    avg_bic_psnr = sum(psnr_bicubic_list) / len(psnr_bicubic_list)
    avg_sr_psnr = sum(psnr_sr_list) / len(psnr_sr_list)
    avg_bic_ssim = sum(ssim_bicubic_list) / len(ssim_bicubic_list)
    avg_sr_ssim = sum(ssim_sr_list) / len(ssim_sr_list)

    return (avg_bic_psnr, avg_sr_psnr, psnr_bicubic_list, psnr_sr_list,
            avg_bic_ssim, avg_sr_ssim, ssim_bicubic_list, ssim_sr_list)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Direct SR')
    parser.add_argument('--ckpt', type=str, required=True)

    # Single-image mode
    parser.add_argument('--lr_image', type=str, default=None)
    parser.add_argument('--hr_image', type=str, default=None)

    # Batch dataset mode
    parser.add_argument('--test_root', type=str, default=None,
                        help='GameIR test dataset root for batch evaluation')
    parser.add_argument('--lr_res', type=str, default='720p')
    parser.add_argument('--hr_res', type=str, default='1440p')
    parser.add_argument('--suffix', type=str, default='_rgb.png')
    parser.add_argument('--save_samples', type=int, default=5,
                        help='Number of sample comparisons to save')

    parser.add_argument('--output', type=str, default='sr_direct_result.png')
    parser.add_argument('--output_dir', type=str, default='eval_results',
                        help='Directory for batch evaluation outputs')
    parser.add_argument('--do_benchmark', action='store_true')
    parser.add_argument('--benchmark_runs', type=int, default=50)
    args = parser.parse_args()

    if not args.lr_image and not args.test_root:
        parser.error('Provide either --lr_image (single) or --test_root (batch)')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device)
    ckpt_args = ckpt.get('args', {})

    model = DirectSRNet(
        scale=ckpt_args.get('scale', 2),
        n_channels=ckpt_args.get('n_channels', 64),
        n_blocks=ckpt_args.get('n_blocks', 16),
        fast_tail=ckpt_args.get('fast_tail', False),
    )
    model.load_state_dict(ckpt['model'])
    model = model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    scale = ckpt_args.get('scale', 2)
    fast_tail = ckpt_args.get('fast_tail', False)
    print(f'Model: {n_params:,} params | Scale: {scale}× | fast_tail: {fast_tail}')

    # ── Batch dataset mode ──────────────────────────────────────────
    if args.test_root:
        pairs = discover_pairs(args.test_root, args.lr_res, args.hr_res,
                               args.suffix)
        if not pairs:
            print(f'ERROR: No image pairs found in {args.test_root}')
            print(f'  Looking for folders with both {args.lr_res}/ and '
                  f'{args.hr_res}/ subfolders,')
            print(f'  containing files ending with "{args.suffix}"')
            return

        print(f'\nFound {len(pairs)} image pairs in test set')
        print(f'{"=" * 70}')

        (avg_bic_psnr, avg_sr_psnr, all_bic, all_sr,
         avg_bic_ssim, avg_sr_ssim, all_bic_ssim, all_sr_ssim) = eval_batch(
            model, pairs, device, scale,
            output_dir=args.output_dir,
            save_samples=args.save_samples,
        )

        print(f'\n{"=" * 70}')
        print(f'RESULTS ({len(pairs)} images)')
        print(f'{"=" * 70}')
        print(f'  Average PSNR (Bicubic):  {avg_bic_psnr:.2f} dB')
        print(f'  Average PSNR (SR ours):  {avg_sr_psnr:.2f} dB')
        print(f'  Average PSNR Δ:          {avg_sr_psnr - avg_bic_psnr:+.2f} dB')
        print(f'  Average SSIM (Bicubic):  {avg_bic_ssim:.4f}')
        print(f'  Average SSIM (SR ours):  {avg_sr_ssim:.4f}')
        print(f'  Average SSIM Δ:          {avg_sr_ssim - avg_bic_ssim:+.4f}')
        print(f'{"=" * 70}')

        # Per-image stats
        deltas = [s - b for s, b in zip(all_sr, all_bic)]
        print(f'\n  Best  PSNR Δ: {max(deltas):+.2f} dB')
        print(f'  Worst PSNR Δ: {min(deltas):+.2f} dB')
        improved = sum(1 for d in deltas if d > 0)
        print(f'  Improved: {improved}/{len(deltas)} images '
              f'({100*improved/len(deltas):.1f}%)')

        if args.save_samples > 0:
            print(f'\n  Sample comparisons saved to: {args.output_dir}/')

        # Benchmark on first image
        if args.do_benchmark and pairs:
            lr_tensor, _ = load_and_prepare(pairs[0][0])
            lr_w, lr_h = Image.open(pairs[0][0]).size
            hr_w, hr_h = lr_w * scale, lr_h * scale
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

        return

    # ── Single-image mode (original) ────────────────────────────────
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
        ssim_bicubic = calc_ssim(bicubic, hr_gt_tensor)
        ssim_sr = calc_ssim(sr_output.cpu(), hr_gt_tensor)

        print(f'\nPSNR:')
        print(f'  Bicubic:   {psnr_bicubic:.2f} dB')
        print(f'  SR (ours): {psnr_sr:.2f} dB')
        print(f'  Δ:         {psnr_sr - psnr_bicubic:+.2f} dB')
        print(f'\nSSIM:')
        print(f'  Bicubic:   {ssim_bicubic:.4f}')
        print(f'  SR (ours): {ssim_sr:.4f}')
        print(f'  Δ:         {ssim_sr - ssim_bicubic:+.4f}')

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
