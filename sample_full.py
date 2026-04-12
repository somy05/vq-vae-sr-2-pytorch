"""
Full-image super-resolution in a single forward pass.
No patching needed — fully convolutional pipeline.

Also benchmarks FPS for real-time performance evaluation.
"""

import argparse
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

from vqvae import VQVAE
from sample import load_model, sample_model


@torch.no_grad()
def sr_full_image(lr_tensor, vqvae, model_top, model_bottom, device, scale=2, temp=0.1):
    """
    Super-resolve a full LR image tensor in a single forward pass.

    Args:
        lr_tensor: [1, 3, H, W] normalized tensor
        vqvae: VQ-VAE model
        model_top: top prior model
        model_bottom: bottom prior model
        device: torch device
        scale: upscaling factor
        temp: sampling temperature

    Returns:
        [1, 3, H*scale, W*scale] super-resolved tensor
    """
    timings = {}

    with torch.cuda.amp.autocast():
        # Step 1: Encode LR image to get LR top codes
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _, _, _, lr_top, _ = vqvae.encode(lr_tensor)
        lr_top = lr_top.long()
        torch.cuda.synchronize()
        timings['encode'] = (time.perf_counter() - t0) * 1000

        lr_h, lr_w = lr_top.shape[-2:]
        hr_top_size = [lr_h * scale, lr_w * scale]
        hr_bottom_size = [lr_h * scale * 2, lr_w * scale * 2]

        # Step 2: Top prior
        t0 = time.perf_counter()
        pred_top = sample_model(model_top, device, 1, hr_top_size, temp, condition=lr_top)
        torch.cuda.synchronize()
        timings['top_prior'] = (time.perf_counter() - t0) * 1000

        # Step 3: Bottom prior -> REPLACED WITH HYBRID APPROACH
        t0 = time.perf_counter()
        
        # Up-sample LR image to HR size using bicubic
        b, c, h, w = lr_tensor.shape
        bicubic_hr = F.interpolate(lr_tensor, size=(h * scale, w * scale), mode='bicubic', align_corners=False)
        
        # Encode the bicubic image to get reliable bottom texture codes
        _, _, _, _, bicubic_bottom = vqvae.encode(bicubic_hr)
        pred_bottom = bicubic_bottom.long()
        
        torch.cuda.synchronize()
        timings['bottom_hybrid'] = (time.perf_counter() - t0) * 1000

        # Step 4: Decode
        t0 = time.perf_counter()
        decoded = vqvae.decode_code(pred_top, pred_bottom)
        decoded = decoded.float().clamp(-1, 1)
        torch.cuda.synchronize()
        timings['decode'] = (time.perf_counter() - t0) * 1000

    return decoded, timings


def benchmark(lr_tensor, vqvae, model_top, model_bottom, device, scale=2,
              temp=0.1, warmup=3, runs=20):
    """Benchmark FPS with warmup."""
    print(f'\nBenchmarking ({warmup} warmup + {runs} timed runs)...')

    # Warmup
    for _ in range(warmup):
        _, _ = sr_full_image(lr_tensor, vqvae, model_top, model_bottom, device, scale, temp)
        torch.cuda.synchronize()

    # Timed runs
    times = []
    all_timings = {}
    for i in range(runs):
        torch.cuda.synchronize()
        start = time.perf_counter()

        _, timings = sr_full_image(lr_tensor, vqvae, model_top, model_bottom, device, scale, temp)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        for k, v in timings.items():
            all_timings.setdefault(k, []).append(v)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    fps = 1000.0 / avg_ms

    print(f'\n{"="*50}')
    print(f'BENCHMARK RESULTS ({lr_tensor.shape[-1]}×{lr_tensor.shape[-2]} → '
          f'{lr_tensor.shape[-1]*scale}×{lr_tensor.shape[-2]*scale})')
    print(f'{"="*50}')
    print(f'Average: {avg_ms:.1f} ms  ({fps:.1f} FPS)')
    print(f'Min:     {min_ms:.1f} ms  ({1000/min_ms:.1f} FPS)')
    print(f'Max:     {max_ms:.1f} ms  ({1000/max_ms:.1f} FPS)')
    print(f'\nPer-step breakdown (avg):')
    for k, v in all_timings.items():
        avg = sum(v) / len(v)
        pct = avg / avg_ms * 100
        print(f'  {k:15s}: {avg:6.1f} ms  ({pct:4.1f}%)')
    print(f'{"="*50}')

    return avg_ms, fps


def main():
    parser = argparse.ArgumentParser(description='Full-image SR (single pass)')
    parser.add_argument('--vqvae', type=str, required=True)
    parser.add_argument('--top', type=str, required=True)
    parser.add_argument('--bottom', type=str, required=True)
    parser.add_argument('--lr_image', type=str, required=True)
    parser.add_argument('--hr_image', type=str, default=None,
                        help='Optional HR ground truth for comparison')
    parser.add_argument('--temp', type=float, default=0.1)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--output', type=str, default='sr_full.png')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run FPS benchmark')
    parser.add_argument('--benchmark_runs', type=int, default=20)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load models
    print('Loading models...')
    vqvae, _ = load_model('vqvae', args.vqvae, device)
    model_top, _ = load_model('pixelsnail_top', args.top, device)
    model_bottom, _ = load_model('pixelsnail_bottom', args.bottom, device)

    # Load and prepare LR image
    lr_img = Image.open(args.lr_image).convert('RGB')
    lr_w, lr_h = lr_img.size
    hr_w, hr_h = lr_w * args.scale, lr_h * args.scale

    # Pad to nearest multiple of 8*scale (needed for VQ-VAE encoder downsampling)
    pad_multiple = 8 * args.scale
    pad_h = (pad_multiple - lr_h % pad_multiple) % pad_multiple
    pad_w = (pad_multiple - lr_w % pad_multiple) % pad_multiple

    lr_tensor = TF.normalize(TF.to_tensor(lr_img), [0.5]*3, [0.5]*3)
    if pad_h > 0 or pad_w > 0:
        lr_tensor = F.pad(lr_tensor, [0, pad_w, 0, pad_h], mode='reflect')
    lr_tensor = lr_tensor.unsqueeze(0).to(device)

    print(f'LR: {lr_w}×{lr_h} (padded to {lr_tensor.shape[-1]}×{lr_tensor.shape[-2]})')
    print(f'HR output: {hr_w}×{hr_h}')

    # Run SR
    print('\nRunning super-resolution...')
    start = time.perf_counter()
    sr_output, timings = sr_full_image(lr_tensor, vqvae, model_top, model_bottom,
                               device, args.scale, args.temp)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f'Done in {elapsed*1000:.1f} ms')
    for k, v in timings.items():
        print(f'  {k}: {v:.1f} ms')

    # Crop padding from output
    sr_output = sr_output[:, :, :hr_h, :hr_w]

    # Build comparison image
    images = []

    # Bicubic upscale for comparison
    lr_orig = TF.normalize(TF.to_tensor(lr_img), [0.5]*3, [0.5]*3).unsqueeze(0)
    lr_bicubic = F.interpolate(lr_orig, size=(hr_h, hr_w),
                                mode='bicubic', align_corners=False)
    images.append(lr_bicubic.squeeze(0))
    images.append(sr_output.cpu().squeeze(0))

    # Add ground truth if provided
    if args.hr_image:
        hr_gt = Image.open(args.hr_image).convert('RGB')
        hr_gt = TF.center_crop(hr_gt, [hr_h, hr_w])
        hr_gt_tensor = TF.normalize(TF.to_tensor(hr_gt), [0.5]*3, [0.5]*3)
        images.append(hr_gt_tensor)

    # Save comparison
    save_image(images, args.output, nrow=len(images),
               normalize=True, value_range=(-1, 1))

    # Save SR-only result
    sr_only_path = args.output.replace('.png', '_only.png')
    save_image(sr_output, sr_only_path, normalize=True, value_range=(-1, 1))

    labels = ['Bicubic', 'SR (ours)']
    if args.hr_image:
        labels.append('Ground Truth')
    print(f'\nSaved comparison ({" | ".join(labels)}): {args.output}')
    print(f'Saved SR only: {sr_only_path}')

    # Benchmark if requested
    if args.benchmark:
        benchmark(lr_tensor, vqvae, model_top, model_bottom, device,
                  args.scale, args.temp, runs=args.benchmark_runs)


if __name__ == '__main__':
    main()
