import argparse
import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image

from sr_model import DirectSRNet, inject_lora, load_lora_weights
from eval_sr import load_and_prepare, calc_psnr, calc_ssim


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description='Compare Base vs LoRA')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to base model checkpoint')
    parser.add_argument('--lora', type=str, required=True, help='Path to LoRA weights file')
    parser.add_argument('--lr', nargs='+', required=True, help='Paths to LR (720p) images')
    parser.add_argument('--output_dir', type=str, default='lora_comparison', help='Where to save images')
    parser.add_argument('--top_n', type=int, default=5, help='Save top N images with best LoRA improvement')
    parser.add_argument('--sample_n', type=int, default=None, help='Randomly sample N images instead of testing all')
    args = parser.parse_args()

    # Randomly sample if requested
    if args.sample_n and args.sample_n < len(args.lr):
        args.lr = random.sample(args.lr, args.sample_n)

    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    # 1. Load Base Model
    print(f'Loading base model: {args.ckpt}')
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

    # 2. Load LoRA Model
    print(f'Loading LoRA model: {args.lora}')
    lora_model = DirectSRNet(
        scale=scale,
        n_channels=ckpt_args.get('n_channels', 64),
        n_blocks=ckpt_args.get('n_blocks', 16),
        fast_tail=ckpt_args.get('fast_tail', False),
    )
    lora_model.load_state_dict(ckpt['model'])
    inject_lora(lora_model)
    load_lora_weights(lora_model, args.lora)
    lora_model = lora_model.to(device)
    lora_model.eval()

    # Lightweight metrics (no tensors) for summary
    all_metrics = []
    # Only keep top_n tensor sets in memory (sorted by delta PSNR)
    import heapq
    top_heap = []  # min-heap of (delta, index, tensor_dict)

    print(f'\nProcessing {len(args.lr)} images...\n')
    print(f'{"Image":>40s} | {"Base PSNR":>10s} | {"LoRA PSNR":>10s} | {"Δ PSNR":>8s} | {"Base SSIM":>10s} | {"LoRA SSIM":>10s} | {"Δ SSIM":>8s}')
    print('-' * 120)

    for idx, lr_path in enumerate(args.lr):
        hr_path = lr_path.replace('720p', '1440p')

        if not os.path.exists(lr_path) or not os.path.exists(hr_path):
            print(f'Skipping, files not found for: {lr_path}')
            continue

        lr_tensor, (lr_w, lr_h) = load_and_prepare(lr_path)
        hr_tensor, (hr_w, hr_h) = load_and_prepare(hr_path)

        expected_h, expected_w = lr_h * scale, lr_w * scale
        hr_tensor = TF.resize(hr_tensor.squeeze(0), [expected_h, expected_w], antialias=True).unsqueeze(0)

        # A. Bicubic
        bicubic = F.interpolate(lr_tensor, size=(expected_h, expected_w), mode='bicubic', align_corners=False)

        # B. Base SR
        base_sr = model(lr_tensor.to(device)).cpu()

        # C. LoRA SR
        lora_sr = lora_model(lr_tensor.to(device)).cpu()

        # Metrics
        psnr_bic = calc_psnr(bicubic, hr_tensor)
        psnr_base = calc_psnr(base_sr, hr_tensor)
        psnr_lora = calc_psnr(lora_sr, hr_tensor)
        delta = psnr_lora - psnr_base
        ssim_bic = calc_ssim(bicubic, hr_tensor)
        ssim_base = calc_ssim(base_sr, hr_tensor)
        ssim_lora = calc_ssim(lora_sr, hr_tensor)
        delta_ssim = ssim_lora - ssim_base

        name = os.path.basename(lr_path)
        print(f'{name:>40s} | {psnr_base:10.2f} | {psnr_lora:10.2f} | {delta:+8.2f} | {ssim_base:10.4f} | {ssim_lora:10.4f} | {delta_ssim:+8.4f}')

        # Store lightweight metrics only
        all_metrics.append({
            'name': name,
            'psnr_bic': psnr_bic, 'psnr_base': psnr_base, 'psnr_lora': psnr_lora, 'delta': delta,
            'ssim_bic': ssim_bic, 'ssim_base': ssim_base, 'ssim_lora': ssim_lora, 'delta_ssim': delta_ssim,
        })

        # Keep only top_n tensors in memory using a min-heap
        entry = (delta, idx, {
            'name': name, 'delta': delta,
            'bicubic': bicubic, 'base_sr': base_sr,
            'lora_sr': lora_sr, 'hr': hr_tensor,
        })
        if len(top_heap) < args.top_n:
            heapq.heappush(top_heap, entry)
        elif delta > top_heap[0][0]:
            heapq.heapreplace(top_heap, entry)

    if not all_metrics:
        print('No images processed.')
        return

    # ── Summary ──
    n = len(all_metrics)
    avg_psnr_bic  = sum(r['psnr_bic']  for r in all_metrics) / n
    avg_psnr_base = sum(r['psnr_base'] for r in all_metrics) / n
    avg_psnr_lora = sum(r['psnr_lora'] for r in all_metrics) / n
    avg_ssim_bic  = sum(r['ssim_bic']  for r in all_metrics) / n
    avg_ssim_base = sum(r['ssim_base'] for r in all_metrics) / n
    avg_ssim_lora = sum(r['ssim_lora'] for r in all_metrics) / n

    print(f'\n{"=" * 78}')
    print(f'TEST SET SUMMARY ({n} images)')
    print(f'{"=" * 78}')
    print(f'{"":>20s} | {"PSNR (dB)":>12s} | {"SSIM":>10s}')
    print(f'  {"Bicubic":>18s} | {avg_psnr_bic:12.3f} | {avg_ssim_bic:10.4f}')
    print(f'  {"Base (no LoRA)":>18s} | {avg_psnr_base:12.3f} | {avg_ssim_base:10.4f}')
    print(f'  {"Base + LoRA":>18s} | {avg_psnr_lora:12.3f} | {avg_ssim_lora:10.4f}')
    print(f'  {"─" * 50}')
    print(f'  {"Δ Base vs Bicubic":>18s} | {avg_psnr_base - avg_psnr_bic:+12.3f} | {avg_ssim_base - avg_ssim_bic:+10.4f}')
    print(f'  {"Δ LoRA vs Base":>18s} | {avg_psnr_lora - avg_psnr_base:+12.3f} | {avg_ssim_lora - avg_ssim_base:+10.4f}')
    print(f'  {"Δ LoRA vs Bicubic":>18s} | {avg_psnr_lora - avg_psnr_bic:+12.3f} | {avg_ssim_lora - avg_ssim_bic:+10.4f}')
    print(f'{"=" * 78}')

    # ── Save top N by biggest LoRA improvement ──
    top_sorted = sorted(top_heap, key=lambda x: x[0], reverse=True)

    print(f'\nSaving top {len(top_sorted)} images with best LoRA improvement...\n')

    for i, (delta, idx, r) in enumerate(top_sorted):
        comparison = [
            r['bicubic'].squeeze(0),
            r['base_sr'].squeeze(0),
            r['lora_sr'].squeeze(0),
            r['hr'].squeeze(0),
        ]

        save_name = f"top{i+1}_delta{r['delta']:+.2f}_{r['name']}"
        save_path = os.path.join(args.output_dir, save_name)
        save_image(comparison, save_path, nrow=4, padding=0, normalize=True, value_range=(-1, 1))
        print(f'  #{i+1} | Δ{r["delta"]:+.2f} dB | {r["name"]} → {save_path}')

    print(f'\n✓ Done! Results saved to: {args.output_dir}')


if __name__ == '__main__':
    main()
