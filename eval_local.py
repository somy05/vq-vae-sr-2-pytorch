import argparse
import glob
import math
import os
import time

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.utils import save_image
from PIL import Image

from sr_model import DirectSRNet, inject_lora, load_lora_weights



def get_device():
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(ckpt_path, device, half=False, compile_model=False):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    ckpt_args = ckpt.get('args', {})

    model = DirectSRNet(
        scale=ckpt_args.get('scale', 2),
        n_channels=ckpt_args.get('n_channels', 64),
        n_blocks=ckpt_args.get('n_blocks', 16),
        fast_tail=ckpt_args.get('fast_tail', False),
    )
    model.load_state_dict(ckpt['model'])
    model.eval()

    if half and device.type != 'cpu':
        model = model.half()
        print('FP16 half precision enabled')

    model = model.to(device)

    if compile_model:
        try:
            model = torch.compile(model)
            print('torch.compile() enabled')
        except Exception as e:
            print(f'torch.compile() failed: {e}')

    n_params = sum(p.numel() for p in model.parameters())
    scale = ckpt_args.get('scale', 2)
    fast_tail = ckpt_args.get('fast_tail', False)
    print(f'  Model: {n_params:,} params | Scale: {scale}× | fast_tail: {fast_tail}')

    return model, scale


def load_image(path, half=False, device='cpu'):
    img = Image.open(path).convert('RGB')
    tensor = TF.normalize(TF.to_tensor(img), [0.5] * 3, [0.5] * 3)
    tensor = tensor.unsqueeze(0)
    if half:
        tensor = tensor.half()
    return tensor.to(device), img.size  # (w, h)


@torch.no_grad()
def upscale(model, lr_tensor):
    return model(lr_tensor)


def find_images(path):
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}

    if os.path.isfile(path):
        return [path]

    images = []
    for ext in extensions:
        images.extend(glob.glob(os.path.join(path, f'*{ext}')))
        images.extend(glob.glob(os.path.join(path, f'*{ext.upper()}')))
    return sorted(set(images))



def export_coreml(model, input_shape, output_path='sr_fast.mlpackage'):
    try:
        import coremltools as ct
    except ImportError:
        print('install coremltools')
        return None

    print(f'\nExporting to CoreML...')
    print(f'  Input shape: {list(input_shape)}')

    model.coreml_export = True

    model_cpu = model.cpu().float()
    example = torch.randn(input_shape)
    traced = torch.jit.trace(model_cpu, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name='input', shape=input_shape)],
        convert_to='mlprogram',
        compute_precision=ct.precision.FLOAT16,
    )
    mlmodel.save(output_path)
    print(f'Saved: {output_path}')
    return mlmodel


def benchmark_coreml(mlmodel, input_shape, warmup=5, runs=50):

    import numpy as np

    dummy = np.random.randn(*input_shape).astype(np.float32)

    print(f'\nCoreML Benchmark ({warmup} warmup + {runs} timed runs)...')
    for _ in range(warmup):
        mlmodel.predict({'input': dummy})

    times = []
    for _ in range(runs):
        start = time.perf_counter()
        mlmodel.predict({'input': dummy})
        times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    fps = 1000.0 / avg_ms
    return avg_ms, min_ms, max_ms, fps




def benchmark_pytorch(model, lr_tensor, device, warmup=5, runs=50):
    print(f'\nPyTorch Benchmark ({warmup} warmup + {runs} timed runs)...')

    # Warmup
    for _ in range(warmup):
        _ = upscale(model, lr_tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

    times = []
    for _ in range(runs):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

        start = time.perf_counter()
        _ = upscale(model, lr_tensor)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        elif device.type == 'mps':
            torch.mps.synchronize()

        times.append(time.perf_counter() - start)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    max_ms = max(times) * 1000
    fps = 1000.0 / avg_ms
    return avg_ms, min_ms, max_ms, fps



def main():
    parser = argparse.ArgumentParser(
        description='Local SR evaluation — upscale any image on Mac'
    )
    parser.add_argument('--ckpt', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to image or folder of images')
    parser.add_argument('--output_dir', type=str, default='upscaled',
                        help='Output directory for upscaled images')

    parser.add_argument('--half', action='store_true',
                        help='Use FP16 half precision')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() JIT optimisation')
    parser.add_argument('--coreml', action='store_true',
                        help='Export to CoreML and benchmark')
    parser.add_argument('--lora', type=str, default=None,
                        help='Path to LoRA weights file')

    parser.add_argument('--benchmark', action='store_true',
                        help='Run speed benchmark')
    parser.add_argument('--benchmark_runs', type=int, default=50)

 
    parser.add_argument('--gt', type=str, default=None,
                        help='Optional ground truth HR image for PSNR calc')

    args = parser.parse_args()

    device = get_device()
    print(f'\n{"=" * 50}')
    print(f'Device: {device}')
    print(f'{"=" * 50}')

    # Load model
    print(f'\nLoading model: {args.ckpt}')
    model, scale = load_model(args.ckpt, device,
                              half=args.half,
                              compile_model=args.compile)

    if args.lora:
        print(f'\nLoading LoRA: {args.lora}')
        lora_ckpt = torch.load(args.lora, map_location='cpu')
        rank = lora_ckpt.get('rank', 4)
        alpha = lora_ckpt.get('alpha', 1.0)
        model = inject_lora(model, rank=rank, alpha=alpha)
        load_lora_weights(model, args.lora, device=device)
        model.eval()


    image_paths = find_images(args.input)
    if not image_paths:
        print(f'ERROR: No images found at {args.input}')
        return

    print(f'\nFound {len(image_paths)} image(s)')
    os.makedirs(args.output_dir, exist_ok=True)


    for img_path in image_paths:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        print(f'\n── {basename} ──')

        lr_tensor, (lr_w, lr_h) = load_image(img_path, half=args.half,
                                              device=device)
        hr_w, hr_h = lr_w * scale, lr_h * scale
        print(f'  {lr_w}×{lr_h} → {hr_w}×{hr_h}')

        
        start = time.perf_counter()
        sr_output = upscale(model, lr_tensor)

        if device.type == 'mps':
            torch.mps.synchronize()
        elif device.type == 'cuda':
            torch.cuda.synchronize()

        elapsed = (time.perf_counter() - start) * 1000
        print(f'  Upscaled in {elapsed:.1f} ms')

        
        sr_path = os.path.join(args.output_dir, f'{basename}_sr.png')
        save_image(sr_output, sr_path, normalize=True, value_range=(-1, 1))
        print(f'  Saved: {sr_path}')

        
        bicubic = F.interpolate(
            lr_tensor.float(), size=(hr_h, hr_w),
            mode='bicubic', align_corners=False
        )
        bic_path = os.path.join(args.output_dir, f'{basename}_bicubic.png')
        save_image(bicubic, bic_path, normalize=True, value_range=(-1, 1))

        
        comparison = [bicubic.cpu().float().squeeze(0),
                      sr_output.cpu().float().squeeze(0)]

        
        if args.gt:
            gt_tensor, _ = load_image(args.gt, device='cpu')
            gt_tensor = F.interpolate(gt_tensor, size=(hr_h, hr_w),
                                      mode='bicubic', align_corners=False)
            psnr_bic = calc_psnr(bicubic.cpu().float(), gt_tensor)
            psnr_sr = calc_psnr(sr_output.cpu().float(), gt_tensor)
            print(f'  PSNR — Bicubic: {psnr_bic:.2f} dB | SR: {psnr_sr:.2f} dB | '
                  f'Δ: {psnr_sr - psnr_bic:+.2f} dB')
            comparison.append(gt_tensor.squeeze(0))

        comp_path = os.path.join(args.output_dir, f'{basename}_comparison.png')
        save_image(comparison, comp_path, nrow=len(comparison),
                   normalize=True, value_range=(-1, 1))

    
    if args.benchmark:

        lr_tensor, (lr_w, lr_h) = load_image(image_paths[0], half=args.half,
                                              device=device)
        hr_w, hr_h = lr_w * scale, lr_h * scale

        avg_ms, min_ms, max_ms, fps = benchmark_pytorch(
            model, lr_tensor, device, runs=args.benchmark_runs
        )

        print(f'\n{"=" * 50}')
        print(f'PYTORCH BENCHMARK ({lr_w}×{lr_h} → {hr_w}×{hr_h})')
        print(f'{"=" * 50}')
        print(f'  Average: {avg_ms:.1f} ms  ({fps:.1f} FPS)')
        print(f'  Min:     {min_ms:.1f} ms  ({1000 / min_ms:.1f} FPS)')
        print(f'  Max:     {max_ms:.1f} ms  ({1000 / max_ms:.1f} FPS)')
        print(f'{"=" * 50}')


    if args.coreml:
        lr_tensor, (lr_w, lr_h) = load_image(image_paths[0])
        input_shape = (1, 3, lr_h, lr_w)

        coreml_path = os.path.join(args.output_dir,
                                   f'sr_model_{lr_w}x{lr_h}.mlpackage')
        mlmodel = export_coreml(model, input_shape, coreml_path)

        if mlmodel and args.benchmark:
            avg_ms, min_ms, max_ms, fps = benchmark_coreml(
                mlmodel, input_shape, runs=args.benchmark_runs
            )
            print(f'\n{"=" * 50}')
            print(f'COREML BENCHMARK ({lr_w}×{lr_h} → {lr_w*scale}×{lr_h*scale})')
            print(f'{"=" * 50}')
            print(f'  Average: {avg_ms:.1f} ms  ({fps:.1f} FPS)')
            print(f'  Min:     {min_ms:.1f} ms  ({1000 / min_ms:.1f} FPS)')
            print(f'  Max:     {max_ms:.1f} ms  ({1000 / max_ms:.1f} FPS)')
            print(f'{"=" * 50}')


def calc_psnr(pred, target):
    pred_255 = ((pred + 1) / 2 * 255).clamp(0, 255)
    target_255 = ((target + 1) / 2 * 255).clamp(0, 255)
    mse = F.mse_loss(pred_255, target_255)
    if mse == 0:
        return float('inf')
    return 10 * math.log10(255 ** 2 / mse.item())


if __name__ == '__main__':
    main()
