"""
Diagnostic script: isolates VQ-VAE quality from Prior quality.
Produces 4 images side-by-side:
  1. Original HR image (ground truth)
  2. VQ-VAE reconstruction of HR (encode→decode, tests codebook quality)
  3. VQ-VAE decode of PREDICTED codes (tests prior quality)
  4. Original LR image (the input condition)
"""

import argparse
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torchvision.utils import save_image
from vqvae import VQVAE
from resnet_prior import ResNetPrior
from sample import load_model, sample_model, load_lr_top_condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vqvae', type=str, required=True)
    parser.add_argument('--top', type=str, required=True)
    parser.add_argument('--bottom', type=str, required=True)
    parser.add_argument('--lr_image', type=str, required=True, help='720p image path')
    parser.add_argument('--hr_image', type=str, required=True, help='Matching 1440p image path')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--temp', type=float, default=0.01)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr_size = args.size // args.scale  # 128

    # --- Load models ---
    model_vqvae, _ = load_model('vqvae', args.vqvae, device)
    model_top, top_args = load_model('pixelsnail_top', args.top, device)
    model_bottom, _ = load_model('pixelsnail_bottom', args.bottom, device)

    # --- Load and preprocess images ---
    hr_img = Image.open(args.hr_image).convert('RGB')
    hr_img = TF.center_crop(hr_img, [args.size, args.size])
    hr_tensor = TF.normalize(TF.to_tensor(hr_img), [0.5]*3, [0.5]*3).unsqueeze(0).to(device)

    lr_img = Image.open(args.lr_image).convert('RGB')
    lr_img = TF.center_crop(lr_img, [lr_size, lr_size])
    lr_tensor = TF.normalize(TF.to_tensor(lr_img), [0.5]*3, [0.5]*3).unsqueeze(0).to(device)

    with torch.no_grad():
        # ============================================
        # TEST 1: VQ-VAE reconstruction of HR image
        # This tests ONLY the codebook quality.
        # If this looks bad → VQ-VAE needs more training.
        # ============================================
        hr_recon, _ = model_vqvae(hr_tensor)
        hr_recon = hr_recon.clamp(-1, 1)

        # ============================================
        # TEST 2: Decode using GROUND-TRUTH codes
        # This shows the best the VQ-VAE can do.
        # ============================================
        _, _, _, gt_top, gt_bottom = model_vqvae.encode(hr_tensor)
        gt_decoded = model_vqvae.decode_code(gt_top, gt_bottom)
        gt_decoded = gt_decoded.clamp(-1, 1)

        # ============================================
        # TEST 3: Decode using PREDICTED codes (from priors)
        # This is what sample.py produces.
        # ============================================
        _, _, _, lr_top, _ = model_vqvae.encode(lr_tensor)
        lr_top = lr_top.long()

        pred_top = sample_model(model_top, device, 1, [32, 32], args.temp, condition=lr_top)
        pred_bottom = sample_model(model_bottom, device, 1, [64, 64], args.temp, condition=pred_top)
        pred_decoded = model_vqvae.decode_code(pred_top, pred_bottom)
        pred_decoded = pred_decoded.clamp(-1, 1)

        # ============================================
        # TEST 4: Accuracy — how many codes did the prior get right?
        # ============================================
        gt_top_long = gt_top.long()
        gt_bottom_long = gt_bottom.long()
        top_acc = (pred_top == gt_top_long).float().mean().item()
        bottom_acc = (pred_bottom == gt_bottom_long).float().mean().item()

        print(f"\n{'='*50}")
        print(f"DIAGNOSTIC RESULTS")
        print(f"{'='*50}")
        print(f"Top prior accuracy (vs ground truth):    {top_acc:.2%}")
        print(f"Bottom prior accuracy (vs ground truth): {bottom_acc:.2%}")
        print(f"HR reconstruction MSE:                   {((hr_recon - hr_tensor)**2).mean().item():.5f}")
        print(f"Predicted SR MSE (vs HR):                {((pred_decoded - hr_tensor)**2).mean().item():.5f}")
        print(f"{'='*50}\n")

    # Upsample LR for visual comparison (nearest neighbor, just for display)
    lr_display = torch.nn.functional.interpolate(lr_tensor, size=(args.size, args.size), mode='nearest')

    # Save all 4 side-by-side:
    # [Original HR] [VQ-VAE Recon] [Prior SR Output] [LR Input]
    grid = torch.cat([hr_tensor, hr_recon, gt_decoded, pred_decoded, lr_display], dim=0)
    save_image(grid, 'diagnostic_result.png', nrow=5, normalize=True, value_range=(-1, 1))
    print("Saved diagnostic_result.png")
    print("Layout: [HR Original] [VQ-VAE Recon] [GT Codes Decoded] [Predicted SR] [LR Input]")


if __name__ == '__main__':
    main()
