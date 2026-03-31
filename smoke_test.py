"""
Smoke test: validates the full training + inference pipeline end-to-end
using tiny synthetic data.  Runs in ~1-2 minutes on CPU/MPS.

Usage:
    python smoke_test.py
"""

import os
import sys
import shutil
import argparse
import random

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import utils as vutils
from PIL import Image, ImageDraw

from dataset import GameIRSuperResolutionDataset
from vqvae import VQVAE
from resnet_prior import ResNetPrior
from sample import sample_model


# ------------------------------------------------------------------ #
#  helpers                                                            #
# ------------------------------------------------------------------ #

SMOKE_DIR = os.path.join(os.path.dirname(__file__), "_smoke_test")
HR_DIR = os.path.join(SMOKE_DIR, "hr")
LR_DIR = os.path.join(SMOKE_DIR, "lr")
CKPT_DIR = os.path.join(SMOKE_DIR, "checkpoints")
SAMPLE_DIR = os.path.join(SMOKE_DIR, "samples")

HR_SIZE = 256
SCALE = 2
LR_SIZE = HR_SIZE // SCALE
NUM_IMAGES = 16          # tiny dataset
VQVAE_EPOCHS = 3
PRIOR_EPOCHS = 3
BATCH = 4


def make_synthetic_pair(idx: int):
    """Create an HR image with random shapes and its downsampled LR version."""
    hr = Image.new("RGB", (HR_SIZE, HR_SIZE), color=(0, 0, 0))
    draw = ImageDraw.Draw(hr)
    for _ in range(random.randint(3, 8)):
        shape_type = random.choice(["rect", "ellipse"])
        x0 = random.randint(0, HR_SIZE - 40)
        y0 = random.randint(0, HR_SIZE - 40)
        x1 = x0 + random.randint(20, 80)
        y1 = y0 + random.randint(20, 80)
        color = tuple(random.randint(30, 255) for _ in range(3))
        if shape_type == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)
    lr = hr.resize((LR_SIZE, LR_SIZE), Image.BICUBIC)
    return lr, hr


def generate_data():
    """Write synthetic LR/HR pairs to disk."""
    os.makedirs(HR_DIR, exist_ok=True)
    os.makedirs(LR_DIR, exist_ok=True)
    for i in range(NUM_IMAGES):
        name = f"smoke_{i:04d}.png"
        lr, hr = make_synthetic_pair(i)
        hr.save(os.path.join(HR_DIR, name))
        lr.save(os.path.join(LR_DIR, name))
    print(f"[data]  Generated {NUM_IMAGES} synthetic LR/HR pairs in {SMOKE_DIR}")


# ------------------------------------------------------------------ #
#  Stage 1: VQ-VAE                                                    #
# ------------------------------------------------------------------ #

def train_vqvae(device):
    print("\n" + "=" * 60)
    print("  Stage 1: VQ-VAE training")
    print("=" * 60)

    dataset = GameIRSuperResolutionDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        suffix=None,
        hr_patch_size=HR_SIZE,
        scale=SCALE,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

    model = VQVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.MSELoss()

    for epoch in range(VQVAE_EPOCHS):
        model.train()
        total_loss = 0.0
        for _, hr_img in loader:
            hr_img = hr_img.to(device)
            out, latent_loss = model(hr_img)
            loss = criterion(out, hr_img) + 0.25 * latent_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(loader)
        print(f"  epoch {epoch + 1}/{VQVAE_EPOCHS}  loss={avg:.4f}")

    # Save checkpoint
    os.makedirs(CKPT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, "vqvae_smoke.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"[vqvae] Saved checkpoint → {ckpt_path}")

    # Quick encode sanity check
    model.eval()
    with torch.no_grad():
        sample_hr = next(iter(loader))[1][:2].to(device)
        sample_lr = next(iter(loader))[0][:2].to(device)
        qt, qb, diff, id_t, id_b = model.encode(sample_hr)
        _, _, _, lr_top, _ = model.encode(sample_lr)
        print(f"[vqvae] HR top codes: {id_t.shape}, HR bottom codes: {id_b.shape}")
        print(f"[vqvae] LR top codes: {lr_top.shape}")
        recon = model.decode(qt, qb)
        print(f"[vqvae] Reconstruction shape: {recon.shape}")
        recon2 = model.decode_code(id_t, id_b)
        print(f"[vqvae] decode_code shape:     {recon2.shape}")

    return ckpt_path


# ------------------------------------------------------------------ #
#  Stage 2: ResNet prior                                              #
# ------------------------------------------------------------------ #

def train_prior(vqvae_ckpt, hier, device):
    print(f"\n{'=' * 60}")
    print(f"  Stage 2: ResNet prior ({hier})")
    print("=" * 60)

    dataset = GameIRSuperResolutionDataset(
        lr_dir=LR_DIR,
        hr_dir=HR_DIR,
        suffix=None,
        hr_patch_size=HR_SIZE,
        scale=SCALE,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True, num_workers=0)

    # Load frozen VQ-VAE
    vqvae = VQVAE()
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae = vqvae.to(device)
    vqvae.eval()

    # Build prior
    channel = 128  # smaller for smoke test
    n_res_channel = 128
    use_lr_condition = (hier == "top")

    if hier == "top":
        model = ResNetPrior(
            [32, 32], 512, channel, 5,
            n_res_block=2,
            dropout=0.1,
            n_cond_res_block=3 if use_lr_condition else 0,
            cond_res_channel=n_res_channel if use_lr_condition else 0,
        )
    else:
        model = ResNetPrior(
            [64, 64], 512, channel, 5,
            n_res_block=2,
            dropout=0.1,
            n_cond_res_block=3,
            cond_res_channel=n_res_channel,
        )
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(PRIOR_EPOCHS):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_pixels = 0
        for lr_img, hr_img in loader:
            lr_img = lr_img.to(device)
            hr_img = hr_img.to(device)

            with torch.no_grad():
                _, _, _, hr_top, hr_bottom = vqvae.encode(hr_img)
                _, _, _, lr_top, _ = vqvae.encode(lr_img)
            hr_top = hr_top.long()
            hr_bottom = hr_bottom.long()
            lr_top = lr_top.long()

            if hier == "top":
                target = hr_top
                out, _ = model(hr_top, condition=lr_top)
            else:
                target = hr_bottom
                out, _ = model(hr_bottom, condition=hr_top)

            loss = criterion(out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = out.max(1)
            total_correct += (pred == target).sum().item()
            total_pixels += target.numel()

        avg = total_loss / len(loader)
        acc = total_correct / total_pixels
        print(f"  epoch {epoch + 1}/{PRIOR_EPOCHS}  loss={avg:.4f}  acc={acc:.4f}")

    ckpt_path = os.path.join(CKPT_DIR, f"prior_{hier}_smoke.pt")
    torch.save({"model": model.state_dict(), "args": {
        "channel": channel,
        "n_res_block": 2,
        "n_res_channel": n_res_channel,
        "dropout": 0.1,
        "n_cond_res_block": 3 if (hier == "top" and use_lr_condition) or hier == "bottom" else 0,
        "n_out_res_block": 0,
        "use_lr_condition": use_lr_condition,
        "model_type": "resnet",
    }}, ckpt_path)
    print(f"[prior] Saved {hier} checkpoint → {ckpt_path}")
    return ckpt_path


# ------------------------------------------------------------------ #
#  Stage 3: Sampling / Inference                                      #
# ------------------------------------------------------------------ #

def test_sampling(vqvae_ckpt, top_ckpt, bottom_ckpt, device):
    print(f"\n{'=' * 60}")
    print("  Stage 3: Sampling / inference")
    print("=" * 60)

    # Load VQ-VAE
    vqvae = VQVAE()
    vqvae.load_state_dict(torch.load(vqvae_ckpt, map_location=device))
    vqvae = vqvae.to(device)
    vqvae.eval()

    # Load top prior
    top_ckpt_data = torch.load(top_ckpt, map_location=device)
    top_args = top_ckpt_data["args"]
    top_model = ResNetPrior(
        [32, 32], 512, top_args["channel"], 5,
        n_res_block=top_args["n_res_block"],
        dropout=top_args["dropout"],
        n_cond_res_block=top_args["n_cond_res_block"],
        cond_res_channel=top_args["n_res_channel"] if top_args.get("use_lr_condition") else 0,
    )
    top_model.load_state_dict(top_ckpt_data["model"])
    top_model = top_model.to(device)
    top_model.eval()

    # Load bottom prior
    bot_ckpt_data = torch.load(bottom_ckpt, map_location=device)
    bot_args = bot_ckpt_data["args"]
    bot_model = ResNetPrior(
        [64, 64], 512, bot_args["channel"], 5,
        n_res_block=bot_args["n_res_block"],
        dropout=bot_args["dropout"],
        n_cond_res_block=bot_args["n_cond_res_block"],
        cond_res_channel=bot_args["n_res_channel"],
    )
    bot_model.load_state_dict(bot_ckpt_data["model"])
    bot_model = bot_model.to(device)
    bot_model.eval()

    # ---- Test 1: unconditional sampling (no LR condition) ----
    print("\n  [sample] Unconditional generation (no LR input)...")
    batch = 2
    top_sample = sample_model(top_model, device, batch, [32, 32], temperature=1.0)
    print(f"    top codes:    {top_sample.shape}  dtype={top_sample.dtype}")
    bottom_sample = sample_model(bot_model, device, batch, [64, 64], temperature=1.0, condition=top_sample)
    print(f"    bottom codes: {bottom_sample.shape}  dtype={bottom_sample.dtype}")
    decoded = vqvae.decode_code(top_sample, bottom_sample)
    print(f"    decoded img:  {decoded.shape}")

    os.makedirs(SAMPLE_DIR, exist_ok=True)
    out_path = os.path.join(SAMPLE_DIR, "uncond_sample.png")
    vutils.save_image(decoded.clamp(-1, 1), out_path, normalize=True, value_range=(-1, 1))
    print(f"    saved → {out_path}")

    # ---- Test 2: LR-conditioned sampling ----
    print("\n  [sample] LR-conditioned generation...")
    dataset = GameIRSuperResolutionDataset(
        lr_dir=LR_DIR, hr_dir=HR_DIR,
        suffix=None,
        hr_patch_size=HR_SIZE, scale=SCALE, augment=False,
    )
    lr_img, hr_img = dataset[0]
    lr_img = lr_img.unsqueeze(0).to(device)
    hr_img = hr_img.unsqueeze(0).to(device)

    with torch.no_grad():
        _, _, _, lr_top, _ = vqvae.encode(lr_img)
    lr_top = lr_top.long()
    print(f"    LR top codes: {lr_top.shape}")

    top_sample = sample_model(top_model, device, 1, [32, 32], temperature=1.0, condition=lr_top)
    bottom_sample = sample_model(bot_model, device, 1, [64, 64], temperature=1.0, condition=top_sample)
    decoded = vqvae.decode_code(top_sample, bottom_sample)

    # Save LR input, HR ground-truth, and SR output side by side
    comparison = torch.cat([
        nn.functional.interpolate(lr_img, size=(HR_SIZE, HR_SIZE), mode="nearest"),
        hr_img,
        decoded.clamp(-1, 1),
    ], dim=0)
    out_path = os.path.join(SAMPLE_DIR, "lr_cond_sample.png")
    vutils.save_image(comparison, out_path, nrow=3, normalize=True, value_range=(-1, 1))
    print(f"    saved → {out_path}  (LR upscaled | HR ground truth | SR output)")

    print("\n  [sample] ✅ Sampling pipeline works!\n")


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="End-to-end smoke test")
    parser.add_argument("--keep", action="store_true", help="Keep smoke test artifacts after completion")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/cuda/mps)")
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = (
            "mps" if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")

    # Seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    try:
        # Step 1: Generate synthetic data
        generate_data()

        # Step 2: Train VQ-VAE
        vqvae_ckpt = train_vqvae(device)

        # Step 3: Train top prior (LR-conditioned)
        top_ckpt = train_prior(vqvae_ckpt, "top", device)

        # Step 4: Train bottom prior (conditioned on top)
        bot_ckpt = train_prior(vqvae_ckpt, "bottom", device)

        # Step 5: Test sampling / inference
        test_sampling(vqvae_ckpt, top_ckpt, bot_ckpt, device)

        print("=" * 60)
        print("  🎉  ALL SMOKE TESTS PASSED")
        print("=" * 60)
        print(f"\nArtifacts in: {SMOKE_DIR}")
        print("You can now train on the real GameIR dataset.\n")

    except Exception as e:
        print(f"\n❌  SMOKE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if not args.keep:
            print(f"Cleaning up {SMOKE_DIR} (use --keep to preserve)...")
            shutil.rmtree(SMOKE_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
