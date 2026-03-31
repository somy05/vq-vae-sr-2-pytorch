import torch
import time
from vqvae import VQVAE
from resnet_prior import ResNetPrior

def measure_speed():
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize models
    # Use torch.compile to aggressively fuse operations and improve speed
    vqvae = VQVAE().to(device)
    if hasattr(torch, 'compile'):
        vqvae = torch.compile(vqvae)
    vqvae.eval()

    # Top prior (32x32)
    # Reducing channel size (from 256 to 128) and layers (from 4 to 2) for real-time speed.
    top_prior = ResNetPrior(
        [32, 32], 512, 128, 5, n_res_block=2, dropout=0.0
    ).to(device)
    if hasattr(torch, 'compile'):
        top_prior = torch.compile(top_prior)
    top_prior.eval()

    # Bottom prior (64x64)
    bottom_prior = ResNetPrior(
        [64, 64], 512, 128, 5, n_res_block=2, dropout=0.0
    ).to(device)
    if hasattr(torch, 'compile'):
        bottom_prior = torch.compile(bottom_prior)
    bottom_prior.eval()

    # Dummy inputs
    batch_size = 1
    dummy_img = torch.randn(batch_size, 3, 256, 256).to(device)
    dummy_top = torch.zeros(batch_size, 32, 32, dtype=torch.int64).to(device)
    dummy_bottom = torch.zeros(batch_size, 64, 64, dtype=torch.int64).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _, _, _, top, bottom = vqvae.encode(dummy_img)
            top_out, _ = top_prior(dummy_top)
            bottom_out, _ = bottom_prior(dummy_bottom, condition=dummy_top)
            vqvae.decode_code(dummy_top, dummy_bottom)

    # Benchmark end-to-end generation
    num_runs = 50
    start_time = time.time()
    
    with torch.no_grad():
        # Use autocast to run inference at float16 (half-precision) where supported
        with torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.float16 if torch.cuda.is_available() else torch.bfloat16):
            for _ in range(num_runs):
                # 1. Generate top latents
                t_out, _ = top_prior(dummy_top)
                t_sample = t_out.argmax(dim=1)
                
                # 2. Generate bottom latents conditioned on top
                b_out, _ = bottom_prior(dummy_bottom, condition=t_sample)
                b_sample = b_out.argmax(dim=1)
                
                # 3. Decode latents to image
                _ = vqvae.decode_code(t_sample, b_sample)
                
                if device == 'cuda':
                    torch.cuda.synchronize()

    total_time = time.time() - start_time
    avg_ms = (total_time / num_runs) * 1000

    print(f"Average time per image generation: {avg_ms:.2f} ms")
    if avg_ms <= 32:
        print("Success! Inference is under 32 ms.")
    else:
        print("Inference is over 32 ms. Might need optimization.")

if __name__ == '__main__':
    measure_speed()
