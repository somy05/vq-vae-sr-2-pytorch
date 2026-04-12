"""
Lightweight Direct Super-Resolution Network.

EDSR-inspired architecture: residual blocks in LR space,
PixelShuffle upscaling, global residual learning from bicubic.

All processing happens at LR resolution for maximum speed.
Only the final PixelShuffle operates at HR resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual block without batch norm (BN hurts SR quality)."""
    def __init__(self, channels, res_scale=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        return x + res * self.res_scale


class DirectSRNet(nn.Module):
    """
    Direct LR → HR super-resolution in a single forward pass.

    Architecture:
        LR image → feature extraction → N residual blocks (LR space)
        → PixelShuffle upscale → HR residual
        → add to bicubic baseline → HR output

    Args:
        scale: upscaling factor (default 2)
        n_channels: feature channels (default 64)
        n_blocks: number of residual blocks (default 16)
    """
    def __init__(self, scale=2, n_channels=64, n_blocks=16):
        super().__init__()
        self.scale = scale

        # Feature extraction from LR input
        self.head = nn.Conv2d(3, n_channels, 3, padding=1)

        # Residual body — all computation in LR space for speed
        body = [ResBlock(n_channels) for _ in range(n_blocks)]
        self.body = nn.Sequential(*body)
        self.body_tail = nn.Conv2d(n_channels, n_channels, 3, padding=1)

        # Upscale: LR features → HR image via sub-pixel convolution
        self.upscale = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * scale * scale, 3, padding=1),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_channels, 3, 3, padding=1),
        )

    def forward(self, x):
        # Bicubic baseline for global residual learning
        bicubic = F.interpolate(
            x, scale_factor=self.scale, mode='bicubic', align_corners=False
        )

        # Feature extraction
        head = self.head(x)

        # Residual blocks with long skip connection
        body = self.body(head)
        body = self.body_tail(body)
        feat = head + body

        # Upscale and add residual to bicubic
        out = self.upscale(feat)
        return out + bicubic
