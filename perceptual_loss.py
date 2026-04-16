"""
VGG19-based Perceptual Loss for Super-Resolution.

Computes feature-space L1 distance between predicted and target images
using intermediate layers of a pre-trained VGG19 network.

Reference: Johnson et al. (2016), "Perceptual Losses for Real-Time
Style Transfer and Super-Resolution."
"""

import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    Extracts features from VGG19 conv layers and computes L1 distance.

    Uses features before the 2nd, 4th, and 5th max-pool layers
    (conv2_2, conv4_4, conv5_4), capturing low, mid, and high-level
    structure respectively.
    """
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features

        # Split VGG into blocks ending at key layers
        # conv2_2 (index 8), conv4_4 (index 26), conv5_4 (index 35)
        self.block1 = nn.Sequential(*vgg[:9])    # → conv2_2 + relu
        self.block2 = nn.Sequential(*vgg[9:27])  # → conv4_4 + relu
        self.block3 = nn.Sequential(*vgg[27:36]) # → conv5_4 + relu

        # Freeze all VGG weights — no training
        for param in self.parameters():
            param.requires_grad = False

        # ImageNet normalisation (VGG expects this)
        self.register_buffer(
            'mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _normalize(self, x):
        """Convert from [-1, 1] (our model range) to ImageNet range."""
        x = (x + 1) / 2  # → [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred, target):
        """Compute perceptual loss between pred and target images."""
        pred = self._normalize(pred)
        target = self._normalize(target)

        loss = 0.0
        # Block 1 — low-level (edges, textures)
        pred = self.block1(pred)
        target = self.block1(target)
        loss += torch.nn.functional.l1_loss(pred, target)

        # Block 2 — mid-level (shapes, patterns)
        pred = self.block2(pred)
        target = self.block2(target)
        loss += torch.nn.functional.l1_loss(pred, target)

        # Block 3 — high-level (structure, semantics)
        pred = self.block3(pred)
        target = self.block3(target)
        loss += torch.nn.functional.l1_loss(pred, target)

        return loss
