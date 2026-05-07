import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):

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
        x = (x + 1) / 2  # → [0, 1]
        return (x - self.mean) / self.std

    def forward(self, pred, target):
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
