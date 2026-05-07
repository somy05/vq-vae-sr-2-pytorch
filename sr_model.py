import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRAConv2d(nn.Module):
    def __init__(self, original_conv, rank=4, alpha=1.0):
        super().__init__()
        self.original_conv = original_conv
        self.rank = rank
        self.scale = alpha / rank

        in_ch = original_conv.in_channels
        out_ch = original_conv.out_channels

        # Down-project: C_in → rank  (1×1 conv)
        self.lora_A = nn.Conv2d(in_ch, rank, 1, bias=False)
        # Up-project:  rank → C_out (1×1 conv)
        self.lora_B = nn.Conv2d(rank, out_ch, 1, bias=False)

        # A gets Kaiming init, B starts at zero → LoRA output is 0 at init
        nn.init.kaiming_normal_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.original_conv(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(model, rank=4, alpha=1.0):
    for block in model.body:
        if isinstance(block, ResBlock):
            block.conv1 = LoRAConv2d(block.conv1, rank, alpha)
            block.conv2 = LoRAConv2d(block.conv2, rank, alpha)

    # Freeze everything, then unfreeze only LoRA params
    for param in model.parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if 'lora_' in name:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'  LoRA injected: {trainable:,} trainable / {total:,} total params '
          f'({100 * trainable / total:.1f}%)')

    return model


def extract_lora_state_dict(model):
    return {k: v for k, v in model.state_dict().items() if 'lora_' in k}


def load_lora_weights(model, lora_path, device='cpu'):
    lora_state = torch.load(lora_path, map_location=device)
    if 'lora_state' in lora_state:
        lora_state = lora_state['lora_state']
    # Merge into the full state dict
    current_state = model.state_dict()
    current_state.update(lora_state)
    model.load_state_dict(current_state)
    print(f'  LoRA weights loaded from {lora_path}')
    return model


class ResBlock(nn.Module):
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
    def __init__(self, scale=2, n_channels=64, n_blocks=16, fast_tail=False):
        super().__init__()
        self.scale = scale

        # Feature extraction from LR input
        self.head = nn.Conv2d(3, n_channels, 3, padding=1)

        # Residual body — all computation in LR space for speed
        body = [ResBlock(n_channels) for _ in range(n_blocks)]
        self.body = nn.Sequential(*body)
        # fast_tail=True uses 1×1 convs where spatial context is less critical
        tail_k = 1 if fast_tail else 3
        tail_p = 0 if fast_tail else 1
        self.body_tail = nn.Conv2d(n_channels, n_channels, tail_k, padding=tail_p)

        # Upscale: LR features → HR image via sub-pixel convolution
        self.upscale = nn.Sequential(
            nn.Conv2d(n_channels, n_channels * scale * scale, tail_k, padding=tail_p),
            nn.PixelShuffle(scale),
            nn.Conv2d(n_channels, 3, tail_k, padding=tail_p),
        )

    def forward(self, x, active_blocks=None):
        # Bicubic baseline for global residual learning
        # CoreML does not support bicubic op, fallback to bilinear during export
        mode = 'bilinear' if getattr(self, 'coreml_export', False) else 'bicubic'
        bicubic = F.interpolate(
            x, scale_factor=self.scale, mode=mode, align_corners=False
        )

        # Feature extraction
        head = self.head(x)

        # Residual blocks with long skip connection
        if active_blocks is not None:
            body = head
            for i, block in enumerate(self.body):
                if i >= active_blocks:
                    break
                body = block(body)
        else:
            body = self.body(head)

        body = self.body_tail(body)
        feat = head + body

        # Upscale and add residual to bicubic
        out = self.upscale(feat)
        return out + bicubic

