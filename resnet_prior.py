import torch
from torch import nn
from torch.nn import functional as F
from pixelsnail import CondResNet, WNConv2d

class SimpleResBlock(nn.Module):
    def __init__(self, channel, kernel_size, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU(inplace=False)

    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        return out + x

class ResNetPrior(nn.Module):
    def __init__(
        self,
        shape,
        n_class,
        channel,
        kernel_size,
        n_block=0, # unused, kept for compatibility
        n_res_block=4,
        res_channel=0, # unused, kept for compatibility
        attention=False, # unused
        dropout=0.1,
        n_cond_res_block=0,
        cond_res_channel=0,
        cond_res_kernel=3,
        n_out_res_block=0, # unused
    ):
        super().__init__()
        self.shape = shape
        self.n_class = n_class
        self.is_autoregressive = False

        self.cond_resnet = None
        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                n_class, cond_res_channel, cond_res_kernel, n_cond_res_block
            )

        self.uncond_embedding = nn.Parameter(torch.randn(1, channel, 1, 1))

        if cond_res_channel > 0:
            self.merge = WNConv2d(channel + cond_res_channel, channel, 1)
        else:
            self.merge = None

        resblocks = []
        for i in range(n_res_block):
            resblocks.append(
                SimpleResBlock(channel, kernel_size, dropout=dropout)
            )

        self.blocks = nn.Sequential(*resblocks)
        self.out = nn.Sequential(
            nn.ELU(inplace=True), WNConv2d(channel, n_class, 1)
        )

    def forward(self, input, condition=None, cache=None):
        """
        Non-autoregressive forward. 
        Ignores 'input' content and just uses it for shape.
        """
        if cache is None:
            cache = {}
        batch, height, width = input.shape
        
        x = self.uncond_embedding.expand(batch, -1, height, width)

        if condition is not None:
            c = F.one_hot(condition, self.n_class).permute(0, 3, 1, 2).type_as(x)
            if self.cond_resnet is not None:
                c = self.cond_resnet(c)
                
            if c.shape[-1] < width:
                c = F.interpolate(c, size=(height, width), mode='nearest')
                
            if self.merge is not None:
                x = torch.cat([x, c], 1)
                x = self.merge(x)

        for block in self.blocks:
            x = block(x)
            
        out = self.out(x)
        
        return out, cache
