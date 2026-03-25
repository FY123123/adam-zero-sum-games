"""
SAGAN: Self-Attention GAN for 64x64 images.
ResNet backbone (identical to wgangp.py) with one self-attention layer
inserted after the second-to-last resolution in both G and D.

Reference: Zhang et al., "Self-Attention Generative Adversarial Networks", ICML 2019.
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init


class SelfAttention(nn.Module):
    """Spectral-norm-free self-attention (keeps parity with the ResNet baseline)."""

    def __init__(self, in_channels):
        super().__init__()
        mid = max(1, in_channels // 8)
        self.query = nn.Conv2d(in_channels, mid, 1)
        self.key = nn.Conv2d(in_channels, mid, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        attn = torch.softmax(q @ k, dim=-1)
        v = self.value(x).view(B, -1, H * W)
        out = (v @ attn.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# ---------------------------------------------------------------------------
# Generator building blocks (same as wgangp.py ResGenBlock)
# ---------------------------------------------------------------------------

class ResGenBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )
        self.shortcut = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )
        self._init()

    def _init(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class SAResGenerator64(nn.Module):
    """ResNet Generator for 64x64 with self-attention at 32x32 feature maps."""

    def __init__(self, z_dim):
        super().__init__()
        self.linear = nn.Linear(z_dim, 4 * 4 * 256)

        self.block1 = ResGenBlock(256, 256)   # 4  -> 8
        self.block2 = ResGenBlock(256, 256)   # 8  -> 16
        self.block3 = ResGenBlock(256, 256)   # 16 -> 32
        self.attn = SelfAttention(256)        # attention at 32x32
        self.block4 = ResGenBlock(256, 256)   # 32 -> 64

        self.output = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 3, 3, 1, 1),
            nn.Tanh(),
        )
        self._init()

    def _init(self):
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)
        for m in self.output.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        x = self.linear(z).view(-1, 256, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.attn(x)
        x = self.block4(x)
        return self.output(x)


# ---------------------------------------------------------------------------
# Discriminator building blocks (same as wgangp.py ResDisBlock)
# ---------------------------------------------------------------------------

class OptimizedResDisblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0),
        )
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.AvgPool2d(2),
        )
        self._init()

    def _init(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class ResDisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=False):
        super().__init__()
        shortcut = []
        if in_channels != out_channels or down:
            shortcut.append(nn.Conv2d(in_channels, out_channels, 1, 1, 0))
        if down:
            shortcut.append(nn.AvgPool2d(2))
        self.shortcut = nn.Sequential(*shortcut)

        residual = [
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        ]
        if down:
            residual.append(nn.AvgPool2d(2))
        self.residual = nn.Sequential(*residual)
        self._init()

    def _init(self):
        for m in self.residual.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
        for m in self.shortcut.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                init.zeros_(m.bias)

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class SAResDiscriminator64(nn.Module):
    """ResNet Discriminator for 64x64 with self-attention at 16x16 feature maps."""

    def __init__(self):
        super().__init__()
        self.block1 = OptimizedResDisblock(3, 128)    # 64 -> 32
        self.block2 = ResDisBlock(128, 128, down=True) # 32 -> 16
        self.attn = SelfAttention(128)                 # attention at 16x16
        self.block3 = ResDisBlock(128, 128, down=True) # 16 -> 8
        self.block4 = ResDisBlock(128, 128)            # 8  -> 8
        self.relu = nn.ReLU()
        self.linear = nn.Linear(128, 1)
        self._init()

    def _init(self):
        init.xavier_uniform_(self.linear.weight)

    def forward(self, x, *args, **kwargs):
        x = self.block1(x)
        x = self.block2(x)
        x = self.attn(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.relu(x)
        x = x.sum(dim=[2, 3])
        return self.linear(x)
