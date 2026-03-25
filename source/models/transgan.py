"""
TransGAN: Pure Transformer-based Generator and Discriminator.

Reference: Jiang et al., "TransGAN: Two Pure Transformers Can Make One Strong GAN,
and That Can Scale Up", NeurIPS 2021. (arXiv:2102.07074)

Adapted for 64x64 generation with WGAN-GP training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.init as init


class PixelNorm(nn.Module):
    """Modified normalization from TransGAN paper (replaces LayerNorm).

    Performs token-wise L2 normalization: y = x / sqrt(mean(x^2) + eps).
    Resembles local response normalization; stabilizes transformer training in GANs.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class TransformerBlock(nn.Module):
    """Standard Transformer encoder block with PixelNorm (TransGAN style)."""

    def __init__(self, dim, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = PixelNorm()
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm2 = PixelNorm()
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class UpSampleBlock(nn.Module):
    """Upsample 2x via PixelShuffle: (B, N, C) -> (B, 4N, C//4) then project back."""

    def __init__(self, in_dim, out_dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.proj = nn.Linear(in_dim // 4, out_dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.H, self.W)
        x = self.pixel_shuffle(x)  # (B, C//4, 2H, 2W)
        x = x.flatten(2).transpose(1, 2)  # (B, 4N, C//4)
        x = self.proj(x)
        return x


class TransGenerator64(nn.Module):
    """TransGAN-style pure-transformer generator for 64x64 images.

    Architecture:
        noise -> MLP -> 8x8 tokens -> [Transformer Blocks] x3 stages
        with PixelShuffle 2x upsampling between stages -> 64x64 RGB
    """

    def __init__(self, z_dim, embed_dim=256, depths=(3, 2, 2), num_heads=4):
        super().__init__()
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.init_size = 8  # start at 8x8

        self.mlp_in = nn.Sequential(
            nn.Linear(z_dim, self.init_size * self.init_size * embed_dim),
        )

        self.pos_embed_8 = nn.Parameter(
            torch.zeros(1, 8 * 8, embed_dim))
        self.pos_embed_16 = nn.Parameter(
            torch.zeros(1, 16 * 16, embed_dim))
        self.pos_embed_32 = nn.Parameter(
            torch.zeros(1, 32 * 32, embed_dim))

        # Stage 1: 8x8
        self.stage1 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[0])]
        )
        self.up1 = UpSampleBlock(embed_dim, embed_dim, 8, 8)

        # Stage 2: 16x16
        self.stage2 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[1])]
        )
        self.up2 = UpSampleBlock(embed_dim, embed_dim, 16, 16)

        # Stage 3: 32x32
        self.stage3 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[2])]
        )
        self.up3 = UpSampleBlock(embed_dim, embed_dim, 32, 32)

        # Final projection: 64x64 tokens -> RGB
        self.to_rgb = nn.Sequential(
            PixelNorm(),
            nn.Linear(embed_dim, 3),
        )

        self.initialize()

    def initialize(self):
        nn.init.trunc_normal_(self.pos_embed_8, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_16, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_32, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        B = z.size(0)
        x = self.mlp_in(z).reshape(B, 64, self.embed_dim)  # (B, 64, C)
        x = x + self.pos_embed_8

        x = self.stage1(x)       # 8x8
        x = self.up1(x)          # -> 16x16
        x = x + self.pos_embed_16

        x = self.stage2(x)       # 16x16
        x = self.up2(x)          # -> 32x32
        x = x + self.pos_embed_32

        x = self.stage3(x)       # 32x32
        x = self.up3(x)          # -> 64x64

        x = self.to_rgb(x)       # (B, 4096, 3)
        x = x.transpose(1, 2).reshape(B, 3, 64, 64)
        x = torch.tanh(x)
        return x


class PatchEmbed(nn.Module):
    """Flatten image patches into token embeddings via linear projection."""

    def __init__(self, patch_size, in_channels=3, embed_dim=64):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        P = self.patch_size
        nH, nW = H // P, W // P
        x = x.reshape(B, C, nH, P, nW, P)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, nH * nW, P * P * C)
        return self.proj(x)


class DownSampleBlock(nn.Module):
    """Downsample 2x via AvgPool: reshape tokens to 2D, pool, flatten back."""

    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.H, self.W)
        x = self.pool(x)  # (B, C, H/2, W/2)
        return x.flatten(2).transpose(1, 2)  # (B, N/4, C)


class TransDiscriminator64(nn.Module):
    """TransGAN-style multi-scale pure-transformer discriminator for 64x64.

    Multi-scale patch embedding:
        - Fine:   4x4 patches -> 256 tokens (stage 1)
        - Medium: 8x8 patches ->  64 tokens (concatenated at stage 2)
        - Coarse: 16x16 patches -> 16 tokens (concatenated at stage 3)
    """

    def __init__(self, embed_dim=256, depths=(2, 2, 2), num_heads=4):
        super().__init__()

        # Multi-scale patch embeddings
        self.patch_embed_fine = PatchEmbed(4, 3, embed_dim // 4)    # 256 tokens, dim/4
        self.patch_embed_med = PatchEmbed(8, 3, embed_dim // 4)     # 64 tokens, dim/4
        self.patch_embed_coarse = PatchEmbed(16, 3, embed_dim // 2) # 16 tokens, dim/2

        # Positional embeddings
        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 256, embed_dim // 4))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 64, embed_dim // 4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 16, embed_dim // 2))

        # Stage 1: 256 tokens (16x16 grid), dim/4
        self.stage1 = nn.Sequential(
            *[TransformerBlock(embed_dim // 4, max(1, num_heads // 4)) for _ in range(depths[0])]
        )
        self.down1 = DownSampleBlock(16, 16)  # 256 -> 64 tokens
        self.merge1 = nn.Linear(embed_dim // 4 + embed_dim // 4, embed_dim // 2)

        # Stage 2: 64 tokens (8x8 grid), dim/2
        self.stage2 = nn.Sequential(
            *[TransformerBlock(embed_dim // 2, max(1, num_heads // 2)) for _ in range(depths[1])]
        )
        self.down2 = DownSampleBlock(8, 8)  # 64 -> 16 tokens
        self.merge2 = nn.Linear(embed_dim // 2 + embed_dim // 2, embed_dim)

        # Stage 3: 16 tokens (4x4 grid), dim
        self.stage3 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[2])]
        )

        # CLS token + classification head
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Linear(embed_dim, 1)

        self.initialize()

    def initialize(self):
        nn.init.trunc_normal_(self.pos_embed_1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_2, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_3, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, *args, **kwargs):
        B = x.size(0)

        # Multi-scale patch tokenization
        fine = self.patch_embed_fine(x) + self.pos_embed_1    # (B, 256, C/4)
        med = self.patch_embed_med(x) + self.pos_embed_2      # (B, 64, C/4)
        coarse = self.patch_embed_coarse(x) + self.pos_embed_3  # (B, 16, C/2)

        # Stage 1
        fine = self.stage1(fine)                  # (B, 256, C/4)
        fine = self.down1(fine)                   # (B, 64, C/4)
        x = self.merge1(torch.cat([fine, med], dim=-1))  # (B, 64, C/2)

        # Stage 2
        x = self.stage2(x)                       # (B, 64, C/2)
        x = self.down2(x)                        # (B, 16, C/2)
        x = self.merge2(torch.cat([x, coarse], dim=-1))  # (B, 16, C)

        # Stage 3 with CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, 17, C)
        x = self.stage3(x)

        return self.head(x[:, 0])                # CLS token -> scalar
