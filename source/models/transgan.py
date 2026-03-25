"""
Minimal TransGAN: Pure Transformer-based Generator and Discriminator.
Kept as simple as possible — no RPB, no special D activation, no tricks.
Drop-in replacement for ResNet G/D under the same WGAN-GP training loop.
"""

import torch
import torch.nn as nn


class PixelNorm(nn.Module):
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
    def __init__(self, in_dim, out_dim, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.proj = nn.Linear(in_dim // 4, out_dim)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.H, self.W)
        x = self.pixel_shuffle(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class TransGenerator64(nn.Module):
    def __init__(self, z_dim, embed_dim=256, depths=(3, 2, 2), num_heads=4):
        super().__init__()
        self.z_dim = z_dim
        self.embed_dim = embed_dim

        self.mlp_in = nn.Linear(z_dim, 8 * 8 * embed_dim)

        self.pos_embed_8 = nn.Parameter(torch.zeros(1, 64, embed_dim))
        self.pos_embed_16 = nn.Parameter(torch.zeros(1, 256, embed_dim))
        self.pos_embed_32 = nn.Parameter(torch.zeros(1, 1024, embed_dim))

        self.stage1 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[0])])
        self.up1 = UpSampleBlock(embed_dim, embed_dim, 8, 8)

        self.stage2 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[1])])
        self.up2 = UpSampleBlock(embed_dim, embed_dim, 16, 16)

        self.stage3 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[2])])
        self.up3 = UpSampleBlock(embed_dim, embed_dim, 32, 32)

        self.to_rgb = nn.Sequential(PixelNorm(), nn.Linear(embed_dim, 3))
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed_8, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_16, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_32, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z, *args, **kwargs):
        B = z.size(0)
        x = self.mlp_in(z).reshape(B, 64, self.embed_dim)
        x = x + self.pos_embed_8
        x = self.stage1(x)
        x = self.up1(x) + self.pos_embed_16
        x = self.stage2(x)
        x = self.up2(x) + self.pos_embed_32
        x = self.stage3(x)
        x = self.up3(x)
        x = self.to_rgb(x)
        x = x.transpose(1, 2).reshape(B, 3, 64, 64)
        return torch.tanh(x)


class PatchEmbed(nn.Module):
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
    def __init__(self, H, W):
        super().__init__()
        self.H = H
        self.W = W
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, self.H, self.W)
        x = self.pool(x)
        return x.flatten(2).transpose(1, 2)


class TransDiscriminator64(nn.Module):
    def __init__(self, embed_dim=256, depths=(2, 2, 2), num_heads=4):
        super().__init__()

        self.patch_embed_fine = PatchEmbed(4, 3, embed_dim // 4)
        self.patch_embed_med = PatchEmbed(8, 3, embed_dim // 4)
        self.patch_embed_coarse = PatchEmbed(16, 3, embed_dim // 2)

        self.pos_embed_1 = nn.Parameter(torch.zeros(1, 256, embed_dim // 4))
        self.pos_embed_2 = nn.Parameter(torch.zeros(1, 64, embed_dim // 4))
        self.pos_embed_3 = nn.Parameter(torch.zeros(1, 16, embed_dim // 2))

        self.stage1 = nn.Sequential(
            *[TransformerBlock(embed_dim // 4, max(1, num_heads // 4)) for _ in range(depths[0])])
        self.down1 = DownSampleBlock(16, 16)
        self.merge1 = nn.Linear(embed_dim // 4 + embed_dim // 4, embed_dim // 2)

        self.stage2 = nn.Sequential(
            *[TransformerBlock(embed_dim // 2, max(1, num_heads // 2)) for _ in range(depths[1])])
        self.down2 = DownSampleBlock(8, 8)
        self.merge2 = nn.Linear(embed_dim // 2 + embed_dim // 2, embed_dim)

        self.stage3 = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads) for _ in range(depths[2])])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.head = nn.Linear(embed_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed_1, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_2, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_3, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, *args, **kwargs):
        B = x.size(0)
        fine = self.patch_embed_fine(x) + self.pos_embed_1
        med = self.patch_embed_med(x) + self.pos_embed_2
        coarse = self.patch_embed_coarse(x) + self.pos_embed_3

        fine = self.stage1(fine)
        fine = self.down1(fine)
        x = self.merge1(torch.cat([fine, med], dim=-1))

        x = self.stage2(x)
        x = self.down2(x)
        x = self.merge2(torch.cat([x, coarse], dim=-1))

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.stage3(x)
        return self.head(x[:, 0])
