#!/usr/bin/env python3
"""
Stress-test GPU memory: one trans64 training step loop matching main.py
(BF16, EMA, DiffAugment, WGAN-GP). Run two instances in parallel to see if
batch_size fits twice on the same GPU.
"""
import argparse
import sys
import os

# Run from repo root
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torch.optim as optim
from copy import deepcopy

import source.losses as losses
from source.models.transgan import TransGenerator64, TransDiscriminator64
from source.diff_aug import DiffAugment


def cacl_gradient_penalty(net_D, real, fake):
    t = torch.rand(real.size(0), 1, 1, 1, device=real.device, dtype=real.dtype)
    t = t.expand(real.size())
    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]
    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    return torch.mean((grad_norm - 1) ** 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--tag', type=str, default='P')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print('ERROR: CUDA not available', flush=True)
        return 1

    device = torch.device('cuda:0')
    z_dim = 128
    bs = args.batch_size
    alpha = 10.0
    policy = 'color,translation,cutout'

    net_G = TransGenerator64(z_dim).to(device)
    net_D = TransDiscriminator64().to(device)
    net_G_ema = deepcopy(net_G)
    net_G_ema.eval()
    for p in net_G_ema.parameters():
        p.requires_grad_(False)

    optim_G = optim.Adam(net_G.parameters(), lr=1e-4, betas=(0.6, 0.9))
    optim_D = optim.Adam(net_D.parameters(), lr=1e-4, betas=(0.6, 0.9))
    loss_fn = losses.Wasserstein()
    amp_dtype = torch.bfloat16

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()

    for _ in range(args.steps):
        with torch.no_grad():
            z = torch.randn(bs, z_dim, device=device)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
                fake = net_G(z).detach()
        real = torch.randn(bs, 3, 64, 64, device=device).clamp(-1.0, 1.0)

        real_aug = DiffAugment(real, policy=policy)
        fake_aug = DiffAugment(fake, policy=policy)

        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            net_D_real = net_D(real_aug)
            net_D_fake = net_D(fake_aug)
            loss = loss_fn(net_D_real, net_D_fake)
        loss_gp = cacl_gradient_penalty(net_D, real.float(), fake.float())
        loss_all = loss + alpha * loss_gp

        optim_D.zero_grad()
        loss_all.backward()
        optim_D.step()

        for p in net_D.parameters():
            p.requires_grad_(False)
        z = torch.randn(bs * 2, z_dim, device=device)
        with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=True):
            gen_out = net_G(z)
            gen_aug = DiffAugment(gen_out, policy=policy)
            loss_g = loss_fn(net_D(gen_aug))
        optim_G.zero_grad()
        loss_g.backward()
        optim_G.step()

        with torch.no_grad():
            for p_ema, p in zip(net_G_ema.parameters(), net_G.parameters()):
                p_ema.data.lerp_(p.data, 1 - 0.999)

        for p in net_D.parameters():
            p.requires_grad_(True)

    peak_alloc = torch.cuda.max_memory_allocated(device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**2
    print(
        f"[{args.tag}] batch={bs} steps={args.steps} peak_allocated_MiB={peak_alloc:.1f} "
        f"peak_reserved_MiB={peak_reserved:.1f}",
        flush=True,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
