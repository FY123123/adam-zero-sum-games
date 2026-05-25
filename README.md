# Adam in Zero-Sum Games

Code release for the ICML 2026 paper:

**Understanding Dynamics of Adam in Zero-Sum Games: An ODE Approach**

Yi Feng, Weiming Ou, Xiao Wang

This repository contains the GAN experiments used to study how Adam descent-ascent behaves in zero-sum games. The code focuses on the empirical predictions from the paper: the effect of the first-order momentum parameter and the second-order momentum parameter on GAN training dynamics, image quality metrics, and gradient norm behavior.

## Contents

- `main.py`: train GANs with Adam, negative-momentum Adam, or SGD.
- `flatness.py`: plot Inception Score, FID, and gradient norm curves from saved `.npz` logs.
- `sample.py`: generate image grids from saved checkpoints.
- `source/models/`: retained generator/discriminator architectures for the paper experiments.
- `scripts/`: reproducibility sweeps for the beta-parameter experiments.

Supported datasets:

- CIFAR-10, downloaded with `torchvision.datasets.CIFAR10`.
- STL-10, downloaded with `torchvision.datasets.STL10`.
- CelebA, streamed from HuggingFace dataset `flwrlabs/celeba`.

Supported architectures:

- `res32`: ResNet-style baseline.
- `cnn32`: CNN baseline.
- `sagan64`: self-attention ResNet GAN at 64x64 resolution.

## Installation

Create an environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For GPU runs, install the PyTorch build matching your CUDA version from the official PyTorch instructions before installing the rest of the requirements. The default `requirements.txt` intentionally avoids pinning an old CUDA wheel.

## Training

Run a short CIFAR-10 ResNet experiment:

```bash
python main.py \
  --dataset cifar10 \
  --arch res32 \
  --optimizer adam_nm \
  --betas=-0.3,0.999 \
  --total_steps 10000 \
  --final_total_steps 100000
```

Run a CelebA self-attention GAN experiment:

```bash
python main.py \
  --dataset celeba \
  --arch sagan64 \
  --optimizer adam_nm \
  --lr_G 1e-4 \
  --lr_D 1e-4 \
  --batch_size 64 \
  --betas=-0.3,0.99 \
  --total_steps 10000 \
  --final_total_steps 100000
```

Useful options:

- `--optimizer adam`: standard Adam.
- `--optimizer adam_nm`: Adam with the custom negative-momentum implementation.
- `--optimizer sgd`: SGD baseline, using `beta1` as the momentum parameter.
- `--enable_fid` / `--noenable_fid`: enable or disable FID evaluation.
- `--record` / `--norecord`: enable or disable metric recording.
- `--diff_aug ""`: disable DiffAugment.
- `--ema_decay 0`: disable generator EMA.

Outputs are written under ignored directories such as `logs/`, `Checkpoint/`, `npz/`, and `fid_stats/`.

## Reproduction Sweeps

The provided scripts run the beta-parameter sweeps used for the public experiments:

```bash
bash scripts/run_sweep_betas_10k.sh
```

This script runs two groups of CelebA `sagan64` experiments:

- beta1 sweep: `beta1` from `-0.5` to `0.5`, with `beta2=0.99`.
- beta2 sweep: `beta1=0`, with `beta2` from `0.9` to `0.3`.

By default it launches at most two parallel jobs. Override this with:

```bash
MAX_JOBS=1 bash scripts/run_sweep_betas_10k.sh
```

For a clean self-attention GAN sweep without DiffAugment or EMA:

```bash
bash scripts/run_sweep_clean.sh
```

## Plotting

After training, plot metric curves from saved `.npz` files:

```bash
python flatness.py npz/sagan64_npz/CelebA/*.npz --output Image/celeba_sagan64
```

If no files are provided, `flatness.py` searches under algorithm-specific folders in `npz/`.

## Sampling

Generate sample image grids from checkpoints:

```bash
python sample.py Checkpoint/sagan64_checkpoint/CelebA/example.pt \
  --output "Sample Images" \
  --num-samples 64 \
  --grid-size 8
```

`sample.py` supports `res32`, `cnn32`, and `sagan64` checkpoints. When a checkpoint contains both `net_G` and `net_G_ema`, it uses `net_G_ema` by default. Pass `--use-raw-generator` to sample from `net_G`.

## License

This project is released under the MIT License. See `LICENSE` for details.
