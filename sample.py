#!/usr/bin/env python3
"""Generate sample image grids from GAN checkpoints."""

import argparse
import os
import re
import sys

import torch
from torchvision.utils import make_grid, save_image

import source.models.sagan as sagan_models
import source.models.transgan as transgan_models
import source.models.wgangp as wgangp_models


DEVICE = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "mps" if (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    else "cpu"
)

KNOWN_ARCHES = ("res32", "cnn32", "trans64", "sagan64")
KNOWN_DATASETS = ("cifar10", "stl10", "celeba")


def _default_output_size(dataset):
    # Mirrors main.py: ResNet/CNN baselines use 64x64 only for STL-10.
    return 64 if dataset == "stl10" else 32


def create_generator(arch, dataset, z_dim):
    """Create the generator architecture used by a saved checkpoint."""
    if arch == "res32":
        return wgangp_models.ResGenerator32(z_dim, _default_output_size(dataset))
    if arch == "cnn32":
        return wgangp_models.Generator32(
            z_dim, output_size=_default_output_size(dataset)
        )
    if arch == "trans64":
        return transgan_models.TransGenerator64(z_dim)
    if arch == "sagan64":
        return sagan_models.SAResGenerator64(z_dim)
    raise ValueError(f"Unknown architecture: {arch}")


def parse_checkpoint_params(checkpoint_path, checkpoint):
    """Infer checkpoint metadata from the file name and saved metadata."""
    stem = os.path.basename(checkpoint_path)
    if stem.endswith(".pt") or stem.endswith(".pth"):
        stem = os.path.splitext(stem)[0]

    params = {
        "learning_rate": None,
        "beta1": None,
        "beta2": None,
        "optimizer": None,
        "arch": None,
        "dataset": None,
        "filename": stem,
    }

    lr_match = re.search(r"lr([0-9.eE+-]+)", stem)
    if lr_match:
        params["learning_rate"] = float(lr_match.group(1))

    beta_match = re.search(r"beta(-?[0-9.]+)_([0-9.]+)", stem)
    if beta_match:
        params["beta1"] = float(beta_match.group(1))
        params["beta2"] = float(beta_match.group(2))

    for arch in KNOWN_ARCHES:
        if f"_{arch}_" in stem or stem.endswith(f"_{arch}"):
            params["arch"] = arch
            break

    lowered_path = checkpoint_path.lower()
    for dataset in KNOWN_DATASETS:
        if f"_{dataset}" in stem.lower() or dataset in lowered_path:
            params["dataset"] = dataset
            break

    if params["arch"] and params["dataset"]:
        suffix = f"_{params['arch']}_{params['dataset']}"
        if stem.endswith(suffix):
            prefix = stem[:-len(suffix)]
            opt_match = re.search(r"beta-?[0-9.]+_[0-9.]+_(.+)$", prefix)
            if opt_match:
                params["optimizer"] = opt_match.group(1)

    params["arch"] = checkpoint.get("arch", params["arch"]) or "res32"
    params["dataset"] = checkpoint.get("dataset", params["dataset"]) or "cifar10"
    params["beta1"] = checkpoint.get("beta1", params["beta1"])
    params["beta2"] = checkpoint.get("beta2", params["beta2"])
    params["optimizer"] = (
        checkpoint.get("optimizer_type", params["optimizer"]) or "adam"
    )

    if params["learning_rate"] is None:
        params["learning_rate"] = 0.0002
    if params["beta1"] is None:
        params["beta1"] = 0.5
    if params["beta2"] is None:
        params["beta2"] = 0.999

    return params


def load_checkpoint(checkpoint_path):
    """Load a checkpoint dictionary."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("Expected a checkpoint dictionary")
    return checkpoint


def load_generator_state(net_G, checkpoint, prefer_ema=True):
    """Load generator weights, preferring EMA weights when requested."""
    generator_keys = []
    if prefer_ema:
        generator_keys.append("net_G_ema")
    generator_keys.extend(["net_G", "generator", "G", "model"])

    for key in generator_keys:
        if key in checkpoint:
            net_G.load_state_dict(checkpoint[key])
            net_G.eval()
            return key

    raise ValueError(
        "Could not find generator weights. "
        f"Available checkpoint keys: {sorted(checkpoint.keys())}"
    )


def _safe_float_token(value):
    return str(value).replace("-", "neg").replace(".", "p")


def generate_sample_images(
    net_G, output_dir, params, z_dim, num_samples=64, grid_size=8
):
    """Generate a grid of samples and save it as a PNG."""
    os.makedirs(output_dir, exist_ok=True)

    net_G.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim, device=DEVICE)
        fake_images = (net_G(z).cpu() + 1) / 2
        grid = make_grid(fake_images, nrow=grid_size, padding=2, normalize=False)

    filename = (
        f"beta{_safe_float_token(params['beta1'])}_"
        f"{_safe_float_token(params['beta2'])}_"
        f"{params['arch']}_{params['dataset']}.png"
    )
    grid_path = os.path.join(output_dir, filename)
    save_image(grid, grid_path)
    print(f"Generated sample grid: {grid_path}")
    return grid_path


def process_checkpoint(
    checkpoint_path, output_dir, num_samples, grid_size, z_dim, prefer_ema
):
    """Process one checkpoint file."""
    try:
        print(f"\nProcessing: {checkpoint_path}")
        print("-" * 60)

        checkpoint = load_checkpoint(checkpoint_path)
        params = parse_checkpoint_params(checkpoint_path, checkpoint)

        print(f"Learning rate: {params['learning_rate']}")
        print(f"Beta1: {params['beta1']}, beta2: {params['beta2']}")
        print(f"Optimizer: {params['optimizer']}")
        print(f"Architecture: {params['arch']}")
        print(f"Dataset: {params['dataset']}")

        net_G = create_generator(params["arch"], params["dataset"], z_dim).to(DEVICE)
        weight_key = load_generator_state(net_G, checkpoint, prefer_ema=prefer_ema)
        print(f"Loaded generator weights from checkpoint key: {weight_key}")

        return generate_sample_images(
            net_G, output_dir, params, z_dim, num_samples, grid_size
        )

    except Exception as exc:
        print(f"Error processing {checkpoint_path}: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate sample image grids from checkpoint files."
    )
    parser.add_argument("checkpoints", nargs="+", help="Checkpoint file path(s).")
    parser.add_argument(
        "--output",
        "-o",
        default="Sample Images",
        help="Output directory. Default: %(default)s",
    )
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=64,
        help="Number of samples per checkpoint. Default: %(default)s",
    )
    parser.add_argument(
        "--grid-size",
        "-g",
        type=int,
        default=8,
        help="Images per row in the grid. Default: %(default)s",
    )
    parser.add_argument(
        "--z-dim",
        type=int,
        default=128,
        help="Latent dimension. Default: %(default)s",
    )
    parser.add_argument(
        "--use-raw-generator",
        action="store_true",
        help="Use net_G even when net_G_ema is available.",
    )

    args = parser.parse_args()

    print(f"Starting sample generation from {len(args.checkpoints)} checkpoint(s)")
    print(f"Output: {args.output}")
    print(f"Samples per checkpoint: {args.num_samples}")
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    successful_generations = []
    failed_generations = []

    for checkpoint_path in args.checkpoints:
        result = process_checkpoint(
            checkpoint_path,
            args.output,
            args.num_samples,
            args.grid_size,
            args.z_dim,
            prefer_ema=not args.use_raw_generator,
        )
        if result:
            successful_generations.append(result)
        else:
            failed_generations.append(checkpoint_path)

    print("\n" + "=" * 60)
    print("Generation summary")
    print("=" * 60)

    if successful_generations:
        print(f"Generated {len(successful_generations)} sample grid(s):")
        for path in successful_generations:
            print(f"  {path}")

    if failed_generations:
        print(f"Failed to generate {len(failed_generations)} sample grid(s):")
        for path in failed_generations:
            print(f"  {path}")

    print(f"\nOutput directory: {os.path.abspath(args.output)}")
    return 0 if successful_generations else 1


if __name__ == "__main__":
    sys.exit(main())
