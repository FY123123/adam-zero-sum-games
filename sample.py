#!/usr/bin/env python3
"""
Script to generate sample images from checkpoint files.
Usage: python sample.py <checkpoint_path1> [checkpoint_path2] ... [options]
"""

import os
import sys
import torch
import argparse
import re
import glob
from pathlib import Path
from torchvision.utils import make_grid, save_image
import source.models.wgangp as models

# Global device setup
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                      else "cpu")

def parse_checkpoint_params(checkpoint_path):
    """
    Parse parameters from checkpoint filename and path
    Format: lr{lr}_beta{beta1}_{beta2}_{optimizer}_{arch}_{dataset}.pt
    """
    filename = os.path.basename(checkpoint_path).replace('.pt', '')
    filepath = checkpoint_path
    
    # Parse learning rate
    lr_pattern = r'lr([0-9.]+)'
    lr_match = re.search(lr_pattern, filename)
    learning_rate = float(lr_match.group(1)) if lr_match else 0.0002
    
    # Parse beta parameters
    beta_pattern = r'beta([0-9.-]+)_([0-9.]+)'
    beta_match = re.search(beta_pattern, filename)
    if beta_match:
        beta1 = float(beta_match.group(1))
        beta2 = float(beta_match.group(2))
    else:
        beta1, beta2 = 0.5, 0.999
    
    # Parse optimizer
    optimizer_pattern = r'beta[0-9.-]+_[0-9.]+_([a-zA-Z_]+)_'
    optimizer_match = re.search(optimizer_pattern, filename)
    optimizer = optimizer_match.group(1) if optimizer_match else 'adam'
    
    # Parse architecture and dataset
    # First try to match pattern from filename: _arch_dataset.pt
    arch_dataset_pattern = r'_([a-zA-Z0-9]+)_([a-zA-Z0-9]+)\.pt$'
    arch_match = re.search(arch_dataset_pattern, filename)
    
    if arch_match:
        arch = arch_match.group(1)
        dataset = arch_match.group(2)
    else:
        # Try to identify architecture from filename
        if 'res32' in filename:
            arch = 'res32'
        elif 'cnn32' in filename:
            arch = 'cnn32'
        else:
            arch = 'res32'  # default
        
        # Identify dataset from file path
        if 'STL10' in filepath or 'stl10' in filepath:
            dataset = 'stl10'
        elif 'cifar10' in filepath or 'CIFAR-10' in filepath:
            dataset = 'cifar10'
        elif 'cifar10' in filename:
            dataset = 'cifar10'
        elif 'stl10' in filename:
            dataset = 'stl10'
        else:
            dataset = 'cifar10'  # default
    
    return {
        'learning_rate': learning_rate,
        'beta1': beta1,
        'beta2': beta2,
        'optimizer': optimizer,
        'arch': arch,
        'dataset': dataset,
        'filename': filename
    }

def create_model_from_params(params, z_dim=128):
    """Create corresponding generator model based on parameters"""
    arch = params['arch']
    dataset = params['dataset']
    
    # Determine output size based on dataset and architecture
    if arch == 'res32':
        # ResGenerator32 now supports dynamic output size
        if dataset == 'stl10':
            output_size = 64  # STL-10 uses 64x64
        else:  # cifar10
            output_size = 32  # CIFAR-10 uses 32x32
        net_G = models.ResGenerator32(z_dim, output_size)
    elif arch == 'cnn32':
        # CNN32 can adjust output size based on dataset
        if dataset == 'stl10':
            output_size = 64  # STL-10 uses 64x64
        else:  # cifar10
            output_size = 32  # CIFAR-10 uses 32x32
        net_G = models.Generator32(z_dim, output_size=output_size)
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    
    print(f"   📐 Output size: {output_size}x{output_size} for {dataset} dataset")
    return net_G.to(device)

def load_checkpoint(checkpoint_path, net_G):
    """Load checkpoint to model"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Try different key names to load generator weights
    generator_keys = ['net_G', 'generator', 'G', 'model']
    net_G_state = None
    
    for key in generator_keys:
        if key in checkpoint:
            net_G_state = checkpoint[key]
            break
    
    if net_G_state is None:
        raise ValueError(f"Could not find generator weights in checkpoint. Available keys: {list(checkpoint.keys())}")
    
    net_G.load_state_dict(net_G_state)
    net_G.eval()
    
    print(f" Successfully loaded checkpoint: {checkpoint_path}")
    return net_G

def generate_sample_images(net_G, output_dir, params, num_samples=64, grid_size=8):
    """Generate sample images and save"""
    os.makedirs(output_dir, exist_ok=True)
    
    net_G.eval()
    with torch.no_grad():
        # Generate random noise
        z = torch.randn(num_samples, 128).to(device)  # Assume z_dim=128
        
        # Generate images
        fake_images = net_G(z)
        
        # Normalize to [0,1]
        fake_images = (fake_images + 1) / 2
        
        # Create grid
        grid = make_grid(fake_images, nrow=grid_size, padding=2, normalize=False)
        
        # Generate filename: betas values + network architecture + dataset name
        beta1_str = str(params['beta1']).replace('-', 'neg').replace('.', '')
        beta2_str = str(params['beta2']).replace('.', '')
        arch = params['arch']
        dataset = params['dataset']
        
        filename = f"beta{beta1_str}_{beta2_str}_{arch}_{dataset}.png"
        grid_path = os.path.join(output_dir, filename)
        
        save_image(grid, grid_path)
        print(f" Generated sample grid: {grid_path}")
    
    return grid_path

def process_checkpoint(checkpoint_path, output_dir, num_samples, grid_size, z_dim):
    """Process single checkpoint file"""
    try:
        print(f"\n Processing: {checkpoint_path}")
        print("-" * 60)
        
        # 1. Parse checkpoint parameters
        params = parse_checkpoint_params(checkpoint_path)
        print(f"   Learning Rate: {params['learning_rate']}")
        print(f"   Beta1: {params['beta1']}, Beta2: {params['beta2']}")
        print(f"   Optimizer: {params['optimizer']}")
        print(f"   Architecture: {params['arch']}")
        print(f"   Dataset: {params['dataset']}")
        
        # 2. Create model
        net_G = create_model_from_params(params, z_dim)
        
        # 3. Load checkpoint
        net_G = load_checkpoint(checkpoint_path, net_G)
        
        # 4. Generate sample images
        grid_path = generate_sample_images(net_G, output_dir, params, num_samples, grid_size)
        
        return grid_path
        
    except Exception as e:
        print(f" Error processing {checkpoint_path}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Generate sample images from checkpoint files')
    parser.add_argument('checkpoints', nargs='+', help='Path(s) to checkpoint file(s)')
    parser.add_argument('--output', '-o', default='Sample Images', 
                       help='Output directory (default: Sample Images)')
    parser.add_argument('--num-samples', '-n', type=int, default=64,
                       help='Number of samples to generate (default: 64)')
    parser.add_argument('--grid-size', '-g', type=int, default=8,
                       help='Grid size for the sample grid (default: 8)')
    parser.add_argument('--z-dim', type=int, default=128,
                       help='Latent dimension (default: 128)')
    
    args = parser.parse_args()
    
    print(f" Starting sample generation from {len(args.checkpoints)} checkpoint(s)")
    print(f" Output: {args.output}")
    print(f" Samples per checkpoint: {args.num_samples}")
    print(f" Grid size: {args.grid_size}x{args.grid_size}")
    print(f" Device: {device}")
    print("=" * 60)
    
    successful_generations = []
    failed_generations = []
    
    # Process each checkpoint
    for checkpoint_path in args.checkpoints:
        result = process_checkpoint(checkpoint_path, args.output, args.num_samples, args.grid_size, args.z_dim)
        if result:
            successful_generations.append(result)
        else:
            failed_generations.append(checkpoint_path)
    
    # Output summary
    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    
    if successful_generations:
        print(f" Successfully generated {len(successful_generations)} sample(s):")
        for path in successful_generations:
            print(f" {path}")
    
    if failed_generations:
        print(f" Failed to generate {len(failed_generations)} sample(s):")
        for path in failed_generations:
            print(f"  {path}")
    
    print(f"\n Output directory: {os.path.abspath(args.output)}")
    
    if successful_generations:
        print(" Sample generation completed successfully!")
        return 0
    else:
        print(" No samples were generated successfully.")
        return 1

if __name__ == "__main__":
    sys.exit(main())