import os
import sys
import numpy as np
import torch
import torch.optim as optim
from absl import flags, app
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from tensorboardX import SummaryWriter
from tqdm import trange
from pytorch_image_generation_metrics import get_inception_score
import time
# from torch.amp import GradScaler, autocast  # Mixed precision training disabled
import glob  # Import glob for scanning
import matplotlib.pyplot as plt # Import matplotlib.pyplot for plotting

# Global device setup - run once only
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
                      else "cpu")

# Print device information directly (execute once only)
if device.type == "cuda":
    print("\r Using CUDA GPU acceleration", end='', flush=True)
elif device.type == "mps":
    print("\r Using MacBook MPS hardware acceleration", end='', flush=True)
else:
    print("\r Using CPU for training", end='', flush=True)

# Disable CUDA warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import source.models.wgangp as models
import source.models.transgan as transgan_models
import source.models.sagan as sagan_models
import source.losses as losses
from source.utils import generate_imgs, infiniteloop, set_seed
from source.diff_aug import DiffAugment
from copy import deepcopy

class Adam_NM(optim.Optimizer):
    def __init__(self, params, lr=0.001, beta1=-0.9, beta2=0.999, eps=1e-8):
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        super(Adam_NM, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for param in group['params']:
                if param.grad is None:
                    continue

                grad = param.grad.data
                state = self.state[param]

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(param.data)
                    state['v'] = torch.zeros_like(param.data)

                m, v = state['m'], state['v']
                beta1, beta2 = group['beta1'], group['beta2']
                state['step'] += 1

                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                v.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                m_hat = m / (1 - beta1 ** state['step'])
                v_hat = v / (1 - beta2 ** state['step'])

                param.data.addcdiv_(m_hat, (v_hat.sqrt() + group['eps']), value=-group['lr'])


        return loss


def create_cnn_generator(z_dim):
    """Create CNN generator, determine output size based on dataset"""
    dataset = FLAGS.dataset
    if dataset == 'stl10':
        output_size = 64  # STL-10 uses 64x64 for non-res32 architectures
    else:  # cifar10
        output_size = 32
    return models.Generator32(z_dim, output_size=output_size)

def create_res_generator(z_dim):
    """Create Res32 generator, determine output size based on dataset"""
    dataset = FLAGS.dataset
    if dataset == 'stl10':
        output_size = 64  # STL-10 uses 64x64
    else:  # cifar10
        output_size = 32  # CIFAR-10 uses 32x32
    return models.ResGenerator32(z_dim, output_size)

def create_trans_generator(z_dim):
    """Create TransGAN generator (64x64 output)"""
    return transgan_models.TransGenerator64(z_dim)

def create_sagan_generator(z_dim):
    """Create SAGAN generator (64x64 output)"""
    return sagan_models.SAResGenerator64(z_dim)

net_G_models = {
    'res32': create_res_generator,
    'cnn32': create_cnn_generator,
    'trans64': create_trans_generator,
    'sagan64': create_sagan_generator,
}

net_D_models = {
    'res32': models.ResDiscriminator32,
    'cnn32': models.Discriminator32,
    'trans64': transgan_models.TransDiscriminator64,
    'sagan64': sagan_models.SAResDiscriminator64,
}

loss_fns = {
    'bce': losses.BCEWithLogits,
    'hinge': losses.Hinge,
    'was': losses.Wasserstein,
    'softplus': losses.Softplus
}

FLAGS = flags.FLAGS
flags.DEFINE_enum('dataset', 'cifar10', ['cifar10', 'stl10', 'celeba'], "dataset")
flags.DEFINE_enum('arch', 'res32', net_G_models.keys(), "architecture")
flags.DEFINE_integer('total_steps',30000, "total number of training steps")
flags.DEFINE_integer('final_total_steps', 100000, "total number of training steps")

flags.DEFINE_integer('batch_size', 128, "batch size")

flags.DEFINE_float('lr_G', 2e-4, "Generator learning rate")
flags.DEFINE_float('lr_D', 2e-4, "Discriminator learning rate")
flags.DEFINE_list('betas', ['0.6', '0.9'], "for Adam (beta1, beta2)")
flags.DEFINE_enum('optimizer', 'adam', ['adam', 'adam_nm', 'sgd'], "optimizer type: adam, adam_nm, or sgd")

flags.DEFINE_integer('n_dis', 1, "update Generator every this steps")
flags.DEFINE_integer('z_dim', 128, "latent space dimension")
flags.DEFINE_float('alpha', 10, "gradient penalty")
flags.DEFINE_enum('loss', 'was', loss_fns.keys(), "loss function")
flags.DEFINE_integer('seed', 0, "random seed")
# logging
flags.DEFINE_integer('print', 1, "print gradient norm")
flags.DEFINE_integer('eval_step', 1000, "evaluate Inception Score")  # Evaluate IS every N steps
flags.DEFINE_integer('gd_step', 1, "evaluate gradient norm")
flags.DEFINE_integer('sample_step', 100, "sample image every this steps")
flags.DEFINE_integer('sample_size', 64, "sampling size of images")
flags.DEFINE_string('logdir', './logs/WGANGP_CIFAR10_RES2', 'logging folder (will be auto-adjusted for STL-10)')

# Training configuration
flags.DEFINE_list('train_modes', ['standard'], "training mode: standard(standard), negative_momentum(negative momentum), betas_variants(beta variants)")
flags.DEFINE_bool('enable_all_modes', False, "enable all training modes")
flags.DEFINE_bool('record', True, "record inception score")
flags.DEFINE_bool('enable_fid', True, "enable FID (Fréchet Inception Distance) evaluation alongside IS")
flags.DEFINE_string('fid_stats', None, 'path to precomputed FID real image stats (.npz). If None, auto-computed from training data and cached in ./fid_stats/.')

# generate
flags.DEFINE_bool('generate', False, 'generate images')
flags.DEFINE_string('pretrain', None, 'path to test model')
flags.DEFINE_string('output', './outputs', 'path to output dir')
flags.DEFINE_integer('num_images', 20000, 'the number of generated images')
flags.DEFINE_bool('amp', True, "enable mixed precision training (BF16) for faster speed")
flags.DEFINE_string('diff_aug', 'color,translation,cutout', "DiffAugment policy applied to D inputs (set '' to disable)")
flags.DEFINE_float('ema_decay', 0.999, "EMA decay rate for generator (0 to disable)")

# Device information already defined and printed at file top


def cacl_gradient_penalty(net_D, real, fake):
    """Calculate gradient penalty, optimized for mixed precision training"""
    t = torch.rand(real.size(0), 1, 1, 1, device=real.device, dtype=real.dtype)
    t = t.expand(real.size())

    interpolates = t * real + (1 - t) * fake
    interpolates.requires_grad_(True)
    
    # Ensure calculation in mixed precision context
    disc_interpolates = net_D(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True, retain_graph=True)[0]

    grad_norm = torch.norm(torch.flatten(grad, start_dim=1), dim=1)
    loss_gp = torch.mean((grad_norm - 1) ** 2)
    return loss_gp


def generate():
    assert FLAGS.pretrain is not None, "set model weight by --pretrain [model]"

    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    ckpt = torch.load(FLAGS.pretrain, weights_only=False)
    if 'net_G_ema' in ckpt:
        net_G.load_state_dict(ckpt['net_G_ema'])
    else:
        net_G.load_state_dict(ckpt['net_G'])
    net_G.eval()

    counter = 0
    os.makedirs(FLAGS.output)
    with torch.no_grad():
        for start in trange(
                0, FLAGS.num_images, FLAGS.batch_size, dynamic_ncols=True):
            batch_size = min(FLAGS.batch_size, FLAGS.num_images - start)
            z = torch.randn(batch_size, FLAGS.z_dim).to(device)
            x = net_G(z).cpu()
            x = (x + 1) / 2
            for image in x:
                save_image(
                    image, os.path.join(FLAGS.output, '%d.png' % counter))
                counter += 1


def auto_plot_training_results(beta1, beta2, optimizer_type, arch=None):
    """Automatically generate visualization charts for training results"""
    try:
        print(" Starting to generate training result charts...")

        # If no architecture specified, use default value from FLAGS
        if arch is None:
            arch = FLAGS.arch

        # Check if corresponding .npz file exists
        npz_filename = f"lr{FLAGS.lr_G}_beta{beta1}_{beta2}_{optimizer_type}_{arch}.npz"

        # Set npz file path based on architecture type
        if arch == 'cnn32':
            # CNN models use dedicated npz directory structure
            npz_dir = os.path.join(os.getcwd(), 'npz', 'cnn_npz')
        else:
            # Other architectures use original logic
            npz_dir = os.path.join(os.getcwd(), f'{optimizer_type}_npz')
        npz_filepath = os.path.join(npz_dir, npz_filename)
        
        if os.path.exists(npz_filepath):
            print(f" Found data file: {npz_filename}")
            
            # Load data
            data = np.load(npz_filepath)
            steps = data["steps"]
            steps_GN = data["steps_GN"]
            IS = data["IS"]
            IS_std = data["IS_std"]
            Total_grad_norms = data["Total_grad_norms"]
            FID_vals = data.get("FID", np.array([]))
            steps_FID = data.get("steps_FID", np.array([]))
            
            # Create chart directory
            plot_dir = os.path.join(FLAGS.logdir, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            
            # 1. Inception Score curve plot
            if len(steps) > 0 and len(IS) > 0:
                plt.figure(figsize=(8, 6))
                plt.plot(steps, IS, label=f'β1={beta1}, β2={beta2}', color='darkblue', linewidth=2)
                if len(IS_std) > 0 and len(IS_std) == len(IS):
                    plt.fill_between(steps, IS - IS_std, IS + IS_std, alpha=0.2, color='darkblue')
                plt.xlabel('Training Steps', fontsize=14)
                plt.ylabel('Inception Score', fontsize=14)
                plt.title(f'Inception Score vs Training Steps ({FLAGS.arch})', fontsize=16)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'IS_curve.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(" Generated Inception Score curve")
            

            # 3. Gradient norm curve plot
            if len(steps_GN) > 0 and len(Total_grad_norms) > 0:
                plt.figure(figsize=(8, 6))
                plt.plot(steps_GN, Total_grad_norms, label=f'β1={beta1}, β2={beta2}', color='darkgreen', linewidth=2)
                plt.xlabel('Training Steps', fontsize=14)
                plt.ylabel('Total Gradient Norms', fontsize=14)
                plt.title(f'Gradient Norms vs Training Steps ({FLAGS.arch})', fontsize=16)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(plot_dir, 'Gradient_Norms_curve.png'), dpi=300, bbox_inches='tight')
                plt.close()
                print(" Generated Gradient Norms curve")
                
                # 4. Cumulative average gradient norm curve plot
                def cumulative_average_after_step(data, start_step=0):
                    """Calculate cumulative average from specified step"""
                    if start_step >= len(data):
                        return np.array([])
                    data_after_step = data[start_step:] 
                    if len(data_after_step) == 0:
                        return np.array([])
                    return np.cumsum(data_after_step) / np.arange(1, len(data_after_step) + 1)
                
                # Calculate cumulative average (from step 0)
                start_steps = 0
                cum_avg_grad_norms = cumulative_average_after_step(Total_grad_norms, start_step=start_steps)
                steps_GN_later = steps_GN[start_steps:]
                
                if len(cum_avg_grad_norms) > 0:
                    plt.figure(figsize=(8, 6))
                    plt.plot(steps_GN_later, cum_avg_grad_norms, 
                            label=f'β1={beta1}, β2={beta2} (Cumulative Avg)', 
                            color='purple', linewidth=2)
                    plt.xlabel('Training Steps', fontsize=14)
                    plt.ylabel('Cumulative Average Gradient Norms', fontsize=14)
                    plt.title(f'Cumulative Average Gradient Norms vs Training Steps ({FLAGS.arch})', fontsize=16)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, 'Cumulative_Avg_Gradient_Norms.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(" Generated Cumulative Average Gradient Norms curve")

            # FID curve plot
            if len(steps_FID) > 0 and len(FID_vals) > 0:
                fid_arr = np.array(FID_vals, dtype=float)
                valid = ~np.isnan(fid_arr)
                if valid.any():
                    plt.figure(figsize=(8, 6))
                    plt.plot(np.array(steps_FID)[valid], fid_arr[valid],
                             label=f'β1={beta1}, β2={beta2}', color='darkorange', linewidth=2)
                    plt.xlabel('Training Steps', fontsize=14)
                    plt.ylabel('FID', fontsize=14)
                    plt.title(f'FID vs Training Steps ({FLAGS.arch})', fontsize=16)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(plot_dir, 'FID_curve.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    print(" Generated FID curve")

            print(f" All charts saved to: {plot_dir}")
            
        else:
            print(f"  Data file not found: {npz_filepath}")
            print("   Please ensure training is completed and .npz file is generated")
            
    except Exception as e:
        print(f" Error generating charts: {e}")
        print("   Please check if matplotlib is properly installed")


def prepare_fid_stats(dataloader):
    """Compute and cache FID real image statistics (runs once, result is saved to disk)."""
    from pytorch_image_generation_metrics.core import get_inception_feature
    import numpy as np

    stats_dir = os.path.join(os.getcwd(), 'fid_stats')
    os.makedirs(stats_dir, exist_ok=True)

    # Use user-supplied stats file if provided
    if FLAGS.fid_stats:
        if os.path.exists(FLAGS.fid_stats):
            print(f"\r📊 FID: Using provided stats: {FLAGS.fid_stats}                     ", end='', flush=True)
            return FLAGS.fid_stats
        raise FileNotFoundError(f"--fid_stats path not found: {FLAGS.fid_stats}")

    stats_filename = f'real_stats_{FLAGS.dataset}.npz'
    stats_path = os.path.join(stats_dir, stats_filename)

    if os.path.exists(stats_path):
        print(f"\r📊 FID: Loaded cached real stats ({stats_filename})                     ", end='', flush=True)
        return stats_path

    print(f"\r⏳ FID: Computing real image statistics (first time only, will be cached)...", end='', flush=True)

    num_samples = 5000
    real_imgs_list = []
    total = 0
    for imgs, _ in dataloader:
        imgs = (imgs + 1) / 2  # [-1, 1] -> [0, 1]
        if FLAGS.dataset in ('stl10', 'celeba'):
            imgs = torch.nn.functional.interpolate(
                imgs, size=(299, 299), mode='bicubic', align_corners=False
            ).clamp(0.0, 1.0)
        real_imgs_list.append(imgs.cpu())
        total += imgs.shape[0]
        if total >= num_samples:
            break

    real_imgs = torch.cat(real_imgs_list)[:num_samples].clamp(0.0, 1.0).float()

    acts, = get_inception_feature(real_imgs, dims=[2048], use_torch=False)
    mu = np.mean(acts, axis=0)
    sigma = np.cov(acts, rowvar=False)
    np.savez(stats_path, mu=mu, sigma=sigma)

    print(f"\r✅ FID: Real image stats saved to {stats_filename}                     ", end='', flush=True)
    return stats_path


def train(beta1=None, beta2=None, optimizer_type=None):
    # Initialize training (mixed precision disabled)
    scaler = None
    # print(" Mixed precision training disabled")  # Remove mixed precision prompt
    # print("   - Using FP32 for all operations")
    # print("   - Better numerical stability")
    # print("   - Higher memory usage but more reliable training")
    
    start_time = time.time()

    # Use passed beta values, if not passed use default values from FLAGS
    if beta1 is None:
        beta1 = float(FLAGS.betas[0])
    if beta2 is None:
        beta2 = float(FLAGS.betas[1])
    if optimizer_type is None:
        optimizer_type = FLAGS.optimizer  # If not specified, use FLAGS.optimizer

    # Dataset-specific: automatically adjust log directory
    if FLAGS.dataset == 'stl10':
        FLAGS.logdir = './logs/STL10_WGANGP_RES2'
        print(f"\r📂 STL-10: Using log directory: {FLAGS.logdir}", end='', flush=True)
    elif FLAGS.dataset == 'celeba':
        FLAGS.logdir = f'./logs/CelebA_WGANGP_{FLAGS.arch.upper()}'
        print(f"\r📂 CelebA: Using log directory: {FLAGS.logdir}", end='', flush=True)

    print(f"\r🎯 Training configuration: optimizer={optimizer_type}, beta1={beta1}, beta2={beta2}                           ", end='', flush=True)

    # Define checkpoint and NumPy file names, use new naming format, include dataset information
    checkpoint_name = f'lr{FLAGS.lr_G}_beta{beta1}_{beta2}_{optimizer_type}_{FLAGS.arch}_{FLAGS.dataset}.pt'

    # Set checkpoint save path based on architecture type
    dataset_dirname = {'cifar10': 'CIFAR-10', 'stl10': 'STL-10', 'celeba': 'CelebA'}[FLAGS.dataset]
    if FLAGS.arch == 'cnn32':
        if FLAGS.dataset == 'stl10':
            checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoint', 'cnn_checkpoint', 'STL-10')
        else:
            checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoint', 'cnn_checkpoint', dataset_dirname)
    elif FLAGS.arch == 'trans64':
        checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoint', 'trans64_checkpoint', dataset_dirname)
    elif FLAGS.arch == 'sagan64':
        checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoint', 'sagan64_checkpoint', dataset_dirname)
    else:
        if FLAGS.dataset == 'stl10':
            checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoint', 'STL10_adam_nm')
        else:
            checkpoint_dir = os.path.join(os.getcwd(), 'Checkpoint', f'{optimizer_type}_Checkpoint')
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)

    # Ensure directory exists before retrieving checkpoint
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Define NumPy data file names and paths
    npz_filename = f"lr{FLAGS.lr_G}_beta{beta1}_{beta2}_{optimizer_type}_{FLAGS.arch}_{FLAGS.dataset}.npz"

    # Set npz save path based on architecture type
    if FLAGS.arch == 'cnn32':
        if FLAGS.dataset == 'stl10':
            npz_dir = os.path.join(os.getcwd(), 'npz', 'cnn_npz', 'STL-10')
        else:
            npz_dir = os.path.join(os.getcwd(), 'npz', 'cnn_npz', dataset_dirname)
    elif FLAGS.arch == 'trans64':
        npz_dir = os.path.join(os.getcwd(), 'npz', 'trans64_npz', dataset_dirname)
    elif FLAGS.arch == 'sagan64':
        npz_dir = os.path.join(os.getcwd(), 'npz', 'sagan64_npz', dataset_dirname)
    else:
        if FLAGS.dataset == 'stl10':
            npz_dir = os.path.join(os.getcwd(), 'npz', 'STL10_adam_nm_npz')
        else:
            npz_dir = os.path.join(os.getcwd(), 'npz', f'{optimizer_type}_npz')
    npz_filepath = os.path.join(npz_dir, npz_filename)

    # Create necessary directories
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)
    
    # Initialize networks
    net_G = net_G_models[FLAGS.arch](FLAGS.z_dim).to(device)
    net_D = net_D_models[FLAGS.arch]().to(device)
    loss_fn = loss_fns[FLAGS.loss]()

    # EMA: keep a shadow copy of generator weights for evaluation
    ema_enabled = FLAGS.ema_decay > 0
    if ema_enabled:
        net_G_ema = deepcopy(net_G)
        net_G_ema.eval()
        for p in net_G_ema.parameters():
            p.requires_grad_(False)

    # Create optimizer based on optimizer type
    if optimizer_type == 'adam_nm':
        optim_G = Adam_NM(net_G.parameters(), lr=FLAGS.lr_G, beta1=beta1, beta2=beta2)
        optim_D = Adam_NM(net_D.parameters(), lr=FLAGS.lr_D, beta1=beta1, beta2=beta2)
    elif optimizer_type == 'sgd':
        # SGD optimizer, use beta1 as momentum parameter, ignore beta2
        optim_G = optim.SGD(net_G.parameters(), lr=FLAGS.lr_G, momentum=beta1)
        optim_D = optim.SGD(net_D.parameters(), lr=FLAGS.lr_D, momentum=beta1)
        print(f" Using SGD optimizer with momentum={beta1}")
    else:  # Default use adam
        optim_G = optim.Adam(net_G.parameters(), lr=FLAGS.lr_G, betas=[beta1, beta2])
        optim_D = optim.Adam(net_D.parameters(), lr=FLAGS.lr_D, betas=[beta1, beta2])
    
    sched_G = optim.lr_scheduler.LambdaLR(
        optim_G, lambda step: 1 - step / FLAGS.final_total_steps)
    sched_D = optim.lr_scheduler.LambdaLR(
        optim_D, lambda step: 1 - step / FLAGS.final_total_steps)

    # Scan directory to find matching files
    possible_checkpoints = glob.glob(os.path.join(checkpoint_dir, checkpoint_name))

    if possible_checkpoints:
        # Select latest file (based on modification time)
        checkpoint_path = max(possible_checkpoints, key=os.path.getmtime)
        found_checkpoint_name = os.path.basename(checkpoint_path)

        # Ensure found filename exactly matches expected filename, including dataset part
        if found_checkpoint_name == checkpoint_name:
            checkpoint_name = found_checkpoint_name
            print(f"\r🎯 Found matching checkpoint: {checkpoint_name}                                                         ", end='', flush=True)
        else:
            print(f"\r⚠️  Found checkpoint with different name: {found_checkpoint_name} (expected: {checkpoint_name})", end='', flush=True)
            # If filename does not exactly match, start training from scratch
            possible_checkpoints = []
    else:
        print(f"\r🆕 No matching checkpoint found, training from scratch                                                         ", end='', flush=True)

    if possible_checkpoints:
        # print(f"📁 Path: {checkpoint_path}")  # Simplified prompt
        # print("⚠️   Note: Using default torch.load to load checkpoint (includes training data)")  # Simplified prompt
        # print("    If higher security is required, you can consider using weights_only=True, but training needs to be restarted")  # Simplified prompt
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        
        # Verify configuration (ignore total_steps)
        if (checkpoint.get('arch') == FLAGS.arch and 
            checkpoint.get('beta1') == beta1 and 
            checkpoint.get('beta2') == beta2 and
            checkpoint.get('optimizer_type') == optimizer_type):
            
            net_G.load_state_dict(checkpoint['net_G'])
            net_D.load_state_dict(checkpoint['net_D'])
            optim_G.load_state_dict(checkpoint['optim_G'])
            optim_D.load_state_dict(checkpoint['optim_D'])
            sched_G.load_state_dict(checkpoint['sched_G'])
            sched_D.load_state_dict(checkpoint['sched_D'])
            if ema_enabled and 'net_G_ema' in checkpoint:
                net_G_ema.load_state_dict(checkpoint['net_G_ema'])
        
            start_step = checkpoint['current_step'] + 1  # Changed to +1, continue from next step
            
            # Try to load existing NumPy data file
            if os.path.exists(npz_filepath):
                # print(f"📊 Loading existing data file: {npz_filename}")  # Simplified prompt
                existing_data = np.load(npz_filepath)
                IS_table = list(existing_data['IS'])
                IS_std_table = list(existing_data['IS_std'])
                step_table = list(existing_data['steps'])
                Total_grad_norms = list(existing_data['Total_grad_norms'])
                step_table_GN = list(existing_data['steps_GN'])
                G_grad_norms = list(existing_data.get('G_grad_norms', []))
                D_grad_norms = list(existing_data.get('D_grad_norms', []))
                FID_table = list(existing_data.get('FID', []))
                fid_step_table = list(existing_data.get('steps_FID', []))

                # Check for duplicate data to avoid duplicate records
                if step_table and step_table[-1] >= start_step:
                    # print(f"⚠️   Detected duplicate data, recording from step {start_step} onwards")  # Simplified prompt
                    # Remove duplicate parts
                    while step_table and step_table[-1] >= start_step:
                        IS_table.pop()
                        IS_std_table.pop()
                        step_table.pop()
                        if Total_grad_norms:
                            Total_grad_norms.pop()
                        if step_table_GN:
                            step_table_GN.pop()
                        if G_grad_norms:
                            G_grad_norms.pop()
                        if D_grad_norms:
                            D_grad_norms.pop()
            else:
                IS_table = []
                IS_std_table = []
                step_table = []
                step_table_GN = []
                G_grad_norms = []
                D_grad_norms = []
                Total_grad_norms = []
                FID_table = []
                fid_step_table = []
            print(f"🚀 Continuing training from step {start_step}...")
        else:
            # print(f"⚠️  Checkpoint configuration does not match, training from scratch")  # Simplified prompt
            # print(f"    Expected: arch={FLAGS.arch}, beta1={beta1}, beta2={beta2}, optimizer={optimizer_type}")
            # print(f"    Actual: arch={checkpoint.get('arch')}, beta1={checkpoint.get('beta1')}, beta2={checkpoint.get('beta2')}, optimizer={checkpoint.get('optimizer_type')}")
            IS_table = []
            IS_std_table = []
            step_table = []
            step_table_GN = []
            G_grad_norms = []
            D_grad_norms = []
            FID_table = []
            fid_step_table = []
            start_step = 1
    else:
        print(f"🆕 No matching checkpoint found, training from scratch")
        # Dynamically generate checkpoint filename, use new naming format, include dataset information
        checkpoint_name = f'lr{FLAGS.lr_G}_beta{beta1}_{beta2}_{optimizer_type}_{FLAGS.arch}_{FLAGS.dataset}.pt'
        print(f"📁 Will save to: {os.path.join(checkpoint_dir, checkpoint_name)}")
        IS_table = []
        IS_std_table = []
        step_table = []
        step_table_GN = []
        G_grad_norms = []
        D_grad_norms = []
        Total_grad_norms = []
        FID_table = []
        fid_step_table = []
        start_step = 1  

    if FLAGS.dataset == 'cifar10':
        dataset = datasets.CIFAR10(
            './data', train=True, download=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
    if FLAGS.dataset == 'stl10':
        # STL-10 specific: optimize training data preprocessing
        # Now all architectures support 64x64 STL-10 training
        dataset = datasets.STL10(
            './data', split='unlabeled', download=True,
            transform=transforms.Compose([
                transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]))
        print(f"📸 STL-10: Using {FLAGS.arch} architecture with optimized preprocessing")
        print(f"   - Training image size: 64x64 (all architectures now support full resolution)")
        print(f"   - IS evaluation will use 299x299 resized images with 10 splits")
    if FLAGS.dataset == 'celeba':
        from source.celeba_dataset import HFCelebA
        celeba_transform = transforms.Compose([
            transforms.Resize(64, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = HFCelebA(transform=celeba_transform)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4,
        drop_last=True)


    # Create sample directory (dataset-specific folders)
    sample_dir_map = {'stl10': 'STL10_sample', 'celeba': 'CelebA_sample'}
    sample_dir = sample_dir_map.get(FLAGS.dataset, 'sample')
    os.makedirs(os.path.join(FLAGS.logdir, sample_dir), exist_ok = True)
    writer = SummaryWriter(os.path.join(FLAGS.logdir))
    sample_z = torch.randn(FLAGS.sample_size, FLAGS.z_dim).to(device)
    with open(os.path.join(FLAGS.logdir, "flagfile.txt"), 'w') as f:
        f.write(FLAGS.flags_into_string())
    # writer.add_text(
    #     "flagfile", FLAGS.flags_into_string().replace('\n', '  \n'))

    real, _ = next(iter(dataloader))
    grid = (make_grid(real[:FLAGS.sample_size]) + 1) / 2
    # writer.add_image('real_sample', grid)

    looper = infiniteloop(dataloader)

    # Prepare FID statistics (computed once from real data and cached to disk)
    fid_stats_path = None
    fid_enabled = FLAGS.enable_fid
    if fid_enabled:
        try:
            fid_stats_path = prepare_fid_stats(dataloader)
        except Exception as e:
            print(f"\r⚠️  FID: Setup failed ({e}). FID evaluation disabled.          ", end='', flush=True)
            fid_enabled = False

    amp_enabled = FLAGS.amp and device.type == 'cuda'
    amp_dtype = torch.bfloat16 if amp_enabled else None
    if amp_enabled:
        print(f"\r⚡ Mixed precision (BF16) enabled", flush=True)

    print(f"\r🚀 Starting training from step {start_step} to {FLAGS.total_steps} ...")

    for step in range(start_step, FLAGS.total_steps + 1):
        # Dynamically display training progress (global progress based on total training steps)
        progress = step / FLAGS.total_steps
        bar_length = 30
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '░' * (bar_length - filled_length)
        percent = progress * 100
        print(f"\r🚀 Training: [{bar}] {percent:.1f}% ({step}/{FLAGS.total_steps})", end='', flush=True)
        
        # Discriminator
        for _ in range(FLAGS.n_dis):
            with torch.no_grad():
                z = torch.randn(FLAGS.batch_size, FLAGS.z_dim).to(device)
                with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                    fake = net_G(z).detach()
            real = next(looper).to(device)

            real_aug = DiffAugment(real, policy=FLAGS.diff_aug)
            fake_aug = DiffAugment(fake, policy=FLAGS.diff_aug)
            
            # Train discriminator
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                net_D_real = net_D(real_aug)
                net_D_fake = net_D(fake_aug)
                loss = loss_fn(net_D_real, net_D_fake)
            # GP must stay in FP32 for numerical stability
            loss_gp = cacl_gradient_penalty(net_D, real.float(), fake.float())
            loss_all = loss + FLAGS.alpha * loss_gp

            optim_D.zero_grad()
            loss_all.backward()
            optim_D.step()

            D_grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in net_D.parameters() if p.grad is not None])).item()
            D_grad_norms.append(D_grad_norm)

            if FLAGS.loss == 'was':
                loss = -loss

            for p in net_D.parameters():
                p.requires_grad_(False)
            z = torch.randn(FLAGS.batch_size * 2, FLAGS.z_dim).to(device)
            
            # Train generator (D sees augmented fakes)
            with torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled):
                gen_out = net_G(z)
                gen_aug = DiffAugment(gen_out, policy=FLAGS.diff_aug)
                loss = loss_fn(net_D(gen_aug))
            optim_G.zero_grad()
            loss.backward()
            optim_G.step()

            # EMA update
            if ema_enabled:
                with torch.no_grad():
                    for p_ema, p in zip(net_G_ema.parameters(), net_G.parameters()):
                        p_ema.data.lerp_(p.data, 1 - FLAGS.ema_decay)

            G_grad_norm = torch.norm(torch.cat([p.grad.view(-1) for p in net_G.parameters() if p.grad is not None])).item()
            G_grad_norms.append(G_grad_norm)

            for p in net_D.parameters():
                p.requires_grad_(True)

            

            # writer.add_scalar('Total_grad_norms', G_grad_norm + D_grad_norm, step)
            Total_grad_norms.append(G_grad_norm + D_grad_norm)
            step_table_GN.append(step)



            sched_G.step()
            sched_D.step()

            if step == 1 or step % FLAGS.sample_step == 0:
                eval_G = net_G_ema if ema_enabled else net_G
                fake = eval_G(sample_z).cpu()
                grid = (make_grid(fake) + 1) / 2
                save_image(grid, os.path.join(
                    FLAGS.logdir, sample_dir, '%d.png' % step))

            # Show training progress (display every 1000 steps)
            if step % 1000 == 0:
                print(f"\r📊 Training progress: {step}/{FLAGS.total_steps} | Loss: {loss:.4f}", end='', flush=True)

            if step == 1 or step % FLAGS.eval_step == 0:
                checkpoint_data = {
                    'net_G': net_G.state_dict(),
                    'net_D': net_D.state_dict(),
                    'optim_G': optim_G.state_dict(),
                    'optim_D': optim_D.state_dict(),
                    'sched_G': sched_G.state_dict(),
                    'sched_D': sched_D.state_dict(),
                    'IS_table': IS_table,
                    'IS_std_table': IS_std_table,
                    'step_table': step_table,
                    'G_grad_norms': G_grad_norms,
                    'D_grad_norms': D_grad_norms,
                    'Total_grad_norms': Total_grad_norms,
                    'step_table_GN': step_table_GN,
                    'FID_table': FID_table,
                    'fid_step_table': fid_step_table,
                    'current_step': step,
                    'arch': FLAGS.arch,
                    'total_steps': FLAGS.total_steps,
                    'beta1': beta1,
                    'beta2': beta2,
                    'optimizer_type': optimizer_type,
                }
                if ema_enabled:
                    checkpoint_data['net_G_ema'] = net_G_ema.state_dict()
                
                # Use new naming format to save, include dataset information
                checkpoint_name = f'lr{FLAGS.lr_G}_beta{beta1}_{beta2}_{optimizer_type}_{FLAGS.arch}_{FLAGS.dataset}.pt'
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                torch.save(checkpoint_data, checkpoint_path)
                
                print(f"\r💾 Saving checkpoint: {checkpoint_name}...", end='', flush=True)


                if FLAGS.record:
                    try:
                        print(f"\r🔍 Evaluating Inception Score at step {step}...", end='', flush=True)
                        if FLAGS.dataset == 'stl10':
                            eval_size = min(3000, FLAGS.num_images)
                            print(f"\r🎯 STL-10: Using {eval_size} samples for IS evaluation", end='', flush=True)
                        else:
                            eval_size = min(5000, FLAGS.num_images)
                        
                        eval_G = net_G_ema if ema_enabled else net_G
                        imgs = generate_imgs(
                            eval_G, device, FLAGS.z_dim,
                            eval_size, FLAGS.batch_size)
                        
                        # Ensure image format is correct
                        if imgs.max() > 1.0 or imgs.min() < 0.0:
                            imgs = torch.clamp(imgs, 0.0, 1.0)
                        
                        # Ensure correct device and data type are used
                        imgs = imgs.float()
                        

                        
                        if FLAGS.dataset in ('stl10', 'celeba'):
                            resize_transform = transforms.Compose([
                                transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BICUBIC),
                                transforms.CenterCrop(299)
                            ])

                            imgs_pil = []
                            for img in imgs:
                                img_pil = transforms.ToPILImage()(img)
                                img_resized = resize_transform(img_pil)
                                imgs_pil.append(transforms.ToTensor()(img_resized))

                            imgs_resized = torch.stack(imgs_pil, dim=0)
                            imgs_for_is = imgs_resized.clamp(0.0, 1.0).float()

                            print(f"\r🔄 Resized images from {imgs.shape[2]}x{imgs.shape[3]} to 299x299 for IS evaluation", end='', flush=True)
                        else:
                            imgs_for_is = imgs.clamp(0.0, 1.0).float()
                        
                        if imgs_for_is.std() < 0.01:
                            print(f"\r⚠️  ({imgs_for_is.std():.3f})", end='', flush=True)
                        
                        if FLAGS.dataset in ('stl10', 'celeba'):
                            IS = get_inception_score(imgs_for_is, splits=10, verbose=False)
                        else:
                            IS = get_inception_score(imgs_for_is, verbose=False)
                        
                        print(f"\r🎯 IS Evaluation Complete! Step {step}/{FLAGS.total_steps} | "
                              f"IS: {IS[0]:.3f} (±{IS[1]:.5f})", end='', flush=True)
                        
                        # writer.add_scalar('Inception_Score', IS[0], step)
                        # writer.add_scalar('Inception_Score_std', IS[1], step)

                        IS_table.append(IS[0])
                        IS_std_table.append(IS[1])
                        step_table.append(step)

                        # FID evaluation (only when enabled)
                        if fid_enabled and fid_stats_path is not None:
                            try:
                                from pytorch_image_generation_metrics import get_fid
                                fid_value = float(get_fid(imgs_for_is, fid_stats_path))
                                FID_table.append(fid_value)
                                fid_step_table.append(step)
                                print(f"\r🎯 Step {step}/{FLAGS.total_steps} | IS: {IS[0]:.3f} (±{IS[1]:.5f}) | FID: {fid_value:.2f}          ", end='', flush=True)
                            except Exception as fid_e:
                                print(f"\r⚠️  FID evaluation failed at step {step}: {fid_e}          ", end='', flush=True)
                                FID_table.append(float('nan'))
                                fid_step_table.append(step)

                    except Exception as e:
                        print(f"\r⚠️  Error during Inception Score evaluation at step {step}: {e}")
                        print(f"\r   Continuing training without metrics...", end='', flush=True)
                        IS_table.append(0.0)
                        IS_std_table.append(0.0)
                        step_table.append(step)
    writer.close()

    checkpoint_data = {
        'net_G': net_G.state_dict(),
        'net_D': net_D.state_dict(),
        'optim_G': optim_G.state_dict(),
        'optim_D': optim_D.state_dict(),
        'sched_G': sched_G.state_dict(),
        'sched_D': sched_D.state_dict(),
        'IS_table': IS_table,
        'IS_std_table': IS_std_table,
        'step_table': step_table,
        'G_grad_norms': G_grad_norms,
        'D_grad_norms': D_grad_norms,
        'Total_grad_norms': Total_grad_norms,
        'step_table_GN': step_table_GN,
        'FID_table': FID_table,
        'fid_step_table': fid_step_table,
        'current_step': FLAGS.total_steps,
        'arch': FLAGS.arch,
        'total_steps': FLAGS.total_steps,
        'beta1': beta1,
        'beta2': beta2,
        'optimizer_type': optimizer_type,
    }
    if ema_enabled:
        checkpoint_data['net_G_ema'] = net_G_ema.state_dict()

    
    checkpoint_name = f'lr{FLAGS.lr_G}_beta{beta1}_{beta2}_{optimizer_type}_{FLAGS.arch}_{FLAGS.dataset}.pt'
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(checkpoint_data, checkpoint_path)

    print(f"\r💾 Final checkpoint saved: {checkpoint_name}")

    np.savez(npz_filepath,
         steps=step_table,
         IS=IS_table,
         IS_std=IS_std_table,
         Total_grad_norms=Total_grad_norms,
         steps_GN=step_table_GN,
         G_grad_norms=G_grad_norms,
         D_grad_norms=D_grad_norms,
         FID=FID_table,
         steps_FID=fid_step_table
         )
    
    print(f"\r💾 Data file saved: {npz_filename}")

    if IS_table:
        final_is = IS_table[-1]
        final_is_std = IS_std_table[-1]
        max_is = max(IS_table)
        max_is_step = step_table[IS_table.index(max_is)]

        print(f"\r📊 Final Results:")
        print(f"\r   • Last IS: {final_is:.3f} (±{final_is_std:.5f})")
        print(f"\r   • Best IS: {max_is:.3f} (at step {max_is_step})")
        print(f"\r   • Total evaluations: {len(IS_table)}")

    # Training completed successfully
    print(" Training completed successfully!")



def main(argv):
    set_seed(FLAGS.seed)
    
    if FLAGS.generate:
        print('[GEN] Entering image generation mode...')
        generate()
    else:
        train()

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit as exc:
        help_flags = {'--help', '-h', '--helpshort', '--helpfull'}
        if exc.code == 1 and help_flags.intersection(sys.argv[1:]):
            raise SystemExit(0) from None
        raise
