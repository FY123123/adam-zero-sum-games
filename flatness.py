import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob
from pathlib import Path

def load_npz_data(filepath):
    """Load data from a .npz file and return a dictionary with the data"""
    try:
        data = np.load(filepath)
        return {
            'filepath': filepath,
            'filename': os.path.basename(filepath),
            'steps': data.get('steps', []),
            'steps_GN': data.get('steps_GN', []),
            'IS': data.get('IS', []),
            'IS_std': data.get('IS_std', []),

            'Total_grad_norms': data.get('Total_grad_norms', []),
            'FID': data.get('FID', []),
            'steps_FID': data.get('steps_FID', [])
        }
    except Exception as e:
        print(f" Error loading {filepath}: {e}")
        return None

def generate_label_from_filename(filename):
    """Generate a readable label from filename showing beta1 (first beta component)"""
    name = filename.replace('.npz', '')

    import re

    # New pattern: lr{lr}_beta{beta1}_{beta2}_{optimizer}_{arch}_{dataset}
    # Example: lr0.0002_beta-0.3_0.999_adam_nm_res32_cifar10
    beta_pattern = r'beta([0-9.-]+)_([0-9.]+)'
    beta_match = re.search(beta_pattern, name)

    # Extract optimizer name
    optimizer_pattern = r'beta[0-9.-]+_[0-9.]+_([a-zA-Z_]+)_'
    optimizer_match = re.search(optimizer_pattern, name)

    if beta_match and optimizer_match:
        beta1_val = float(beta_match.group(1))
        beta2_val = float(beta_match.group(2))
        optimizer = optimizer_match.group(1).replace('_', ' ').upper()

        # Only show beta1 value (first beta component) as beta
        return f'$\\beta$={beta1_val}'

    # If no beta found, use a simple fallback
    return 'Unknown'

def plot_metrics(data_list, output_dir='./plots'):
    """Plot all metrics for multiple data files"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors and line styles for better distinguishability
    colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf',  # cyan
        '#ff9896',  # light red
        '#98df8a',  # light green
    ]

    # Define line styles for additional distinction
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':']

    # Define markers for different curves
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'd']
    
    # 1. Inception Score Plot
    plt.figure(figsize=(10, 7))
    for i, data in enumerate(data_list):
        if data and len(data['steps']) > 0 and len(data['IS']) > 0:
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            label = generate_label_from_filename(data['filename'])

            plt.plot(data['steps'], data['IS'], label=label, color=color,
                    linestyle=linestyle, linewidth=2.5, marker=marker, markersize=3,
                    markevery=max(1, len(data['steps'])//20))
            
            # Add error bands if standard deviation exists
            if len(data['IS_std']) > 0:
                plt.fill_between(data['steps'], 
                               data['IS'] - data['IS_std'], 
                               data['IS'] + data['IS_std'], 
                               alpha=0.2, color=color)
    
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Inception Score', fontsize=16)
    # plt.title('Inception Score vs Iterations', fontsize=18)  # Title removed
    plt.legend(fontsize=14, loc='lower left', bbox_to_anchor=(0.0, 0.0),
               ncol=2, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 8.5)  # Set IS y-axis range from 0 to 8  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'IS_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Generated Inception Score curve")
    

    # 3. Gradient Norms Plot
    plt.figure(figsize=(10, 7))
    for i, data in enumerate(data_list):
        if data and len(data['steps_GN']) > 0 and len(data['Total_grad_norms']) > 0:
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            label = generate_label_from_filename(data['filename'])

            plt.plot(data['steps_GN'], data['Total_grad_norms'],
                    label=label, color=color, alpha=0.8,linestyle=linestyle,
                    linewidth=2.5, marker=marker, markersize=3,
                    markevery=max(1, len(data['steps_GN'])//20))
    
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Total Gradient Norms', fontsize=16)
    plt.legend(fontsize=14, loc='lower left', bbox_to_anchor=(0.0, 0.0),
               ncol=2, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 15000)  
    plt.ylim(10, 1300)
    plt.yscale('log') 
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Gnorm_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated Gradient Norms curve")
    
    # 4. Cumulative Average Gradient Norms Plot
    plt.figure(figsize=(10, 7))
    for i, data in enumerate(data_list):
        if data and len(data['steps_GN']) > 0 and len(data['Total_grad_norms']) > 0:
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            label = generate_label_from_filename(data['filename'])

            # Calculate cumulative average
            grad_norms = np.array(data['Total_grad_norms'])
            if len(grad_norms) > 0:
                cum_avg = np.cumsum(grad_norms) / np.arange(1, len(grad_norms) + 1)
                plt.plot(data['steps_GN'], cum_avg,
                        label=label, color=color, linestyle=linestyle,
                        linewidth=2.5, marker=marker, markersize=8,
                        markevery=max(1, len(data['steps_GN'])//20))
    
    plt.xlabel('Iterations', fontsize=16)
    plt.ylabel('Cumulative Average Gradient Norms', fontsize=16)
    # plt.title('Cumulative Average Gradient Norms vs Iterations', fontsize=18)  # Title removed
    plt.legend(fontsize=14, loc='lower left', bbox_to_anchor=(0.0, 0.0),
               ncol=2, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3)
    plt.xlim(100, 15000)  # Display to 16000, leave 15000-16000 as blank  
    plt.yscale('log')  
    plt.ylim(20, 300)  # Set y-axis range from 20 to 180  
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Avg_Gnorm_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(" Generated Cumulative Average Gradient Norms curve")
    
    # FID Plot (only drawn when at least one file contains FID data)
    any_fid = any(
        data and len(data.get('steps_FID', [])) > 0 and len(data.get('FID', [])) > 0
        for data in data_list
    )
    if any_fid:
        plt.figure(figsize=(10, 7))
        for i, data in enumerate(data_list):
            if not data:
                continue
            steps_fid = data.get('steps_FID', [])
            fid_vals = data.get('FID', [])
            if len(steps_fid) == 0 or len(fid_vals) == 0:
                continue
            fid_arr = np.array(fid_vals, dtype=float)
            valid = ~np.isnan(fid_arr)
            if not valid.any():
                continue
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            label = generate_label_from_filename(data['filename'])
            plt.plot(np.array(steps_fid)[valid], fid_arr[valid],
                     label=label, color=color, linestyle=linestyle,
                     linewidth=2.5, marker=marker, markersize=3,
                     markevery=max(1, int(valid.sum()) // 20))

        plt.xlabel('Iterations', fontsize=16)
        plt.ylabel('FID', fontsize=16)
        plt.legend(fontsize=14, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                   ncol=2, frameon=True, fancybox=True, shadow=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'FID_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(" Generated FID curve")

    print(f" All charts saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Plot training metrics from multiple .npz files')
    parser.add_argument('files', nargs='*', help='.npz files to plot (can use glob patterns). If not specified, will use all .npz files in algorithm-specific npz directories')
    parser.add_argument('--output', '-o', default='./Image', help='Output directory for plots (default: ./Image)')
    parser.add_argument('--npz-dir', default='.', help='Root directory to search for algorithm-specific npz folders (default: current directory)')
    
    args = parser.parse_args()
    
    # If no files specified, use all .npz files in algorithm-specific npz directories
    if not args.files:
        npz_dir = os.path.abspath(args.npz_dir)
        if os.path.exists(npz_dir):
            # Look for the npz directory
            npz_main_dir = os.path.join(npz_dir, 'npz')
            if os.path.exists(npz_main_dir):
                # Search for folders ending with '_npz' inside npz/
                algorithm_npz_folders = []
                for item in os.listdir(npz_main_dir):
                    item_path = os.path.join(npz_main_dir, item)
                    if os.path.isdir(item_path) and item.endswith('_npz'):
                        algorithm_name = item.replace('_npz', '')
                        print(f"🔍 Found algorithm npz folder: {item} (algorithm: {algorithm_name})")
                        algorithm_npz_folders.append(item_path)
                
                if algorithm_npz_folders:
                    args.files = [os.path.join(folder, '*.npz') for folder in algorithm_npz_folders]
                    print(f" No files specified, using all .npz files in {len(algorithm_npz_folders)} algorithm npz directories")
                else:
                    print(f" No algorithm-specific npz directories found in: {npz_main_dir}")
                    print(" Please specify files explicitly or ensure algorithm-specific npz directories exist")
                    return
            else:
                print(f" npz directory not found: {npz_main_dir}")
                print(" Please specify files explicitly or create the npz directory")
                return
        else:
            print(f" Root directory not found: {npz_dir}")
            print(" Please specify files explicitly or provide a valid root directory")
            return
    
    # Expand glob patterns and collect all files
    all_files = []
    for file_pattern in args.files:
        if '*' in file_pattern or '?' in file_pattern:
            # This is a glob pattern
            matched_files = glob.glob(file_pattern)
            all_files.extend(matched_files)
        else:
            # This is a direct file path
            all_files.append(file_pattern)
    
    # Remove duplicates and sort by beta1 values (largest to smallest)
    all_files = list(set(all_files))

    def extract_beta1_value(filepath):
        """Extract beta1 value from filename for sorting"""
        filename = os.path.basename(filepath).replace('.npz', '')
        import re
        beta_pattern = r'beta([0-9.-]+)_([0-9.]+)'
        beta_match = re.search(beta_pattern, filename)
        if beta_match:
            beta1_val = float(beta_match.group(1))  # beta1 is the first beta component
            return beta1_val
        return 0.0  # fallback if no beta found

    # Sort by beta1 value in descending order (largest first)
    all_files = sorted(all_files, key=extract_beta1_value, reverse=True)
    
    if not all_files:
        print(" No files found!")
        return
    
    print(f" Found {len(all_files)} files to plot:")
    for f in all_files:
        print(f"   - {f}")
    
    # Load data from all files
    data_list = []
    for filepath in all_files:
        if os.path.exists(filepath):
            data = load_npz_data(filepath)
            if data:
                data_list.append(data)
                print(f" Loaded: {filepath}")
        else:
            print(f" File not found: {filepath}")
    
    if not data_list:
        print(" No valid data files loaded!")
        return
    
    print(f"\n Plotting data from {len(data_list)} files...")
    
    # Generate plots
    plot_metrics(data_list, args.output)
    
    print("\n Plotting completed successfully!")

if __name__ == "__main__":
    main()
