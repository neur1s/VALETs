import sys, os
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_memory_data(run_path):
    """Load memory accuracy data from a run."""
    files_dir = os.path.join(run_path, 'files')
    
    data = {}
    files_to_load = {
        'train_wm': 'train_wm_accuracy_valence.pt',
        'train_rm': 'train_rm_accuracy_valence.pt',
        'val_wm': 'val_wm_accuracy_valence.pt',
        'val_rm': 'val_rm_accuracy_valence.pt'
    }
    
    for key, filename in files_to_load.items():
        filepath = os.path.join(files_dir, filename)
        if os.path.exists(filepath):
            try:
                data[key] = torch.load(filepath, map_location='cpu', weights_only=False)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
    
    return data if data else None

def extract_weight_from_name(run_name):
    """Extract contrastive weight from run name."""
    if run_name == 'contrastive_mlp':
        return 1.0
    elif 'contrastive_mlp_weight_' in run_name:
        try:
            weight = float(run_name.split('weight_')[1])
            return weight
        except Exception:
            return None
    return None

def plot_comparison(all_data, save_path='./comparison_contrastive_weights.png'):
    """Plot comparison of all contrastive weights."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Memory Accuracy by Valence - Contrastive Weight Comparison', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plot_configs = [
        ('train_wm', 'Working Memory - Familiar Map (Train)', axes[0, 0]),
        ('train_rm', 'Reference Memory - Familiar Map (Train)', axes[0, 1]),
        ('val_wm', 'Working Memory - Novel Map (Validation)', axes[1, 0]),
        ('val_rm', 'Reference Memory - Novel Map (Validation)', axes[1, 1])
    ]
    
    # color scheme for different weights
    colors = {
        1.0: '#1f77b4',    # blue
        2.5: '#9467bd',    # purple
        5.0: '#ff7f0e',    # orange
        20.0: '#2ca02c',   # green
        40.0: '#d62728'    # red
    }
    
    # sort by weight for consistent plotting
    sorted_weights = sorted(all_data.keys())
    
    for data_key, title, ax in plot_configs:
        for weight in sorted_weights:
            run_data = all_data[weight]
            
            if data_key in run_data:
                data = run_data[data_key]
                
                # convert string keys to float and sort
                valences = sorted([float(v) for v in data.keys()])
                accuracies = [data[str(v)] for v in valences]
                
                color = colors.get(weight, 'gray')
                label = f'Weight = {weight}'
                
                ax.plot(valences, accuracies, 'o-', linewidth=2.5, markersize=7, 
                       color=color, label=label, alpha=0.8)
        
        ax.set_xlabel('Valence', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.set_ylim([0, 1.05])
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        
        # add more ticks for better readability
        ax.set_xticks(np.arange(-1, 1.1, 0.25))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison plot to: {save_path}")
    plt.close()

def save_summary_statistics(all_data, save_path='./contrastive_weight_summary.txt'):
    """Save summary statistics for each weight."""
    with open(save_path, 'w') as f:
        f.write("Contrastive Weight Comparison - Summary Statistics\n")
        f.write("=" * 70 + "\n\n")
        
        sorted_weights = sorted(all_data.keys())
        
        for weight in sorted_weights:
            f.write(f"Contrastive Weight = {weight}\n")
            f.write("-" * 70 + "\n")
            
            run_data = all_data[weight]
            
            for data_key in ['train_wm', 'train_rm', 'val_wm', 'val_rm']:
                if data_key in run_data:
                    data = run_data[data_key]
                    
                    map_type = "Familiar" if 'train' in data_key else "Novel"
                    mem_type = "Working" if 'wm' in data_key else "Reference"
                    
                    valences = sorted([float(v) for v in data.keys()])
                    accuracies = [data[str(v)] for v in valences]
                    
                    mean_acc = np.mean(accuracies)
                    std_acc = np.std(accuracies)
                    min_acc = np.min(accuracies)
                    max_acc = np.max(accuracies)
                    
                    f.write(f"  {mem_type} Memory - {map_type} Map:\n")
                    f.write(f"    Mean: {mean_acc:.4f}, Std: {std_acc:.4f}, ")
                    f.write(f"Min: {min_acc:.4f}, Max: {max_acc:.4f}\n")
            
            f.write("\n")
    
    print(f"✓ Saved summary statistics to: {save_path}")

def main():
    """Main comparison function."""
    wandb_dir = './wandb'
    
    # map run names to weights
    target_runs = {
        'contrastive_mlp': 1.0,
        'contrastive_mlp_weight_2.5': 2.5,
        'contrastive_mlp_weight_5': 5.0,
        'contrastive_mlp_weight_20': 20.0,
        'contrastive_mlp_weight_40': 40.0
    }
    
    print("Contrastive Weight Comparison Analysis")
    print("=" * 70)
    print(f"Target runs: {list(target_runs.keys())}")
    
    # load data for each run
    all_data = {}
    
    for run_name, weight in target_runs.items():
        run_path = os.path.join(wandb_dir, run_name)
        
        if not os.path.exists(run_path):
            print(f"⚠ Warning: Run '{run_name}' not found at {run_path}")
            continue
        
        print(f"\nLoading data from: {run_name} (weight={weight})")
        data = load_memory_data(run_path)
        
        if data:
            all_data[weight] = data
            print(f"  ✓ Loaded {len(data)} data types")
        else:
            print(f"  ✗ No data found")
    
    if len(all_data) == 0:
        print("\n✗ No data loaded. Cannot create comparison plot.")
        return
    
    print(f"\n{'='*70}")
    print(f"Successfully loaded data for {len(all_data)} weights: {sorted(all_data.keys())}")
    
    # create output directory
    output_dir = './contrastive_weight_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # generate comparison plot
    print("\nGenerating comparison plot...")
    plot_comparison(all_data, os.path.join(output_dir, 'comparison_contrastive_weights.png'))
    
    # save summary statistics
    print("Generating summary statistics...")
    save_summary_statistics(all_data, os.path.join(output_dir, 'summary_statistics.txt'))
    
    print(f"\n{'='*70}")
    print(f"Analysis complete! Results saved to: {output_dir}/")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
