import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

def load_data(runs_dir):
    data = {
        'train_wm': [],
        'train_rm': [],
        'val_wm': [],
        'val_rm': []
    }
    
    # find all run directories in runs/wandb/
    wandb_dir = os.path.join(runs_dir, 'wandb')
    if not os.path.exists(wandb_dir):
        print(f"Directory not found: {wandb_dir}")
        return data
        
    # look for run-* directories
    run_paths = glob.glob(os.path.join(wandb_dir, 'run-*'))
    
    print(f"Found {len(run_paths)} runs in {runs_dir}")
    
    for run_path in run_paths:
        files_dir = os.path.join(run_path, 'files')
        
        # files to load
        files_map = {
            'train_wm': 'train_wm_accuracy_valence.pt',
            'train_rm': 'train_rm_accuracy_valence.pt',
            'val_wm': 'val_wm_accuracy_valence.pt',
            'val_rm': 'val_rm_accuracy_valence.pt'
        }
        
        # check if run has all files
        if all(os.path.exists(os.path.join(files_dir, f)) for f in files_map.values()):
            try:
                for key, filename in files_map.items():
                    path = os.path.join(files_dir, filename)
                    # Load data (dictionaries of valence -> accuracy)
                    d = torch.load(path, map_location='cpu', weights_only=False)
                    # Convert keys (strings) to floats for sorting
                    acc = {float(k): float(v) for k, v in d.items()}
                    data[key].append(acc)
                    
                print(f"Loaded data from {os.path.basename(run_path)}")
            except Exception as e:
                print(f"Error loading {run_path}: {e}")
        else:
            print(f"Skipping {os.path.basename(run_path)} (missing files)")
            
    return data

def process_data(data_list):
    if not data_list:
        return None, None, None, None
        
    # get all unique valence values present in any of the runs
    all_valences = set()
    for d in data_list:
        all_valences.update(d.keys())
    valences = sorted(list(all_valences))
    
    means = []
    stds = []
    sems = []
    
    for v in valences:
        # collect values for this valence across all runs that have it
        values = [d[v] for d in data_list if v in d]
        
        if values:
            means.append(np.mean(values))
            std = np.std(values)
            stds.append(std)
            sems.append(std / np.sqrt(len(values))) # Standard Error of Mean
        else:
            means.append(np.nan)
            stds.append(np.nan)
            sems.append(np.nan)
        
    return valences, means, stds, sems

def plot_baseline(runs_dir='./runs', save_path='baseline_accuracy_grid.png', show_error='sem'):
    print(f"Loading data from {runs_dir}...")
    data = load_data(runs_dir)
    
    # check if we have data (any key having non-empty list)
    if not any(data.values()):
        print("No valid data found to plot!")
        return

    print("Processing runs...")
    
    # create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    configs = [
        ('train_wm', 'Familiar Maps - Working Memory', 'blue'),
        ('train_rm', 'Familiar Maps - Reference Memory', 'orange'),
        ('val_wm', 'Novel Maps - Working Memory', 'green'),
        ('val_rm', 'Novel Maps - Reference Memory', 'red')
    ]
    
    for i, (key, title, color) in enumerate(configs):
        ax = axes[i]
        if not data[key]:
            print(f"No data for {key}, skipping...")
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
            continue
            
        val, mean, std, sem = process_data(data[key])
        err = sem if show_error == 'sem' else std
        
        ax.grid(True, alpha=0.3)
        ax.errorbar(val, mean, yerr=err, fmt='-o', 
                    capsize=5, linewidth=2, markersize=8, color=color)
        
        ax.set_xlabel('Valence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=14, pad=10)
        
        # Ensure valences on x-axis are shown
        if val:
            ax.set_xticks(val)
            ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Combined plot saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    plot_baseline()
