import os
import sys
import traceback
import torch
import numpy as np
import matplotlib.pyplot as plt

# fix numpy compatibility issue
try:
    import numpy.core as np_core
    sys.modules['numpy._core'] = np_core
except Exception:
    pass

from valence_embedding import ValenceEmbedder

def load_checkpoint(run_path):
    """Load checkpoint with compatibility fixes."""
    checkpoint_path = os.path.join(run_path, 'files', 'checkpoint.pt')
    try:
        data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return data
    except Exception as e:
        print(f"Error loading {checkpoint_path}: {e}")
        return None

def compute_embedding_similarity_centered(valence_embedder, center_valences, all_valences, device='cpu'):
    """
    Compute cosine similarity between embeddings centered at specific valences
    and all other valence embeddings.
    """
    valence_embedder.eval()
    
    # convert to tensors
    center_vals_tensor = torch.tensor(center_valences, dtype=torch.float32).to(device)
    all_vals_tensor = torch.tensor(all_valences, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # get embeddings for center valences
        center_embeddings = valence_embedder(center_vals_tensor)  # (num_centers, embed_dim)
        
        # get embeddings for all valences
        all_embeddings = valence_embedder(all_vals_tensor)  # (num_valences, embed_dim)
        
        # compute cosine similarity
        # mormalize embeddings
        center_norm = center_embeddings / (center_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        all_norm = all_embeddings / (all_embeddings.norm(dim=1, keepdim=True) + 1e-8)
        
        # similarity matrix: (num_centers, num_valences)
        similarity_matrix = torch.matmul(center_norm, all_norm.T)
    
    # organize results
    results = {}
    for i, center_val in enumerate(center_valences):
        results[center_val] = {
            float(all_valences[j]): float(similarity_matrix[i, j]) 
            for j in range(len(all_valences))
        }
    
    return results, similarity_matrix.cpu().numpy()

def plot_similarity_heatmap(center_valences, all_valences, similarity_matrix, run_name, save_dir):
    """Plot heatmap of similarities."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    im = ax.imshow(similarity_matrix, aspect='auto', cmap='RdYlBu_r', vmin=-1, vmax=1)
    
    # set ticks
    ax.set_yticks(range(len(center_valences)))
    ax.set_yticklabels([f"{v:.1f}" for v in center_valences])
    ax.set_ylabel('Center Valence', fontsize=12)
    
    ax.set_xticks(range(len(all_valences)))
    ax.set_xticklabels([f"{v:.1f}" for v in all_valences], rotation=45, ha='right')
    ax.set_xlabel('Target Valence', fontsize=12)
    
    ax.set_title(f'Valence Embedding Similarity\n{run_name}', fontsize=14, fontweight='bold')
    
    # add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', rotation=270, labelpad=20)
    
    # add text annotations
    for i in range(len(center_valences)):
        for j in range(len(all_valences)):
            text = ax.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'valence_similarity_heatmap_{run_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved heatmap to: {save_path}")
    plt.close()

def plot_similarity_curves(center_valences, all_valences, similarity_matrix, run_name, save_dir):
    """Plot similarity curves for each center valence."""
    fig, axes = plt.subplots(len(center_valences), 1, figsize=(10, 3 * len(center_valences)))
    
    if len(center_valences) == 1:
        axes = [axes]
    
    for i, (ax, center_val) in enumerate(zip(axes, center_valences)):
        similarities = similarity_matrix[i, :]
        
        ax.plot(all_valences, similarities, 'o-', linewidth=2, markersize=8)
        ax.axvline(x=center_val, color='red', linestyle='--', alpha=0.7, label=f'Center: {center_val}')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5)
        
        ax.set_xlabel('Valence', fontsize=11)
        ax.set_ylabel('Cosine Similarity', fontsize=11)
        ax.set_title(f'Similarity from Center Valence = {center_val}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-1.0, 1.0])
    
    plt.suptitle(f'Valence Embedding Similarity Curves\n{run_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'valence_similarity_curves_{run_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved curves to: {save_path}")
    plt.close()

def load_memory_accuracies(run_path):
    """Load working/reference memory accuracy data by valence."""
    files_dir = os.path.join(run_path, 'files')
    
    memory_data = {}
    
    # load train (familiar map) data
    train_wm_path = os.path.join(files_dir, 'train_wm_accuracy_valence.pt')
    train_rm_path = os.path.join(files_dir, 'train_rm_accuracy_valence.pt')
    
    # load validation (novel map) data
    val_wm_path = os.path.join(files_dir, 'val_wm_accuracy_valence.pt')
    val_rm_path = os.path.join(files_dir, 'val_rm_accuracy_valence.pt')
    
    try:
        if os.path.exists(train_wm_path):
            memory_data['train_wm'] = torch.load(train_wm_path, map_location='cpu', weights_only=False)
        if os.path.exists(train_rm_path):
            memory_data['train_rm'] = torch.load(train_rm_path, map_location='cpu', weights_only=False)
        if os.path.exists(val_wm_path):
            memory_data['val_wm'] = torch.load(val_wm_path, map_location='cpu', weights_only=False)
        if os.path.exists(val_rm_path):
            memory_data['val_rm'] = torch.load(val_rm_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: Could not load memory accuracy files: {e}")
        return None
    
    return memory_data if memory_data else None

def plot_memory_accuracy_by_valence(memory_data, run_name, save_dir):
    """Plot working/reference memory accuracy by valence for familiar and novel maps."""
    if not memory_data:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Memory Accuracy by Valence\n{run_name}', fontsize=14, fontweight='bold')
    
    plot_configs = [
        ('train_wm', 'Working Memory - Familiar Map (Train)', axes[0, 0]),
        ('train_rm', 'Reference Memory - Familiar Map (Train)', axes[0, 1]),
        ('val_wm', 'Working Memory - Novel Map (Validation)', axes[1, 0]),
        ('val_rm', 'Reference Memory - Novel Map (Validation)', axes[1, 1])
    ]
    
    for key, title, ax in plot_configs:
        if key in memory_data:
            data = memory_data[key]
            
            # Convert string keys to float and sort
            valences = sorted([float(v) for v in data.keys()])
            accuracies = [data[str(v)] for v in valences]
            
            ax.plot(valences, accuracies, 'o-', linewidth=2, markersize=8, color='steelblue')
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (0.5)')
            ax.set_xlabel('Valence', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
            ax.legend()
        else:
            ax.text(0.5, 0.5, f'No data for {key}', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'memory_accuracy_by_valence_{run_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved memory accuracy plot to: {save_path}")
    plt.close()

def save_memory_accuracy_summary(memory_data, run_name, save_dir):
    """Save numerical summary of memory accuracies."""
    if not memory_data:
        return
    
    summary_path = os.path.join(save_dir, 'memory_accuracy_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(f"Memory Accuracy by Valence Summary\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"{'='*60}\n\n")
        
        for key in ['train_wm', 'train_rm', 'val_wm', 'val_rm']:
            if key in memory_data:
                data = memory_data[key]
                
                map_type = "Familiar Map (Train)" if 'train' in key else "Novel Map (Validation)"
                mem_type = "Working Memory" if 'wm' in key else "Reference Memory"
                
                f.write(f"{mem_type} - {map_type}\n")
                f.write(f"{'-'*40}\n")
                
                # Sort by valence
                valences = sorted([float(v) for v in data.keys()])
                for val in valences:
                    acc = data[str(val)]
                    f.write(f"  Valence {val:6.1f}: {acc:.4f}\n")
                
                # Calculate mean accuracy
                mean_acc = np.mean([data[str(v)] for v in valences])
                f.write(f"  Mean accuracy: {mean_acc:.4f}\n")
                f.write("\n")
    
    print(f"✓ Saved memory accuracy summary to: {summary_path}")

def analyze_run(run_path, center_valences=[-1, -0.5, 0, 0.5, 1], device='cpu'):
    """Analyze a single wandb run."""
    run_name = os.path.basename(run_path)
    print(f"\n{'='*60}")
    print(f"Analyzing run: {run_name}")
    print(f"{'='*60}")
    
    # load checkpoint
    checkpoint = load_checkpoint(run_path)
    if checkpoint is None:
        print(f"Skipping {run_name} - could not load checkpoint")
        return None
    
    print(f"✓ Loaded checkpoint")
    
    # get config
    config = checkpoint['config']
    
    # get valence embedding dimensions
    # IMPORTANT: The standalone valence_embedder uses config.d_embed, not config.valence_embed_dim!
    # (See main_wce.py:265)
    if 'valence_embedder_state_dict' in checkpoint:
        # new checkpoints: infer from actual saved weights
        sample_weight = checkpoint['valence_embedder_state_dict']['mlp.2.weight']
        valence_embed_dim = sample_weight.shape[0]  # output dimension
        valence_hidden_dim = sample_weight.shape[1]  # input dimension
        print(f"✓ Inferred from checkpoint: embed_dim={valence_embed_dim}, hidden_dim={valence_hidden_dim}")
    else:
        # old checkpoints: use config values
        valence_embed_dim = getattr(config, 'valence_embed_dim', config.d_embed)
        valence_hidden_dim = getattr(config, 'valence_hidden_dim', 32)

    valence_activation = getattr(config, 'valence_activation', 'relu')
    
    print(f"✓ Config loaded - valence_embed_dim: {valence_embed_dim}, hidden_dim: {valence_hidden_dim}")
    
    # initialize valence embedder with correct config
    valence_embedder = ValenceEmbedder(
        embed_dim=valence_embed_dim,
        hidden_dim=valence_hidden_dim,
        activation=valence_activation
    ).to(device)

    # load the STANDALONE valence_embedder (the one actually trained with contrastive loss)
    if 'valence_embedder_state_dict' in checkpoint:
        print("✓ Loading standalone valence_embedder (trained with contrastive loss)")
        valence_embedder.load_state_dict(checkpoint['valence_embedder_state_dict'])
        print(f"✓ Loaded valence embedder weights")
    else:
        # fallback: extract from model state dict (old checkpoints)
        print("⚠ Fallback: Loading from model.valence_embedder (may not have contrastive training)")
        state_dict = checkpoint['state_dict']
        valence_keys = {k: v for k, v in state_dict.items() if 'valence' in k.lower()}

        if len(valence_keys) == 0:
            print("⚠ Warning: No valence embedder weights found in checkpoint")
            return None

        # load weights (need to match the key names)
        valence_state = {}
        for k, v in valence_keys.items():
            # remove any prefix to match ValenceEmbedder architecture
            new_key = k.replace('valence_embedder.', '').replace('module.', '')
            valence_state[new_key] = v

        valence_embedder.load_state_dict(valence_state, strict=False)
        print(f"✓ Loaded valence embedder weights from model state_dict")
    
    # determine all possible valences for analysis
    # use steps of 0.05 from -1 to 1
    all_valences = np.linspace(-1, 1, num=41).tolist()  # 41 points gives step size of 0.05

    print(f"✓ Using {len(all_valences)} valences from {all_valences[0]:.3f} to {all_valences[-1]:.3f} (step=0.05)")
    
    # compute similarities
    results, similarity_matrix = compute_embedding_similarity_centered(
        valence_embedder, center_valences, all_valences, device
    )
    
    print(f"✓ Computed similarity matrix")
    
    # create output directory
    output_dir = os.path.join(run_path, 'valence_analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # plot results
    plot_similarity_heatmap(center_valences, all_valences, similarity_matrix, run_name, output_dir)
    plot_similarity_curves(center_valences, all_valences, similarity_matrix, run_name, output_dir)
    
    # load and plot memory accuracy data
    print(f"✓ Loading memory accuracy data...")
    memory_data = load_memory_accuracies(run_path)
    if memory_data:
        plot_memory_accuracy_by_valence(memory_data, run_name, output_dir)
        save_memory_accuracy_summary(memory_data, run_name, output_dir)
    else:
        print("  No memory accuracy data found")
    
    # save numerical results
    results_path = os.path.join(output_dir, 'similarity_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Valence Embedding Similarity Analysis\n")
        f.write(f"Run: {run_name}\n")
        f.write(f"{'='*60}\n\n")
        
        for center_val in center_valences:
            f.write(f"Center Valence: {center_val}\n")
            f.write(f"{'-'*40}\n")
            sims = results[center_val]
            for val in sorted(sims.keys()):
                f.write(f"  {val:6.1f} -> similarity: {sims[val]:7.4f}\n")
            f.write("\n")
    
    print(f"✓ Saved numerical results to: {results_path}")
    
    return results

def plot_comparison_across_runs(all_memory_data, output_dir):
    """Create comparison plots across multiple runs (like Baseline, WCE β=0.5, WCE β=1)."""
    if not all_memory_data:
        print("No memory data available for comparison")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Memory Accuracy by Valence - Comparison Across Runs', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plot_configs = [
        ('train_wm', 'Working memory on familiar maps', axes[0, 0]),
        ('train_rm', 'Reference memory on familiar maps', axes[0, 1]),
        ('val_wm', 'Working memory on novel maps', axes[1, 0]),
        ('val_rm', 'Reference memory on novel maps', axes[1, 1])
    ]
    
    # define colors and markers for different runs
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for key, title, ax in plot_configs:
        color_idx = 0
        
        for run_name, memory_data in sorted(all_memory_data.items()):
            if key in memory_data:
                data = memory_data[key]
                
                # convert string keys to float and sort
                valences = sorted([float(v) for v in data.keys()])
                accuracies = [data[str(v)] for v in valences]
                
                # calculate standard error for error bars (if multiple values available)
                # for now, just plot the line
                ax.plot(valences, accuracies, 
                       marker=markers[color_idx % len(markers)], 
                       color=colors[color_idx % len(colors)],
                       linewidth=2, markersize=8, 
                       label=run_name, alpha=0.8)
                
                color_idx += 1
        
        # styling
        ax.set_xlabel('Valence', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10, loc='best')
        ax.set_xlim([-1.05, 1.05])
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, 'memory_accuracy_comparison_all_runs.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to: {save_path}")
    plt.close()

def main():
    """Main analysis function."""
    # configuration
    wandb_dir = './wandb'
    center_valences = [-1, -0.5, 0, 0.5, 1]
    device = 'cpu'  # use 'cuda' if available and needed
    
    print("Valence Embedding Similarity Analysis")
    print("=" * 60)
    print(f"Center valences: {center_valences}")
    print(f"Device: {device}")
    
    # find all runs
    run_dirs = [os.path.join(wandb_dir, d) for d in os.listdir(wandb_dir) 
                if os.path.isdir(os.path.join(wandb_dir, d))]
    
    print(f"\nFound {len(run_dirs)} runs to analyze")
    
    # analyze each run
    all_results = {}
    all_memory_data = {}
    
    for run_dir in sorted(run_dirs):
        try:
            results = analyze_run(run_dir, center_valences, device)
            if results is not None:
                run_name = os.path.basename(run_dir)
                all_results[run_name] = results
                
                # Load memory data for comparison plot
                memory_data = load_memory_accuracies(run_dir)
                if memory_data:
                    all_memory_data[run_name] = memory_data
        except Exception as e:
            print(f"Error analyzing {run_dir}: {e}")
            traceback.print_exc()
    
    # create comparison plot across all runs
    if all_memory_data:
        comparison_output_dir = './wandb'
        plot_comparison_across_runs(all_memory_data, comparison_output_dir)
    
    print(f"\n{'='*60}")
    print(f"Analysis complete! Analyzed {len(all_results)} runs successfully")
    print(f"Check the 'valence_analysis' folder in each run directory for results")
    print(f"Check './wandb' for the comparison plot across all runs")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
