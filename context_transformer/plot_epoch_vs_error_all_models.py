import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
SCRIPT_DIR = os.environ["BASE_DIR"]

BETA_VALUES = [0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]

MODEL_CONFIGS = {
    "Baseline": {"subdir": "baseline", "has_beta": False},
    "WCE": {"subdir": "wce", "has_beta": True},
    "WCE + Val Embed (MLP)": {"subdir": "wce_valence_embed_mlp", "has_beta": True},
    "WCE + Val Embed + Attn": {"subdir": "wce_valence_embed_attn", "has_beta": True},
    "Val Embed + Attn (No WCE)": {"subdir": "no_wce_valence_embed_attn", "has_beta": False}
}

model_colors = {
    "Baseline": "grey",
    "WCE": "#a3b7c2",
    "WCE + Val Embed (MLP)": "#51748e",
    "WCE + Val Embed + Attn": "#d13262",
    "Val Embed + Attn (No WCE)": "#1e4667"
}

def plot_memory_comparison():
    for mem_type in ["wm", "rm"]:
        # increased figsize height slightly to accommodate the legend lift
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        title_prefix = "Working Memory" if mem_type == "wm" else "Reference Memory"
        
        print(f"\n--- Plotting {title_prefix} ---")

        for i, beta in enumerate(BETA_VALUES):
            ax = axes[i]
            ax.set_title(f"β = {beta}", fontweight='bold', fontsize=16)
            
            for model_name, cfg in MODEL_CONFIGS.items():
                subdir = cfg["subdir"]
                all_seeds = []
                
                for seed in SEEDS:
                    if cfg["has_beta"]:
                        fname = f"checkpoint_seed{seed}_beta{beta}.pt"
                    else:
                        fname = f"checkpoint_seed{seed}.pt"
                    
                    path = os.path.join(SCRIPT_DIR, subdir, fname)
                    if not os.path.exists(path): continue
                    
                    try:
                        data = torch.load(path, map_location='cpu', weights_only=False)
                        h = data['history']
                        
                        possible_keys = [f"val_{mem_type}_error", f"train_{mem_type}_error", f"{mem_type}_error"]
                        found_key = next((k for k in possible_keys if k in h), None)
                        
                        if found_key:
                            all_seeds.append(np.array(h[found_key]) * 100)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")

                if len(all_seeds) > 0:
                    arr = np.array(all_seeds)
                    mean = np.mean(arr, axis=0)
                    sem = np.std(arr, axis=0) / np.sqrt(len(all_seeds))
                    epochs = np.arange(len(mean))
                    
                    ax.plot(epochs, mean, label=model_name, color=model_colors[model_name], lw=2.5)
                    ax.fill_between(epochs, mean - sem, mean + sem, color=model_colors[model_name], alpha=0.1)
            
            ax.set_xlabel("Epoch")
            ax.grid(True, linestyle=':', alpha=0.6)
            if i == 0:
                ax.set_ylabel("Error rate")
            
            ax.set_ylim(85, 100) 
            ax.set_xlim(80, 100) 

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, 
                   bbox_to_anchor=(0.5, 0.02), fontsize=11, frameon=False)
        
        plt.tight_layout(rect=[0, 0.15, 1, 0.95])
        
        out_file = f"Epoch_vs_error_{mem_type.upper()}.pdf"
        plt.savefig(out_file, bbox_inches='tight')
        print(f"Saved → {out_file}")

if __name__ == "__main__":
    plot_memory_comparison()