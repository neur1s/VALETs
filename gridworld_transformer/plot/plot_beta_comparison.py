import torch
import matplotlib.pyplot as plt
import numpy as np
import os

BASE_DIR = os.environ["BASE_DIR"]

config_groups = [
    {
        "name": "WCE",
        "base_dir": os.path.join(BASE_DIR, "wce/results"),
        "betas": ["0_25", "0_5", "1"], # from folder names 0_wce_loss_beta_0_5
        "beta_label": ["0.25", "0.5", "1.0"],
        "folder_template": "{}_wce_loss_beta_{}", 
        "color": "blue"
    },
    {
        "name": "WCE + Val Embed (MLP)",
        "base_dir": os.path.join(BASE_DIR, "wce_valence_embed_mlp/results"),
        # check folders: 0_wce_loss_beta_0.25, etc.
        "betas": ["0.25", "0.5", "1"], # derived from typical sweep names
        "beta_label": ["0.25", "0.5", "1.0"],
        "folder_template": "{}_wce_loss_beta_{}",
        "color": "green"
    },
    {
        "name": "WCE + Val Embed + Attn",
        "base_dir": os.path.join(BASE_DIR, "wce_valence_embed_attn/results"),
        "betas": ["1"], 
        "beta_label": ["1.0"],
        "folder_template": "{}_wce_beta_{}",
        "color": "red"
    }
]

metrics = {
    "train_wm": "train_wm_accuracy_valence.pt",
    "val_wm": "val_wm_accuracy_valence.pt",
    "train_rm": "train_rm_accuracy_valence.pt",
    "val_rm": "val_rm_accuracy_valence.pt"
}

titles = {
    "train_wm": "Train WM Accuracy by Valence",
    "val_wm": "Validation WM Accuracy by Valence",
    "train_rm": "Train RM Accuracy by Valence",
    "val_rm": "Validation RM Accuracy by Valence"
}

for group in config_groups:
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    print(f"Processing group: {group['name']}")
    
    for i, (metric_key, metric_file) in enumerate(metrics.items()):
        ax = axes[i]
        
        # color map for betas - shades of the base color
        colors = plt.cm.get_cmap(f"{group['color'].title()}s")(np.linspace(0.5, 1, len(group['betas'])))

        for b_idx, beta in enumerate(group['betas']):
            all_seeds_data = []
            beta_lbl = group['beta_label'][b_idx]
            
            for seed in [0, 1, 2]:
                folder_name = group["folder_template"].format(seed, beta)
                file_path = os.path.join(group["base_dir"], folder_name, metric_file)
                
                if os.path.exists(file_path):
                    try:
                        data = torch.load(file_path, map_location=torch.device('cpu'), weights_only=False)
                        all_seeds_data.append(data)
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    pass 

            if all_seeds_data:
                # aggregate data
                valences = set()
                for d in all_seeds_data:
                    valences.update(d.keys())
                
                val_map = {}
                for v in valences:
                    try:
                        val_map[float(v)] = v
                    except Exception:
                        continue
                
                sorted_vals = sorted(val_map.keys())
                means, stderrs, valid_vals = [], [], []
                
                for v_float in sorted_vals:
                    v_key = val_map[v_float]
                    vals_for_v = []
                    for d in all_seeds_data:
                        if v_key in d:
                            vals_for_v.append(d[v_key])
                    
                    if vals_for_v:
                        vals_for_v = np.array(vals_for_v)
                        means.append(np.mean(vals_for_v))
                        stderrs.append(np.std(vals_for_v) / np.sqrt(len(vals_for_v)))
                        valid_vals.append(v_float)
                
                if valid_vals:
                    ax.errorbar(valid_vals, means, yerr=stderrs, label=f"Beta={beta_lbl}", 
                                color=colors[b_idx], capsize=5, marker='o')
            else:
                print(f"No data for {group['name']} Beta={beta}")

        ax.set_title(titles[metric_key])
        ax.set_xlabel("Valence")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle='--', alpha=0.7)
        if i == 0:
            ax.legend()

    plt.suptitle(f"{group['name']} - Accuracy vs Valence across Betas")
    plt.tight_layout()
    safe_name = group['name'].replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    plt.savefig(os.path.join(BASE_DIR, f"comparison_betas_{safe_name}.png"))
    print(f"Saved comparison_betas_{safe_name}.png")
