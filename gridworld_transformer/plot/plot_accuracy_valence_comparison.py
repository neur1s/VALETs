import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as mcolors

BASE_DIR = os.environ["BASE_DIR"]

# standardized models configuration
models = {
    "Baseline": {
        "base_dir": os.path.join(BASE_DIR, "baseline/results"),
        "seed_template": "{}_baseline",
        "betas": [None],
        "color": "grey"
    },
    "WCE": {
        "base_dir": os.path.join(BASE_DIR, "wce/results"),
        "seed_template": "{}_wce_loss_beta_{}",
        "betas": ["0_25", "0_5", "1"],
        "color": "#a3b7c2"
    },
    "WCE + Val Embed (MLP)": {
        "base_dir": os.path.join(BASE_DIR, "wce_valence_embed_mlp/results"),
        "seed_template": "{}_wce_loss_beta_{}",
        "betas": ["0_25", "0_5", "1"],
        "color": "#51748e"
    },
    "WCE + Val Embed + Attn": {
        "base_dir": os.path.join(BASE_DIR, "wce_valence_embed_attn/results"),
        "seed_template": "{}_wce_valence_embed_attn_beta_{}",
        "betas": ["0_25", "0_5", "1"],
        "color": "#d13262"
    },
    "Val Embed + Attn (No WCE)": {
        "base_dir": os.path.join(BASE_DIR, "no_wce_valence_embed_attn/results"),
        "seed_template": "{}_no_wce_embed_attn",
        "betas": [None],
        "color": "#1e4667"
    }
}

# only for train RM Accuracy
metric_file = "train_rm_accuracy_valence.pt"

target_betas = ["0.25", "0.5", "1"] # columns 

beta_map = {"0.25": ["0.25", "0_25"], "0.5": ["0.5", "0_5"], "1": ["1"]}

plt.rcParams.update({'font.size': 14}) 

fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

for idx, target_beta in enumerate(target_betas):
    ax = axes[idx]
    print(f"\n--- Processing Subplot Beta: {target_beta} ---")
    
    for model_name, config in models.items():
        model_betas = config["betas"]
        
        active_beta = None
        if None in model_betas:
            active_beta = None
        else:
            for b_variant in beta_map[target_beta]:
                if b_variant in model_betas:
                    active_beta = b_variant
                    break
            
            if active_beta is None:
                continue

        print(f"  > Model: {model_name} | Active Beta variant: {active_beta}")

        beta_seeds_data = [] 
        for seed in [0, 1, 2]:
            if active_beta:
                folder_name = config["seed_template"].format(seed, active_beta)
            else:
                folder_name = config["seed_template"].format(seed)
                
            file_path = os.path.join(config["base_dir"], folder_name, metric_file)
            
            if os.path.exists(file_path):
                try:
                    data = torch.load(file_path, map_location='cpu', weights_only=False)
                    beta_seeds_data.append(data)
                    if seed == 0:
                        keys_sample = list(data.keys())[:3]
                        print(f"    [Seed 0] Found {len(data)} keys. Sample: {keys_sample}")
                except Exception as e:
                    print(f"    Error loading {file_path}: {e}")
            else:
                print(f"    MISSING: {file_path}")

        if beta_seeds_data:
            normalized_data = []
            all_v_vals = set()
            
            for d in beta_seeds_data:
                new_d = {float(k): v for k, v in d.items()}
                normalized_data.append(new_d)
                all_v_vals.update(new_d.keys())
            
            sorted_vals = sorted(list(all_v_vals))
            
            means, stderrs, valid_x = [], [], []
            for v_val in sorted_vals:
                vals_for_v = []
                for d in normalized_data:
                    if v_val in d:
                        val = d[v_val]
                        if torch.is_tensor(val):
                            val = val.item()
                        vals_for_v.append(val)
                
                if vals_for_v:
                    means.append(np.mean(vals_for_v))
                    stderrs.append(np.std(vals_for_v) / np.sqrt(len(vals_for_v)))
                    valid_x.append(v_val)
            
            if valid_x:
                # add Jitter to make overlapping lines visible
                # each model gets a slight shift
                model_index = list(models.keys()).index(model_name)
                jitter = model_index * 0.01 
                jittered_x = [x + jitter for x in valid_x]

                # use different linestyles for WCE models to help distinguish
                current_linestyle = '-'

                ax.errorbar(jittered_x, means, yerr=stderrs, label=model_name, 
                            color=config["color"], capsize=4, marker='o', 
                            markersize=5, linestyle=current_linestyle, 
                            linewidth=1.5, alpha=0.8) 
                
    ax.set_title(f"β = {target_beta.replace('_', '.')}", fontweight='bold')
    ax.set_xlabel("Valence")
    if idx == 0:
        ax.set_ylabel("Long-term \n memory accuracy")
    ax.grid(True, linestyle=':', alpha=0.6)
    
# single legend for the whole figure
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=13)

plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # adjust for legend and title
save_name = os.path.join(BASE_DIR, "comparison_valence_accuracy.pdf")
plt.savefig(save_name, bbox_inches='tight')
print(f"Saved {save_name}")
plt.close()