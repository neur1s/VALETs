import torch
import matplotlib.pyplot as plt
import seaborn as sns
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
        "color": "grey",
        "marker": None
    },
    "WCE": {
        "base_dir": os.path.join(BASE_DIR, "wce/results"),
        "seed_template": "{}_wce_loss_beta_{}",
        "betas": ["0_25", "0_5", "1"],
        "color": "#a3b7c2",
        "marker": "o"
    },
    "WCE + Val Embed (MLP)": {
        "base_dir": os.path.join(BASE_DIR, "wce_valence_embed_mlp/results"),
        "seed_template": "{}_wce_loss_beta_{}",
        "betas": ["0_25", "0_5", "1"],
        "color": "#51748e",
        "marker": "s"
    },
    "WCE + Val Embed + Attn": {
        "base_dir": os.path.join(BASE_DIR, "wce_valence_embed_attn/results"),
        "seed_template": "{}_wce_valence_embed_attn_beta_{}",
        "betas": ["0_25", "0_5", "1"],
        "color": "#d13262",
        "marker": "^"
    },
    "Val Embed + Attn (No WCE)": {
        "base_dir": os.path.join(BASE_DIR, "no_wce_valence_embed_attn/results"),
        "seed_template": "{}_no_wce_embed_attn",
        "betas": [None],
        "color": "#1e4667",
        "marker": "D"
    }
}

metrics = {
    "train_rm_certainty": "train_rm_certainty_valence.pt",
    "train_rm_loss": "train_rm_raw_loss_valence.pt",
    "distractor_task": "distractor_task_results.pt",
    "frequency_task": "frequency_task_results.pt",
    "spatial_info": "valence_vs_spatial_info.pt",
    "gini_index": "valence_vs_gini_idx.pt"
}

titles = {
    "train_rm_certainty": "Train reference memory certainty",
    "train_rm_loss": "Train reference memory raw loss",
    "distractor_task": "Distractor task accuracy",
    "frequency_task": "Frequency task accuracy",
    "spatial_info": "Probability density",
    "gini_index": "Self-attention Gini index"
}

target_betas = ["0.25", "0.5", "1"]
beta_map     = {"0.25": ["0.25", "0_25"], "0.5": ["0.5", "0_5"], "1": ["1"]}
v_colors     = {1.0: "red", -1.0: "blue"}
MIN_SAMPLES  = 20   # minimum number of raw accuracy values per bin to include

plt.rcParams.update({"font.size": 14})


def try_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


for metric_key, metric_file in metrics.items():
    is_si_dist   = (metric_key == "spatial_info")
    is_distractor = (metric_key == "distractor_task")
    is_frequency  = (metric_key == "frequency_task")

    if is_si_dist:
        valid_models = []
        for model_name, cfg in models.items():
            active_beta = "1" if "1" in cfg["betas"] or "1_0" in cfg["betas"] else None # do spatial info analysis for beta=1
            if None in cfg["betas"]:
                active_beta = None
            for seed in [0, 1, 2]:
                folder_name = cfg["seed_template"].format(seed, active_beta) if active_beta else cfg["seed_template"].format(seed)
                if os.path.exists(os.path.join(cfg["base_dir"], folder_name, metric_file)):
                    valid_models.append(model_name)
                    break
        if not valid_models:
            continue
        fig, axes = plt.subplots(len(valid_models), 1,
                                  figsize=(5, 3 * len(valid_models)),
                                  squeeze=False, sharex=True)
        axes        = axes.flatten()
        column_loop = valid_models
    else:
        fig, axes   = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        column_loop = target_betas

    for idx, col_item in enumerate(column_loop):
        ax            = axes[idx]
        active_models = [col_item] if is_si_dist else models.keys()

        # for frequency task: find the global x-axis cutoff across all models 
        # We collect, per n, the total number of raw samples across all seeds/models
        # active in this panel, then cap at the last n where ALL models have >= MIN_SAMPLES.
        if is_frequency:
            # First pass: for each model, find the last n where it has >= MIN_SAMPLES
            # x_cutoff = minimum across all models (so every model has data up to that n)
            per_model_max_n = []

            for model_name in active_models:
                cfg         = models[model_name]
                target_beta = col_item
                active_beta = None
                if None in cfg["betas"]:
                    active_beta = None
                else:
                    for b_variant in beta_map[target_beta]:
                        if b_variant in cfg["betas"]:
                            active_beta = b_variant
                            break
                    if active_beta is None:
                        continue

                # collect per-n sample counts for this model across all seeds and valences
                model_n_counts = {}
                for seed in [0, 1, 2]:
                    folder_name = cfg["seed_template"].format(seed, active_beta) if active_beta else cfg["seed_template"].format(seed) 
                    file_path   = os.path.join(cfg["base_dir"], folder_name, metric_file)
                    if not os.path.exists(file_path):
                        continue
                    data = torch.load(file_path, map_location="cpu", weights_only=False)
                    for v_key in data:
                        for n_key, accs in data[v_key].items():
                            n_int     = int(n_key)
                            n_samples = len(accs) if isinstance(accs, list) else 1
                            model_n_counts[n_int] = model_n_counts.get(n_int, 0) + n_samples

                # last n with sufficient samples for this model
                valid_ns = sorted([n for n, cnt in model_n_counts.items() if cnt >= MIN_SAMPLES])
                if valid_ns:
                    per_model_max_n.append(valid_ns[-1])

            # global cutoff: the minimum across all models so every model has data
            x_cutoff = min(per_model_max_n) if per_model_max_n else None

        for model_name in active_models:
            cfg         = models[model_name]
            target_beta = col_item
            active_beta = None

            if is_si_dist:
                active_beta = "1" if "1" in cfg["betas"] or "1_0" in cfg["betas"] else None
                if None in cfg["betas"]:
                    active_beta = None
            else:
                if None in cfg["betas"]:
                    active_beta = None
                else:
                    for b_variant in beta_map[target_beta]:
                        if b_variant in cfg["betas"]:
                            active_beta = b_variant
                            break
                    if active_beta is None:
                        continue

            beta_seeds_data = []
            for seed in [0, 1, 2]:
                folder_name = cfg["seed_template"].format(seed, active_beta) if active_beta else cfg["seed_template"].format(seed)
                file_path   = os.path.join(cfg["base_dir"], folder_name, metric_file)
                if os.path.exists(file_path):
                    beta_seeds_data.append(
                        torch.load(file_path, map_location="cpu", weights_only=False)
                    )
            if not beta_seeds_data:
                continue

            # spatial info plot
            if is_si_dist:
                custom_colors = mcolors.LinearSegmentedColormap.from_list("custom_valence", ['#18364f', '#4d858f', '#9cd07e', '#edc070'])
                all_valences  = sorted(set().union(*(d.keys() for d in beta_seeds_data)))
                for v_val in all_valences:
                    combined = [s for d in beta_seeds_data if v_val in d for s in d[v_val]]
                    if not combined:
                        continue
                    norm_v = (float(v_val) + 1) / 2
                    sns.kdeplot(combined, ax=ax, color=custom_colors(norm_v),
                                label=f"$v={v_val:.1f}$", linewidth=2, alpha=0.8)
                ax.set_title(model_name, fontweight="bold")
                ax.set_xlim(0, 1)
                ax.set_xlabel("Spatial information")

            # distractor task plot 
            elif is_distractor:
                if model_name == "Baseline":
                    all_n        = set().union(*(d[v].keys() for d in beta_seeds_data for v in d))
                    sorted_n     = sorted([int(n) for n in all_n])
                    points_by_n  = [[d[v][n] for d in beta_seeds_data for v in d if n in d[v]]
                                    for n in sorted_n]
                    means = np.array([np.mean(p) for p in points_by_n])
                    sems  = np.array([np.std(p) / np.sqrt(len(p)) for p in points_by_n])
                    ax.plot(sorted_n, means, label="Baseline", color="black",
                            linestyle="--", linewidth=3)
                    ax.fill_between(sorted_n, means - sems, means + sems,
                                    color="black", alpha=0.1)
                else:
                    for v_val in [1.0, -1.0]:
                        all_n = set().union(*(d[v_val].keys() for d in beta_seeds_data if v_val in d))
                        if not all_n:
                            continue
                        sorted_n    = sorted([int(n) for n in all_n])
                        points_by_n = [[d[v_val][n] for d in beta_seeds_data
                                        if v_val in d and n in d[v_val]]
                                       for n in sorted_n]
                        means = np.array([np.mean(p) for p in points_by_n])
                        sems  = np.array([np.std(p) / np.sqrt(len(p)) for p in points_by_n])
                        ax.plot(sorted_n, means,
                                label=f"$v={v_val:.1f}$",
                                color=v_colors[v_val], marker=cfg["marker"], alpha=0.8)

            elif is_frequency:
                if x_cutoff is None:
                    continue

                def get_moving_avg(x, y, window=5):
                    if len(y) < window: 
                        return x, y
                    
                    # use a simple rolling mean that doesn't pad with zeros
                    y_smooth = np.zeros_like(y, dtype=float)
                    half_w = window // 2
                    
                    for i in range(len(y)):
                        # calculate bounds for the window at this specific index
                        start = max(0, i - half_w)
                        end = min(len(y), i + half_w + 1)
                        # average only the available data in the current window
                        y_smooth[i] = np.mean(y[start:end])
                        
                    return x, y_smooth

                if model_name == "Baseline":
                    all_n = set()
                    for d in beta_seeds_data:
                        for v in d:
                            all_n |= set(int(n) for n in d[v].keys())
                    sorted_n = sorted([n for n in all_n if n <= x_cutoff])

                    points_by_n = []
                    for n in sorted_n:
                        pts = []
                        for d in beta_seeds_data:
                            for v in d:
                                accs = d[v].get(n, d[v].get(str(n)))
                                if accs is not None:
                                    pts.extend(accs if isinstance(accs, list) else [accs])
                        points_by_n.append(pts)

                    filtered = [(n, p) for n, p in zip(sorted_n, points_by_n) if len(p) >= MIN_SAMPLES]
                    if filtered:
                        sn, pn = zip(*filtered)
                        means = np.array([np.mean(p) for p in pn])
                        sems  = np.array([np.std(p) / np.sqrt(len(p)) for p in pn])
                        
                        # apply Smoothing
                        sn_plot, means_smooth = get_moving_avg(sn, means, window=3)
                        
                        ax.plot(sn_plot, means_smooth, label="Baseline", color="black",
                                linestyle="--", linewidth=3)
                        ax.fill_between(sn_plot, means_smooth - sems, means_smooth + sems,
                                        color="black", alpha=0.1)

                else:
                    for target_v in [1.0, -1.0]:
                        all_n = set()
                        for d in beta_seeds_data:
                            v_key = next((k for k in d if try_float(k) == target_v), None)
                            if v_key:
                                all_n |= set(int(n) for n in d[v_key].keys())
                        sorted_n = sorted([n for n in all_n if n <= x_cutoff])

                        points_by_n = []
                        for n in sorted_n:
                            pts = []
                            for d in beta_seeds_data:
                                v_key = next((k for k in d if try_float(k) == target_v), None)
                                if v_key:
                                    accs = d[v_key].get(n, d[v_key].get(str(n)))
                                    if accs is not None:
                                        pts.extend(accs if isinstance(accs, list) else [accs])
                            points_by_n.append(pts)

                        filtered = [(n, p) for n, p in zip(sorted_n, points_by_n) if len(p) >= MIN_SAMPLES]
                        if not filtered: continue
                        
                        sn_f, pn_f = zip(*filtered)
                        means = np.array([np.mean(p) for p in pn_f])
                        sems  = np.array([np.std(p) / np.sqrt(len(p)) for p in pn_f])
                        
                        # apply Smoothing
                        sn_plot, means_smooth = get_moving_avg(sn_f, means, window=3)

                        ax.plot(sn_plot, means_smooth,
                                label=f"{model_name} (V={target_v:+})",
                                color=v_colors[target_v],
                                marker=cfg["marker"], alpha=0.8, markersize=4)
                        ax.fill_between(sn_plot,
                                        means_smooth - sems, means_smooth + sems,
                                        color=v_colors[target_v], alpha=0.15)

            # reference memory plots
            else:
                val_keys    = set().union(*(d.keys() for d in beta_seeds_data))
                sorted_vals = sorted([float(v) for v in val_keys])
                means, stderrs, valid_x = [], [], []
                for v_val in sorted_vals:
                    vals_for_v = [d[k] for d in beta_seeds_data
                                  for k in [v_val, str(v_val), round(v_val, 2)] if k in d]
                    if vals_for_v:
                        means.append(np.mean(vals_for_v))
                        stderrs.append(np.std(vals_for_v) / np.sqrt(len(vals_for_v)))
                        valid_x.append(v_val)
                if valid_x:
                    ax.errorbar(valid_x, means, yerr=stderrs,
                                label=model_name, color=cfg["color"],
                                capsize=5, marker="o")

        # axis formatting 
        if not is_si_dist:
            ax.set_title(f"β = {col_item}", fontweight="bold")
        if not is_si_dist and not is_distractor and not is_frequency:
            ax.set_xlabel("Valence")
        if is_frequency:
            ax.set_xlabel("# previous visits")
        if idx == 0:
            ax.set_ylabel(titles[metric_key])
        if is_si_dist:
            ax.set_ylabel(titles[metric_key])
        ax.grid(True, linestyle="--", alpha=0.5)

    # legend 
    handles, labels = axes[0].get_legend_handles_labels()
    if is_si_dist:
        by_label = dict(zip(labels, handles))
        fig.legend(by_label.values(), by_label.keys(),
                loc="center left", bbox_to_anchor=(1.01, 0.5),
                ncol=1, fontsize=14, frameon=True)
    else:
        fig.legend(handles, labels,
                   loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(os.path.join(BASE_DIR, f"comparison_{metric_key}.pdf"), bbox_inches="tight")
    plt.close()