import os
import sys
import importlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})

SCRIPT_DIR = os.environ["BASE_DIR"]

BETA_VALUES = [0.25, 0.5, 1.0]
SEEDS = [0, 1, 2]
POOL_SEED = 42

num_contexts = 32
num_observations = 1024
seq_len = 256
mem_len = 64
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
num_train_samples = 1000

model_colors = {
    "Baseline": "grey",
    "WCE": "#a3b7c2",
    "WCE + Val Embed (MLP)": "#51748e",
    "WCE + Val Embed + Attn": "#d13262",
    "Val Embed + Attn (No WCE)": "#1e4667",
}


def _import_from(subdir):
    module_dir = os.path.join(SCRIPT_DIR, subdir)
    for mod_name in ['model', 'train', 'ctx_episodic_memory_task']:
        sys.modules.pop(mod_name, None)
    sys.path.insert(0, module_dir)
    model_mod = importlib.import_module('model')
    train_mod = importlib.import_module('train')
    task_mod  = importlib.import_module('ctx_episodic_memory_task')
    sys.path.remove(module_dir)
    return model_mod, train_mod, task_mod


def _make_train_loader(task_mod, seed=0):
    train_dataset = task_mod.CtxtMemoryDataset(
        num_contexts=num_contexts,
        num_observations=num_observations,
        num_samples=num_train_samples,
        seq_len=seq_len,
        seed=seed,
        pool_seed=POOL_SEED,
    )
    obs_to_valence = train_dataset.core.obs_to_valence
    loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
    return loader, obs_to_valence


def calculate_gini(x):
    x = np.asarray(x, dtype=float)
    if np.sum(x) == 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))


def _subsequent_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).to(torch.uint8)
    return mask == 0


def _gini_valence_binned(attn_row, key_valences, valence_levels):
    bin_rate = np.array([
        attn_row[key_valences == v].mean() if np.any(key_valences == v) else 0.0
        for v in valence_levels
    ])
    return calculate_gini(bin_rate)


def _accumulate_gini(avg_attn, query_obs_ids, src_obs_ids,
                     obs_to_valence, accumulator, q_range, k_range):
    B, tgt_obs, src_obs = avg_attn.shape
    
    t_start = int(tgt_obs * q_range[0])
    t_end   = int(tgt_obs * q_range[1])
    
    ks, ke = int(src_obs * k_range[0]), int(src_obs * k_range[1]) # key window

    for b in range(B):
        current_keys = src_obs_ids[b, ks:ke]
        current_attn = avg_attn[b, :, ks:ke] #slice attention to match the keys
        
        #re-normalize over the window so attention sums to 1 
        row_sums = current_attn.sum(axis=-1, keepdims=True)
        current_attn = np.divide(current_attn, row_sums, out=np.zeros_like(current_attn), where=row_sums != 0)

        key_vals = np.array([round(float(obs_to_valence[int(o)]), 2) for o in current_keys])
        valence_levels = sorted(set(key_vals.tolist()))

        for t in range(t_start, t_end):
            row   = current_attn[t, :]
            q_obs = int(query_obs_ids[b, t])
            q_val = round(float(obs_to_valence[q_obs]), 2)
            g     = _gini_valence_binned(row, key_vals, valence_levels)
            accumulator.setdefault(q_val, []).append(g)


def _compute_gini_baseline(model, loader, obs_to_valence, device, cfg):
    SOS      = cfg["num_observations"]
    CTX_OFF  = cfg["num_observations"] + 1
    _mem_len = cfg['mem_len']

    accumulator = {}
    model.eval()

    with torch.no_grad():
        for ctx_ids, obs_ids in loader:
            obs_ids, ctx_ids = obs_ids.to(device), ctx_ids.to(device)
            B, T = obs_ids.size()

            src    = torch.cat([(ctx_ids + CTX_OFF).unsqueeze(1), obs_ids], dim=1)
            memory = model.encode(src, src_mask=None)
            src_obs_np = obs_ids.cpu().numpy()

            for start in range(0, T - _mem_len + 1, _mem_len):
                end = start + _mem_len
                sos = torch.full((B, 1), SOS, dtype=torch.long, device=device)
                tgt_window = torch.cat([sos, obs_ids[:, start:end - 1]], dim=1)
                tgt_mask   = _subsequent_mask(tgt_window.size(1)).to(device)

                _ = model.decode(memory, None, tgt_window, tgt_mask)

                p_attn    = model.decoder.layers[-1].src_attn.attn
                cross_obs = p_attn[:, :, :, 1:].cpu().numpy()  
                avg_attn  = cross_obs.mean(axis=1)              

                avg_attn_no_sos = avg_attn[:, 1:, :]
                query_obs = obs_ids[:, start:start + (_mem_len - 1)].cpu().numpy()

                _accumulate_gini(avg_attn_no_sos, query_obs, src_obs_np,
                                  obs_to_valence, accumulator, q_range=(0.75, 1.0), k_range=(0.7, 1.0))

    return {v: float(np.mean(gs)) for v, gs in accumulator.items() if gs}

def _compute_gini_valence_attn(model, valence_embedder, loader, obs_to_valence, device):
    accumulator = {}
    model.eval()
    valence_embedder.eval()

    with torch.no_grad():
        for ctx_ids, obs_ids in loader:
            obs_ids, ctx_ids = obs_ids.to(device), ctx_ids.to(device)
            B, T = obs_ids.size()

            src_vals_full = torch.tensor(
                [[0.0] + [obs_to_valence[o.item()] for o in row] for row in obs_ids],
                device=device, dtype=torch.float32)
            tgt_vals_full = torch.tensor(
                [[0.0] + [obs_to_valence[o.item()] for o in row[:-1]] for row in obs_ids],
                device=device, dtype=torch.float32)

            _, _, _, cross_w = model.forward_with_weights(
                ctx_ids, obs_ids, src_vals_full, tgt_vals_full, valence_embedder)

            cross_obs       = cross_w[:, :, :, 1:]           
            if isinstance(cross_obs, torch.Tensor):
                cross_obs = cross_obs.cpu().numpy()
            avg_attn        = cross_obs.mean(axis=1)          
            avg_attn_no_sos = avg_attn[:, 1:, :]              

            query_obs  = obs_ids[:, T - mem_len: T - 1].cpu().numpy()
            src_obs_np = obs_ids.cpu().numpy()

            _accumulate_gini(avg_attn_no_sos, query_obs, src_obs_np,
                              obs_to_valence, accumulator, q_range=(0.75, 1.0), k_range=(0.75, 1.0))

    return {v: float(np.mean(gs)) for v, gs in accumulator.items() if gs}

def _evaluate_baseline(ckpt_path, device, seed, compute_gini=False):
    model_mod, train_mod, task_mod = _import_from("baseline")
    cp  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cp['config']
    model = model_mod.make_model(
        num_contexts=cfg['num_contexts'], num_observations=cfg['num_observations'],
        N=cfg['num_encoder_layers'], d_model=cfg['d_model'],
        d_ff=cfg['dim_feedforward'], h=cfg['nhead'],
        max_len=cfg['seq_len'] + 1, decoder_max_len=cfg['mem_len'],
    )
    model.load_state_dict(cp['model_state'])
    model.to(device).eval()
    loader, obs_to_valence = _make_train_loader(task_mod, seed=seed)
    SOS, CTX_OFF = cfg["num_observations"], cfg["num_observations"] + 1
    criterion = nn.CrossEntropyLoss()
    _, _, _, _, vs = train_mod.evaluate_memory(
        model, loader, criterion, SOS, CTX_OFF, device,
        obs_to_valence=obs_to_valence, mem_len=cfg['mem_len'],
    )
    gini_pv = None
    if compute_gini:
        gini_pv = _compute_gini_baseline(model, loader, obs_to_valence, device, cfg)
    return vs, gini_pv


def _evaluate_wce(ckpt_path, beta, device, seed):
    model_mod, train_mod, task_mod = _import_from("wce")
    cp  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cp['config']
    model = model_mod.make_model(
        num_contexts=cfg['num_contexts'], num_observations=cfg['num_observations'],
        N=cfg['num_encoder_layers'], d_model=cfg['d_model'],
        d_ff=cfg['dim_feedforward'], h=cfg['nhead'],
        max_len=cfg['seq_len'] + 1, decoder_max_len=cfg['mem_len'],
    )
    model.load_state_dict(cp['model_state'])
    model.to(device).eval()
    loader, obs_to_valence = _make_train_loader(task_mod, seed=seed)
    SOS, CTX_OFF = cfg["num_observations"], cfg["num_observations"] + 1
    criterion = model_mod.WCELoss(valence_scaling=beta)
    _, _, _, _, vs = train_mod.evaluate_memory(
        model, loader, criterion, SOS, CTX_OFF, device,
        obs_to_valence=obs_to_valence, mem_len=cfg['mem_len'],
    )
    return vs


def _evaluate_valence_mlp(ckpt_path, beta, device, seed):
    model_mod, train_mod, task_mod = _import_from("wce_valence_embed_mlp")
    cp  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cp['config']
    model = model_mod.make_model(
        num_contexts=cfg['num_contexts'], num_observations=cfg['num_observations'],
        N=cfg['num_encoder_layers'], d_model=cfg['d_model'],
        d_ff=cfg['dim_feedforward'], h=cfg['nhead'],
        max_len=cfg['seq_len'] + 1, decoder_max_len=cfg['mem_len'],
    )
    model.load_state_dict(cp['model_state'])
    model.to(device).eval()
    loader, obs_to_valence = _make_train_loader(task_mod, seed=seed)
    SOS, CTX_OFF = cfg["num_observations"], cfg["num_observations"] + 1
    criterion = model_mod.WCELoss(valence_scaling=beta)
    _, _, _, _, vs = train_mod.evaluate_memory(
        model, loader, criterion, SOS, CTX_OFF, device,
        obs_to_valence=obs_to_valence, mem_len=cfg['mem_len'],
    )
    return vs


def _evaluate_valence_attn(subdir, ckpt_path, beta, device, seed, compute_gini=False):
    model_mod, train_mod, task_mod = _import_from(subdir)
    cp  = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = cp['config']
    model = model_mod.make_model(
        num_contexts=cfg['num_contexts'], num_observations=cfg['num_observations'],
        N=cfg['num_encoder_layers'], d_model=cfg['d_model'],
        d_ff=cfg['dim_feedforward'], h=cfg['nhead'],
        max_len=cfg['seq_len'] + 1, decoder_max_len=cfg['mem_len'],
    )
    model.load_state_dict(cp['model_state'])
    model.to(device).eval()
    valence_embedder = model_mod.ValenceEmbedder().to(device).eval()
    train_mod.valence_embedder = valence_embedder
    loader, obs_to_valence = _make_train_loader(task_mod, seed=seed)
    SOS, CTX_OFF = cfg["num_observations"], cfg["num_observations"] + 1
    if subdir == "no_wce_valence_embed_attn":
        _ce_fn = nn.CrossEntropyLoss()
        criterion = lambda logits, targets, valences=None: _ce_fn(
            logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    else:
        criterion = model_mod.WCELoss(valence_scaling=beta)
    _, _, _, _, vs = train_mod.evaluate_memory(
        model, valence_embedder, loader, criterion, SOS, CTX_OFF, device,
        obs_to_valence=obs_to_valence, mem_len=cfg['mem_len'],
    )
    gini_pv = None
    if compute_gini:
        gini_pv = _compute_gini_valence_attn(
            model, valence_embedder, loader, obs_to_valence, device)
    return vs, gini_pv

def compute_valence_accuracy(valence_stats):
    valences = sorted(valence_stats.keys())
    wm_acc, rm_acc = {}, {}
    for v in valences:
        s = valence_stats[v]
        wm_acc[v] = s['wm_correct'] / s['wm_total'] if s['wm_total'] > 0 else None
        rm_acc[v] = s['rm_correct'] / s['rm_total'] if s['rm_total'] > 0 else None
    return valences, wm_acc, rm_acc


def aggregate_across_seeds(list_of_stats):
    all_valences = sorted({v for vs in list_of_stats for v in vs.keys()})
    wm_all, rm_all = [], []
    for vs in list_of_stats:
        _, wm, rm = compute_valence_accuracy(vs)
        wm_all.append(wm)
        rm_all.append(rm)
    wm_mean, wm_sem, rm_mean, rm_sem = {}, {}, {}, {}
    for v in all_valences:
        wm_vals = [w[v] for w in wm_all if v in w and w[v] is not None]
        rm_vals = [r[v] for r in rm_all if v in r and r[v] is not None]
        if wm_vals:
            wm_mean[v] = np.mean(wm_vals)
            wm_sem[v]  = np.std(wm_vals) / np.sqrt(len(wm_vals)) if len(wm_vals) > 1 else 0
        if rm_vals:
            rm_mean[v] = np.mean(rm_vals)
            rm_sem[v]  = np.std(rm_vals) / np.sqrt(len(rm_vals)) if len(rm_vals) > 1 else 0
    return all_valences, wm_mean, wm_sem, rm_mean, rm_sem


def aggregate_scalar_across_seeds(list_of_dicts):
    all_valences = sorted({v for d in list_of_dicts for v in d.keys()})
    mean_d, sem_d = {}, {}
    for v in all_valences:
        vals = [d[v] for d in list_of_dicts if v in d]
        if vals:
            mean_d[v] = float(np.mean(vals))
            sem_d[v]  = float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return all_valences, mean_d, sem_d

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    results      = {b: {} for b in BETA_VALUES}
    gini_results = {b: {} for b in BETA_VALUES}

    for beta in BETA_VALUES:
        print(f"\n{'='*60}\n  Beta = {beta}\n{'='*60}")

        print("  Evaluating Baseline ...")
        bl_stats, bl_gini = [], []
        for seed in SEEDS:
            path = os.path.join(SCRIPT_DIR, "baseline", f"checkpoint_seed{seed}.pt")
            vs, gini_pv = _evaluate_baseline(path, device, seed, compute_gini=True)
            bl_stats.append(vs)
            if gini_pv:
                bl_gini.append(gini_pv)
        results[beta]["Baseline"] = aggregate_across_seeds(bl_stats)
        if bl_gini:
            gini_results[beta]["Baseline"] = aggregate_scalar_across_seeds(bl_gini)

        print(f"  Evaluating WCE β={beta} ...")
        wce_stats = []
        for seed in SEEDS:
            path = os.path.join(SCRIPT_DIR, "wce", f"checkpoint_seed{seed}_beta{beta}.pt")
            wce_stats.append(_evaluate_wce(path, beta, device, seed))
        results[beta]["WCE"] = aggregate_across_seeds(wce_stats)

        print(f"  Evaluating WCE + Val Embed (MLP) β={beta} ...")
        mlp_stats = []
        for seed in SEEDS:
            path = os.path.join(SCRIPT_DIR, "wce_valence_embed_mlp",
                                f"checkpoint_seed{seed}_beta{beta}.pt")
            mlp_stats.append(_evaluate_valence_mlp(path, beta, device, seed))
        results[beta]["WCE + Val Embed (MLP)"] = aggregate_across_seeds(mlp_stats)

        print(f"  Evaluating WCE + Val Embed + Attn β={beta} ...")
        attn_stats, attn_gini = [], []
        for seed in SEEDS:
            path = os.path.join(SCRIPT_DIR, "wce_valence_embed_attn",
                                f"checkpoint_seed{seed}_beta{beta}.pt")
            vs, gini_pv = _evaluate_valence_attn(
                "wce_valence_embed_attn", path, beta, device, seed, compute_gini=True)
            attn_stats.append(vs)
            if gini_pv:
                attn_gini.append(gini_pv)
        results[beta]["WCE + Val Embed + Attn"] = aggregate_across_seeds(attn_stats)
        if attn_gini:
            gini_results[beta]["WCE + Val Embed + Attn"] = \
                aggregate_scalar_across_seeds(attn_gini)

        print("  Evaluating Val Embed + Attn (No WCE) ...")
        no_wce_stats = []
        for seed in SEEDS:
            path = os.path.join(SCRIPT_DIR, "no_wce_valence_embed_attn",
                                f"checkpoint_seed{seed}.pt")
            vs, _ = _evaluate_valence_attn(
                "no_wce_valence_embed_attn", path, beta, device, seed,
                compute_gini=False)
            no_wce_stats.append(vs)
        results[beta]["Val Embed + Attn (No WCE)"] = aggregate_across_seeds(no_wce_stats)

    # ── Accuracy plots ──────────────────────────────────────────────────────────
    for mem_key, out_name in [
        ("wm", "comparison_all_models_wm_train.pdf"),
        ("rm", "comparison_all_models_rm_train.pdf"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for col, beta in enumerate(BETA_VALUES):
            ax = axes[col]
            ax.set_title(f"β = {beta}", fontsize=16, fontweight='bold')
            for i, (model_name, data) in enumerate(results[beta].items()):
                plot_color = model_colors[model_name]
                valences, wm_mean, wm_sem, rm_mean, rm_sem = data
                mean_dict = wm_mean if mem_key == "wm" else rm_mean
                sem_dict  = wm_sem  if mem_key == "wm" else rm_sem
                xvals = np.array([v for v in valences
                                  if v in mean_dict and mean_dict[v] is not None])
                yvals = np.array([mean_dict[v] for v in xvals])
                yerr  = np.array([sem_dict.get(v, 0) for v in xvals])
                if len(xvals) > 0:
                    ax.errorbar(xvals + i * 0.01, yvals, yerr=yerr,
                                marker='o', linewidth=1.5, markersize=5, capsize=4,
                                color=plot_color, label=model_name, alpha=0.8)
            ax.set_xlabel("Valence", fontsize=12)
            if col == 0:
                ylabel = ("Working memory\naccuracy" if mem_key == "wm"
                          else "Long-term\nmemory accuracy")
                ax.set_ylabel(ylabel, fontsize=12)
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle=':', alpha=0.6)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center',
                   bbox_to_anchor=(0.5, 0.05), ncol=3, fontsize=11)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        fig.savefig(os.path.join(SCRIPT_DIR, out_name), bbox_inches="tight")
        plt.close(fig)

    #plot Gini index 
    gini_models = ["Baseline", "WCE + Val Embed + Attn"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for col, beta in enumerate(BETA_VALUES):
        ax = axes[col]
        ax.set_title(f"β = {beta}", fontsize=16, fontweight='bold')
        for i, model_name in enumerate(gini_models):
            if model_name not in gini_results[beta]: continue
            plot_color = model_colors[model_name]
            valences, mean_d, sem_d = gini_results[beta][model_name]
            xvals = np.array([v for v in valences if v in mean_d])
            yvals = np.array([mean_d[v] for v in xvals])
            yerr  = np.array([sem_d.get(v, 0.0) for v in xvals])
            if len(xvals) > 0:
                ax.errorbar(xvals + i * 0.01, yvals, yerr=yerr,
                            marker='o', linewidth=1.5, markersize=5, capsize=4,
                            color=plot_color, label=model_name, alpha=0.8)
        ax.set_xlabel("Query valence", fontsize=12)
        if col == 0:
            ax.set_ylabel("Gini index", fontsize=11)
        ax.grid(True, linestyle=':', alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 0.05), ncol=2, fontsize=11)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    fig.savefig(os.path.join(SCRIPT_DIR, "comparison_gini_cross_attn.pdf"), bbox_inches="tight")
    plt.close(fig)

if __name__ == "__main__":
    main()