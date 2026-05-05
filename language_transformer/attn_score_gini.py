import argparse
import importlib.util
import inspect
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from flair.models import SequenceTagger

from language_task import (LanguageTask, TextDataset, CausalLMCollator,
                            _find_sentence_boundaries, _find_nnp_repeat_pair)
from preprocess import load_task_dataset

# constants 

D_MODEL     = 512
D_FF        = 2048
N_LAYERS    = 6
N_HEADS     = 8
MAX_LEN     = 512
MODEL_SEED  = 0
BETA        = 1.0

# full valence range including ±1.0 from NNP injection
VALENCE_LEVELS = [-1.0, -0.8, -0.64, -0.56, -0.48, -0.4, -0.32, -0.24,
                  -0.08, 0.08, 0.24, 0.32, 0.4, 0.48, 0.56, 0.64,
                   0.8, 1.0]

# match train.py colormap 
HEX_LIST    = ['#18364f', '#4d858f', '#9cd07e', '#fffed6']
CUSTOM_CMAP = mcolors.LinearSegmentedColormap.from_list("custom_valence", HEX_LIST)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))


# binning helpers 
def get_bin_indices(valences_np):
    """Nearest-neighbour bin for each valence value."""
    lvls = np.array(VALENCE_LEVELS)
    diff = np.abs(valences_np[..., np.newaxis] - lvls)
    return np.argmin(diff, axis=-1)

def gini_from_sum(weights):
    w = np.sort(weights)
    n = len(w)
    if n == 0 or w.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return ((2 * idx - n - 1) * w).sum() / (n * w.sum())


# model loader 
def import_make_model(folder):
    path = os.path.join(SCRIPT_DIR, folder, "model.py")
    spec = importlib.util.spec_from_file_location(f"model_{folder}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "make_model", None) or getattr(mod, "make_mlm_model")

def load_model(folder, ckpt_name, tokenizer, device):
    make_fn = import_make_model(folder)
    model   = make_fn(tokenizer.vocab_size, d_model=D_MODEL, d_ff=D_FF,
                      N=N_LAYERS, h=N_HEADS, max_len=MAX_LEN)
    ckpt    = os.path.join(SCRIPT_DIR, folder, ckpt_name)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device).eval()
    has_val = "valences" in inspect.signature(model.forward).parameters
    return model, has_val


# NNP injection 
def inject_valences(ids_list, vals_list, tokenizer):
    """
    Find one NNP repeat pair and inject ±1.0 on both sentences.
    Returns the injected valence list, or the original if no pair found.
    """
    pair = _find_nnp_repeat_pair(ids_list, vals_list, tokenizer)
    if pair is None:
        return vals_list

    first_pos, repeat_pos, sign = pair
    new_v     = list(vals_list)
    sentences = _find_sentence_boundaries(ids_list, tokenizer)
    for ss, se in sentences:
        if ss <= first_pos < se or ss <= repeat_pos < se:
            for i in range(ss, se):
                new_v[i] = sign
    return new_v


# attention extraction 
@torch.no_grad()
def collect_attention_stats(model, has_val, dataloader, tokenizer, device,
                             n_batches):
    """
    Runs forward passes with NNP-injected valences and aggregates:
      - heatmap_sum / heatmap_count  →  avg attention per (query_bin, key_bin)
      - gini_per_bin                 →  attention Gini per query_bin

    For the WCE model the self_attn module returns (output, p_attn).
    We capture p_attn via a hook that reads the module's output tuple,
    NOT module.attn (which is never assigned in the WCE model).
    """
    n = len(VALENCE_LEVELS)
    heatmap_sum   = np.zeros((n, n), dtype=np.float64)
    heatmap_count = np.zeros((n, n), dtype=np.float64)
    gini_per_bin  = [[] for _ in range(n)]

    # storage for the hook: use a list so the closure can mutate it
    captured = [None]

    def hook_fn(module, inp, out):
        # baseline: out is a plain tensor (attention output)
        #   module.attn holds p_attn (set inside attention())
        # WCE model: out is (attention_output, p_attn) tuple
        if isinstance(out, tuple):
            # WCE model: second element is p_attn (B, H, T, T)
            captured[0] = out[1].detach().cpu()
        elif hasattr(module, "attn") and module.attn is not None:
            # baseline: p_attn stored in module.attn
            captured[0] = module.attn.detach().cpu()

    # hook the last decoder layer's self-attention
    last_self_attn = model.decoder_stack.layers[-1].self_attn
    handle = last_self_attn.register_forward_hook(hook_fn)

    batch_idx = 0
    for batch in tqdm(dataloader, desc="Collecting attn", total=n_batches):
        if batch_idx >= n_batches:
            break

        ids_b  = batch["input_ids"]
        mask_b = batch["attention_mask"]
        vals_b = batch["valences"]  # raw POS valences from dataset
        B      = ids_b.shape[0]

        # inject ±1.0 for each example in the batch
        injected_vals = []
        for b in range(B):
            real_len  = int(mask_b[b].sum().item())
            ids_list  = ids_b[b, :real_len].tolist()
            vals_list = vals_b[b, :real_len].tolist()
            new_v     = inject_valences(ids_list, vals_list, tokenizer)
            # Pad back to full length
            new_v    += [0.0] * (ids_b.shape[1] - len(new_v))
            injected_vals.append(new_v[:ids_b.shape[1]])

        val_t = torch.tensor(injected_vals, dtype=torch.float)
        vals_np = val_t.numpy()  # (B, T) — includes ±1.0 where injected

        ids_d  = ids_b.to(device)
        mask_d = mask_b.to(device)
        val_d  = val_t.to(device)

        captured[0] = None
        if has_val:
            _ = model(ids_d, mask_d, valences=val_d)
        else:
            _ = model(ids_d, mask_d)

        if captured[0] is None:
            print(f"  WARNING: no attention captured for batch {batch_idx}")
            batch_idx += 1
            continue

        # p_attn: (B, H, T, T) — average over heads -> (B, T, T)
        p_attn = captured[0].mean(dim=1).numpy()
        bin_ids = get_bin_indices(vals_np)   # (B, T)

        for b in range(B):
            real_len = int(mask_b[b].sum().item())
            for t_q in range(1, real_len):   # skip position 0 ([CLS])
                qi  = bin_ids[b, t_q]
                row = p_attn[b, t_q, :t_q + 1]  # causal keys only

                if random.random() < 0.2:
                    gini_per_bin[qi].append(gini_from_sum(row))

                for t_k in range(t_q + 1):
                    ki = bin_ids[b, t_k]
                    heatmap_sum[qi, ki]   += p_attn[b, t_q, t_k]
                    heatmap_count[qi, ki] += 1

        batch_idx += 1

    handle.remove()

    heatmap = np.divide(heatmap_sum, heatmap_count,
                        out=np.zeros_like(heatmap_sum),
                        where=heatmap_count != 0)
    gini_vals = [np.mean(g) if g else np.nan for g in gini_per_bin]
    return heatmap, np.array(gini_vals)


# plotting 

def plot_heatmap(heatmap, levels, model_label, output_path):
    """
    Single heatmap panel. Matches train.py plot_valence_similarity_heatmap:
      figsize=(7,6), font.size=12, black spines, local vmin/vmax 98th pct.
    """
    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(7, 6))
    matrix  = np.nan_to_num(heatmap, nan=0.0)

    pos_vals = matrix[matrix > 0]
    vmax = float(np.percentile(pos_vals, 98)) if len(pos_vals) > 0 else 0.01
    vmin = 0.0

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im   = ax.imshow(matrix, cmap=CUSTOM_CMAP, norm=norm,
                     aspect='equal', origin='upper')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(0.7)

    ticks = np.arange(len(levels))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{l:.2f}" for l in levels],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{l:.2f}" for l in levels], fontsize=8)

    ax.set_xlabel("Key Valence", fontsize=13)
    ax.set_ylabel("Query Valence", fontsize=13)
    ax.set_title(f"{model_label}\n(Self-Attention Heatmap)",
                 fontweight='bold', fontsize=14, pad=10)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Avg Attention Score', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved → {output_path}")


def plot_gini_combined(gini_bl, gini_wce, levels, output_path):
    """
    Gini index line plot. Matches train.py accuracy plot style:
      figsize=(8,5), font.size=14, grid alpha=0.3.
    """
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.array(levels)

    # only plot where we have data (not nan)
    bl_mask  = ~np.isnan(gini_bl)
    wce_mask = ~np.isnan(gini_wce)

    ax.plot(x[bl_mask], gini_bl[bl_mask], marker='o', color='grey',
            label='Baseline', alpha=0.8, linewidth=2)
    ax.plot(x[wce_mask], gini_wce[wce_mask], marker='s', color='#d13262',
            label=f'WCE + Val + Attn (β={BETA})', linewidth=2)

    ax.set_xlabel('Query Valence', fontsize=14)
    ax.set_ylabel('Gini Index', fontsize=14)
    ax.set_title('Attention Sparsity by Query Valence', fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    print(f"Gini plot saved → {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=200,
                   help="Val documents to load")
    p.add_argument("--n_batches", type=int, default=50,
                   help="Batches to process per model")
    p.add_argument("--batch_size", type=int, default=8)
    args = p.parse_args()

    device     = "cuda" if torch.cuda.is_available() else "cpu"
    pos_tagger = SequenceTagger.load("flair/pos-english")
    task       = LanguageTask(model_name="bert-base-uncased", max_len=MAX_LEN,
                              tagger=pos_tagger,
                              upgrade_sentence_position=None)
    tokenizer  = task.tokenizer

    print(f"Loading {args.n_samples} val documents...")
    all_texts = load_task_dataset(split="val",
                                  max_samples=args.n_samples, min_len=200)
    ds = TextDataset(all_texts, tokenizer, MAX_LEN, pos_tagger,
                     upgrade_sentence_position=None,
                     cache_suffix="_attn_gini")
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=CausalLMCollator(tokenizer.pad_token_id or 0),
                    num_workers=0)

    configs = [
        {"folder": "baseline",
         "ckpt": f"checkpoint_seed{MODEL_SEED}_mlm.pt",
         "label": "Baseline"},
        {"folder": "wce_valence_embed_attn",
         "ckpt": f"checkpoint_seed{MODEL_SEED}_beta{BETA}_mlm.pt",
         "label": f"WCE Beta={BETA}"},
    ]

    gini_results = []
    for cfg in configs:
        print(f"\n{'='*50}\n  {cfg['label']}\n{'='*50}")
        model, has_val = load_model(cfg["folder"], cfg["ckpt"],
                                    tokenizer, device)
        print(f"  has_val={has_val}")

        hm, gn = collect_attention_stats(
            model, has_val, dl, tokenizer, device, args.n_batches)

        out_name = os.path.join(SCRIPT_DIR,
                                f"attn_heatmap_{cfg['folder']}.pdf")
        plot_heatmap(hm, VALENCE_LEVELS, cfg["label"], out_name)
        gini_results.append(gn)

        # Print summary
        print(f"  Heatmap max: {hm.max():.5f}  "
              f"non-zero cells: {(hm > 0).sum()}/{hm.size}")
        valid_g = [(VALENCE_LEVELS[i], gn[i])
                   for i in range(len(VALENCE_LEVELS))
                   if not np.isnan(gn[i])]
        print(f"  Gini non-NaN bins: {len(valid_g)}/{len(VALENCE_LEVELS)}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(gini_results) == 2:
        plot_gini_combined(
            gini_results[0], gini_results[1],
            VALENCE_LEVELS,
            os.path.join(SCRIPT_DIR, "gini_by_valence.pdf"),
        )


if __name__ == "__main__":
    main()