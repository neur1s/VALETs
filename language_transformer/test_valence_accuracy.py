import argparse
import importlib.util
import inspect
import os
import json
import random
import textwrap
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 14})
import torch
from torch.utils.data import DataLoader
from flair.models import SequenceTagger

from language_task import (LanguageTask, TextDataset, CausalLMCollator,
                            _find_sentence_boundaries,
                            MIN_REPEAT_DIST)
from preprocess import load_task_dataset


# hyperparameters 

D_MODEL  = 512
D_FF     = 2048
N_LAYERS = 6
N_HEADS  = 8
MAX_LEN  = 512

MODEL_SEED = 0            # single trained-model seed for both models
EVAL_SEEDS = [3, 5, 2]   # three data sub-samples → error bars
BETAS      = [0.25, 0.5, 1.0]
QUAL_BETA  = 1          # beta used for WCE qualitative examples
N_QUAL     = 2

EVAL_DEPTHS      = [0.0, 0.25, 0.5, 0.75]
DEPTH_LABELS     = ["0%", "25%", "50%", "75%"]
DEPTH_BINS_RIGHT = [0.125, 0.375, 0.625, 0.875]

VARIANTS = [
    {"label": "baseline", "folder": "baseline", "betas": [None]},
    {"label": "wce_val_attn", "folder": "wce_valence_embed_attn", "betas": BETAS},
]
MODEL_COLORS = {
    "Baseline": "grey",
    "WCE + Val + Attn": "#d13262",
}


# model loader 

def import_make_model(base_dir, folder):
    path = os.path.join(base_dir, folder, "model.py")
    spec = importlib.util.spec_from_file_location(f"model_{folder}", path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "make_model", None) or getattr(mod, "make_mlm_model")


def load_checkpoint(make_fn, ckpt_path, tokenizer, device):
    model   = make_fn(tokenizer.vocab_size,
                      d_model=D_MODEL, d_ff=D_FF,
                      N=N_LAYERS, h=N_HEADS, max_len=MAX_LEN)
    has_val = "valences" in inspect.signature(model.forward).parameters
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    return model, has_val


# valence injection

def inject_both_sentences(vals, first_pos, repeat_pos, ids, tokenizer, sign):
    """
    Overwrites target sentences with +/- 1.0 while preserving 
    original POS-based valences for all other tokens.
    """
    # create a copy of original valences (preserves 0.8 NNP signal)
    v = list(vals) 
    
    # convert ids to list for boundary detection
    ids_list = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    
    # use the fast sentence boundary logic 
    sentences = _find_sentence_boundaries(ids_list, tokenizer)
    
    for ss, se in sentences:
        if (ss <= first_pos < se) or (ss <= repeat_pos < se): # if sentence contains either the first or repeat occurrence
            for i in range(ss, se): 
                v[i] = sign
    return v


# core evaluation 

@torch.no_grad()
def evaluate_recall(model, has_valences, dataloader, tokenizer, device,
                    collect_examples=False):
    n_depth  = len(EVAL_DEPTHS)
    correct  = [0] * n_depth
    total    = [0] * n_depth
    examples = [] if collect_examples else None

    for batch in dataloader:
        input_ids_b = batch["input_ids"]
        pad_mask_b  = batch["attention_mask"]
        labels_b    = batch["labels"]
        valences_b  = batch["valences"]
        B, T        = input_ids_b.shape

        for b in range(B):
            ids      = input_ids_b[b]
            mask     = pad_mask_b[b]
            labels   = labels_b[b]
            vals     = valences_b[b].tolist()
            real_len = int(mask.sum().item())
            if real_len < 4:
                continue

            ids_list  = ids[:real_len].tolist()
            vals_real = vals[:real_len]

            first_seen = {}
            all_pairs  = []
            for t, (tid, val) in enumerate(zip(ids_list, vals_real)):
                if abs(val) >= 0.75:
                    if tid in first_seen:
                        first_t = first_seen[tid]
                        if t - first_t >= MIN_REPEAT_DIST:
                            all_pairs.append(
                                (first_t, t, 1.0 if val > 0 else -1.0))
                            first_seen[tid] = t  # Bug 2 fix: chain to next gap
                    else:
                        first_seen[tid] = t

            for first_pos, repeat_pos, sign in all_pairs:
                if labels[repeat_pos].item() == -100:
                    continue

                injected = inject_both_sentences(
                    vals, first_pos, repeat_pos, ids, tokenizer, sign)

                ids_t = ids.unsqueeze(0).to(device)
                msk_t = mask.unsqueeze(0).to(device)
                val_t = torch.tensor(injected, dtype=torch.float
                                     ).unsqueeze(0).to(device)

                logits  = (model(ids_t, msk_t, valences=val_t)
                           if has_valences else model(ids_t, msk_t))
                pred_id = logits[0, repeat_pos - 1].argmax().item()
                true_id = ids[repeat_pos].item()
                is_corr = int(pred_id == true_id)

                frac  = first_pos / (MAX_LEN - 1)
                d_bin = min(np.searchsorted(DEPTH_BINS_RIGHT, frac),
                            n_depth - 1)
                correct[d_bin] += is_corr
                total[d_bin]   += 1

                if collect_examples:
                    examples.append({
                        "ids": ids_list,
                        "first_pos": first_pos,
                        "repeat_pos": repeat_pos,
                        "sign": sign,
                        "pred_id": pred_id,
                        "true_id": true_id,
                        "correct": is_corr,
                        "real_len": real_len,
                    })

    acc = [correct[d] / total[d] if total[d] > 0 else float("nan")
           for d in range(n_depth)]
    return acc, total, examples


# multi-seed evaluation 

def evaluate_variant_multi_seed(base_dir, folder, betas, make_fn,
                                 all_texts, tokenizer, pos_tagger, device,
                                 n_per_seed, collect_qual=False):
    results   = {}
    qual_data = {}

    for beta in betas:
        ckpt = (
            os.path.join(base_dir, folder,
                         f"checkpoint_seed{MODEL_SEED}_mlm.pt")
            if beta is None else
            os.path.join(base_dir, folder,
                         f"checkpoint_seed{MODEL_SEED}_beta{beta}_mlm.pt")
        )
        if not os.path.exists(ckpt):
            print(f"  [SKIP] {ckpt}")
            results[beta] = None
            continue

        print(f"  Loading {os.path.basename(ckpt)} ...")
        model, has_val = load_checkpoint(make_fn, ckpt, tokenizer, device)

        per_seed_acc = []
        per_seed_tot = []
        all_examples = []

        for eval_seed in EVAL_SEEDS:
            rng      = random.Random(eval_seed)
            sub      = rng.sample(all_texts, min(n_per_seed, len(all_texts)))
            suffix   = f"_qualeval_s{eval_seed}"
            ds = TextDataset(sub, tokenizer, MAX_LEN, pos_tagger,
                             upgrade_sentence_position=None,
                             cache_suffix=suffix)
            dl = DataLoader(ds, batch_size=16, shuffle=False,
                            collate_fn=CausalLMCollator(
                                tokenizer.pad_token_id or 0),
                            num_workers=0)

            want_ex = collect_qual and (beta == QUAL_BETA or beta is None)
            acc, tot, exs = evaluate_recall(
                model, has_val, dl, tokenizer, device,
                collect_examples=want_ex)
            per_seed_acc.append(acc)
            per_seed_tot.append(tot)
            if want_ex and exs:
                all_examples.extend(exs)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        arr  = np.array(per_seed_acc, dtype=float)
        mean = np.nanmean(arr, axis=0)
        sem  = np.nanstd(arr, axis=0, ddof=0) / max(np.sqrt(len(EVAL_SEEDS)), 1)
        tot  = np.array(per_seed_tot, dtype=float).sum(axis=0).tolist()

        results[beta] = {
            "mean": mean.tolist(),
            "sem": sem.tolist(),
            "total": tot,
            "per_seed": per_seed_acc,
        }
        if collect_qual and all_examples:
            qual_data[beta] = all_examples

    return results, qual_data


# qualitative example printer 

def _sent_span(ids, pos, tokenizer):
    for ss, se in _find_sentence_boundaries(ids, tokenizer):
        if ss <= pos < se:
            return ss, se
    return max(0, pos - 5), min(len(ids), pos + 6)


def print_qualitative_examples(examples, tokenizer, model_label,
                                want_correct, n=N_QUAL):
    filtered = [e for e in examples if e["correct"] == int(want_correct)]
    random.shuffle(filtered)
    filtered = filtered[:n]

    tag = "CORRECT ✓" if want_correct else "WRONG ✗"
    print(f"\n{'='*70}")
    print(f"  {model_label}  —  {len(filtered)} examples where prediction is {tag}")
    print(f"{'='*70}")

    for i, ex in enumerate(filtered, 1):
        ids       = ex["ids"]
        fp        = ex["first_pos"]
        rp        = ex["repeat_pos"]
        pred_word = tokenizer.decode([ex["pred_id"]],
                                     skip_special_tokens=True).strip()
        true_word = tokenizer.decode([ex["true_id"]],
                                     skip_special_tokens=True).strip()

        f_ss, f_se = _sent_span(ids, fp, tokenizer)
        r_ss, r_se = _sent_span(ids, rp, tokenizer)

        first_sent = tokenizer.decode(ids[f_ss:f_se],
                                      skip_special_tokens=True).strip()

        # bridge: up to 2 sentences strictly between the two sentence spans
        all_sents  = _find_sentence_boundaries(ids, tokenizer)
        bridge_tok = [
            (ss, se) for ss, se in all_sents
            if ss >= f_se and se <= r_ss
        ][:2]
        bridge = " ".join(
            tokenizer.decode(ids[ss:se], skip_special_tokens=True).strip()
            for ss, se in bridge_tok
        )
        if not bridge and r_ss > f_se:
            snip  = tokenizer.decode(ids[f_se:min(f_se + 40, r_ss)],
                                     skip_special_tokens=True).strip()
            bridge = (snip + " [...]") if snip else ""

        # repeat sentence with annotation
        before = tokenizer.decode(ids[r_ss:rp],
                                  skip_special_tokens=True).strip()
        after  = tokenizer.decode(ids[rp + 1:r_se],
                                  skip_special_tokens=True).strip()
        ann    = (f"[✓ {pred_word}]" if want_correct
                  else f"[✗ pred={pred_word} | true={true_word}]")
        repeat_sent = f"{before} {ann} {after}".strip()

        depth_pct = int(round(fp / (MAX_LEN - 1) * 100))  # Bug 1 fix: absolute depth
        print(f"\n  [{i}]  first-occ depth ≈ {depth_pct}%")
        print(f"  FIRST : {textwrap.fill(first_sent,  80, subsequent_indent='          ')}")
        if bridge:
            print(f"  ...   : {textwrap.fill(bridge,      80, subsequent_indent='          ')}")
        print(f"  SECOND: {textwrap.fill(repeat_sent, 80, subsequent_indent='          ')}")


# plot

def plot_recall_facet(baseline_res, valence_res, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)

    x       = np.arange(len(EVAL_DEPTHS))
    bl      = (baseline_res or {}).get(None)
    bl_mean = np.array(bl["mean"]) if bl else None
    bl_sem  = np.array(bl["sem"])  if bl else None

    for col, beta in enumerate(BETAS):
        ax  = axes[col]
        res = (valence_res or {}).get(beta)
        ax.set_title(f"β = {beta}", fontsize=16, fontweight="bold")

        if bl_mean is not None:
            m = ~np.isnan(bl_mean)
            ax.errorbar(x[m], bl_mean[m], yerr=bl_sem[m],
                        marker="o", linewidth=1.5, markersize=5, capsize=4,
                        color=MODEL_COLORS["Baseline"],
                        label="Baseline", alpha=0.8)

        if res is not None:
            mean = np.array(res["mean"])
            sem  = np.array(res["sem"])
            m    = ~np.isnan(mean)
            ax.errorbar(x[m] + 0.05, mean[m], yerr=sem[m],
                        marker="o", linewidth=1.5, markersize=5, capsize=4,
                        color=MODEL_COLORS["WCE + Val + Attn"],
                        label="WCE + Val + Attn", alpha=0.8)

        ax.set_xlabel("Depth of first occurrence")
        if col == 0:
            ax.set_ylabel("Recall accuracy\n(second occurrence)")
        ax.set_xticks(x)
        ax.set_xticklabels(DEPTH_LABELS)
        ax.set_ylim(0.3, 0.5) 
        ax.grid(True, linestyle=":", alpha=0.6)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper center",
               bbox_to_anchor=(0.5, 0.05), ncol=2)
      
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved → {output_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_samples", type=int, default=500,
                   help="Total val docs to load")
    p.add_argument("--n_per_seed", type=int, default=200,
                   help="Docs per eval-seed sub-sample (≤ n_samples)")
    p.add_argument("--output_plot", type=str,
                   default="recall_by_first_depth.pdf")
    p.add_argument("--output_json", type=str,
                   default="recall_by_first_depth.json")
    args = p.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Model seed: {MODEL_SEED}  "
          f"|  Eval seeds: {EVAL_SEEDS}")

    pos_tagger = SequenceTagger.load("flair/pos-english")
    task = LanguageTask(model_name="bert-base-uncased", max_len=MAX_LEN,
                        tagger=pos_tagger, upgrade_sentence_position=None)
    tokenizer = task.tokenizer

    print(f"Loading {args.n_samples} val documents...")
    all_texts = load_task_dataset(split="val",
                                  max_samples=args.n_samples, min_len=400)
    print(f"Loaded {len(all_texts)} documents.")

    # Pre-cache sub-samples so TextDataset hits disk on all subsequent loads
    print("Pre-caching eval sub-samples...")
    for eval_seed in EVAL_SEEDS:
        rng = random.Random(eval_seed)
        sub = rng.sample(all_texts, min(args.n_per_seed, len(all_texts)))
        TextDataset(sub, tokenizer, MAX_LEN, pos_tagger,
                    upgrade_sentence_position=None,
                    cache_suffix=f"_qualeval_s{eval_seed}")
    print()

    json_out     = {}
    baseline_res = None
    valence_res  = None
    baseline_qual = {}
    valence_qual  = {}

    for variant in VARIANTS:
        label  = variant["label"]
        folder = variant["folder"]
        betas  = variant["betas"]

        if not os.path.isdir(os.path.join(base_dir, folder)):
            print(f"[SKIP] folder not found: {folder}")
            continue

        print(f"\n{'='*55}\n  {label}\n{'='*55}")
        make_fn = import_make_model(base_dir, folder)
        res, qual = evaluate_variant_multi_seed(
            base_dir, folder, betas, make_fn,
            all_texts, tokenizer, pos_tagger, device,
            n_per_seed=args.n_per_seed,
            collect_qual=True,
        )

        for beta, r in res.items():
            tag = label if beta is None else f"{label} β={beta}"
            print(f"\n  {tag}")
            if r is None:
                print("    no checkpoints found"); continue
            for depth, m, s, t in zip(
                    DEPTH_LABELS, r["mean"], r["sem"], r["total"]):
                val_str = "N/A" if np.isnan(m) else f"{m:.4f} ±{s:.4f}"
                print(f"    first-occ depth {depth:>4}: {val_str}  (n={int(t)})")

        json_out[label] = {str(b): r for b, r in res.items()}

        if label == "baseline":
            baseline_res  = res
            baseline_qual = qual
        else:
            valence_res  = res
            valence_qual = qual

    # qualitative examples 
    bl_ex  = baseline_qual.get(None, [])
    wce_ex = valence_qual.get(QUAL_BETA, [])

    if bl_ex:
        print_qualitative_examples(
            bl_ex, tokenizer,
            f"BASELINE  (model seed {MODEL_SEED})",
            want_correct=False, n=N_QUAL)
    else:
        print("\n[No baseline examples collected — check checkpoint path]")

    if wce_ex:
        print_qualitative_examples(
            wce_ex, tokenizer,
            f"WCE + Val + Attn  β={QUAL_BETA}  (model seed {MODEL_SEED})",
            want_correct=True, n=N_QUAL)
    else:
        print("\n[No WCE examples collected — check checkpoint path]")

    # plot 
    if not baseline_res and not valence_res:
        print("No results to plot.")
        return

    plot_recall_facet(
        baseline_res or {},
        valence_res  or {},
        os.path.join(base_dir, args.output_plot),
    )

    with open(os.path.join(base_dir, args.output_json), "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"JSON → {args.output_json}")


if __name__ == "__main__":
    main()