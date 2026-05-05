import argparse
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import make_model, subsequent_mask
from ctx_episodic_memory_task import CtxtMemoryDataset
from train import (
    get_special_tokens,
    run_batch,
    eval_memory_types,
    evaluate_memory,
    plot_memory_errors,
    plot_epoch_accuracy,
)


# checkpoint loading 

def load_checkpoint(path="checkpoint.pt", device="cpu"):
    ckpt = torch.load(path, map_location=device)
    cfg  = ckpt["config"]

    SOS_TOKEN, CTX_OFFSET = get_special_tokens(cfg["num_observations"], cfg["num_contexts"])
    max_len = cfg["seq_len"] + 1
    decoder_max_len = cfg.get("mem_len", None)  # decoder PE window limit

    model = make_model(
        num_contexts=cfg["num_contexts"],
        num_observations=cfg["num_observations"],
        N=cfg["num_encoder_layers"],
        d_model=cfg["d_model"],
        d_ff=cfg["dim_feedforward"],
        h=cfg["nhead"],
        dropout=cfg.get("dropout", 0.1),
        max_len=max_len,
        decoder_max_len=decoder_max_len,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    print(f"Loaded checkpoint from {path}")
    print(f"  Trained for {cfg['num_epochs']} epochs")
    print(f"  num_contexts={cfg['num_contexts']}  num_observations={cfg['num_observations']}  seq_len={cfg['seq_len']}")
    if decoder_max_len:
        print(f"  decoder_max_len={decoder_max_len} (mem_len)")

    return model, cfg, ckpt["history"], SOS_TOKEN, CTX_OFFSET


def build_val_loader(cfg):
    val_dataset = CtxtMemoryDataset(
        num_contexts=cfg["num_contexts"],
        num_observations=cfg["num_observations"],
        num_samples=cfg["num_val_samples"],
        seq_len=cfg["seq_len"],
        seed=cfg["val_seed"],
        pool_seed=cfg["pool_seed"],
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg["batch_size"], shuffle=False)
    obs_to_valence = val_dataset.core.obs_to_valence
    context_pools = val_dataset.core.context_pools
    return val_loader, obs_to_valence, context_pools, val_dataset.core


def build_train_loader(cfg):
    train_dataset = CtxtMemoryDataset(
        num_contexts=cfg["num_contexts"],
        num_observations=cfg["num_observations"],
        num_samples=cfg["num_train_samples"],
        seq_len=cfg["seq_len"],
        seed=cfg["train_seed"],
        pool_seed=cfg["pool_seed"],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg["batch_size"], shuffle=False)
    obs_to_valence = train_dataset.core.obs_to_valence
    context_pools = train_dataset.core.context_pools
    return train_loader, obs_to_valence, context_pools, train_dataset.core

def eval_full_retrieval(model, val_loader, cfg, SOS_TOKEN, CTX_OFFSET,
                        obs_to_valence, device):
    criterion = nn.CrossEntropyLoss()
    mem_len = cfg.get("mem_len", None)
    loss, wm_err, rm_err, acc, valence_stats = evaluate_memory(
        model, val_loader, criterion, SOS_TOKEN, CTX_OFFSET, device, 
        obs_to_valence, mem_len=mem_len)

    print(f"\n=== Full Retrieval (Mode 1) ===")
    print(f"  Loss:    {loss:.4f}")
    print(f"  WM Acc:  {100*(1-wm_err):.1f}%")
    print(f"  RM Acc:  {100*(1-rm_err):.1f}%")
    print(f"  Overall: {100*acc:.1f}%")

    return valence_stats

def eval_pattern_completion(model, cfg, context_pools, obs_to_valence,
                            SOS_TOKEN, CTX_OFFSET, device,
                            encoder_fraction=0.5, num_samples=200, seed=77777):
    rng = np.random.RandomState(seed)
    num_contexts    = cfg["num_contexts"]
    num_observations = cfg["num_observations"]
    seq_len         = cfg["seq_len"]

    in_pool_correct = 0
    in_pool_total   = 0
    valence_pc_stats = {}   # {valence: {correct, total}}

    model.eval()
    with torch.no_grad():
        for _ in tqdm(range(num_samples), desc="Pattern completion"):
            ctx_id = rng.randint(0, num_contexts)
            pool   = context_pools[ctx_id]

            # split pool: encoder sees encoder_fraction, rest are held out
            n_enc  = max(1, int(len(pool) * encoder_fraction))
            perm   = rng.permutation(len(pool))
            enc_obs = pool[perm[:n_enc]]
            held_out = set(pool[perm[n_enc:]].tolist())

            if len(held_out) == 0:
                continue

            # build encoder input: [ctx_token, enc_obs...]
            ctx_token = torch.tensor([ctx_id + CTX_OFFSET], dtype=torch.long, device=device)
            enc_tensor = torch.tensor(enc_obs, dtype=torch.long, device=device)
            src = torch.cat([ctx_token, enc_tensor]).unsqueeze(0)   # [1, n_enc+1]

            # encode once
            memory = model.encode(src, src_mask=None)               # [1, n_enc+1, d]

            # autoregressive decode from SOS — generate one token
            tgt = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)  # [1, 1]
            tgt_emb  = model.tgt_embed(tgt)
            tgt_mask = subsequent_mask(1).to(device)
            dec_out  = model.decoder(tgt_emb, memory, src_mask=None, tgt_mask=tgt_mask)
            logits   = model.generator(dec_out[:, -1, :])           # [1, num_obs]

            # restrict prediction to pool membership
            pred_id = logits.argmax(-1).item()
            pool_set = set(pool.tolist())

            is_in_pool = pred_id in pool_set
            is_held_out_correct = pred_id in held_out

            in_pool_correct += int(is_in_pool)
            in_pool_total   += 1

            # valence of the predicted token
            if 0 <= pred_id < num_observations:
                v = obs_to_valence[pred_id]
                if v not in valence_pc_stats:
                    valence_pc_stats[v] = {"correct": 0, "total": 0}
                valence_pc_stats[v]["total"]   += 1
                valence_pc_stats[v]["correct"] += int(is_in_pool)

    pool_acc = 100 * in_pool_correct / max(in_pool_total, 1)
    chance   = 100 * len(pool) / num_observations
    print(f"\n=== Pattern Completion (Mode 2, encoder sees {int(encoder_fraction*100)}% of pool) ===")
    print(f"  Pool membership accuracy: {pool_acc:.1f}%  (chance: {chance:.1f}%)")

    return valence_pc_stats, pool_acc, chance


def plot_valence_accuracy(valence_stats, title_wm, title_rm, fname):
    """Plot WM and RM accuracy vs valence from full-retrieval stats."""
    valences      = sorted(valence_stats.keys())
    wm_accs, rm_accs = [], []

    for v in valences:
        s = valence_stats[v]
        wm_accs.append(100 * s["wm_correct"] / s["wm_total"] if s["wm_total"] > 0 else None)
        rm_accs.append(100 * s["rm_correct"] / s["rm_total"] if s["rm_total"] > 0 else None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, accs, title, color, marker in [
        (axes[0], wm_accs, title_wm, "tab:blue", "o"),
        (axes[1], rm_accs, title_rm, "tab:red", "s"),
    ]:
        vals = [v for v, a in zip(valences, accs) if a is not None]
        accs_f = [a for a in accs if a is not None]
        if vals:
            ax.plot(vals, accs_f, marker=marker, linewidth=2, markersize=8, color=color)
        ax.set_xlabel("Valence", fontsize=12)
        ax.set_ylabel("Accuracy (%)", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def plot_pattern_completion_valence(valence_pc_stats, chance, fname):
    """Pool membership accuracy vs valence for pattern completion."""
    valences = sorted(valence_pc_stats.keys())
    accs = [
        100 * valence_pc_stats[v]["correct"] / valence_pc_stats[v]["total"]
        if valence_pc_stats[v]["total"] > 0 else None
        for v in valences
    ]
    vals  = [v for v, a in zip(valences, accs) if a is not None]
    accs_f = [a for a in accs if a is not None]

    plt.figure(figsize=(8, 5))
    if vals:
        plt.plot(vals, accs_f, marker="D", linewidth=2, markersize=8, color="tab:purple")
    plt.axhline(chance, color="gray", linestyle="--", label=f"Chance ({chance:.1f}%)")
    plt.xlabel("Valence of Predicted Token", fontsize=12)
    plt.ylabel("Pool Membership Accuracy (%)", fontsize=12)
    plt.title("Pattern Completion: Valence vs Pool Accuracy", fontsize=13, fontweight="bold")
    plt.ylim(-5, 105)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {fname}")


def print_valence_table(valence_stats):
    print(f"\n{'Valence':<10} {'WM Acc (%)':<15} {'WM N':<10} {'RM Acc (%)':<15} {'RM N':<10}")
    print("-" * 62)
    for v in sorted(valence_stats.keys()):
        s = valence_stats[v]
        wm = 100 * s["wm_correct"] / s["wm_total"] if s["wm_total"] > 0 else 0.0
        rm = 100 * s["rm_correct"] / s["rm_total"] if s["rm_total"] > 0 else 0.0
        print(f"{v:<10.2f} {wm:<15.1f} {s['wm_total']:<10} {rm:<15.1f} {s['rm_total']:<10}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoint.pt")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # load model + config
    model, cfg, history, SOS_TOKEN, CTX_OFFSET = load_checkpoint(args.checkpoint, device)

    # rebuild train and val loaders using same seeds as training
    train_loader, obs_to_valence, context_pools, task_core = build_train_loader(cfg)
    val_loader, _, _, _ = build_val_loader(cfg)

    # training curves (from saved history) 
    plot_memory_errors(history)
    plot_epoch_accuracy(history)

    # valence vs accuracy (WM and RM, full retrieval) 
    print("\n" + "=" * 70)
    print("TRAINING DATA EVALUATION")
    print("=" * 70)
    train_valence_stats = eval_full_retrieval(
        model, train_loader, cfg, SOS_TOKEN, CTX_OFFSET, obs_to_valence, device)
    print_valence_table(train_valence_stats)
    plot_valence_accuracy(
        train_valence_stats,
        title_wm="Training WM: Valence vs Accuracy",
        title_rm="Training RM: Valence vs Accuracy",
        fname="train_valence_accuracy.png",
    )

    print("\n" + "=" * 70)
    print("VALIDATION DATA EVALUATION")
    print("=" * 70)
    val_valence_stats = eval_full_retrieval(
        model, val_loader, cfg, SOS_TOKEN, CTX_OFFSET, obs_to_valence, device)
    print_valence_table(val_valence_stats)
    plot_valence_accuracy(
        val_valence_stats,
        title_wm="Validation WM: Valence vs Accuracy",
        title_rm="Validation RM: Valence vs Accuracy",
        fname="val_valence_accuracy.png",
    )

    # pattern completion — pool membership vs valence 
    print("\n" + "=" * 70)
    print("PATTERN COMPLETION EVALUATION")
    print("=" * 70)
    valence_pc_stats, pool_acc, chance = eval_pattern_completion(
        model, cfg, context_pools, obs_to_valence,
        SOS_TOKEN, CTX_OFFSET, device,
        encoder_fraction=0.5,
        num_samples=500,
    )

    plot_pattern_completion_valence(valence_pc_stats, chance, "pattern_completion.png")

    print("\nAll charts saved:")
    print("  memory_accuracy.png          — WM/RM accuracy over training epochs")
    print("  training_accuracy.png        — overall train/val accuracy over epochs")
    print("  train_valence_accuracy.png   — training data valence vs WM/RM accuracy")
    print("  val_valence_accuracy.png     — validation data valence vs WM/RM accuracy")
    print("  pattern_completion.png       — valence vs pool accuracy (partial encoder)")