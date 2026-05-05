import sys, os
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))

import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import make_model, subsequent_mask
from ctx_episodic_memory_task import CtxtMemoryTask, CtxtMemoryDataset
from train import (
    get_special_tokens,
    train_loop,
    evaluate_memory,
)


# reproducibility helper 

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# hyperparameters 

SEEDS = [0, 1, 2]

d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
num_epochs = 100
batch_size = 32
num_train_samples = 1000
num_val_samples = 200
learning_rate = 0.0001

num_contexts = 32
num_observations = 1024
seq_len = 256   # full episode length
mem_len = 64    # WM window / decoder PE limit
POOL_SEED = 42    # fixed so vocab is identical across seeds

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_histories = {}

    for seed in SEEDS:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}")
        print(f"{'='*60}\n")

        set_seed(seed)

        # datasets (pool_seed fixed so vocab is shared across seeds) 
        train_dataset = CtxtMemoryDataset(
            num_contexts=num_contexts,
            num_observations=num_observations,
            num_samples=num_train_samples,
            seq_len=seq_len,
            seed=seed,           # varies per run → different episode ordering
            pool_seed=POOL_SEED, # fixed  → same vocabulary across all seeds
        )
        val_dataset = CtxtMemoryDataset(
            num_contexts=num_contexts,
            num_observations=num_observations,
            num_samples=num_val_samples,
            seq_len=seq_len,
            seed=seed + 100000,  # distinct from train, but deterministic per seed
            pool_seed=POOL_SEED,
        )

        # re-seed after dataset creation so DataLoader shuffle is also seeded
        set_seed(seed)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        obs_to_valence = train_dataset.core.obs_to_valence

        print(f"Training samples:   {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")

        model = make_model(
            num_contexts=num_contexts,
            num_observations=num_observations,
            N=num_encoder_layers,
            d_model=d_model,
            d_ff=dim_feedforward,
            h=nhead,
            max_len=seq_len + 1,
            decoder_max_len=mem_len,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        history = train_loop(
            model, train_loader, val_loader, optimizer, criterion, num_epochs,
            device=device,
            num_observations=num_observations,
            num_contexts=num_contexts,
            obs_to_valence=obs_to_valence,
            mem_len=mem_len,
        )
        all_histories[seed] = history

        # rename per-seed plots 
        for fname in ["memory_accuracy.png", "training_accuracy.png",
                      "train_valence_accuracy.png", "val_valence_accuracy.png"]:
            if os.path.exists(fname):
                base, ext = os.path.splitext(fname)
                os.rename(fname, f"{base}_seed{seed}{ext}")

        # save per-seed checkpoint 
        config = {
            "seed": seed,
            "num_contexts": num_contexts,
            "num_observations": num_observations,
            "seq_len": seq_len,
            "d_model": d_model,
            "nhead": nhead,
            "num_encoder_layers": num_encoder_layers,
            "num_decoder_layers": num_decoder_layers,
            "dim_feedforward": dim_feedforward,
            "batch_size": batch_size,
            "num_train_samples": num_train_samples,
            "num_val_samples": num_val_samples,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "pool_seed": POOL_SEED,
            "mem_len": mem_len,
        }
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "history": history,
            "config": config,
        }, f"checkpoint_seed{seed}.pt")
        print(f"Checkpoint saved → checkpoint_seed{seed}.pt")

    # aggregate across seeds
    print(f"\n{'='*60}")
    print("  SEED SWEEP SUMMARY  (mean ± std across seeds)")
    print(f"{'='*60}\n")

    keys = ["train_wm_error", "train_rm_error",
            "val_wm_error", "val_rm_error",
            "train_loss", "val_loss",
            "train_acc", "val_acc"]

    agg = {}   # key -> {"mean": [...], "std": [...]}
    for k in keys:
        arrays = np.array([all_histories[s][k] for s in SEEDS])  # (n_seeds, n_epochs)
        agg[k] = {"mean": arrays.mean(axis=0).tolist(),
                  "std": arrays.std(axis=0).tolist()}

    # print final-epoch summary
    print(f"{'Metric':<25} {'Mean':>10} {'Std':>10}")
    print("-" * 47)
    for k in keys:
        mean_final = agg[k]["mean"][-1]
        std_final  = agg[k]["std"][-1]
        print(f"{k:<25} {mean_final:>10.4f} {std_final:>10.4f}")

    torch.save({
        "seeds": SEEDS,
        "all_histories": all_histories,
        "aggregated": agg,
    }, "seed_results_summary.pt")
    print("\nAggregated results saved → seed_results_summary.pt")


if __name__ == "__main__":
    main()
