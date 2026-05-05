import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import make_model, WCELoss, ValenceContrastiveLoss
from language_task import LanguageTask


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


D_MODEL = 512
D_FF = 2048
N_LAYERS = 6
N_HEADS = 8
MAX_LEN = 512
BATCH_SIZE = 128
LR_FINETUNE = 5e-5
FINETUNE_EPOCHS = 30


def build_tensor_dataset(dataset_list, cache_path, max_len=512):
    if os.path.exists(cache_path):
        print(f"  Loading pre-built TensorDataset: {cache_path}")
        ids, mask, labels, vals = torch.load(cache_path)
        return TensorDataset(ids, mask, labels, vals)

    print(f"  Building TensorDataset ({len(dataset_list)} samples)...")
    t0 = time.time()

    def to_tensor(x, key, dtype, pad_val, length=max_len):
        v = x[key]
        if isinstance(v, torch.Tensor):
            v = v.detach()
        else:
            v = torch.tensor(v, dtype=dtype)
        v = v.to(dtype)
        if v.shape[0] < length:
            pad = torch.full((length - v.shape[0],), pad_val, dtype=dtype)
            v   = torch.cat([v, pad])
        return v[:length]

    ids    = torch.stack([to_tensor(x, "input_ids", torch.long, 0)     for x in dataset_list])
    mask   = torch.stack([to_tensor(x, "attention_mask", torch.bool, False) for x in dataset_list])
    labels = torch.stack([to_tensor(x, "labels", torch.long, -100)  for x in dataset_list])
    vals   = torch.stack([to_tensor(x, "valences", torch.float, 0.0)   for x in dataset_list])

    print(f"  Stacking done in {(time.time()-t0)/60:.1f} min — saving to {cache_path}")
    torch.save((ids, mask, labels, vals), cache_path)
    return TensorDataset(ids, mask, labels, vals)


def fast_valence_sample(valences_tensor, sample_size=512):
    flat  = valences_tensor.reshape(-1)
    valid = flat[flat.abs() > 1e-6]
    if len(valid) <= sample_size:
        return valid
    idx = torch.randperm(len(valid), device=valid.device)[:sample_size]
    return valid[idx]


def run_epoch(model, loader, gc_sample, optimizer,
              wce_fn, gc_fn, gc_weight, device,
              train=True, max_steps=None):
    model.train() if train else model.eval()
    total_loss, count, step = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch in loader:
            if train:
                optimizer.zero_grad(set_to_none=True)

            input_ids = batch[0].to(device, non_blocking=True)
            pad_mask = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)
            valences = batch[3].to(device, non_blocking=True)

            logits = model(input_ids, pad_mask, valences=valences)
            wce = wce_fn(logits, labels, valences)

            gc = (gc_fn(model.val_embedder(gc_sample), gc_sample)
                  if gc_sample is not None and len(gc_sample) > 1
                  else torch.tensor(0.0, device=device))

            loss = wce + gc_weight * gc

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            count += 1; step += 1
            if max_steps and step >= max_steps:
                break

    return total_loss / max(count, 1)


def main(args):
    torch.set_float32_matmul_precision('high')   # TF32 on H100
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cache_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n[{time.strftime('%H:%M:%S')}] Loading .pt caches...")
    ret_train_list = torch.load(f"{cache_base}/ret_aug_cache_2631998d_train_base.pt")
    ret_val_list = torch.load(f"{cache_base}/ret_aug_cache_370e996d_val_base.pt")
    raw_train_list = torch.load(f"{cache_base}/ar_cache_f77f0f24_len512_15000__train_raw.pt")

    print(f"[{time.strftime('%H:%M:%S')}] Building TensorDatasets...")
    ret_train_ds = build_tensor_dataset(
        ret_train_list, f"{cache_base}/tensor_ret_train.pt")
    ret_val_ds = build_tensor_dataset(
        ret_val_list, f"{cache_base}/tensor_ret_val.pt")
    raw_train_ds = build_tensor_dataset(
        raw_train_list, f"{cache_base}/tensor_raw_train.pt")

    ret_train_loader = DataLoader(ret_train_ds, batch_size=args.batch_size,
                                  shuffle=True, pin_memory=True, num_workers=0)
    ret_val_loader = DataLoader(ret_val_ds, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True, num_workers=0)
    raw_train_loader = DataLoader(raw_train_ds, batch_size=512,
                                  shuffle=True, pin_memory=True, num_workers=0)

    print(f"  Retrieval train: {len(ret_train_ds)}  val: {len(ret_val_ds)}")
    print(f"  Steps/epoch (train): {len(ret_train_ds) // args.batch_size}")

    print(f"[{time.strftime('%H:%M:%S')}] Building GC sample...")
    raw_batch = next(iter(raw_train_loader))
    gc_sample_cpu = fast_valence_sample(raw_batch[3], sample_size=512)

    task = LanguageTask(model_name="bert-base-uncased", max_len=MAX_LEN)
    vocab_size = task.tokenizer.vocab_size
    gc_fn = ValenceContrastiveLoss()

    for seed in args.seeds:
        for beta in args.betas:
            ckpt_final = os.path.join(out_dir, f"checkpoint_seed{seed}_beta{beta}_mlm.pt")
            phase1_ckpt = os.path.join(out_dir, f"checkpoint_seed{seed}_phase1.pt")

            if not os.path.exists(phase1_ckpt):
                print(f"\n[ERROR] Phase 1 checkpoint not found: {phase1_ckpt}")
                continue

            print(f"\n{'='*60}\n  SEED {seed}  BETA {beta}\n{'='*60}")
            set_seed(seed)

            model = make_model(
                vocab_size, d_model=args.embed_dim, d_ff=args.hidden_dim,
                N=args.num_layers, h=args.num_heads, max_len=MAX_LEN,
            ).to(device)
            model.load_state_dict(torch.load(phase1_ckpt, map_location=device))

            wce_fn = WCELoss(beta=beta)
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr_finetune,
                weight_decay=0.05, fused=True)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", patience=args.patience, factor=0.5)

            gc_sample = gc_sample_cpu.to(device)

            best_val, no_improve = float("inf"), 0

            print(f"--- Phase 2: {args.finetune_epochs} epochs ---")
            for epoch in range(1, args.finetune_epochs + 1):
                t0 = time.time()

                tr = run_epoch(model, ret_train_loader, gc_sample, optimizer,
                               wce_fn, gc_fn, args.gc_weight, device,
                               train=True, max_steps=args.max_steps)
                vl = run_epoch(model, ret_val_loader, gc_sample, optimizer,
                               wce_fn, gc_fn, args.gc_weight, device,
                               train=False, max_steps=args.max_steps)

                scheduler.step(vl)
                mins = (time.time() - t0) / 60
                print(f"  ep {epoch:3d} | train={tr:.4f}  val={vl:.4f} | {mins:.1f} min")

                if vl < best_val:
                    best_val, no_improve = vl, 0
                    torch.save(model.state_dict(), ckpt_final)
                    print(f"    saved -> {ckpt_final}")
                else:
                    no_improve += 1
                    if no_improve >= args.patience:
                        print(f"  Early stop (patience={args.patience})")
                        break

            print(f"\nSeed {seed} beta {beta} done. Best val: {best_val:.4f}")

    print("\nAll done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--betas", type=float, nargs="+", default=[0.5])
    p.add_argument("--finetune_epochs", type=int, default=FINETUNE_EPOCHS)
    p.add_argument("--embed_dim", type=int, default=D_MODEL)
    p.add_argument("--hidden_dim", type=int, default=D_FF)
    p.add_argument("--num_heads", type=int, default=N_HEADS)
    p.add_argument("--num_layers", type=int, default=N_LAYERS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr_finetune", type=float, default=LR_FINETUNE)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--gc_weight", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=None)
    main(p.parse_args())