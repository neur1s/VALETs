import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse, random, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from flair.models import SequenceTagger

from model import make_model, ValenceContrastiveLoss
from language_task import LanguageTask, TextDataset, CausalLMCollator
from preprocess import load_task_dataset


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
MAX_SAMPLES = 15000
N_VAL = 1000
BATCH_SIZE = 32
LR_PRETRAIN = 1e-4
PRETRAIN_EPOCHS = 20


def get_diverse_valence_sample(valences, sample_size=128):
    flat = valences.view(-1)
    flat = flat[flat != 0.0]
    if len(flat) == 0:
        return flat
    unique_vals = torch.unique(flat)
    per_val_count = max(1, sample_size // len(unique_vals))
    samples = []
    for uv in unique_vals:
        indices = (flat == uv).nonzero(as_tuple=True)[0]
        perm = torch.randperm(len(indices), device=flat.device)[:per_val_count]
        samples.append(flat[indices[perm]])
    return torch.cat(samples)


def compute_ar_loss(logits, labels):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return nn.CrossEntropyLoss(ignore_index=-100)(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def run_pretrain_epoch(model, ar_loader, raw_loader, optimizer,
                       gc_fn, gc_weight, device, train=True, max_steps=None):
    model.train() if train else model.eval()
    total_loss, count, step = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()

    with ctx:
        for batch_ar, batch_raw in zip(ar_loader, raw_loader):
            if train:
                optimizer.zero_grad()

            logits = model(
                batch_ar["input_ids"].to(device),
                batch_ar["attention_mask"].to(device),
                valences=None,
            )
            ar_loss = compute_ar_loss(logits, batch_ar["labels"].to(device))

            v_raw = batch_raw["valences"].to(device)
            v_sample = get_diverse_valence_sample(v_raw, sample_size=128)
            if len(v_sample) > 1:
                embs = model.val_embedder(v_sample)
                gc_loss = gc_fn(embs, v_sample)
            else:
                gc_loss = torch.tensor(0.0, device=device)

            loss = ar_loss + gc_weight * gc_loss

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            count += 1; step += 1
            if max_steps and step >= max_steps:
                break

    return total_loss / max(count, 1)


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    all_texts   = load_task_dataset(
        split="train", max_samples=args.max_samples + args.n_val, min_len=200)
    train_texts = all_texts[:args.max_samples]
    val_texts   = all_texts[args.max_samples:args.max_samples + args.n_val]
    print(f"Train: {len(train_texts)}  Val: {len(val_texts)}")

    pos_tagger = SequenceTagger.load("flair/pos-english")
    task = LanguageTask(
        model_name="bert-base-uncased", max_len=args.max_len,
        tagger=pos_tagger, upgrade_sentence_position=True,
    )
    tokenizer = task.tokenizer
    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id or 0

    def make_ar_loader(texts, shuffle, suffix):
        ds = TextDataset(texts, tokenizer, args.max_len, pos_tagger,
                         upgrade_sentence_position=True, cache_suffix=suffix)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=CausalLMCollator(pad_id), num_workers=4)

    def make_raw_loader(texts, shuffle, suffix):
        ds = TextDataset(texts, tokenizer, args.max_len, pos_tagger,
                         upgrade_sentence_position=None, cache_suffix=suffix)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          collate_fn=CausalLMCollator(pad_id), num_workers=4)

    print(f"\n[{time.strftime('%H:%M:%S')}] Building AR loaders...")
    ar_train_loader = make_ar_loader(train_texts, shuffle=True, suffix="_train_inj")
    ar_val_loader = make_ar_loader(val_texts, shuffle=False, suffix="_val_inj")

    print(f"[{time.strftime('%H:%M:%S')}] Building raw loaders (GC)...")
    raw_train_loader = make_raw_loader(train_texts, shuffle=True, suffix="_train_raw")
    raw_val_loader = make_raw_loader(val_texts, shuffle=False, suffix="_val_raw")

    gc_fn   = ValenceContrastiveLoss()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    for seed in args.seeds:
        print(f"\n{'='*55}\n  SEED {seed}\n{'='*55}")
        set_seed(seed)

        model = make_model(
            vocab_size, d_model=args.embed_dim, d_ff=args.hidden_dim,
            N=args.num_layers, h=args.num_heads, max_len=args.max_len,
        ).to(device)
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr_pretrain, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=3, factor=0.5)

        best_val, no_improve = float("inf"), 0

        for epoch in range(1, args.pretrain_epochs + 1):
            tr = run_pretrain_epoch(
                model, ar_train_loader, raw_train_loader,
                optimizer, gc_fn, args.gc_weight,
                device, train=True, max_steps=args.max_steps)
            vl = run_pretrain_epoch(
                model, ar_val_loader, raw_val_loader,
                optimizer, gc_fn, args.gc_weight,
                device, train=False, max_steps=args.max_steps)
            scheduler.step(vl)
            print(f"  ep {epoch:3d} | train={tr:.4f}  val={vl:.4f}")

            if vl < best_val:
                best_val, no_improve = vl, 0
                torch.save(model.state_dict(),
                           os.path.join(out_dir,
                                        f"checkpoint_seed{seed}_phase1.pt"))
                print(f"    saved (best val={best_val:.4f})")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"  Early stop (best val={best_val:.4f})")
                    break

        print(f"\nSeed {seed} done.")

    print("\nAll seeds complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--pretrain_epochs", type=int, default=PRETRAIN_EPOCHS)
    p.add_argument("--embed_dim", type=int, default=D_MODEL)
    p.add_argument("--hidden_dim", type=int, default=D_FF)
    p.add_argument("--num_heads", type=int, default=N_HEADS)
    p.add_argument("--num_layers", type=int, default=N_LAYERS)
    p.add_argument("--max_len", type=int, default=MAX_LEN)
    p.add_argument("--max_samples", type=int, default=MAX_SAMPLES)
    p.add_argument("--n_val", type=int, default=N_VAL)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr_pretrain", type=float, default=LR_PRETRAIN)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--gc_weight", type=float, default=0.1)
    p.add_argument("--max_steps", type=int, default=None)
    args = p.parse_args()
    train(args)