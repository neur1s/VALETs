import sys, os
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))
from model import make_model, subsequent_mask, WCELoss
from ctx_episodic_memory_task import CtxtMemoryTask, CtxtMemoryDataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# token ID conventions (must match model.py)
def get_special_tokens(num_observations, num_contexts):
    SOS_TOKEN = num_observations
    CTX_OFFSET = num_observations + 1  # ctx token i → CTX_OFFSET + i
    return SOS_TOKEN, CTX_OFFSET


def make_src(ctx_ids, obs_ids, CTX_OFFSET, device):
    ctx_tokens = (ctx_ids + CTX_OFFSET).unsqueeze(1).to(device)  # [B, 1]
    return torch.cat([ctx_tokens, obs_ids.to(device)], dim=1)    # [B, seq+1]


def make_tgt(obs_ids, SOS_TOKEN, device):
    B = obs_ids.size(0)
    sos = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
    return torch.cat([sos, obs_ids[:, :-1].to(device)], dim=1)   # [B, seq]


def eval_memory_types(pred, obs_ids, obs_to_valence=None, mem_len=None):
    batch_size, seq_len = obs_ids.shape
    wm_correct = wm_total = rm_correct = rm_total = 0
    valence_stats = {} if obs_to_valence is not None else None

    for pos in range(seq_len):
        target_col = obs_ids[:, pos]
        pred_col = pred[:, pos]

        if pos == 0:
            seen_before = torch.zeros(batch_size, dtype=torch.bool, device=obs_ids.device)
        else:
            window_start = max(0, pos - mem_len) if mem_len is not None else 0
            prior = obs_ids[:, window_start:pos]
            seen_before = (prior == target_col.unsqueeze(1)).any(dim=1)

        correct = (pred_col == target_col)
        wm_mask = seen_before
        rm_mask = ~seen_before

        wm_correct += (correct & wm_mask).sum().item()
        wm_total   += wm_mask.sum().item()
        rm_correct += (correct & rm_mask).sum().item()
        rm_total   += rm_mask.sum().item()

        if obs_to_valence is not None:
            for b in range(batch_size):
                obs_id  = target_col[b].item()
                valence = obs_to_valence[obs_id]
                if valence not in valence_stats:
                    valence_stats[valence] = {'wm_correct': 0, 'wm_total': 0,
                                              'rm_correct': 0, 'rm_total': 0}
                is_correct = correct[b].item()
                if wm_mask[b].item():
                    valence_stats[valence]['wm_total'] += 1
                    if is_correct:
                        valence_stats[valence]['wm_correct'] += 1
                else:
                    valence_stats[valence]['rm_total'] += 1
                    if is_correct:
                        valence_stats[valence]['rm_correct'] += 1

    result = {"wm_correct": wm_correct, "wm_total": wm_total,
              "rm_correct": rm_correct, "rm_total": rm_total}
    if valence_stats is not None:
        result["valence_stats"] = valence_stats
    return result


def run_batch(model, ctx_ids, obs_ids, valences, SOS_TOKEN, CTX_OFFSET, device, mem_len=64):
    B, seq_len = obs_ids.shape

    # encoder always sees the full episode
    src = make_src(ctx_ids, obs_ids, CTX_OFFSET, device)   # [B, seq+1]
    memory = model.encode(src, src_mask=None)               # [B, seq+1, d_model]

    all_outputs = []
    all_valences = []

    for chunk_start in range(0, seq_len, mem_len):
        chunk_end = min(chunk_start + mem_len, seq_len)
        chunk_obs = obs_ids[:, chunk_start:chunk_end]       # [B, chunk_len]
        chunk_val = valences[:, chunk_start:chunk_end]      # [B, chunk_len]

        # decoder input: SOS for first chunk, else last token of previous chunk
        if chunk_start == 0:
            sos = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
            tgt = torch.cat([sos, chunk_obs[:, :-1]], dim=1)
        else:
            prev_last = obs_ids[:, chunk_start - 1].unsqueeze(1)  # [B, 1]
            tgt = torch.cat([prev_last, chunk_obs[:, :-1]], dim=1)

        tgt_mask = subsequent_mask(tgt.size(1)).to(device)
        tgt_emb = model.tgt_embed(tgt)
        chunk_out = model.decoder(tgt_emb, memory, src_mask=None, tgt_mask=tgt_mask)
        chunk_logits = model.generator(chunk_out)           # [B, chunk_len, num_obs]
        all_outputs.append(chunk_logits)
        all_valences.append(chunk_val)

    all_outputs = torch.cat(all_outputs, dim=1)             # [B, seq, num_obs]
    all_valences = torch.cat(all_valences, dim=1)           # [B, seq]
    return all_outputs, obs_ids, all_valences


def train_loop(model, train_loader, val_loader, optimizer, criterion, num_epochs,
               device=None, num_observations=800, num_contexts=32, obs_to_valence=None,
               mem_len=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    SOS_TOKEN, CTX_OFFSET = get_special_tokens(num_observations, num_contexts)

    history = {"train_wm_error": [], "train_rm_error": [],
               "val_wm_error": [], "val_rm_error": [],
               "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        wm_correct = wm_total = rm_correct = rm_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (ctx_ids, obs_ids) in enumerate(pbar):
            obs_ids = obs_ids.to(device)
            
            # get valences for each observation
            valences = torch.tensor(
                [[obs_to_valence[obs_id.item()] for obs_id in row] for row in obs_ids],
                dtype=torch.float32, device=device
            )

            optimizer.zero_grad()
            output, dec_obs, dec_valences = run_batch(
                model, ctx_ids, obs_ids, valences, SOS_TOKEN, CTX_OFFSET, device, mem_len=mem_len or 64
            )

            # use WCELoss with valences
            loss = criterion(output, dec_obs, dec_valences)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            total_loss += loss.item()

            pred = output.argmax(dim=-1)
            mem = eval_memory_types(pred, dec_obs, mem_len=mem_len)
            wm_correct += mem["wm_correct"]
            wm_total   += mem["wm_total"]
            rm_correct += mem["rm_correct"]
            rm_total   += mem["rm_total"]

            pbar.set_postfix({'loss': f'{loss.item():.4f}',
                              'avg_loss': f'{total_loss/(batch_idx+1):.4f}'})

        train_loss = total_loss / max(len(train_loader), 1)
        train_wm_err  = 1 - (wm_correct / wm_total) if wm_total > 0 else 0.0
        train_rm_err  = 1 - (rm_correct / rm_total) if rm_total > 0 else 0.0
        train_acc = (wm_correct + rm_correct) / max(wm_total + rm_total, 1)

        # validation 
        val_loss, val_wm_err, val_rm_err, val_acc, _ = evaluate_memory(
            model, val_loader, criterion, SOS_TOKEN, CTX_OFFSET, device, obs_to_valence, mem_len=mem_len)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_wm_error"].append(train_wm_err)
        history["train_rm_error"].append(train_rm_err)
        history["val_wm_error"].append(val_wm_err)
        history["val_rm_error"].append(val_rm_err)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch+1}/{num_epochs} — "
              f"Train Loss: {train_loss:.4f} | "
              f"Train WM Err: {100*train_wm_err:.1f}% | Train RM Err: {100*train_rm_err:.1f}% | "
              f"Val WM Err: {100*val_wm_err:.1f}% | Val RM Err: {100*val_rm_err:.1f}%")

    plot_memory_errors(history)
    plot_epoch_accuracy(history)

    if obs_to_valence is not None:
        print("\nEvaluating valence-based accuracy on training data...")
        plot_valence_accuracy(model, train_loader, criterion, SOS_TOKEN, CTX_OFFSET,
                              device, obs_to_valence, mem_len=mem_len, 
                              filename_prefix='train')
        print("\nEvaluating valence-based accuracy on validation data...")
        plot_valence_accuracy(model, val_loader, criterion, SOS_TOKEN, CTX_OFFSET,
                              device, obs_to_valence, mem_len=mem_len,
                              filename_prefix='val')

    return history


def evaluate_memory(model, dataloader, criterion, SOS_TOKEN, CTX_OFFSET, device,
                    obs_to_valence=None, mem_len=None):
    model.eval()
    total_loss = 0.0
    wm_correct = wm_total = rm_correct = rm_total = 0
    all_valence_stats = {}

    with torch.no_grad():
        for ctx_ids, obs_ids in dataloader:
            obs_ids = obs_ids.to(device)
            
            # get valences for each observation
            valences = torch.tensor(
                [[obs_to_valence[obs_id.item()] for obs_id in row] for row in obs_ids],
                dtype=torch.float32, device=device
            )

            output, dec_obs, dec_valences = run_batch(
                model, ctx_ids, obs_ids, valences, SOS_TOKEN, CTX_OFFSET, device, mem_len=mem_len or 64
            )
            loss = criterion(output, dec_obs, dec_valences)
            total_loss += loss.item()

            pred = output.argmax(dim=-1)
            mem  = eval_memory_types(pred, dec_obs, obs_to_valence, mem_len=mem_len)
            wm_correct += mem["wm_correct"]
            wm_total   += mem["wm_total"]
            rm_correct += mem["rm_correct"]
            rm_total   += mem["rm_total"]

            if obs_to_valence is not None and "valence_stats" in mem:
                for valence, stats in mem["valence_stats"].items():
                    if valence not in all_valence_stats:
                        all_valence_stats[valence] = {'wm_correct': 0, 'wm_total': 0,
                                                      'rm_correct': 0, 'rm_total': 0}
                    for k in all_valence_stats[valence]:
                        all_valence_stats[valence][k] += stats[k]

    avg_loss = total_loss / max(len(dataloader), 1)
    wm_err = 1 - (wm_correct / wm_total) if wm_total > 0 else 0.0
    rm_err = 1 - (rm_correct / rm_total) if rm_total > 0 else 0.0
    overall_acc = (wm_correct + rm_correct) / max(wm_total + rm_total, 1)

    if obs_to_valence is not None:
        return avg_loss, wm_err, rm_err, overall_acc, all_valence_stats
    return avg_loss, wm_err, rm_err, overall_acc


def plot_valence_accuracy(model, dataloader, criterion, SOS_TOKEN, CTX_OFFSET,
                          device, obs_to_valence, mem_len=None, filename_prefix='val'):
    _, _, _, _, valence_stats = evaluate_memory(
        model, dataloader, criterion, SOS_TOKEN, CTX_OFFSET, device, obs_to_valence, mem_len=mem_len)

    valences = sorted(valence_stats.keys())
    wm_accuracies = []
    rm_accuracies = []

    for v in valences:
        s = valence_stats[v]
        wm_accuracies.append(100 * s['wm_correct'] / s['wm_total'] if s['wm_total'] > 0 else None)
        rm_accuracies.append(100 * s['rm_correct'] / s['rm_total'] if s['rm_total'] > 0 else None)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    wm_vals = [v for v, a in zip(valences, wm_accuracies) if a is not None]
    wm_accs = [a for a in wm_accuracies if a is not None]
    if wm_vals:
        axes[0].plot(wm_vals, wm_accs, marker='o', linewidth=2, markersize=8, color='tab:blue')
        axes[0].set_xlabel('Valence', fontsize=12)
        axes[0].set_ylabel('Accuracy (%)', fontsize=12)
        axes[0].set_title('Working Memory: Valence vs Accuracy', fontsize=14, fontweight='bold')
        axes[0].set_ylim(-5, 105)
        axes[0].grid(True, alpha=0.3)

    rm_vals = [v for v, a in zip(valences, rm_accuracies) if a is not None]
    rm_accs = [a for a in rm_accuracies if a is not None]
    if rm_vals:
        axes[1].plot(rm_vals, rm_accs, marker='s', linewidth=2, markersize=8, color='tab:red')
        axes[1].set_xlabel('Valence', fontsize=12)
        axes[1].set_ylabel('Accuracy (%)', fontsize=12)
        axes[1].set_title('Reference Memory: Valence vs Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_ylim(-5, 105)
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_valence_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Valence accuracy plot saved to {filename_prefix}_valence_accuracy.png")

    print("\n=== Valence-Based Accuracy Summary ===")
    print(f"{'Valence':<10} {'WM Acc (%)':<15} {'WM Count':<12} {'RM Acc (%)':<15} {'RM Count':<12}")
    print("-" * 70)
    for v in valences:
        s = valence_stats[v]
        wm_acc = 100 * s['wm_correct'] / s['wm_total'] if s['wm_total'] > 0 else 0.0
        rm_acc = 100 * s['rm_correct'] / s['rm_total'] if s['rm_total'] > 0 else 0.0
        print(f"{v:<10.2f} {wm_acc:<15.1f} {s['wm_total']:<12} {rm_acc:<15.1f} {s['rm_total']:<12}")


def plot_memory_errors(history):
    epochs = range(1, len(history["train_wm_error"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, [100 * (1 - e) for e in history["train_wm_error"]],
                 marker='o', label='Train WM Acc', color='tab:blue')
    axes[0].plot(epochs, [100 * (1 - e) for e in history["val_wm_error"]],
                 marker='s', label='Val WM Acc', color='tab:cyan')
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Working Memory Accuracy")
    axes[0].set_ylim(-2, 102); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(epochs, [100 * (1 - e) for e in history["train_rm_error"]],
                 marker='o', label='Train RM Acc', color='tab:red')
    axes[1].plot(epochs, [100 * (1 - e) for e in history["val_rm_error"]],
                 marker='s', label='Val RM Acc', color='tab:orange')
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_title("Reference Memory Accuracy")
    axes[1].set_ylim(-2, 102); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("memory_accuracy.png", dpi=150)
    plt.close()
    print("Memory accuracy plots saved to memory_accuracy.png")


def plot_epoch_accuracy(history):
    epochs = range(1, len(history.get("train_acc", [])) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [100 * a for a in history["train_acc"]],
             marker='o', color='tab:green', label='Train Accuracy')
    plt.plot(epochs, [100 * a for a in history["val_acc"]],
             marker='s', color='tab:purple', label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Epoch vs Accuracy')
    plt.ylim(-2, 102); plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig('training_accuracy.png', dpi=150)
    plt.close()
    print('Epoch vs accuracy plot saved to training_accuracy.png')


if __name__ == "__main__":
    # hyperparameters
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

    # task parameters
    num_contexts = 32
    num_observations = 1024
    seq_len = 256  # full episode length; encoder + decoder see all 256 tokens
    mem_len = 64   # WM window: matches decoder positional encoding limit

    # beta sweep values
    beta_values = [0.25, 0.5, 1.0]

    SOS_TOKEN, CTX_OFFSET = get_special_tokens(num_observations, num_contexts)
    max_len = 64  # decoder positional encoding window = short-term memory limit

    print("Creating datasets with shared context pools...")
    train_dataset = CtxtMemoryDataset(
        num_contexts=num_contexts,
        num_observations=num_observations,
        num_samples=num_train_samples,
        seq_len=seq_len,
        seed=42,
        pool_seed=42,
    )
    val_dataset = CtxtMemoryDataset(
        num_contexts=num_contexts,
        num_observations=num_observations,
        num_samples=num_val_samples,
        seq_len=seq_len,
        seed=99999,
        pool_seed=42,   # same pools as train
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    obs_to_valence = train_dataset.core.obs_to_valence

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Episode length:     {seq_len}  (encoder input: {max_len} tokens)")
    print(f"Vocab size:         {num_observations + 1 + num_contexts}")
    print(f"\n{'='*60}")
    print(f"Starting Beta Sweep: {beta_values}")
    print(f"{'='*60}\n")

    # track best model across sweep
    best_train_acc = 0.0
    best_model_state = None
    best_config = None

    # beta sweep loop
    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Training with Beta = {beta}")
        print(f"{'='*60}\n")

        # create fresh model for each beta
        model = make_model(
            num_contexts=num_contexts,
            num_observations=num_observations,
            N=num_encoder_layers,
            d_model=d_model,
            d_ff=dim_feedforward,
            h=nhead,
            max_len=seq_len + 1,      # encoder: full episode + ctx token
            decoder_max_len=mem_len,  # decoder: short-term memory window
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = WCELoss(valence_scaling=beta)

        history = train_loop(
            model, train_loader, val_loader, optimizer, criterion, num_epochs,
            num_observations=num_observations,
            num_contexts=num_contexts,
            obs_to_valence=obs_to_valence,
            mem_len=mem_len,
        )

        final_train_acc = history["train_acc"][-1]
        final_val_acc = history["val_acc"][-1]
        print(f"\nBeta {beta} - Final Train Accuracy: {100*final_train_acc:.2f}%, Val Accuracy: {100*final_val_acc:.2f}%")
        
        if final_train_acc > best_train_acc:
            best_train_acc = final_train_acc
            best_beta = beta
            best_model_state = model.state_dict()
            best_config = {
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
                "pool_seed": 42,
                "train_seed": 42,
                "val_seed": 99999,
                "mem_len": mem_len,
                "beta": beta,
                "final_train_acc": final_train_acc,
                "final_val_acc": final_val_acc,
            }

        # rename plots for this beta
        import os
        if os.path.exists("memory_accuracy.png"):
            os.rename("memory_accuracy.png", f"memory_accuracy_beta_{beta}.png")
        if os.path.exists("training_accuracy.png"):
            os.rename("training_accuracy.png", f"training_accuracy_beta_{beta}.png")
        if os.path.exists("train_valence_accuracy.png"):
            os.rename("train_valence_accuracy.png", f"train_valence_accuracy_beta_{beta}.png")
        if os.path.exists("val_valence_accuracy.png"):
            os.rename("val_valence_accuracy.png", f"val_valence_accuracy_beta_{beta}.png")

        torch.save({
            "model_state": best_model_state,
            "config": best_config,
        }, f"checkpoint_{beta}.pt")
        print(f"Checkpoint saved for Beta {beta} with Train Acc: {100*final_train_acc:.2f}% and Val Acc: {100*final_val_acc:.2f}%\n")
