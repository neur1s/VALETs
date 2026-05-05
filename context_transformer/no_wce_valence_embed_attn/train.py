import sys, os
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))
from model import make_model, subsequent_mask, ValenceEmbedder, ValenceContrastiveLoss
from ctx_episodic_memory_task import CtxtMemoryTask, CtxtMemoryDataset
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

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


def run_batch(model, valence_embedder, ctx_ids, obs_ids, valences, SOS_TOKEN, CTX_OFFSET, device, mem_len=64):
    B, seq_len = obs_ids.shape

    # generate valence embeddings
    obs_v_embeddings = valence_embedder(valences) 

    # prepare encoder valence
    ctx_v = torch.zeros(B, 1, 64, device=device)
    src_v = torch.cat([ctx_v, obs_v_embeddings], dim=1) 

    src = make_src(ctx_ids, obs_ids, CTX_OFFSET, device)
    
    # change src_v to src_val
    memory = model.encode(src, src_mask=None) 

    all_outputs = []
    all_valences = []

    for chunk_start in range(0, seq_len, mem_len):
        chunk_end = min(chunk_start + mem_len, seq_len)
        chunk_obs = obs_ids[:, chunk_start:chunk_end]
        chunk_val = valences[:, chunk_start:chunk_end]
        
        chunk_v_embeddings = obs_v_embeddings[:, chunk_start:chunk_end]
        
        if chunk_start == 0:
            sos_v = torch.zeros(B, 1, 64, device=device)
            tgt_v = torch.cat([sos_v, chunk_v_embeddings[:, :-1]], dim=1)
            sos = torch.full((B, 1), SOS_TOKEN, dtype=torch.long, device=device)
            tgt = torch.cat([sos, chunk_obs[:, :-1]], dim=1)
        else:
            prev_v = obs_v_embeddings[:, chunk_start - 1].unsqueeze(1)
            tgt_v = torch.cat([prev_v, chunk_v_embeddings[:, :-1]], dim=1)
            prev_last = obs_ids[:, chunk_start - 1].unsqueeze(1)
            tgt = torch.cat([prev_last, chunk_obs[:, :-1]], dim=1)

        tgt_mask = subsequent_mask(tgt.size(1)).to(device)
        
        # decoder pass
        chunk_out = model.decode(
            memory=memory, 
            src_mask=None, 
            tgt=tgt, 
            tgt_mask=tgt_mask, 
            src_val=src_v,  
            tgt_val=tgt_v   
        )
        
        chunk_logits = model.generator(chunk_out)
        all_outputs.append(chunk_logits)
        all_valences.append(chunk_val)

    all_outputs = torch.cat(all_outputs, dim=1)
    all_valences = torch.cat(all_valences, dim=1)
    
    return all_outputs, obs_ids, all_valences

def train_loop(model, valence_embedder, train_loader, val_loader, optimizer_model, optimizer_valence, criterion, valence_loss, num_epochs,
               device=None, num_observations=800, num_contexts=32, obs_to_valence=None,
               mem_len=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    valence_embedder.to(device)

    SOS_TOKEN, CTX_OFFSET = get_special_tokens(num_observations, num_contexts)

    history = {"train_wm_error": [], "train_rm_error": [],
               "val_wm_error": [], "val_rm_error": [],
               "train_loss": [], "val_loss": [],
               "train_acc": [], "val_acc": [],
               "train_valence_loss": []
               }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_valence_loss = 0.0
        wm_correct = wm_total = rm_correct = rm_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (ctx_ids, obs_ids) in enumerate(pbar):
            obs_ids = obs_ids.to(device)
            
            # fet valences for each observation
            valences = torch.tensor(
                [[obs_to_valence[obs_id.item()] for obs_id in row] for row in obs_ids],
                dtype=torch.float32, device=device
            )

            optimizer_model.zero_grad()
            optimizer_valence.zero_grad()

            # main model pass
            output, dec_obs, dec_valences = run_batch(
                model, valence_embedder, ctx_ids, obs_ids, valences, SOS_TOKEN, CTX_OFFSET, device, mem_len=mem_len or 64
            )
            loss = criterion(output, dec_obs, dec_valences)
            loss.backward() # Gradients for 'model'

            # valence Pass (using detached valences to prevent leakage)
            detached_valences = dec_valences.detach()
            dec_val_embeddings = valence_embedder(detached_valences)
            v_loss, val_loss_dict = valence_loss(dec_val_embeddings.view(-1, dec_val_embeddings.size(-1)), detached_valences.reshape(-1))
            v_loss.backward() # gradients for 'valence_embedder'

            # now step both together
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer_model.step()
            optimizer_valence.step()
            
            total_loss += loss.item()
            total_valence_loss += val_loss_dict['contrastive']

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
            model, valence_embedder, val_loader, criterion, SOS_TOKEN, CTX_OFFSET, device, obs_to_valence, mem_len=mem_len)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_valence_loss"].append(total_valence_loss / max(len(train_loader), 1))
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
        plot_valence_accuracy(model, valence_embedder, train_loader, criterion, SOS_TOKEN, CTX_OFFSET,
                              device, obs_to_valence, mem_len=mem_len, 
                              filename_prefix='train')
        print("\nEvaluating valence-based accuracy on validation data...")
        plot_valence_accuracy(model, valence_embedder, val_loader, criterion, SOS_TOKEN, CTX_OFFSET,
                              device, obs_to_valence, mem_len=mem_len,
                              filename_prefix='val')

    return history


def evaluate_memory(model, valence_embedder, dataloader, criterion, SOS_TOKEN, CTX_OFFSET, device,
                    obs_to_valence=None, mem_len=None):
    model.eval()
    total_loss = 0.0
    wm_correct = wm_total = rm_correct = rm_total = 0
    all_valence_stats = {}

    with torch.no_grad():
        for ctx_ids, obs_ids in dataloader:
            obs_ids = obs_ids.to(device)
            
            # Get valences for each observation
            valences = torch.tensor(
                [[obs_to_valence[obs_id.item()] for obs_id in row] for row in obs_ids],
                dtype=torch.float32, device=device
            )

            output, dec_obs, dec_valences = run_batch(
                model, valence_embedder, ctx_ids, obs_ids, valences, SOS_TOKEN, CTX_OFFSET, device, mem_len=mem_len or 64
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


def plot_valence_accuracy(model, valence_embedder, dataloader, criterion, SOS_TOKEN, CTX_OFFSET,
                          device, obs_to_valence, mem_len=None, filename_prefix='val'):
    _, _, _, _, valence_stats = evaluate_memory(
        model, valence_embedder, dataloader, criterion, SOS_TOKEN, CTX_OFFSET, device, obs_to_valence, mem_len=mem_len)

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

def plot_valence_similarity_heatmap(valence_embedder, obs_to_valence, device, filename='valence_heatmap.png'):
    valence_embedder.eval()

    unique_valences = np.unique(obs_to_valence).tolist()
    unique_valences.sort()
    val_tensor = torch.tensor(unique_valences, dtype=torch.float32, device=device).unsqueeze(1)
    
    with torch.no_grad():
        # val_tensor is [10, 1]
        embeddings = valence_embedder(val_tensor) 
        
        # flatten to 2D [num_valences, embedding_dim]
        embeddings = embeddings.view(len(unique_valences), -1)
            
        # normalize along the embedding dimension (dim=1)
        norm_emb = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # now norm_emb is [10, 64], so norm_emb.t() is [64, 10]
        sim_matrix = torch.mm(norm_emb, norm_emb.t()).cpu().numpy()

    plt.figure(figsize=(10, 8))
    im = plt.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
    plt.colorbar(im, label='Cosine Similarity')
    
    # label the axes with the actual valence values
    ticks = np.arange(len(unique_valences))
    plt.xticks(ticks, [f"{v:.2f}" for v in unique_valences], rotation=45)
    plt.yticks(ticks, [f"{v:.2f}" for v in unique_valences])
    
    plt.title("Valence Embedder", fontsize=16)
    plt.xlabel("Valence", fontsize=16)
    plt.ylabel("Valence", fontsize=16)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def get_empirical_attn_maps(model, dataloader, valence_embedder, obs_to_valence, device, num_samples=5):
    model.eval()
    valence_embedder.eval()
    
    # standardize valence set using only real observations
    unique_valences = sorted(list(set(round(float(v), 2) for v in obs_to_valence.tolist())))
    val_to_idx = {v: i for i, v in enumerate(unique_valences)}
    num_v = len(unique_valences)

    maps = {k: torch.zeros(num_v, num_v, device=device) for k in ["Encoder Self-Attn", "Decoder Self-Attn", "Cross-Attn"]}
    counts = {k: torch.zeros(num_v, num_v, device=device) for k in maps.keys()}

    with torch.no_grad():
        for i, (ctx_ids, obs_ids) in enumerate(dataloader):
            if i >= num_samples: break
            obs_ids = obs_ids.to(device)
            ctx_ids = ctx_ids.to(device)
            
            # full sequences for the model (including special tokens at index 0)
            src_vals_full = torch.tensor([[0.0] + [obs_to_valence[o.item()] for o in row] for row in obs_ids], device=device)
            tgt_vals_full = torch.tensor([[0.0] + [obs_to_valence[o.item()] for o in row[:-1]] for row in obs_ids], device=device)

            _, enc_w, self_w, cross_w = model.forward_with_weights(ctx_ids, obs_ids, src_vals_full, tgt_vals_full, valence_embedder)
            
            # sliced sequences for the heatmap (Observations Only)
            # we slice [:, 1:] to remove the 0.0 valence assigned to special tokens
            src_vals_obs = src_vals_full[:, 1:] 
            tgt_vals_obs = tgt_vals_full[:, 1:]

            def aggregate_obs(attn_weights, q_vals, k_vals, map_key, q_slice=None, k_slice=None):
                # slicing the attention weights to ignore SOS/Context tokens
                # [B, H, Q, K] -> [B, Q_obs, K_obs] (averaged over heads)
                qs = q_slice if q_slice is not None else slice(0, None)
                ks = k_slice if k_slice is not None else slice(0, None)
                avg_attn = attn_weights[:, :, qs, ks].mean(dim=1)
                
                q_indices = torch.tensor([[val_to_idx[round(v.item(), 2)] for v in row] for row in q_vals], device=device)
                k_indices = torch.tensor([[val_to_idx[round(v.item(), 2)] for v in row] for row in k_vals], device=device)

                for b in range(avg_attn.size(0)):
                    q_len_curr = avg_attn.size(1)
                    k_len_curr = avg_attn.size(2)

                    # match indices to the current attention window
                    idx_q = q_indices[b, -q_len_curr:].view(-1, 1).expand(-1, k_len_curr)
                    idx_k = k_indices[b, -k_len_curr:].view(1, -1).expand(q_len_curr, -1)
                    
                    maps[map_key].index_put_((idx_q.reshape(-1), idx_k.reshape(-1)), avg_attn[b].reshape(-1), accumulate=True)
                    counts[map_key].index_put_((idx_q.reshape(-1), idx_k.reshape(-1)), torch.ones_like(avg_attn[b].reshape(-1)), accumulate=True)

            # encoder: skip context token (idx 0) in both Q and K
            aggregate_obs(enc_w, src_vals_obs, src_vals_obs, "Encoder Self-Attn", q_slice=slice(1, None), k_slice=slice(1, None))
            
            # decoder: Skip SOS token (idx 0) in both Q and K
            aggregate_obs(self_w, tgt_vals_obs, tgt_vals_obs, "Decoder Self-Attn", q_slice=slice(1, None), k_slice=slice(1, None))
            
            # cross: Skip SOS token (idx 0) in Q and context token (idx 0) in K
            aggregate_obs(cross_w, tgt_vals_obs, src_vals_obs, "Cross-Attn", q_slice=slice(1, None), k_slice=slice(1, None))

    final_maps = {k: (maps[k] / counts[k].clamp(min=1)).cpu().numpy() for k in maps.keys()}
    return final_maps, unique_valences

def plot_attention_valence_heatmaps(maps, labels, beta):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (name, matrix) in enumerate(maps.items()):
        im = axes[i].imshow(matrix, cmap='viridis')
        axes[i].set_title(f"{name} (β={beta})", fontsize=15)
        
        ticks = np.arange(len(labels))
        axes[i].set_xticks(ticks)
        axes[i].set_xticklabels([f"{l:.1f}" for l in labels])
        axes[i].set_yticks(ticks)
        axes[i].set_yticklabels([f"{l:.1f}" for l in labels])
        
        axes[i].set_xlabel("Key Valence")
        axes[i].set_ylabel("Query Valence")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"attn_heatmap_beta_{beta}.png")
    plt.show()

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

        rng_state = torch.get_rng_state() # capture RNG state for comparability with the baseline
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None

        valence_embedder = ValenceEmbedder()
        valence_loss = ValenceContrastiveLoss()

        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)

        optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)
        optimizer_valence = torch.optim.Adam(valence_embedder.parameters(), lr=learning_rate)

        # use regular CE loss
        _ce_fn = nn.CrossEntropyLoss()
        criterion = lambda logits, targets, valences=None: _ce_fn(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        history = train_loop(
            model, valence_embedder, train_loader, val_loader, optimizer_model, optimizer_valence, criterion, valence_loss, num_epochs,
            num_observations=num_observations,
            num_contexts=num_contexts,
            obs_to_valence=obs_to_valence,
            mem_len=mem_len,
        )

        print(f"Generating final valence heatmap for Beta {beta}...")
        plot_valence_similarity_heatmap(
            valence_embedder, 
            obs_to_valence,  
            device=next(model.parameters()).device, 
            filename=f"valence_heatmap_beta_{beta}.png"
        )

        print(f"Generating attention valence heatmaps for Beta {beta}...")
        device = next(model.parameters()).device
        
        # 1. Generate the data
        attn_data, valence_labels = get_empirical_attn_maps(
            model, 
            val_loader, 
            valence_embedder, 
            obs_to_valence, 
            device
        )

        # plot the data
        plot_attention_valence_heatmaps(
            maps=attn_data, 
            labels=valence_labels, 
            beta=beta
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
            "model_state": {k: v.half() for k, v in best_model_state.items()},
            "config": best_config,
        }, f"checkpoint_{beta}.pt")
        print(f"Checkpoint saved for Beta {beta} with Train Acc: {100*final_train_acc:.2f}% and Val Acc: {100*final_val_acc:.2f}%\n")
