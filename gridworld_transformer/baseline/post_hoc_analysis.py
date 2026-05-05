import os
import sys
import os.path as osp
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../include"))
from misc import create_data
from model import xlTEM
from random_walk import RandomWalker

def calculate_gini(x):
    """
    Computes the Gini coefficient of a distribution.
    x: array of attention weights (must be non-negative)
    """
    if np.sum(x) == 0: return 0.0
    # sort the weights
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x))) # gini formula for non-negative values

def calculate_spatial_information(rho, visit_counts):
    """
    Computes Skaggs Spatial Information (bits per spike).
    rho: Rate map (L x L)
    visit_counts: Occupancy map (L x L)
    """
    rho = rho.flatten()
    occ = visit_counts.flatten()
    
    total_visits = np.sum(occ)
    if total_visits == 0: return 0.0
    
    p_i = occ / total_visits # occupancy probability per bin
    
    mean_lambda = np.sum(p_i * rho) # mean activity across the entire environment
    
    if mean_lambda <= 1e-9: return 0.0
    
    nonzero = rho > 1e-9
    ratio = rho[nonzero] / mean_lambda
    info = np.sum(p_i[nonzero] * ratio * np.log2(ratio)) # Skaggs spatial info: I = sum( P_i * (Li/L) * log2(Li/L) )
    
    return float(max(0, info))

def extract_si_per_valence(attn_accum, visit_counts, valences):
    """
    Aggregates SI scores for every attention unit per valence.
    """
    si_dist = {v: [] for v in valences}
    for v in valences:
        if v not in attn_accum or np.sum(visit_counts[v]) == 0:
            continue
        
        layer_maps = attn_accum[v] / (visit_counts[v] + 1e-9) # mean rate maps: [units, L, L] normalized by occupancy
        
        for unit_idx in range(layer_maps.shape[0]):
            score = calculate_spatial_information(layer_maps[unit_idx], visit_counts[v])
            si_dist[v].append(score)
    return si_dist

def recover_train_rm_stats(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    L = config.side_len
 
    model = xlTEM(config).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    rw = RandomWalker(L, config.num_envs, config.seed, config.vocab_size, config.n_a)
    rw.x = ckpt['maps'] 
    
    seed_idx = config.training_epoch + 42 
    obs, vals, act, pos = create_data(rw, config, seed_idx, validation=False)
    vals_tensor = torch.tensor(vals, dtype=torch.float32, device=device)

    raw_ce_criterion = nn.CrossEntropyLoss(reduction='none')
    all_probs, all_losses, all_valences, all_rm_masks = [], [], [], []

    unique_valence_levels = sorted(np.unique(vals).tolist())
    rounded_valences = [round(float(v), 2) for v in unique_valence_levels]
    val_to_idx = {v: i for i, v in enumerate(rounded_valences)}
    num_v = len(unique_valence_levels)
    
    max_k = config.mem_len + config.tgt_len + 1
    num_attn_units = config.n_head * max_k
    attn_rate_accum = {v: np.zeros((num_attn_units, L, L)) for v in rounded_valences}
    pos_visit_counts = {v: np.zeros((L, L)) for v in rounded_valences}
    
    attn_accumulator = np.zeros((num_v, num_v))
    attn_counts = np.zeros((num_v, num_v))

    gini_per_valence = {v: [] for v in rounded_valences} # to store gini scores of the attention vector at each step per valence

    with torch.no_grad():
        init_pos = pos[:, 0].to(device)
        prev_hidden = model.rnn.init_hidden(init_pos)
        mems = None
        
        chunks = [[i, min(i + config.tgt_len, config.walk_length)] 
                  for i in range(0, config.walk_length, config.tgt_len)]
        
        for j, [start, stop] in enumerate(chunks):
            src_x = obs[:, start:stop].to(device)
            trg_x = obs[:, stop].to(device)
            a = act[:, start:stop].to(device)
            
            outputs = model(a, src_x, prev_hidden, mems)
            last_layer_attn = outputs.attentions[-1]

            if stop >= config.mem_len + config.tgt_len:
                for b in range(config.num_envs):
                    v_label = round(float(vals_tensor[b, stop]), 2)
                    _x, _y = int(pos[b, stop] % L), int(pos[b, stop] // L)
                    
                    pos_visit_counts[v_label][_x, _y] += 1
                    softmax = last_layer_attn[b, :, -1, :].cpu().numpy()
                    
                    full_softmax = np.zeros((config.n_head, max_k)) # align softmax to full memory width
                    k_actual = softmax.shape[1]
                    full_softmax[:, -k_actual:] = softmax

                    attn_rate_accum[v_label][:, _x, _y] += full_softmax.flatten()

                    # calculate Gini for current attention (mean across heads)
                    mean_attn_weights = softmax.mean(axis=0) # take the mean across heads to see the overall sparsity of the layer
                    gini_per_valence[v_label].append(calculate_gini(mean_attn_weights))

            layer_attns = last_layer_attn.mean(dim=1) 
            qlen, klen = layer_attns.size(1), layer_attns.size(2)
            full_q_vals = vals_tensor[:, start:stop]
            if full_q_vals.size(1) > qlen: full_q_vals = full_q_vals[:, -qlen:]
            elif full_q_vals.size(1) < qlen:
                pad = torch.zeros((full_q_vals.size(0), qlen - full_q_vals.size(1)), device=device)
                full_q_vals = torch.cat([pad, full_q_vals], dim=1)
            mem_start = max(0, start - config.mem_len)
            full_k_vals = vals_tensor[:, mem_start:stop]
            if full_k_vals.size(1) > klen: full_k_vals = full_k_vals[:, -klen:]
            elif full_k_vals.size(1) < klen:
                pad = torch.zeros((full_k_vals.size(0), klen - full_k_vals.size(1)), device=device)
                full_k_vals = torch.cat([pad, full_k_vals], dim=1)

            for b in range(config.num_envs):
                for q_i in range(qlen):
                    q_val = round(float(full_q_vals[b, q_i]), 2)
                    if q_val not in val_to_idx: continue
                    for k_i in range(klen):
                        k_val = round(float(full_k_vals[b, k_i]), 2)
                        if k_val not in val_to_idx: continue
                        v_row, v_col = val_to_idx[q_val], val_to_idx[k_val]
                        attn_accumulator[v_row, v_col] += layer_attns[b, q_i, k_i].item()
                        attn_counts[v_row, v_col] += 1
            
            mems, prev_hidden = outputs.mems, outputs.rnn_hidden

            # masking for reference memory
            mlen = config.mem_len
            visited_pos = pos[:, max(0, start - mlen) : stop]
            c = visited_pos[..., None, None] == pos[:, stop : stop + 1]
            _unvisited_before = (~c).all(1).squeeze(-1).diagonal() 

            probs = torch.softmax(outputs.logits, dim=-1)
            max_probs, _ = torch.max(probs, dim=-1)
            loss = raw_ce_criterion(outputs.logits, trg_x)

            all_probs.extend(max_probs.cpu().tolist())
            all_losses.extend(loss.cpu().tolist())  
            all_valences.extend(vals[:, stop].tolist())
            all_rm_masks.extend(_unvisited_before.cpu().tolist())

    all_probs, all_losses, all_valences = np.array(all_probs), np.array(all_losses), np.array(all_valences)
    all_rm_masks = np.array(all_rm_masks).astype(bool)
    rm_probs, rm_losses, rm_valences = all_probs[all_rm_masks], all_losses[all_rm_masks], all_valences[all_rm_masks]

    certainty_results = {v: np.mean(rm_probs[rm_valences == v]) for v in np.unique(rm_valences)}
    loss_results = {v: np.mean(rm_losses[rm_valences == v]) for v in np.unique(rm_valences)}
    final_attn_map = np.divide(attn_accumulator, attn_counts, out=np.zeros_like(attn_accumulator), where=attn_counts != 0)

    gini_summary = {v: np.mean(scores) if scores else 0.0 for v, scores in gini_per_valence.items()}
    si_scores = extract_si_per_valence(attn_rate_accum, pos_visit_counts, rounded_valences)

    return certainty_results, loss_results, (final_attn_map, unique_valence_levels), si_scores, gini_summary


def save_heatmap(attn_map, labels, folder):
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(7, 6))
    
    hex_list = ['#18364f', '#4d858f', '#9cd07e', '#fffed6']
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_valence", hex_list)

    vmin, vmax = np.min(attn_map), np.percentile(attn_map, 95)
 
    if vmin < 0 < vmax:
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(attn_map, cmap=custom_cmap, norm=norm, aspect='equal')

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)

    ticks = np.arange(len(labels))
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{l:.2f}" for l in labels], rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{l:.2f}" for l in labels])
    
    fig.colorbar(im, ax=ax, label='Similarity Score')
    ax.set_title("Normalized Attention Score Similarity", pad=20)
    ax.set_xlabel("Key Valence")
    ax.set_ylabel("Query Valence")

    plt.tight_layout()
    plt.savefig(osp.join(folder, "valence_attention_heatmap.pdf"), bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    script_dir = Path(__file__).resolve().parent
    subfolders = [f.path for f in os.scandir(script_dir / "results") if f.is_dir()]

    for folder in tqdm(subfolders, desc="Processing folders"):
        checkpoint_file = osp.join(folder, "checkpoint.pt")
        
        if osp.exists(checkpoint_file):
            cert_data, loss_data, (attn_map, labels), si_scores, gini_summary = recover_train_rm_stats(checkpoint_file, device=device)

            torch.save(cert_data, osp.join(folder, "train_rm_certainty_valence.pt"))
     
            torch.save(loss_data, osp.join(folder, "train_rm_raw_loss_valence.pt"))
     
            torch.save(si_scores, osp.join(folder, "valence_vs_spatial_info.pt")) 
            
            torch.save(gini_summary, osp.join(folder, "valence_vs_gini_idx.pt")) 
            save_heatmap(attn_map, labels, folder)
        else:
            print(f"Skipping {folder}: No checkpoint.pt found.")

    print("\nBatch processing complete.")