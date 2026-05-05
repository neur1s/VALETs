import os
from tqdm import tqdm
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import os.path as osp
from model import xlTEM
from random_walk import RandomWalker
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../include"))
from misc import create_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def recover_train_rm_stats(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    # reconstruct the model
    model = xlTEM(config).to(device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()
    
    # reconstruct environment with the same training maps
    rw = RandomWalker(config.side_len, config.num_envs, config.seed, config.vocab_size, config.n_a)
    rw.x = ckpt['maps'] 
    
    seed_idx = config.training_epoch + 42 # to match training
    obs, vals, act, pos = create_data(rw, config, seed_idx, validation=False) # create data in the training mode

    vals_tensor = torch.tensor(vals, dtype=torch.float32, device=device)
    raw_ce_criterion = nn.CrossEntropyLoss(reduction='none')

    all_probs, all_losses, all_valences, all_rm_masks = [], [], [], []

    unique_valence_levels = sorted(np.unique(vals).tolist())
    rounded_valences = [round(float(v), 2) for v in unique_valence_levels]
    val_to_idx = {v: i for i, v in enumerate(rounded_valences)}

    attn_accumulator = np.zeros((len(unique_valence_levels), len(unique_valence_levels)))
    attn_counts = np.zeros((len(unique_valence_levels), len(unique_valence_levels)))
    
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

            mems = outputs.mems
            prev_hidden = outputs.rnn_hidden

            # masking logic for train reference memory 
            mlen = config.mem_len
            visited_pos = pos[:, max(0, start - mlen) : stop]

            c = visited_pos[..., None, None] == pos[:, stop : stop + 1]
            _unvisited_before = (~c).all(1).squeeze(-1).diagonal() # if all are false, it's reference memory

            # get certainty 
            probs = torch.softmax(outputs.logits, dim=-1) # extract softmax probabilities
            max_probs, _ = torch.max(probs, dim=-1) # get max probability (certainty)

            # get raw loss
            loss = raw_ce_criterion(outputs.logits, trg_x)

            all_probs.extend(max_probs.cpu().tolist())
            all_losses.extend(loss.cpu().tolist())  
            all_valences.extend(vals[:, stop].tolist())
            all_rm_masks.extend(_unvisited_before.cpu().tolist())

    # convert to np array
    all_probs = np.array(all_probs)
    all_losses = np.array(all_losses)
    all_valences = np.array(all_valences)
    all_rm_masks = np.array(all_rm_masks).astype(bool) # training reference memory mask

    # filtering
    rm_probs = all_probs[all_rm_masks]
    rm_losses = all_losses[all_rm_masks]
    rm_valences = all_valences[all_rm_masks]

    certainty_results = {}
    loss_results = {}

    unique_vals = np.unique(rm_valences)
    for v in unique_vals:
        mask = (rm_valences == v)
        if mask.any():
            certainty_results[round(float(v), 2)] = np.mean(rm_probs[mask])
            loss_results[round(float(v), 2)] = np.mean(rm_losses[mask])

    final_attn_map = np.divide(attn_accumulator, attn_counts, out=np.zeros_like(attn_accumulator), where=attn_counts != 0)

    return certainty_results, loss_results, (final_attn_map, unique_valence_levels)

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
    
    fig.colorbar(im, ax=ax, label='Attention Score')
    ax.set_xlabel("Key valence")
    ax.set_ylabel("Query valence")
    
    plt.tight_layout()
    plt.savefig(osp.join(folder, "valence_attention_heatmap.pdf"), bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get list of all subdirectories in results
    script_dir = Path(__file__).resolve().parent
    subfolders = [f.path for f in os.scandir(script_dir / "results") if f.is_dir()]

    for folder in tqdm(subfolders, desc="Processing folders"):
        checkpoint_file = osp.join(folder, "checkpoint.pt")
        
        if osp.exists(checkpoint_file):
            cert_data, loss_data, (attn_map, labels) = recover_train_rm_stats(checkpoint_file, device=device)
            # save certainty data
            torch.save(cert_data, osp.join(folder, "train_rm_certainty_valence.pt"))
            # save raw loss data
            torch.save(loss_data, osp.join(folder, "train_rm_raw_loss_valence.pt"))

            save_heatmap(attn_map, labels, folder)
        else:
            print(f"Skipping {folder}: No checkpoint.pt found.")

print("\nBatch processing complete.")