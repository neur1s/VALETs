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

def recover_train_rm_stats(checkpoint_path, device='cuda'):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    
    # reconstruct the model
    model = xlTEM(config).to(device)
    model.load_state_dict(ckpt['state_dict'])
    model.eval()
    
    # reconstruct environment with the same training maps
    rw = RandomWalker(config.side_len, config.num_envs, config.seed, config.vocab_size, config.n_a)
    rw.x = ckpt['maps'] 
    
    seed_idx = config.training_epoch + 42 # to match training
    obs, vals, act, pos = create_data(rw, config, seed_idx, validation=False) # create data in the training mode

    raw_ce_criterion = nn.CrossEntropyLoss(reduction='none')

    all_probs, all_losses, all_valences, all_rm_masks = [], [], [], []
    
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
    return certainty_results, loss_results

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # get list of all subdirectories in results
    script_dir = Path(__file__).resolve().parent
    subfolders = [f.path for f in os.scandir(script_dir / "results") if f.is_dir()]

    for folder in tqdm(subfolders, desc="Processing folders"):
        checkpoint_file = osp.join(folder, "checkpoint.pt")
        
        if osp.exists(checkpoint_file):
            cert_data, loss_data = recover_train_rm_stats(checkpoint_file, device=device)
            # save certainty data
            torch.save(cert_data, osp.join(folder, "train_rm_certainty_valence.pt"))
            # save raw loss data
            torch.save(loss_data, osp.join(folder, "train_rm_raw_loss_valence.pt"))
        else:
            print(f"Skipping {folder}: No checkpoint.pt found.")

    print("\nBatch processing complete.")