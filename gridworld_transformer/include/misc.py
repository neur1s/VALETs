import sys, os
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))
from copy import deepcopy
from typing import Dict
import numpy as np
import torch

def create_data(rw, config, act_seed, validation=False): # wrapper for RandomWalker (rw object)
    g, x, v, a = rw.run(config.batch_size, config.walk_length, act_seed, validation)
    return (
        torch.tensor(x).long(), # observations
        torch.tensor(v).float(), # valences
        torch.tensor(a).long(), # actions
        torch.tensor(g).long(), # 2D graph positions
    )

def rate_map(L, rw, model, config, device, validation=False, use_valence_embeddings=False):
    """ Evaluates the model by running it on a long random walk and then analyzing its internal activations,
    such as the position embeddings, hidden states of the FFN and attention softmax values.
    Generates rate maps to model place cells."""
    config = deepcopy(config)
    config.walk_length = config.walk_length * 50 # very long random walk
    obs, vals, act, pos = create_data(rw, config, 0, validation)

    # processing the walk in chunks
    # tgt_len is the length of each chunk (size of segment processed at a time)
    chunks = [
        [i, min(i + config.tgt_len, config.walk_length)]
        for i in range(0, config.walk_length, config.tgt_len)
    ]
    config.batch_size = 1
    _obs, _act, _pos = obs[:1], act[:1], pos[:1]
    counts = np.zeros((L, L))
    pos_counts = np.zeros((L, L)) # to track how many times each location has been visited

    # initiate return dictionary to store the activations of different internal activations at different visited locations
    ret = {"pos_emb": np.zeros((config.d_embed, L, L))}

    for l in range(1, config.n_layer + 1):
        ret["layer%d/ffn_hid" % l] = np.zeros((config.d_inner, L, L))
        ret["layer%d/softmax" % l] = np.zeros((config.n_head, config.mem_len + config.tgt_len + 1, L, L))

    model.eval()
    with torch.no_grad():
        init_pos = _pos[:, 0].to(device)
        prev_hidden = model.rnn.init_hidden(init_pos)
        prev_outputs = None
        mems = None
        # run through all chunks that we are going to backprop for
        for j, [start, stop] in enumerate(chunks):
            src_x = _obs[:, start:stop].to(device) # current chunk of observations
            a = _act[:, start:stop].to(device) # current chunk of actions
            if prev_outputs is not None:
                prev_hidden = model.correction(prev_hidden, prev_outputs)

            if use_valence_embeddings:
                v_curr = torch.zeros((src_x.size(1), src_x.size(0), 64), device=device)
                outputs = model(a, src_x, prev_hidden, mems, v_curr=v_curr)
            else:
                outputs = model(a, src_x, prev_hidden, mems)

            if stop >= config.mem_len + config.tgt_len:
                _x, _y = _pos[0, stop].item() % L, _pos[0, stop].item() // L
                pos_idx = _pos[0, start + 1 : stop + 1].numpy()
                pos_emb = outputs.grid_activations[0, 1:].detach().cpu().numpy()
                for n, _pos_idx in enumerate(pos_idx):
                    # position is used to map the activations to a specific (x, y) coordinate in the 2D grid
                    # the horizontal position (along x) in the grid is given by _pos_idx % L (modulus operation gives the column idx)
                    # the vertical position (along y) in the grid is given by _pos_idx // L (integer division)
                    ret["pos_emb"][:, _pos_idx % L, _pos_idx // L] = (ret["pos_emb"][:, _pos_idx % L, _pos_idx // L] + pos_emb[n])

                    # update pos_counts to keep track of how many times each location has been visited
                    pos_counts[_pos_idx % L, _pos_idx // L] = (pos_counts[_pos_idx % L, _pos_idx // L] + 1)

                # extract activations from the output object & store them in the ret dictionary
                for l in range(1, config.n_layer + 1): # iterating over layers
                    ffn_hid = (outputs.ffn_activations[l - 1][0, -1].detach().cpu().numpy())

                    # %d will specift the layer number in the key name
                    ret["layer%d/ffn_hid" % l][:, _x, _y] = (ret["layer%d/ffn_hid" % l][:, _x, _y] + ffn_hid)

                    softmax = outputs.attentions[l - 1][0, :, -1].detach().cpu().numpy()
                    ret["layer%d/softmax" % l][..., _x, _y] = (ret["layer%d/softmax" % l][..., _x, _y] + softmax)

                counts[_x, _y] = counts[_x, _y] + 1

            # update the mems and prev_hidden for the next chunk
            mems = outputs.mems
            prev_hidden = outputs.rnn_hidden
            if config.correction:
                prev_outputs = outputs.last_hidden_state[:, -1].detach()

    ret["pos_emb"] = ret["pos_emb"] / pos_counts

    # average activations for each unique location on the grid, to produce rate maps
    for l in range(1, config.n_layer + 1):
        ret["layer%d/ffn_hid" % l] = ret["layer%d/ffn_hid" % l] / counts
        ret["layer%d/softmax" % l] = ret["layer%d/softmax" % l] / counts

    return ret

def eval_memory_correct(logits, target, pos, start, stop, mlen):
    """ Evaluate the model's performance by calculating the accuracy of its predictions based on whether the position has been previously visited or not. """
    visited_pos = pos[:, max(0, start - mlen) : stop] # positions visited within the memory window
    c = visited_pos[..., None, None] == pos[:, stop : stop + 1]
    _visited_before = c.any(1).squeeze(-1).diagonal()
    _unvisited_before = (~c).all(1).squeeze(-1).diagonal()

    visited_correct = torch.sum((torch.argmax(logits[_visited_before], dim=-1) == target[_visited_before]).float()).item()
    unvisited_correct = torch.sum((torch.argmax(logits[_unvisited_before], dim=-1) == target[_unvisited_before]).float()).item()

    correct = torch.sum((torch.argmax(logits, dim=-1) == target).float()).item()

    return {
        "visited_correct": visited_correct,
        "unvisited_correct": unvisited_correct,
        "correct": correct,
        "visited_tot": _visited_before.sum(),
        "unvisited_tot": _unvisited_before.sum(),
        ## return masks for visited and unvisited locations:
        "wm_mask": _visited_before.cpu(),
        "rm_mask": _unvisited_before.cpu()

    }

def all_pred_and_targets(logits, target):
    all_pred = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    all_targets = target.detach().cpu().tolist()

    return{
        "all_pred": all_pred,
        "all_targets": all_targets,
    }

def rm_loss_by_valence(all_logits, all_trgs, all_vals, all_rm_masks, device, criterion):
    all_logits_tensor = torch.cat(all_logits, dim=0).to(device)
    all_trgs_tensor = torch.tensor(all_trgs).to(device)
    all_vals_tensor = torch.tensor(all_vals, dtype=torch.float32).to(device)
    all_rm_masks_tensor = torch.tensor(all_rm_masks).to(device)

    # filter all aggregated data to get only the RM subset
    rm_logits = all_logits_tensor[all_rm_masks_tensor]
    rm_trgs = all_trgs_tensor[all_rm_masks_tensor]
    rm_vals = all_vals_tensor[all_rm_masks_tensor]

    # calculate loss per calence on thefFiltered RM Subset
    rm_loss_valence_level = {}

    # iterate over only the unique valence values present in the RM subset
    for val_float in np.unique(all_vals[all_rm_masks]).tolist():
        val_tensor = torch.tensor(val_float, dtype=torch.float32, device=device)
        val_str = str(val_float)

        # create a mask for the current valence level *within the RM subset*
        valence_mask = torch.isclose(rm_vals, val_tensor)
        rm_val_count = valence_mask.sum().item()

        if rm_val_count > 0:
            # filter RM subset by valence
            rm_val_logits = rm_logits[valence_mask]
            rm_val_trgs = rm_trgs[valence_mask]
            rm_val_vals = rm_vals[valence_mask]

            # calculate the raw CE Loss (average loss per sample)
            # we don't use WCE here because it obscures the true loss value and pushes salient samples towerd larger loss, resulting in ~uniform loss across valence levels
            rm_loss_val = criterion(rm_val_logits, rm_val_trgs).item()

            rm_loss_valence_level[val_str] = rm_loss_val

        else:
            rm_loss_valence_level[val_str] = 0.0

    return rm_loss_valence_level


def rm_certainty_by_valence(all_logits, all_rm_masks, all_vals, device):
    all_logits_tensor = torch.cat(all_logits, dim=0).to(device)
    all_rm_masks_tensor = torch.tensor(all_rm_masks).to(device)
    all_vals_tensor = torch.tensor(all_vals, dtype=torch.float32).to(device)

    rm_logits = all_logits_tensor[all_rm_masks_tensor]
    rm_vals = all_vals_tensor[all_rm_masks_tensor]

    if rm_logits.numel() == 0:
        return {}

    rm_probs = torch.softmax(rm_logits, dim=-1)

    # find the probability of the highest-scoring token for each RM sample
    # torch.max returns (values, indices); we only need the values (the max probability)
    max_prob_values, _ = torch.max(rm_probs, dim=-1) # max_prob_values has shape [num_rm_samples]

    rm_certainty_valence: Dict[str, float] = {}

    unique_rm_vals = np.unique(all_vals[np.array(all_rm_masks) == 1]).tolist()

    for val_float in unique_rm_vals:
        val_tensor = torch.tensor(val_float, dtype=torch.float32, device=device)
        val_str = str(val_float)

        # mask for the current valence level within the RM subset
        valence_mask = (rm_vals == val_tensor)

        if valence_mask.sum().item() > 0:
            val_certainties = max_prob_values[valence_mask]

            # calculate the average prediction certainty for this valence group
            avg_certainty = val_certainties.mean().item()
            rm_certainty_valence[val_str] = avg_certainty
        else:
            rm_certainty_valence[val_str] = 0.0

    return rm_certainty_valence
