import os
import sys
import os.path as osp
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../include"))
from model import xlTEM
from random_walk import RandomWalker
from wce_loss import WCELoss
plt.rcParams['savefig.bbox'] = 'tight'

def walk_simulation_gif(vis_map, nodes, L, target_id, target_val):
    fig, ax = plt.subplots(figsize=(9, 7))
    
    display_map = vis_map.copy()

    v_limit = abs(target_val) * 1.25 # target node has a distinct shade
    vmin, vmax = -v_limit, v_limit

    override_val = vmax if target_val >= 0 else vmin # dark red if target is +1, otherwise the darkest blue
    display_map[0, 0] = override_val

    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) # 0 is the visual center (white) of the 'coolwarm' map

    distractor_data = vis_map.copy().flatten()
    distractor_vals = np.delete(distractor_data, 0) # remove the target's original valence (idx 0) to find the distractor extremes
    actual_min = np.min(distractor_vals)
    actual_max = np.max(distractor_vals)
    
    im = ax.imshow(display_map, cmap='coolwarm', origin='upper', norm=norm)

    # valence colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.9) # specific ticks represent the range of valences in stim_to_val
    cbar.set_label('Valence', rotation=270, labelpad=25, fontsize=12)
    
    # define points to label
    ticks = [actual_min, 0, actual_max, override_val]
    ticks = sorted(list(set(ticks)))
    
    labels = []
    for t in ticks:
        if t == override_val:
            labels.append("Target Spot")
        elif np.isclose(t, actual_min):
            labels.append(f"{t:.2f}")
        elif np.isclose(t, actual_max):
            labels.append(f"{t:.2f}")
        else:
            labels.append(f"{t:.2f}")

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    
    fig.subplots_adjust(right=0.82, left=0.08)
    
    agent_m = ax.scatter([], [], color='green', s=250, edgecolors='black', zorder=5, label='Agent') # markers
    ax.scatter([0], [0], color='darkgrey', marker='*', s=300, label=f'Target (V={target_val})', zorder=4)

    ax.legend(loc='upper right', 
              labelspacing=2.0, 
              markerscale=0.7,
              handletextpad=1.2, 
              fontsize='medium', 
              framealpha=0.9)
    
    ax.set_xticks(np.arange(-.5, L, 1), minor=True)
    ax.set_yticks(np.arange(-.5, L, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.legend(loc='upper right')

    def update(i):
        curr_node = nodes[i]
        x, y = curr_node % L, curr_node // L
        agent_m.set_offsets(np.c_[x, y])
        ax.set_title(f"Step {i}/30 | Node: {curr_node} | Target ID: {target_id}")
        return agent_m,

    ani = FuncAnimation(fig, update, frames=len(nodes), blit=True, interval=250)
    ani.save("distractor_walk_simulation.gif", writer='pillow')
    plt.close()
    print("--> Distractor walk GIF saved")

def get_clean_gap_path(start_node, L, total_N, adjacency_matrix):
    """
    Generates a path of length total_N that:
    - starts at start_node
    - never revisits start_node until the final step (total_N)
    - uses action mapping: 0:Stay, 1:E, 2:S, 3:N, 4:W
    """
    actions = []
    curr_pos = start_node
    # manhattan distance to start_node
    trgt_x, trgt_y = start_node % L, start_node // L 
    def dist(node):
        return abs(node % L - trgt_x) + abs(node // L - trgt_y)
    
    for step in range(total_N):
        steps_left = total_N - step

        # identify all valid neighbors from the Adjacency Matrix
        valid_neighbor_indices = np.where(adjacency_matrix[curr_pos] == 1)[0] # row curr_pos in self.A has a 1 where a connection exists
        
        possible_moves = []
        for nxt in valid_neighbor_indices:
            if step == total_N - 1: # Last step must be start_node
                if nxt == start_node:
                    possible_moves.append(nxt)
            else: 
                if nxt != start_node: # intermediate steps: MUST NOT be start_node
                    possible_moves.append(nxt)
        
        # if the agent gets stuck (only neighbour is start_node but not last step)
        if not possible_moves:
            possible_moves = [curr_pos] if curr_pos != start_node else [n for n in valid_neighbor_indices if n != start_node] # force it to stay if n_a == 5, otherwise pick any non-start move

        dist_to_home = dist(curr_pos)

        if (steps_left - dist_to_home) % 2 != 0 and curr_pos != start_node:
            selected_nxt = curr_pos # force a 'stay' action to fix parity
        
        elif dist_to_home >= steps_left:
            # forced Homing
            possible_moves.sort(key=dist)
            best_dist = dist(possible_moves[0])
            possible_moves = [m for m in possible_moves if dist(m) == best_dist]
            selected_nxt = possible_moves[0]
        else:
            # random Exploration
            selected_nxt = np.random.choice(possible_moves)
            
        # map the node transition back to action IDs
        diff = selected_nxt - curr_pos
        if diff == 0:
            act = 0 # stay
        elif diff == 1:
            act = 1 # east
        elif diff == L:
            act = 2 # south
        elif diff == -L:
            act = 3 # north
        elif diff == -1:
            act = 4 # west
        else:
            # this should not happen on a standard grid
            act = 0
            
        actions.append(act)
        curr_pos = selected_nxt
        
    return actions

def memory_distractor_task(checkpoint_path, device, max_N=30, consolidation_t=10):

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt['config']
    L = config.side_len
    training_maps = ckpt['maps']
    
    rw = RandomWalker(config.side_len, config.num_envs, config.seed, config.vocab_size, config.n_a)
    rw.x = training_maps
    stim_to_val = rw.stimulus_to_valence


    # identifying salient observations
    abs_vals = np.abs(stim_to_val)
    unique_abs = np.unique(np.round(abs_vals, 4)) # Rounding to avoid float precision issues
    unique_abs = sorted(unique_abs, reverse=True)
    
    v_max = unique_abs[0] 
    v_second_max = unique_abs[1] if len(unique_abs) > 1 else None
    
    # IDs to exclude from the environment during the walk
    ids_to_exclude = np.where(np.isclose(abs_vals, v_max, atol=1e-4))[0].tolist()
    if v_second_max is not None:
        ids_to_exclude.extend(np.where(np.isclose(abs_vals, v_second_max, atol=1e-4))[0].tolist())
        
    # salient targets
    id_plus_1 = np.where(np.isclose(stim_to_val, 1.0, atol=1e-5))[0][0]
    id_minus_1 = np.where(np.isclose(stim_to_val, -1.0, atol=1e-5))[0][0]
    salient_targets = [id_plus_1, id_minus_1]

    # sistractor pool: strictly lower-valence IDs
    distractor_ids = [i for i in range(config.vocab_size) if i not in ids_to_exclude]

    results_log = []
    gif_generated = False
    n_range = range(1, max_N + 1, 1)

    for N in tqdm(n_range, desc="Increasing distractor gap"):
        for target_id in salient_targets:
            target_valence = stim_to_val[target_id]
            trial_accuracies = []
            trial_losses = []
            
            for trial in range(3):
                model = xlTEM(config).to(device)
                model.load_state_dict(ckpt['state_dict'], strict=False) # we reload model for every N to assess performance from scratch
                model.valence_embedder.eval()
                for param in model.valence_embedder.parameters():
                    param.requires_grad = False
                model.train() 

                criterion = WCELoss(valence_scaling=config.valence_weight).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate * 0.5)

                ### phase 1: consolidation ###
                study_obs = torch.full((config.batch_size, consolidation_t + 1), target_id, device=device).long()
                study_act = torch.zeros((config.batch_size, consolidation_t), device=device).long()
                study_val = torch.full((config.batch_size, consolidation_t + 1), target_valence, device=device).float()
                
                h_study = model.rnn.init_hidden(torch.zeros(config.batch_size, dtype=torch.long, device=device))

                optimizer.zero_grad()
                v_study_full = model.valence_embedder(study_val).transpose(0, 1)
                v_study_input = v_study_full[:-1, :, :]

                study_out = model(study_act, study_obs[:, :-1], h_study, mems=None, v_curr=v_study_input)
                
                target_tensor = torch.full((config.batch_size,), target_id, device=device).long()
                target_val_tensor = torch.full((config.batch_size,), target_valence, device=device)
                
                loss_study = criterion(study_out.logits, target_tensor, target_val_tensor)
                loss_study.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                final_h = study_out.rnn_hidden.detach()
                final_mems = [m.detach() for m in study_out.mems] if study_out.mems else None
                final_v_mems = v_study_full.detach()

                ### phase 2: distractor walk ###
                batch_obs = np.zeros((config.batch_size, N + 1), dtype=np.int32)
                batch_act = np.zeros((config.batch_size, N), dtype=np.int32)
                
                env_ids = np.random.choice(np.arange(config.num_envs), config.batch_size)

                for b in range(config.batch_size):
                    b_acts = get_clean_gap_path(0, L, N, rw.A)
                    curr = 0
                    b_nodes = [curr]
                    for a in b_acts:
                        if a == 1: curr += 1
                        elif a == 4: curr -= 1
                        elif a == 2: curr += L
                        elif a == 3: curr -= L
                        b_nodes.append(curr)

                    actual_map = training_maps[env_ids[b]].copy()
                    for i in range(len(actual_map)):
                        if actual_map[i] in salient_targets: 
                            actual_map[i] = np.random.choice(distractor_ids)

                    b_observations = actual_map[b_nodes]
                    b_observations[0] = target_id 
                    batch_obs[b], batch_act[b] = b_observations, b_acts

                    # capture for GIF if N=30 and it's the first sample in batch
                    if N == 30 and trial == 0 and b == 0 and not gif_generated:
                        captured_nodes = b_nodes
                    
                        grid_stimuli = actual_map.copy()
                        grid_stimuli[0] = target_id # ensure target is at node 0
                        
                        raw_valences = np.array([stim_to_val[s] for s in grid_stimuli])
                        vis_map = raw_valences.reshape(L, L)
                        captured_valence_map = vis_map
                
                # gif generation
                if N == 30 and trial == 0 and not gif_generated and captured_valence_map is not None:
                    walk_simulation_gif(captured_valence_map, captured_nodes, L, target_id, stim_to_val[target_id])
                    gif_generated = True

                obs_batch = torch.tensor(batch_obs, device=device).long()
                act_batch = torch.tensor(batch_act, device=device).long()
                val_seq = torch.tensor(stim_to_val[batch_obs], device=device).float()

                v_walk_full = model.valence_embedder(val_seq).transpose(0, 1)
                v_walk_input = v_walk_full[:-1, :, :]
                
                with torch.no_grad():
                    model.eval()
                    outputs = model(act_batch, obs_batch[:, :-1], final_h, 
                                    mems=final_mems, v_curr=v_walk_input, v_mems=final_v_mems)
                    
                    acc = (outputs.logits.argmax(dim=-1) == target_tensor).float().mean().item()
                    loss = criterion(outputs.logits, target_tensor, target_val_tensor).item()

                trial_accuracies.append(acc)
                trial_losses.append(loss)

            results_log.append({
                "N": N,
                "target_valence": target_valence,
                "avg_accuracy": np.mean(trial_accuracies),
                "avg_loss": np.mean(trial_losses)
            })
            
    return results_log

if __name__ == "__main__":
    # get list of all subdirectories in results
    script_dir = Path(__file__).resolve().parent
    subfolders = [f.path for f in os.scandir(script_dir / "results") if f.is_dir()]

    for folder in tqdm(subfolders, desc="Processing folders"):
        checkpoint_file = osp.join(folder, "checkpoint.pt")
        
        if osp.exists(checkpoint_file):
            results_list = memory_distractor_task(checkpoint_file, device="cuda" if torch.cuda.is_available() else "cpu")
        
            structured_results = {}
            for entry in results_list:
                v_key = float(entry["target_valence"])
                n_key = int(entry["N"])
                acc = float(entry["avg_accuracy"])
                
                if v_key not in structured_results:
                    structured_results[v_key] = {}
                
                structured_results[v_key][n_key] = acc
            
            output_path = osp.join(folder, "distractor_task_results.pt")
            torch.save(structured_results, output_path)
            
            print(f"Done. Systematic results saved to {output_path}")

        else:
            print(f"Skipping {folder}: No checkpoint.pt found.")

    print("\nBatch processing complete.")