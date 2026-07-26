import sys
import os
# ABLATION: this directory's own model.py must win over wce_valence_embed_attn's
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import argparse
import os.path as osp
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import wandb
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from configuration_transfo_xl import TransfoXLConfig
from random_walk import RandomWalker
from misc import create_data, eval_memory_correct, all_pred_and_targets, rate_map
from model import xlTEM
from wce_loss import WCELoss
from valence_embedding import ValenceEmbedder
from valence_contrastive_loss import ValenceContrastiveLoss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.autograd.set_detect_anomaly(True)


def train(seed, total_steps, rw, model, valence_embedder, optimizer_model, optimizer_valence, lr_scheduler, criterion, valence_loss, config, device):
    chunks = [[i, min(i + config.tgt_len, config.walk_length)] for i in range(0, config.walk_length, config.tgt_len)]

    obs, vals, act, pos = create_data(rw, config, seed + 42,)

    init_pos = pos[:, 0].to(device)
    prev_hidden = model.rnn.init_hidden(init_pos)
    mems = None
    prev_outputs = None

    visited_tot = 0
    visited_correct = 0
    unvisited_tot = 0
    unvisited_correct = 0
    _tot = 0
    _correct = 0
    _loss = 0
    _valence_contrastive_loss = 0

    all_preds, all_trgs, all_vals = [], [], []
    all_wm_masks, all_rm_masks = [], []
    
    model.train()

    v_mems = None
    for j, [start, stop] in enumerate(chunks):
        v_context_list = [] # to store valence embeddings of the observed stimuli in the current chunk, to be fed to transformer's attn layer
        for i in range(start + 1, stop + 1):
            optimizer_model.zero_grad()
            optimizer_valence.zero_grad()

            if len(v_context_list) > 0:
                v_curr_step = torch.stack(v_context_list, dim=0)
            else:
                # ABLATION: placeholder holds RAW scalar valences, [0, bsz]
                v_curr_step = torch.zeros((0, config.batch_size)).to(device) # 1st step in chunk has no context yet, handled by padding

            _stop = i
            src_x = obs[:, start:_stop].to(device)
            trg_x = obs[:, _stop].to(device)
            trg_vals = vals[:, _stop].to(device)  # get valences for targets
            a = act[:, start:_stop].to(device)

            if prev_outputs is not None:
                prev_hidden = model.correction(prev_hidden, prev_outputs)

            outputs = model(a, src_x, prev_hidden, mems, v_curr=v_curr_step, v_mems=v_mems)

            ### learning loop
            trg_val_embeddings = valence_embedder(trg_vals)  # learnable MLP call

            # main WCE loss
            wce_loss = criterion(outputs.logits, trg_x, trg_vals)

            # valence contrastive loss
            val_loss, val_loss_dict = valence_loss(trg_val_embeddings, trg_vals)

            # backward pass for the main transformer stream (memory task)
            optimizer_model.zero_grad()
            wce_loss.backward(retain_graph=True) # retain graph is needed for the second backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm) # clip and step for the main model parameters
            optimizer_model.step()

            # backward pass for the valence embedding task (contrastive loss)
            optimizer_valence.zero_grad()
            val_loss.backward()
            torch.nn.utils.clip_grad_norm_(valence_embedder.parameters(), config.max_grad_norm) # clip and step for the valence embedder parameters
            optimizer_valence.step()

            # ABLATION: cache the RAW target valences for the attention term
            v_context_list.append(trg_vals.detach().clone())

        mems = outputs.mems
        v_mems = outputs.v_mems
        prev_hidden = outputs.rnn_hidden
        if config.correction:
            prev_outputs = outputs.last_hidden_state[:, -1].detach()
        with torch.no_grad():
            _loss += wce_loss.item() * config.batch_size
            _valence_contrastive_loss += val_loss_dict['contrastive'] * config.batch_size
            ret = eval_memory_correct(outputs.logits, trg_x, pos, start, stop, config.mem_len)
            visited_correct += ret["visited_correct"]
            unvisited_correct += ret["unvisited_correct"]
            _correct += ret["correct"]
            visited_tot += ret["visited_tot"]
            unvisited_tot += ret["unvisited_tot"]
            _tot += config.batch_size

            if seed == total_steps - 1:
                all_pred_and_trg = all_pred_and_targets(outputs.logits, trg_x)
                all_preds.extend(all_pred_and_trg["all_pred"])
                all_trgs.extend(all_pred_and_trg["all_targets"])
                all_vals.extend(vals[:, stop].detach().cpu().tolist())

                all_wm_masks.extend(ret["wm_mask"].tolist())
                all_rm_masks.extend(ret["rm_mask"].tolist())

   
    wm_accuracy_valence = {}
    rm_accuracy_valence = {}

    if seed == total_steps - 1 and len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_trgs = np.array(all_trgs)
        all_vals = np.array(all_vals)

        all_wm_masks = np.array(all_wm_masks)
        all_rm_masks = np.array(all_rm_masks)

        overall_accuracy = all_preds == all_trgs
        
        for val in np.unique(all_vals):
            valence_mask = all_vals == val
            wm_valence_mask = valence_mask & all_wm_masks
            if wm_valence_mask.sum() > 0:
                wm_acc = overall_accuracy[wm_valence_mask].mean()
                wm_accuracy_valence[f"{val}"] = wm_acc
            
            rm_valence_mask = valence_mask & all_rm_masks
            if rm_valence_mask.sum() > 0:
                rm_acc = overall_accuracy[rm_valence_mask].mean()
                rm_accuracy_valence[f"{val}"] = rm_acc

    if lr_scheduler is not None:
        lr_scheduler.step()

    return {
        "train_tot_error": 1 - _correct / _tot,
        "train_working_memory_error": 1 - visited_correct / visited_tot,
        "train_reference_memory_error": 1 - unvisited_correct / unvisited_tot,
        "loss": _loss / _tot,
        "valence_contrastive_loss": _valence_contrastive_loss / _tot,
        "learning rate": config.learning_rate
        if lr_scheduler is None
        else lr_scheduler.get_last_lr()[0],
        "final_pred": all_preds,
        "final_targets": all_trgs,
        "final_valences": trg_vals,
        "wm_accuracy_valence": wm_accuracy_valence,
        "rm_accuracy_valence": rm_accuracy_valence
    }
    

def validate(seed, total_steps, rw, model, valence_embedder, valence_loss, config, device):
    chunks = [[i, min(i + config.tgt_len, config.walk_length)] 
              for i in range(0, config.walk_length, config.tgt_len)]
    
    obs, vals, act, pos = create_data(rw, config, seed, True)

    _tot = 0
    _correct = 0
    visited_tot = 0
    visited_correct = 0
    unvisited_tot = 0
    unvisited_correct = 0
    _valence_contrastive_loss = 0

    model.eval()
    valence_embedder.eval() # ensure valence embedder doesn't use dropout/batchnorm

    with torch.no_grad():
        init_pos = pos[:, 0].to(device)
        prev_hidden = model.rnn.init_hidden(init_pos)
        prev_outputs = None
        mems = None
        v_mems = None
        all_preds, all_trgs, all_vals = [], [], []
        all_wm_masks, all_rm_masks = [], []
        # Run through all chunks 
        for j, [start, stop] in enumerate(chunks):
            src_x = obs[:, start:stop].to(device)
            trg_x = obs[:, stop].to(device)
            trg_vals = vals[:, stop].to(device)
            a = act[:, start:stop].to(device)

            segment_vals = vals[:, start:stop].to(device)
            # ABLATION: feed the RAW scalar valences to attention
            v_curr = segment_vals.transpose(0, 1).contiguous()

            if prev_outputs is not None:
                prev_hidden = model.correction(prev_hidden, prev_outputs)

            outputs = model(a, src_x, prev_hidden, mems, v_curr=v_curr, v_mems=v_mems)

            # update memories
            mems = outputs.mems
            v_mems = outputs.v_mems
            prev_hidden = outputs.rnn_hidden

            if config.correction:
                prev_outputs = outputs.last_hidden_state[:, -1].detach()

            # compute valence loss for monitoring (no backprop in validation)
            trg_val_embeddings = valence_embedder(vals[:, stop].to(device))
            val_loss, val_loss_dict = valence_loss(trg_val_embeddings, vals[:, stop].to(device))
            _valence_contrastive_loss += val_loss_dict['contrastive'] * config.batch_size

            ret = eval_memory_correct(outputs.logits, trg_x, pos, start, stop, config.mem_len)
            visited_correct += ret["visited_correct"]
            unvisited_correct += ret["unvisited_correct"]
            _correct += ret["correct"]
            visited_tot += ret["visited_tot"]
            unvisited_tot += ret["unvisited_tot"]
            _tot += config.batch_size

            if seed == total_steps - 1:
                all_pred_and_trg = all_pred_and_targets(outputs.logits, trg_x)
                all_preds.extend(all_pred_and_trg["all_pred"])
                all_trgs.extend(all_pred_and_trg["all_targets"])
                all_vals.extend(vals[:, stop].detach().cpu().tolist())

                all_wm_masks.extend(ret["wm_mask"].tolist())
                all_rm_masks.extend(ret["rm_mask"].tolist())

   
    wm_accuracy_valence = {}
    rm_accuracy_valence = {}

    if seed == total_steps - 1 and len(all_preds) > 0:
        all_preds = np.array(all_preds)
        all_trgs = np.array(all_trgs)
        all_vals = np.array(all_vals)

        all_wm_masks = np.array(all_wm_masks)
        all_rm_masks = np.array(all_rm_masks)

        overall_accuracy = all_preds == all_trgs
        
        for val in np.unique(all_vals):
            valence_mask = all_vals == val
            wm_valence_mask = valence_mask & all_wm_masks
            if wm_valence_mask.sum() > 0:
                wm_acc = overall_accuracy[wm_valence_mask].mean()
                wm_accuracy_valence[f"{val}"] = wm_acc
            
            rm_valence_mask = valence_mask & all_rm_masks
            if rm_valence_mask.sum() > 0:
                rm_acc = overall_accuracy[rm_valence_mask].mean()
                rm_accuracy_valence[f"{val}"] = rm_acc

    return {
        "val_tot_error": 1 - _correct / _tot,
        "val_working_memory_error": 1 - visited_correct / visited_tot,
        "val_reference_memory_error": 1 - unvisited_correct / unvisited_tot,
        "val_valence_contrastive_loss": _valence_contrastive_loss / _tot,
        "final_pred": all_preds,
        "final_targets": all_trgs,
        "final_valences": all_vals,
        "wm_accuracy_valence": wm_accuracy_valence,
        "rm_accuracy_valence": rm_accuracy_valence
    }

def experiment(config):
    config = TransfoXLConfig(**config)
    torch.manual_seed(config.seed)

    device = torch.device("cuda:%d" % config.gpu if torch.cuda.is_available() else "mps")

    config.walk_length = config.steps_per_epoch * config.tgt_len

    if config.log_to_wandb:
        run = wandb.init(dir = config.run_dir, group=f"{config.group_name}", project=config.proj_name, config=config)

    rw = RandomWalker(config.side_len, config.num_envs, config.seed, config.vocab_size, config.n_a)

    model = xlTEM(config).to(device)

    rng_state = torch.get_rng_state()
    if torch.cuda.is_available():
        cuda_rng_state = torch.cuda.get_rng_state()
    else:
        cuda_rng_state = None

    valence_embedder = ValenceEmbedder().to(device)
    valence_loss = ValenceContrastiveLoss().to(device)

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    criterion = WCELoss(valence_scaling=config.valence_weight).to(device)
    

    optimizer_model = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer_valence = torch.optim.Adam(valence_embedder.parameters(), lr=config.learning_rate)

    if config.warmup_epoch >= 0:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer=optimizer_model, num_warmup_steps = config.warmup_epoch * config.steps_per_epoch, num_training_steps = config.training_epoch * config.steps_per_epoch)
    else:
        lr_scheduler = None
    
    total_steps = 1 + config.training_epoch
    t = tqdm(range(1, 1 + config.training_epoch))

    train_wm_error_per_step, train_rm_error_per_step = [], []
    val_wm_error_per_step, val_rm_error_per_step = [], []
    for n in t:
        train_logs = train(n, total_steps, rw, model, valence_embedder, optimizer_model, optimizer_valence, lr_scheduler, criterion, valence_loss, config, device)
        val_logs = validate(n, total_steps, rw, model, valence_embedder, valence_loss, config, device)

        train_wm_error_per_step.append(train_logs["train_working_memory_error"])
        train_rm_error_per_step.append(train_logs["train_reference_memory_error"])
        val_wm_error_per_step.append(val_logs["val_working_memory_error"])
        val_rm_error_per_step.append(val_logs["val_reference_memory_error"])

        if config.log_to_wandb:
            wandb.log(train_logs, step=n)
            wandb.log(val_logs, step=n) 

            if n % config.log_image_interval == 0:
                L = config.side_len
                ret1 = rate_map(L, rw, model, config, device, use_valence_embeddings=True)
                ret2 = rate_map(L, rw, model, config, device, True, use_valence_embeddings=True)
                m = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
                for k, ret in zip(["train", "valid"], [ret1, ret2]):
                    for key, val in ret.items():
                        if "softmax" in key:
                            fig = plt.figure(figsize=(8,8))
                            for i in range(64):
                                head = i // config.n_head
                                tmp = val[head]
                                plt.subplot(8, 8, i + 1)
                                img = m(torch.from_numpy(tmp[i % 8, :, :].reshape(1, 1, L, L)))
                                plt.imshow(img[0, 0])
                                plt.axis("off")
                            wandb.log({"%s/%s" % (k, key): fig}, step=n)
                            plt.close(fig)
                        else:
                            fig = plt.figure(figsize=(8, 8))
                            for i in range(64):
                                plt.subplot(8, 8, i + 1)
                                img = m(torch.from_numpy(val[i, :, :].reshape(1, 1, L, L)))
                                plt.imshow(img[0, 0])
                                plt.axis("off")
                            wandb.log({"%s/%s" % (k, key): fig}, step=n)
                            plt.close(fig)

        train_working_memory_error = 100 * train_logs["train_working_memory_error"]
        train_reference_memory_error = 100 * train_logs["train_reference_memory_error"]
        val_working_memory_error = 100 * val_logs["val_working_memory_error"]
        t.set_description(desc=f"WM error: {train_working_memory_error:.3f}%, RM error: {train_reference_memory_error:.3f}%, loss: {train_logs['loss']:.4f}")

    if config.log_to_wandb:
        L = config.side_len
        ret1 = rate_map(L, rw, model, config, device, use_valence_embeddings=True)
        ret2 = rate_map(L, rw, model, config, device, True, use_valence_embeddings=True)
        torch.save({
            "state_dict": model.state_dict(),
            "valence_embedder_state_dict": valence_embedder.state_dict(),
            "criterion_state_dict": criterion.state_dict(),
            "valence_loss_state_dict": valence_loss.state_dict(),
            "maps": rw.x,
            "config": config
        }, osp.join(run.dir, "checkpoint.pt"))
        torch.save({"train_rate_map": ret1, "valid_rate_map": ret2, "train_wm_error": train_working_memory_error, "train_rm_error": train_reference_memory_error, "valid_wm_error": val_working_memory_error, "config": config}, osp.join(run.dir, "rate_maps.pt"))
        wandb.save("*rate_maps.pt")

        torch.save(train_logs["wm_accuracy_valence"], osp.join(run.dir, "train_wm_accuracy_valence.pt"))
        wandb.save(osp.join(run.dir, "train_wm_accuracy_valence.pt"))

        torch.save(train_logs["rm_accuracy_valence"], osp.join(run.dir, "train_rm_accuracy_valence.pt"))
        wandb.save(osp.join(run.dir, "train_rm_accuracy_valence.pt"))

        torch.save(val_logs["wm_accuracy_valence"], osp.join(run.dir, "val_wm_accuracy_valence.pt"))
        wandb.save(osp.join(run.dir, "val_wm_accuracy_valence.pt"))

        torch.save(val_logs["rm_accuracy_valence"], osp.join(run.dir, "val_rm_accuracy_valence.pt"))
        wandb.save(osp.join(run.dir, "val_rm_accuracy_valence.pt"))

        torch.save(train_wm_error_per_step, osp.join(run.dir, "train_wm_accuracy_per_step.pt"))
        wandb.save(osp.join(run.dir, "train_wm_accuracy_per_step.pt"))

        torch.save(train_rm_error_per_step, osp.join(run.dir, "train_rm_accuracy_per_step.pt"))
        wandb.save(osp.join(run.dir, "train_rm_accuracy_per_step.pt"))

        torch.save(val_wm_error_per_step, osp.join(run.dir, "val_wm_accuracy_per_step.pt"))
        wandb.save(osp.join(run.dir, "val_wm_accuracy_per_step.pt"))

        torch.save(val_rm_error_per_step, osp.join(run.dir, "val_rm_accuracy_per_step.pt"))
        wandb.save(osp.join(run.dir, "val_rm_accuracy_per_step.pt"))
    
        run.finish()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./")
    parser.add_argument("--proj_name", type=str, default="predictive coding")
    parser.add_argument("--group_name", type=str, default="Swish")
    parser.add_argument("--num_envs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--tgt_len", "-K", type=int, default=32)
    parser.add_argument("--mem_len", type=int, default=32)
    #parser.add_argument("--attn_type", type=int, default=1) # 0: relative positional encoding, 1: lrelative positional encoding with sinusoidal position embeddings and learnable biases, 2: fully learned positional embeddings
    parser.add_argument("--steps_per_epoch", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=10)
    parser.add_argument("--side_len", type=int, default=11)
    parser.add_argument("--n_layer", type=int, default=2)
    parser.add_argument("--n_a", type=int, default=5)
    parser.add_argument(
        "--ffn_act_ftn",
        type=str,
        default="nmda",
        choices=["nmda", "gelu", "swish", "linear"],
    )
    parser.add_argument(
        "--rnn_act_ftn",
        type=str,
        default="tanh",
        choices=["tanh", "linear"],
    )
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--beta", type=float, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--valence_weight", type=float, default=1.0)
    parser.add_argument("--contrastive_weight", type=float, default=1.0, help="Weight for valence contrastive loss")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", type=float, default=0.25)
    parser.add_argument("--training_epoch", type=int, default=200)
    parser.add_argument("--warmup_epoch", type=int, default=0)
    parser.add_argument("--log_image_interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--log_to_wandb", "-w", action="store_true", default=False)

    args = parser.parse_args()
    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    experiment(config=vars(args))