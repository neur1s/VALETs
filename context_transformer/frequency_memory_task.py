import os
import sys
import torch
import numpy as np
import importlib.util
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 14})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_REPEATS = [1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64]
REPEATS = 20
SEEDS = [0, 1, 2]
RECENCY_LAG = 32  # target's last occurrence is always this many tokens from end


def load_unique_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def evaluate_step(model, model_module, task, target_id, ctx_id, n_repeats,
                  is_attn, cfg, trial_seed, v_embedder=None):
    model.eval()
    if v_embedder:
        v_embedder.eval()

    seq_len = cfg.get('seq_len', 256)
    mem_len = cfg.get('mem_len', 64)
    num_observations = cfg['num_observations']
    SOS_TOKEN = num_observations
    CTX_OFFSET = num_observations + 1

    rng = np.random.RandomState(trial_seed)

    with torch.no_grad():
        ctx_pool_idx = ctx_id - CTX_OFFSET
        pool = task.context_pools[ctx_pool_idx]
        filler_pool = pool[pool != target_id]

        pool = task.context_pools[ctx_pool_idx]
        neutral_ids = np.where(np.abs(task.obs_to_valence) < 1.0)[0]
        filler_pool = np.intersect1d(neutral_ids, pool)
        filler_pool = filler_pool[filler_pool != target_id]
        if len(filler_pool) < 5:  # fallback if context pool has too few neutrals
            filler_pool = pool[pool != target_id]

        # build obs_list with controlled recency (target pinned at seq_len - RECENCY_LAG - 1)
        RECENCY_LAG = 32
        n_filler = seq_len - n_repeats
        filler = rng.choice(filler_pool, n_filler, replace=True).tolist()

        prefix_len = seq_len - RECENCY_LAG - 1
        extra_targets = [target_id] * (n_repeats - 1)
        prefix_filler_count = prefix_len - len(extra_targets)
        if prefix_filler_count < 0:
            extra_targets = extra_targets[:prefix_len]
            prefix_filler_count = 0

        prefix_items = extra_targets + filler[:prefix_filler_count]
        rng.shuffle(prefix_items)
        suffix_filler = filler[prefix_filler_count: prefix_filler_count + RECENCY_LAG]
        while len(suffix_filler) < RECENCY_LAG:
            suffix_filler.append(int(rng.choice(filler_pool)))

        obs_list = prefix_items + [target_id] + suffix_filler
        assert len(obs_list) == seq_len

        obs_ids = torch.tensor([obs_list], dtype=torch.long).to(DEVICE)  # [1, seq_len]
        ctx_ids = torch.tensor([ctx_pool_idx], dtype=torch.long).to(DEVICE)  # [1]

        valences_list = [task.obs_to_valence[t] for t in obs_list]
        valences = torch.tensor([valences_list], dtype=torch.float32).to(DEVICE)  # [1, seq_len]

        ctx_tokens = (ctx_ids + CTX_OFFSET).unsqueeze(1)  # [1, 1]
        src = torch.cat([ctx_tokens, obs_ids], dim=1)     # [1, 257]

        if is_attn and v_embedder is not None:
            obs_v_embeddings = v_embedder(valences)        # [1, seq_len, 64]
            ctx_v = torch.zeros(1, 1, 64, device=DEVICE)
            src_v = torch.cat([ctx_v, obs_v_embeddings], dim=1)  # [1, 257, 64]
        
        memory = model.encode(src, src_mask=None)
        if isinstance(memory, (tuple, list)):
            memory = memory[0]

        all_logits = []
        for chunk_start in range(0, seq_len, mem_len):
            chunk_end = min(chunk_start + mem_len, seq_len)
            chunk_obs = obs_ids[:, chunk_start:chunk_end]          # [1, chunk]

            if is_attn and v_embedder is not None:
                chunk_v_emb = obs_v_embeddings[:, chunk_start:chunk_end]

            if chunk_start == 0:
                sos = torch.full((1, 1), SOS_TOKEN, dtype=torch.long, device=DEVICE)
                tgt = torch.cat([sos, chunk_obs[:, :-1]], dim=1)
                if is_attn and v_embedder is not None:
                    sos_v = torch.zeros(1, 1, 64, device=DEVICE)
                    tgt_v = torch.cat([sos_v, chunk_v_emb[:, :-1]], dim=1)
            else:
                prev_last = obs_ids[:, chunk_start - 1].unsqueeze(1)
                tgt = torch.cat([prev_last, chunk_obs[:, :-1]], dim=1)
                if is_attn and v_embedder is not None:
                    prev_v = obs_v_embeddings[:, chunk_start - 1].unsqueeze(1)
                    tgt_v = torch.cat([prev_v, chunk_v_emb[:, :-1]], dim=1)

            tgt_mask = model_module.subsequent_mask(tgt.size(1)).to(DEVICE)

            if is_attn and v_embedder is not None:
                chunk_out = model.decode(
                    memory=memory, src_mask=None, tgt=tgt,
                    tgt_mask=tgt_mask, src_val=src_v, tgt_val=tgt_v
                )
            else:
                tgt_emb = model.tgt_embed(tgt)
                res_dec = model.decoder(tgt_emb, memory, None, tgt_mask)
                chunk_out = res_dec[0] if isinstance(res_dec, (tuple, list)) else res_dec

            chunk_logits = model.generator(chunk_out)  # [1, chunk, vocab]
            all_logits.append(chunk_logits)

        all_logits = torch.cat(all_logits, dim=1)  # [1, seq_len, vocab]
        preds = all_logits.argmax(dim=-1)[0]       # [seq_len]

        # score: did the model correctly predict the target at its position(s)?
        obs_tensor = obs_ids[0]  # [seq_len]
        target_positions = (obs_tensor == target_id).nonzero(as_tuple=True)[0]

        if len(target_positions) == 0:
            return 0.0

        # accuracy at target positions only
        correct = (preds[target_positions] == target_id).float().mean().item()
        return correct


def main():
    results = {
        "Baseline_Pos": {n: [] for n in N_REPEATS},
        "Baseline_Neg": {n: [] for n in N_REPEATS},
        "Full_Pos": {n: [] for n in N_REPEATS},
        "Full_Neg": {n: [] for n in N_REPEATS},
    }

    base_path = os.path.join(os.environ["BASE_DIR"], "baseline")
    full_path = os.path.join(os.environ["BASE_DIR"], "wce_valence_embed_attn")

    for s in SEEDS:
        print(f"\n>>> Processing Seed {s}")

        # baseline 
        sys.path.insert(0, base_path)
        m_mod_b = load_unique_module(f"m_base_{s}", os.path.join(base_path, "model.py"))
        t_mod_b = load_unique_module(f"t_base_{s}", os.path.join(os.environ["BASE_DIR"], "include", "ctx_episodic_memory_task.py"))

        fpath_b = os.path.join(base_path, f"checkpoint_seed{s}.pt")
        ckpt_b = torch.load(fpath_b, map_location=DEVICE, weights_only=False)
        cfg_b = ckpt_b['config']

        model_b = m_mod_b.make_model(
            num_contexts=cfg_b['num_contexts'], num_observations=cfg_b['num_observations'],
            N=cfg_b['num_encoder_layers'], d_model=cfg_b['d_model'],
            max_len=cfg_b.get('seq_len', 256) + 1, decoder_max_len=cfg_b.get('mem_len', 64)
        ).to(DEVICE)
        model_b.load_state_dict(ckpt_b['model_state'])
        task_b = t_mod_b.CtxtMemoryTask(
            num_contexts=cfg_b['num_contexts'], num_observations=cfg_b['num_observations'],
            seed=s, pool_seed=42
        )
        sys.path.pop(0)

        # full model 
        sys.path.insert(0, full_path)
        m_mod_f = load_unique_module(f"m_full_{s}", os.path.join(full_path, "model.py"))
        t_mod_f = load_unique_module(f"t_full_{s}", os.path.join(os.environ["BASE_DIR"], "include", "ctx_episodic_memory_task.py"))

        fpath_f = os.path.join(full_path, f"checkpoint_1.0.pt")
        ckpt_f = torch.load(fpath_f, map_location=DEVICE, weights_only=False)
        cfg_f = ckpt_f['config']

        model_f = m_mod_f.make_model(
            num_contexts=cfg_f['num_contexts'], num_observations=cfg_f['num_observations'],
            N=cfg_f['num_encoder_layers'], d_model=cfg_f['d_model'],
            max_len=cfg_f.get('seq_len', 256) + 1, decoder_max_len=cfg_f.get('mem_len', 64)
        ).to(DEVICE)
        model_f.load_state_dict(ckpt_f['model_state'])

        v_emb = m_mod_f.ValenceEmbedder().to(DEVICE)
        if 'valence_embedder_state' in ckpt_f:
            v_emb.load_state_dict(ckpt_f['valence_embedder_state'])
        else:
            v_state = {k.replace('v_embedder.', ''): v
                    for k, v in ckpt_f['model_state'].items()
                    if k.startswith('v_embedder.')}
            if not v_state:
                # Print available keys to find the right prefix
                all_keys = list(ckpt_f['model_state'].keys())[:20]
                raise KeyError(f"Could not find valence embedder weights. Top-level ckpt keys: {list(ckpt_f.keys())}. "
                            f"First 20 model_state keys: {all_keys}")
            v_emb.load_state_dict(v_state)

        task_f = t_mod_f.CtxtMemoryTask(
            num_contexts=cfg_f['num_contexts'], num_observations=cfg_f['num_observations'],
            seed=s, pool_seed=42
        )
        sys.path.pop(0)

        pos_targets = np.where(task_f.obs_to_valence == 1.0)[0]
        neg_targets = np.where(task_f.obs_to_valence == -1.0)[0]

        for n_rep in tqdm(N_REPEATS, desc=f"Seed {s}"):
            for r in range(REPEATS):
                trial_seed = s * 100000 + n_rep * 100 + r
                rng = np.random.RandomState(trial_seed)

                ctx_pool_idx = rng.randint(0, cfg_f['num_contexts'])
                ctx_id = cfg_f['num_observations'] + 1 + ctx_pool_idx

                # Targets must be in the context's pool
                ctx_pool = task_f.context_pools[ctx_pool_idx]
                pos_in_pool = np.intersect1d(pos_targets, ctx_pool)
                neg_in_pool = np.intersect1d(neg_targets, ctx_pool)
                if len(pos_in_pool) == 0 or len(neg_in_pool) == 0:
                    continue

                t_pos = rng.choice(pos_in_pool)
                t_neg = rng.choice(neg_in_pool)

                results["Baseline_Pos"][n_rep].append(
                    evaluate_step(model_b, m_mod_b, task_b, t_pos, ctx_id,
                                  n_rep, False, cfg_b, trial_seed))
                results["Baseline_Neg"][n_rep].append(
                    evaluate_step(model_b, m_mod_b, task_b, t_neg, ctx_id,
                                  n_rep, False, cfg_b, trial_seed))
                results["Full_Pos"][n_rep].append(
                    evaluate_step(model_f, m_mod_f, task_f, t_pos, ctx_id,
                                  n_rep, True, cfg_f, trial_seed, v_emb))
                results["Full_Neg"][n_rep].append(
                    evaluate_step(model_f, m_mod_f, task_f, t_neg, ctx_id,
                                  n_rep, True, cfg_f, trial_seed, v_emb))

    plot_results(results)


def plot_results(results):
    fig, ax = plt.subplots(figsize=(6, 4))

    def get_stats(data_dict):
        means, sems = [], []
        for n in N_REPEATS:
            data = np.array(data_dict[n])
            means.append(np.mean(data) if len(data) > 0 else 0)
            sems.append(np.std(data) / np.sqrt(len(data)) if len(data) > 0 else 0)
        return np.array(means), np.array(sems)

    plot_configs = [
        ("Baseline_Pos", results["Baseline_Pos"], "#7f8c8d", "--", "Baseline (v=+1)"),
        ("Baseline_Neg", results["Baseline_Neg"], "#95a5a6", ":", "Baseline (v=−1)"),
        ("Full_Pos", results["Full_Pos"], "#e14d75", "-", "WCE Valence-Embed Attn (β=1, v=+1)"),
        ("Full_Neg", results["Full_Neg"], "#21448d", "-", "WCE Valence-Embed Attn (β=1, v=−1)"),
    ]

    x = list(range(len(N_REPEATS)))
    for key, data, color, ls, label in plot_configs:
        means, sems = get_stats(data)
        ax.plot(x, means, label=label, color=color, linewidth=2.5,
                linestyle=ls, marker='o', markersize=4)
        ax.fill_between(x, means - sems, means + sems, color=color, alpha=0.15)

    ax.set_xticks(x)
    ax.set_xticklabels(N_REPEATS)
    ax.set_xlabel("Max. |v| observation frequency")
    ax.set_ylabel("Recall accuracy")
    ax.set_xlim(0, len(N_REPEATS) - 1) 
    ax.set_ylim(0, 0.9)  
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=2, fontsize=14)
    plt.tight_layout(rect=[0, 0.15, 1, 1])
    ax.grid(True, alpha=0.2, linestyle=':')
    plt.savefig("frequency_memory_curve.pdf")
    plt.show()


if __name__ == "__main__":
    main()