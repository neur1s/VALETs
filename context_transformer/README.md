# context_transformer

## Setup

Set the `BASE_DIR` environment variable to the absolute path of this directory. All scripts use it to locate checkpoints, the shared `include/` folder, and output files.

```bash
cd context_transformer
export BASE_DIR=`pwd`
```

Install dependencies:

```bash
pip install -r ../requirements.txt
```

---

## Training

Each model variant has its own subdirectory. Run `train_seeds.py` from within the variant directory — it runs seeds 0, 1, 2 automatically and saves one checkpoint per (seed, beta) combination.

### Baseline (standard cross-entropy)

```bash
cd baseline
python train_seeds.py
```

Saves `checkpoint_seed{0,1,2}.pt` and `seed_results_summary.pt`.

### WCE (weighted cross-entropy)

```bash
cd wce
python train_seeds.py
```

Sweeps `BETA_VALUES = [0.25, 0.5, 1.0]` × 3 seeds. Saves `checkpoint_seed{seed}_beta{beta}.pt`.

To change the beta sweep, edit `BETA_VALUES` at the top of `train_seeds.py`.

### WCE + Valence Embeddings (MLP)

```bash
cd wce_valence_embed_mlp
python train_seeds.py
```

### WCE + Valence Embeddings (Attention)

```bash
cd wce_valence_embed_attn
python train_seeds.py
```

### Valence Embeddings in Attention, no WCE

```bash
cd no_wce_valence_embed_attn
python train_seeds.py
```

### Evaluating a checkpoint

`include/test.py` evaluates any saved checkpoint:

```bash
python include/test.py --checkpoint <variant>/checkpoint_seed0.pt
```

---

## Plotting

Run plotting scripts from the `context_transformer/` root.

| Script | Output | Description |
|--------|--------|-------------|
| `plot_comparison_all_models.py` | `comparison_all_models_wm_train.pdf`, `comparison_all_models_rm_train.pdf`, `comparison_gini_cross_attn.pdf` | Working memory and reference memory accuracy across all model variants; cross-attention Gini index |
| `plot_epoch_vs_error_all_models.py` | `Epoch_vs_error_WM.pdf`, `Epoch_vs_error_RM.pdf` | Training error curves per epoch for all models |
| `frequency_memory_task.py` | `frequency_memory_curve.pdf` | Memory accuracy as a function of how often a context was repeated during training |
