# gridworld_transformer

## Setup

Set the `BASE_DIR` environment variable to the absolute path of this directory. All plotting scripts read results from and write outputs to paths relative to `BASE_DIR`.

```bash
cd gridworld_transformer
export BASE_DIR=`pwd`
```

Add this line to `~/.zshrc` or `~/.bashrc` to make it permanent.

Install dependencies:

```bash
pip install -r ../requirements.txt
```

---

## Training

Each model variant has its own subdirectory. Run training from within the variant directory.

### Baseline (standard cross-entropy, no valence weighting)

```bash
cd baseline
python ../include/main_baseline.py --run_dir ./runs --group_name baseline --num_envs 32 --seed 0 --training_epoch 200 --log_to_wandb
```

### WCE (weighted cross-entropy)

```bash
cd wce
python main_wce.py --run_dir ./runs --group_name wce --num_envs 32 --seed 0 --valence_weight 1.0 --training_epoch 200 --log_to_wandb
```

`--valence_weight` (default `1.0`) controls the strength of valence-based upweighting in the loss.

### WCE + Valence Embeddings (MLP)

```bash
cd wce_valence_embed_mlp
python main_wce.py --run_dir ./runs --group_name wce_valence_embed_mlp --num_envs 32 --seed 0 --valence_weight 1.0 --contrastive_weight 1.0 --log_to_wandb
```

`--contrastive_weight` controls the weight of the geometric valence contrastive loss, which enforces that cosine similarity between valence embeddings matches the cosine of their angle difference.

### WCE + Valence Embeddings (Attention)

```bash
cd wce_valence_embed_attn
python main_wce.py --run_dir ./runs --group_name wce_valence_embed_attn --num_envs 32 --seed 0 --valence_weight 1.0 --contrastive_weight 1.0 --log_to_wandb
```

Same flags as the MLP variant; valence embeddings are injected into the attention mechanism instead of the MLP stream.

### Valence Embeddings in Attention, no WCE

```bash
cd no_wce_valence_embed_attn
python main_wce.py --run_dir ./runs --group_name no_wce_valence_embed_attn --num_envs 32 --seed 0 --contrastive_weight 1.0 --log_to_wandb
```

Key shared flags across all variants:

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `0` | Random seed |
| `--training_epoch` | `200` | Number of training epochs |
| `--learning_rate` | `1e-4` | Learning rate |
| `--batch_size` | `512` | Batch size |
| `--num_envs` | `32` | Number of gridworld environments |
| `--run_dir` | `./` | Directory to save checkpoints and results |
| `--log_to_wandb` | `False` | Enable Weights & Biases logging |
| `--gpu` | `0` | GPU index |

---

## Plotting

All scripts are in `plot/` and require `BASE_DIR` to be set. Run from the `gridworld_transformer/` root:

```bash
python plot/<script_name>.py
```

| Script | Output | Description |
|--------|--------|-------------|
| `plot_accuracy_valence_comparison.py` | `comparison_valence_accuracy.pdf` | Accuracy vs. valence across all model variants |
| `plot_post_hoc_analyses.py` | `comparison_<metric>.pdf` for each metric | Reference memory certainty, raw loss, distractor task accuracy, frequency task accuracy, spatial information, and Gini index — one plot per metric |
| `plot_comparison_all_models.py` | `comparison_all_models_grid.png` | Grid of all models × all metrics in a single figure |
| `plot_comparison_all_models_subset.py` | `comparison_individual_seeds_<model>.png` per model | Per-seed breakdown for each model variant |
| `plot_beta_comparison.py` | `comparison_betas_<model>.png` per model | Accuracy vs. valence across beta sweep values within each variant |
| `plot_embedding_similarity.py` | `embedding_similarity_heatmap_<variant>.png` and `embedding_similarity_vs_diff_<variant>.png` | Dot-product similarity between learned valence embeddings, shown as a heatmap and as a function of valence difference |

All outputs are saved to `$BASE_DIR`.
