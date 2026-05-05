# language_transformer

## Setup

Install dependencies:

```bash
pip install -r ../requirements.txt
```

Download and preprocess the dataset:

```bash
python download_pile.py        # downloads raw Pile data
python preprocess.py           # tokenizes and builds train/val splits
```

---

## Training

Each model variant has its own subdirectory. Training is split into two stages: pre-training (autoregressive language modeling) and fine-tuning.

### Baseline (standard cross-entropy)

**Pre-train:**
```bash
cd baseline
python train_seeds.py
```

Runs seeds 0, 1, 2. Saves `checkpoint_seed{seed}_mlm.pt` and `checkpoint_seed{seed}_phase1.pt` per seed.

**Fine-tune:**
```bash
python fine_tuning.py
```

Key flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--seeds` | `0 1 2` | Seeds to run |
| `--pretrain_epochs` | `20` | Pre-training epochs |
| `--finetune_epochs` | varies | Fine-tuning epochs |
| `--lr_pretrain` | `1e-4` | Pre-training learning rate |
| `--lr_finetune` | varies | Fine-tuning learning rate |
| `--patience` | `5` / `10` | Early stopping patience |

### WCE + Valence Embeddings (Attention)

**Pre-train:**
```bash
cd wce_valence_embed_attn
python train_seeds.py --gc_weight 0.1
```

Runs seeds 0, 1, 2. The `--gc_weight` flag controls the weight of the geometric valence contrastive loss relative to the autoregressive loss.

**Fine-tune:**
```bash
python fine_tuning.py --betas 0.25 0.5 1.0 --gc_weight 0.1
```

`--betas` sweeps the WCE loss scaling factor. Saves `checkpoint_seed{seed}_beta{beta}_mlm.pt` per (seed, beta) combination.

---

## Analysis

Run from the `language_transformer/` root.

| Script | Output | Description |
|--------|--------|-------------|
| `test_valence_accuracy.py` | `recall_by_first_depth.pdf`, `recall_by_first_depth.json` | Repeated proper-noun recall accuracy by depth of first occurrence, with qualitative examples comparing baseline vs. WCE model |
| `attn_score_gini.py` | `attn_heatmap_baseline.pdf`, `attn_heatmap_wce_valence_embed_attn.pdf`, `gini_by_valence.pdf` | Self-attention heatmap (query valence × key valence) and Gini index of attention distributions bucketed by valence |
