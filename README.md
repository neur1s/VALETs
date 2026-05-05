# VALETs

This is the codebase for the paper:

> Zbaranska S., Rajeev A., Josselyn S.A., Laschowski B. "A Brain-Inspired Framework for Memory Prioritization in Neural Networks Based on Valence." 2026

All experiments were run on an NVIDIA H100 with 32 GB RAM.

Three experimental settings are included, each with a `baseline` variant and one or more valence-augmented variants:

| Directory | Setting |
|-----------|---------|
| `gridworld_transformer/` | Agent navigating a gridworld; stimuli have associated valences |
| `context_transformer/` | Episodic context-matching transformer |
| `language_transformer/` | Autoregressive language model on naturalistic text |

Each directory has its own README with setup, training, and analysis instructions.

## Setup

```bash
pip install -r requirements.txt
```

`gridworld_transformer/` and `context_transformer/` each require a `BASE_DIR` environment variable pointing to their respective directory. Set it before running any training or plotting scripts:

```bash
export BASE_DIR=/path/to/gridworld_transformer  # when working in gridworld_transformer/
export BASE_DIR=/path/to/context_transformer    # when working in context_transformer/
```

## Model variants

All three settings share the same set of model variants:
- `baseline`: standard cross-entropy loss, not using valence signal 
- `wce`: weighted cross-entropy loss scaless error by the valence magnitude
- `wce_valence_embed_mlp`: WCE and a learned valence embedder (MLP) which maps scalar valences to multidimensional valence embeddings
- `wce_valence_embed_attn`: same as above but with valence embeddings injected into the attention mechanism
- `no_wce_valence_embed_attn`: valence embedder and valence injection into attention but without WCE

See the per-directory READMEs for implementation details.
