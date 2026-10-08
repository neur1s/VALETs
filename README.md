# VALETs

This is the codebase for the paper:
> Sofiya Zbaranska, Aditya Rajeev, Sheena A. Josselyn, Brokoslaw Laschowski. "A Brain-Inspired Framework for Memory Prioritization in Neural Networks Based on Valence." 2026, University of Toronto

**Abstract:** Improving long-term memory in artificial neural networks remains an open challenge. To address this, we developed a novel brain-inspired framework for memory prioritization based on the principle of emotional valence. Our framework includes: (i) a valence-weighted cross-entropy loss that scales the learning signal by the valence magnitude, analogous to neuromodulation; (ii) an amygdala-inspired module that learns high-dimensional valence embeddings; and (iii) a hippocampus-inspired module that integrates valence embeddings into the attention mechanism to modulate information retrieval. We demonstrated the generalization of our framework across spatial, episodic, and language-based memory tasks, consistently improving memory prioritization and long-term retention of high-salience information. In addition to improving long-term memory, we also showed that our framework can help mitigate the “lost-in-the-middle” problem in language modeling. More generally, this research provides further evidence of the potential of brain-inspired algorithms to advance the field of machine learning.

[[Paper]](https://www.biorxiv.org/content/10.64898/2026.05.05.723022v1.full.pdf)

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

Supervised by Prof. Laschowski: [https://github.com/DrLaschowski](https://github.com/DrLaschowski)
