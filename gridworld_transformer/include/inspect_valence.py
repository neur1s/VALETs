import torch
import numpy as np
import matplotlib.pyplot as plt

# load the similarities data
sim_data = torch.load('valence_embedding_similarities.pt')
# structure: {'train': {'per_sample_sims': [...], 'mean_sim_by_init': {...}, 'pearson_sim_vs_correct': float}, 'valid': {...}}

# extract mean similarities by initial valence
train_mean_sim = sim_data['train']['mean_sim_by_init']  # {valence: mean_similarity}
valid_mean_sim = sim_data['valid']['mean_sim_by_init']

# plot
init_valences = sorted(set(train_mean_sim.keys()) | set(valid_mean_sim.keys()))
train_sims = [train_mean_sim.get(v, np.nan) for v in init_valences]
valid_sims = [valid_mean_sim.get(v, np.nan) for v in init_valences]

plt.figure()
plt.plot(init_valences, train_sims, 'o-', label='train')
plt.plot(init_valences, valid_sims, 's-', label='valid')
plt.xlabel('Initial Valence')
plt.ylabel('Mean Embedding Similarity (init vs target)')
plt.title('Embedding Similarity by Initial Valence')
plt.legend()
plt.savefig('embedding_similarity_by_initial_valence.png', dpi=300, bbox_inches='tight')
plt.show()

# 1. EMBEDDING QUALITY: Do similar valences have similar embeddings?
print("=" * 50)
print("VALENCE EMBEDDING QUALITY ANALYSIS")
print("=" * 50)

# get all unique valences 
all_valences = set()
if sim_data.get('train', {}).get('mean_sim_by_init'):
    all_valences.update(sim_data['train']['mean_sim_by_init'].keys())
if sim_data.get('valid', {}).get('mean_sim_by_init'):
    all_valences.update(sim_data['valid']['mean_sim_by_init'].keys())

valences = sorted(list(all_valences))
print(f"Found valences: {valences}")

# analyze what we have from the similarities data
try:
    print("Analyzing available data from valence_embedding_similarities.pt...")
    
    # extract accuracy data from sim_data if available
    train_acc = {}
    valid_acc = {}
       
    # create visualization 
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # show embedding similarities by initial valence
    ax1.plot(init_valences, train_sims, 'o-', label='Train', linewidth=2)
    ax1.plot(init_valences, valid_sims, 's-', label='Valid', linewidth=2)
    ax1.set_xlabel('Initial Valence')
    ax1.set_ylabel('Mean Embedding Similarity (initial→target)')
    ax1.set_title('EMBEDDING SIMILARITY BY INITIAL VALENCE\n(Valence preservation over time)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    train_accs = train_sims  # Use for summary
    valid_accs = valid_sims
    
    # plot 2: embedding similarity vs valence distance
    ax2.text(0.5, 0.5, 'Load actual model\nto show embedding\nsimilarity matrix', 
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('EMBEDDING SIMILARITY MATRIX\n(Are similar valences similar embeddings?)')
    
    # plot 3: does embedding similarity predict accuracy?
    train_corr = sim_data.get('train', {}).get('pearson_sim_vs_correct', np.nan)
    valid_corr = sim_data.get('valid', {}).get('pearson_sim_vs_correct', np.nan)
    
    corrs = [train_corr, valid_corr]
    labels = ['Train', 'Valid']
    colors = ['blue', 'orange']
    
    bars = ax3.bar(labels, corrs, color=colors, alpha=0.7)
    ax3.set_ylabel('Pearson Correlation')
    ax3.set_title('EMBEDDING SIMILARITY ↔ ACCURACY\n(Do better embeddings = better predictions?)')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # add value labels on bars
    for bar, corr in zip(bars, corrs):
        if not np.isnan(corr):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
                    f'{corr:.3f}', ha='center', va='bottom')
    
    # plot 4: Summary stats
    ax4.axis('off')
    summary_text = f"""
    SUMMARY INSIGHTS:
    
    Similarity varies by valence: {'Yes' if len(set(train_accs)) > 1 else 'No'}
    
    Embedding-accuracy correlation:
       Train: {train_corr:.3f}
       Valid: {valid_corr:.3f}
    
    What this means:
       • High correlation = embeddings capture 
         useful valence information
       • Low correlation = embeddings not 
         helping much for prediction
    
    Next steps:
       • Check embedding similarity matrix
       • Try different embedding dimensions
       • Visualize embedding space (t-SNE/PCA)
    """
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
             fontsize=11, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('valence_embedding_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
except Exception as e:
    print(f"Could not load model: {e}")
    print("Run this after training to get full embedding analysis")