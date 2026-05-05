import torch
import torch.nn as nn

class WCELoss(nn.Module):
    def __init__(self, valence_scaling=1.0, embed_dim=64):
        """
        Weighted CE loss with learnable valence embeddings
        """
        super(WCELoss, self).__init__()
        self.valence_scaling = valence_scaling
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        
        # learnable projection: 64-d embedding -> scalar weight
        self.weight_projection = nn.Linear(embed_dim, 1)

    def forward(self, logits, targets, valence_embeddings):
        """
        Function: L_WCE = -1/N * sum_{i=1}^{N} [1 + Beta * |w(e_i)| * y_i * log(y_i)]
        """
        ce_loss = self.cross_entropy(logits, targets)  # shape: (batch_size,)
        
        # project embedding to scalar weight
        weight_scores = self.weight_projection(valence_embeddings).squeeze(-1)  # (batch_size,)
        valence_weights = 1.0 + self.valence_scaling * torch.abs(weight_scores)  # (batch_size,)
        
        weighted_loss = ce_loss * valence_weights  # (batch_size,)
        return weighted_loss.mean()