import torch
import torch.nn as nn

class WCELoss(nn.Module):
    def __init__(self, valence_scaling=1.0):
        """
        Weighted CE loss with valence
        """
        super(WCELoss, self).__init__()
        self.valence_scaling = valence_scaling
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, valences):
        """
        Function: L_WCE = -1/N * sum_{i=1}^{N} [1 + Beta * |v_i| * y_i * log(y_i)]
        """
        ce_loss = self.cross_entropy(logits, targets)  # shape: (batch_size, seq_len)
        valence_weights = 1.0 + self.valence_scaling * torch.abs(valences).float()  # shape: (batch_size, seq_len)
        weighted_loss = ce_loss * valence_weights  # shape: (batch_size, seq_len)
        return weighted_loss.mean()