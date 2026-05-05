import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ValenceContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embeddings, valences):
        batch_size = embeddings.size(0)
        embeddings_norm = F.normalize(embeddings, dim=1)
        actual_similarity = torch.matmul(embeddings_norm, embeddings_norm.T)
        angles = (valences + 1) * math.pi / 2  # shape: (batch_size,)
        
        angle_diff = angles.unsqueeze(0) - angles.unsqueeze(1)  # (batch_size, batch_size)
        
        target_similarity = torch.cos(angle_diff)
        eye_mask = torch.eye(batch_size, device=embeddings.device).bool()
        
        valid_pairs = ~eye_mask
        loss = F.mse_loss(
            actual_similarity[valid_pairs], 
            target_similarity[valid_pairs]
        )
        
        avg_error = torch.abs(actual_similarity[valid_pairs] - target_similarity[valid_pairs]).mean()
        
        loss_dict = {'contrastive': loss.item(), 'avg_sim_error': avg_error.item()}
        
        return loss, loss_dict
