import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ValenceContrastiveLoss(nn.Module):
    def __init__(self):
        """
        Geometric loss for valence embeddings using angular mapping.
        
        Maps valence ∈ [-1, 1] to angles, then enforces that cosine similarity
        between embeddings matches the cosine of angle differences.
        
        Mapping: valence → angle
        - valence = -1 → angle = π (180°)
        - valence =  0 → angle = π/2 (90°)  
        - valence = +1 → angle = 0 (0°)
        
        This gives natural geometric properties:
        - v1=+1, v2=+1: angle_diff=0 → cos(0)=1 (same direction)
        - v1=+1, v2=-1: angle_diff=π → cos(π)=-1 (opposite direction)
        - v1=+1, v2=0: angle_diff=π/2 → cos(π/2)=0 (orthogonal)
        - v1=+0.5, v2=-0.5: angle_diff=π/2 → cos(π/2)=0 (orthogonal)
        """
        super().__init__()

    def forward(self, embeddings, valences):
        """
        Compute valence contrastive loss.
        """
        batch_size = embeddings.size(0)

        # normalize embeddings for cosine similarity
        embeddings_norm = F.normalize(embeddings, dim=1)

        # compute actual pairwise cosine similarities
        actual_similarity = torch.matmul(embeddings_norm, embeddings_norm.T)

        angles = (valences + 1) * math.pi / 2  # shape: (batch_size,); this gives: v=-1 -> 0, v=0 -> pi/2, v=+1 -> pi
        
        # compute angle differences for all pairs
        angle_diff = angles.unsqueeze(0) - angles.unsqueeze(1)  # (batch_size, batch_size)
        
        # target similarity = cos(angle_diff)
        target_similarity = torch.cos(angle_diff)
        
        # remove self-pairs (diagonal) from loss computation
        eye_mask = torch.eye(batch_size, device=embeddings.device).bool()
        
        # compute MSE loss only on off-diagonal elements
        valid_pairs = ~eye_mask
        loss = F.mse_loss(
            actual_similarity[valid_pairs], 
            target_similarity[valid_pairs]
        )
        
        # for logging: compute average absolute error
        avg_error = torch.abs(actual_similarity[valid_pairs] - target_similarity[valid_pairs]).mean()
        
        loss_dict = {
            'contrastive': loss.item(),
            'avg_sim_error': avg_error.item()
        }
        
        return loss, loss_dict
