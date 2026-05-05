import torch
import torch.nn as nn
import torch.nn.functional as F

class ValenceEmbedder(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=32, activation='relu',
                 use_signed=True, apply_normalization=False):
        """2-layer MLP to embed scalar valence values into vectors."""
        super(ValenceEmbedder, self).__init__()

        self.use_signed = use_signed
        self.apply_normalization = apply_normalization

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, valences):
        if self.use_signed:
            v = valences.float()
        else:
            v = valences.abs().float()

        v = v.unsqueeze(-1)  # (..., 1)

        embeddings = self.mlp(v)

        if self.apply_normalization:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            magnitude_scale = 1.0 + torch.abs(v)
            embeddings = embeddings * magnitude_scale

        return embeddings
