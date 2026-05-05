import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size, device=None):
    """Causal lower-triangular mask. True = attend."""
    mask = torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))
    return mask.unsqueeze(0)   # (1, T, T)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        result = sublayer(self.norm(x))
        if isinstance(result, tuple):          # attention returns (out, weights)
            output, _ = result
            return x + self.dropout(output)
        return x + self.dropout(result)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)



class ValenceEmbedder(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, valences):
        v = valences.float().unsqueeze(-1)
        e = self.mlp(v)
        e = F.normalize(e, p=2, dim=-1)
        magnitude = 1.0 + torch.abs(v)
        return e * magnitude


class ValenceContrastiveLoss(nn.Module):
    """Geometric contrastive loss (Eq. 2 in paper): MSE between learned and target cosine similarities."""
    def forward(self, embeddings, valences):
        e_norm = F.normalize(embeddings, p=2, dim=1)
        sim     = torch.matmul(e_norm, e_norm.T)              # (B, B)

        angles      = (valences.float() + 1.0) * math.pi / 2  # (B,)
        angle_diff  = angles.unsqueeze(0) - angles.unsqueeze(1)
        target_sim  = torch.cos(angle_diff)                    # (B, B)

        mask = ~torch.eye(embeddings.size(0), dtype=torch.bool,
                          device=embeddings.device)
        loss = F.mse_loss(sim[mask], target_sim[mask])
        return loss


class WCELoss(nn.Module):
    """Valence-weighted cross-entropy (Eq. 1 in paper): L_WCE = -1/N Σ (1 + β|v_i|) log p̂_{i,c_i}."""
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.ce   = nn.CrossEntropyLoss(reduction='none', ignore_index=-100)

    def forward(self, logits, targets, valences):
        shift_logits = logits[:, :-1].contiguous()
        shift_targets = targets[:, 1:].contiguous()
        shift_valences = valences[:, 1:].contiguous()

        B, T, V = shift_logits.shape
        ce = self.ce(shift_logits.view(-1, V),
                     shift_targets.view(-1)).view(B, T)

        weights = 1.0 + self.beta * shift_valences.abs().float()
        return (ce * weights).mean()

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, valence_dim=64):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.v_proj = nn.Linear(valence_dim, self.d_k, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, v_emb=None, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        B = query.size(0)

        q, k, v = [
            lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        if v_emb is not None:
            vp = self.v_proj(v_emb).unsqueeze(1).expand(-1, self.h, -1, -1)
            with torch.no_grad():
                q_std = q.std().clamp(min=1e-6)
                k_std = k.std().clamp(min=1e-6)
            vp_q = (vp - vp.mean(dim=-1, keepdim=True)) * q_std
            vp_k = (vp - vp.mean(dim=-1, keepdim=True)) * k_std

            q = q + vp_q
            k = k + vp_k

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(~mask, -1e9)
        p_attn = scores.softmax(dim=-1)
        x = torch.matmul(self.dropout(p_attn), v)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        del q, k, v
        return self.linears[-1](x), p_attn

class DecoderOnlyLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask, v_emb=None):
        x = self.sublayer[0](
            x, lambda xn: self.self_attn(xn, xn, xn, v_emb=v_emb, mask=mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x


class DecoderOnlyStack(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask, v_emb=None):
        for layer in self.layers:
            x = layer(x, mask, v_emb=v_emb)
        return self.norm(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)   # (B, T, vocab)

class DecoderOnlyLM(nn.Module):
    def __init__(self, embed, decoder_stack, generator, val_embedder):
        super().__init__()
        self.embed = embed
        self.decoder_stack = decoder_stack
        self.generator = generator
        self.val_embedder = val_embedder

    def forward(self, x, pad_mask=None, valences=None):
        T = x.size(1)
        device = x.device
        causal = subsequent_mask(T, device=device)
        if pad_mask is not None:
            mask = causal & pad_mask.unsqueeze(1)
        else:
            mask = causal
        h = self.embed(x)
        v_emb = None
        if valences is not None:
            v_emb = self.val_embedder(valences)
        h = self.decoder_stack(h, mask, v_emb=v_emb)
        return self.generator(h)


def make_model(vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.2,
               max_len=512, valence_dim=64):
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model, dropout, valence_dim=valence_dim)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    pos_enc = PositionalEncoding(d_model, dropout, max_len)

    layer = DecoderOnlyLayer(d_model, c(attn), c(ff), dropout)
    stack = DecoderOnlyStack(layer, N)
    embed = nn.Sequential(Embeddings(d_model, vocab), c(pos_enc))
    gen = Generator(d_model, vocab)
    val_emb = ValenceEmbedder(embed_dim=valence_dim, hidden_dim=32)

    model = DecoderOnlyLM(embed, stack, gen, val_emb)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model