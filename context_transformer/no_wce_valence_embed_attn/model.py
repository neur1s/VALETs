import sys, os
sys.path.insert(0, os.path.join(os.environ["BASE_DIR"], "include"))
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from ctx_episodic_memory_task import CtxtMemoryTask


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask, src_val=None, tgt_val=None):
        memory = self.encode(src, src_mask)
        return self.decode(memory, src_mask, tgt, tgt_mask, src_val, tgt_val)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, src_val=None, tgt_val=None):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src_val, tgt_val)
    
    def forward_with_weights(self, ctx_ids, obs_ids, src_val, tgt_val, valence_embedder):
        num_obs = self.generator.proj.out_features 
        SOS_TOKEN = num_obs
        CTX_OFFSET = num_obs + 1
        
        max_decoder_pos = self.tgt_embed[1].pe.size(1)
        src = torch.cat([(ctx_ids + CTX_OFFSET).unsqueeze(1), obs_ids], dim=1)
        full_tgt = torch.cat([torch.full((obs_ids.size(0), 1), SOS_TOKEN, device=obs_ids.device),
                               obs_ids[:, :-1]], dim=1)
        tgt = full_tgt[:, -max_decoder_pos:]
        tgt_mask = subsequent_mask(tgt.size(1)).to(obs_ids.device)

        s_v_emb = valence_embedder(src_val)
        t_v_emb = valence_embedder(tgt_val)[:, -max_decoder_pos:]

        x = self.src_embed(src)
        enc_w = None
        for layer in self.encoder.layers:
            x, enc_w = layer(x, None)
        memory = self.encoder.norm(x)

        y = self.tgt_embed(tgt)
        self_w = None
        cross_w = None
        for layer in self.decoder.layers:
            y, self_w, cross_w = layer(y, memory, None, tgt_mask, s_v_emb, t_v_emb)

        out = self.generator(self.decoder.norm(y))
        return out, enc_w, self_w, cross_w


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x)


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x, _ = layer(x, mask) 
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer_fn):
        result = sublayer_fn(self.norm(x))
        if isinstance(result, tuple):
            output, attn = result
            return x + self.dropout(output), attn
        return x + self.dropout(result)

class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x, weights = self.sublayer[0](x, lambda x_norm: self.self_attn(
            x_norm, x_norm, x_norm, mask=mask))
        x = self.sublayer[1](x, self.feed_forward)
        return x, weights
    
class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, src_val=None, tgt_val=None):
        for layer in self.layers:
            x, _, _ = layer(x, memory, src_mask, tgt_mask, src_val, tgt_val)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, src_val, tgt_val):
        x, self_attn_weights = self.sublayer[0](x, lambda x_n: self.self_attn(
            x_n, x_n, x_n, mask=tgt_mask))
        x, cross_attn_weights = self.sublayer[1](x, lambda x_n: self.src_attn(
            x_n, memory, memory, v_q_emb=tgt_val, v_k_emb=src_val, mask=src_mask))
        x = self.sublayer[2](x, self.feed_forward)
        return x, self_attn_weights, cross_attn_weights

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)
    return subsequent_mask == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, valence_dim=64):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.v_proj = nn.Linear(valence_dim, self.d_k, bias=False)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, v_q_emb=None, v_k_emb=None, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        q, k, v = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        if v_q_emb is not None and v_k_emb is not None:
            v_q = self.v_proj(v_q_emb).unsqueeze(1).expand(-1, self.h, -1, -1)
            v_k = self.v_proj(v_k_emb).unsqueeze(1).expand(-1, self.h, -1, -1)
            with torch.no_grad():
                q_std, k_std = q.std(), k.std()
            v_q_clean = (v_q - v_q.mean(dim=-1, keepdim=True)) * q_std
            v_k_clean = (v_k - v_k.mean(dim=-1, keepdim=True)) * k_std
            q = q + (v_q * v_q_clean)
            k = k + (v_k * v_k_clean)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        p_attn = scores.softmax(dim=-1)
        x = torch.matmul(self.dropout(p_attn), v)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), p_attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class ValenceEmbedder(nn.Module):
    def __init__(self, embed_dim=64, hidden_dim=32, activation='relu'):
        super(ValenceEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU() if activation == 'relu' else nn.Tanh(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, valences):
        valences_float = valences.float().unsqueeze(-1)
        embeddings = self.mlp(valences_float)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        magnitude_scale = 1.0 + torch.abs(valences_float)
        return embeddings * magnitude_scale

class ValenceContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.15):
        super().__init__()
        self.temperature = temperature

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
        
        # for logging: compute average absolute error
        avg_error = torch.abs(actual_similarity[valid_pairs] - target_similarity[valid_pairs]).mean()
        
        loss_dict = {
            'contrastive': loss.item(),
            'avg_sim_error': avg_error.item()
        }
        
        return loss, loss_dict

def make_model(num_contexts, num_observations, N=6, d_model=512, d_ff=2048, h=8,
               dropout=0.17, max_len=512, decoder_max_len=None):
    c = copy.deepcopy
    vocab_size = num_observations + 1 + num_contexts  # obs + SOS + ctx tokens

    attn = MultiHeadedAttention(h, d_model, dropout)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)

    # encoder: full episode length positional encoding
    enc_position = PositionalEncoding(d_model, dropout, max_len=max_len)
    # decoder: short-term window positional encoding (caps self-attention range)
    dec_max = decoder_max_len if decoder_max_len is not None else max_len
    dec_position = PositionalEncoding(d_model, dropout, max_len=dec_max)

    # separate embeddings: shared token embedding table, separate PE
    token_embed = Embeddings(d_model, vocab_size)
    src_embed = nn.Sequential(token_embed, enc_position)
    tgt_embed = nn.Sequential(token_embed, dec_position)

    model = EncoderDecoder(
        encoder=Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        decoder=Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        src_embed=src_embed,
        tgt_embed=tgt_embed,
        generator=Generator(d_model, num_observations),  # output over obs only
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model
