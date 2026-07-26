import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../include"))
from valence_embedding import ValenceEmbedder
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils.generic import ModelOutput

@dataclass # automatically generates __init__(), __repr__(), and __eq__() methods for the class below
class TransfoXLModelOutput(ModelOutput):
    last_hidden_state: torch.FloatTensor
    logits: torch.FloatTensor = None
    mems: List[torch.FloatTensor] = None
    v_mems: torch.FloatTensor = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    ffn_activations: Optional[Tuple[torch.FloatTensor]] = None
    grid_activations: Optional[Tuple[torch.FloatTensor]] = None
    rnn_hidden: Optional[Tuple[torch.FloatTensor]] = None

class NMDA(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(NMDA, self).__init__()
        self.alpha = alpha
        if alpha <= 0:
            self.a = None
        else:
            self.a = math.log(self.alpha)
        self.beta = beta 

    def forward(self, x):
        if self.a is None:
            return x
        else:
            return x * torch.sigmoid(self.beta * x - self.a)
        
class PositionwiseFF(nn.Module): # the feed-forward network with two linear layers
    def __init__(self, d_model, d_inner, dropout, pre_lnorm=False, ffn_act_ftn = 'gelu', alpha=1.0, beta=1.0):
        super(PositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        act_nn = NMDA(alpha, beta)

        self.fc1 = nn.Sequential(nn.Linear(d_model, d_inner, bias=False), act_nn)
        self.fc2 = nn.Sequential(nn.Linear(d_inner, d_model, bias=False), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm
    
    def forward(self, inp): # inp = input
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            ffn_hid = self.fc1(self.layer_norm(inp))
            core_out = self.fc2(ffn_hid)
            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            ffn_hid = self.fc1(inp)
            core_out = self.fc2(ffn_hid)
            # residual connection + layer normalization 
            output = self.layer_norm(inp + core_out)

        return [output, ffn_hid]


class RelMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0, tgt_len = None, ext_len = None, mem_len = None, pre_lnorm=False, **kwargs):
        
        super(RelMultiHeadAttn, self).__init__()
        self.n_head = n_head
        self.d_model = d_model # [seq_len x batch_size x d_model]
        self.d_head = d_head
        self.dropout = dropout
        
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False) # features_in = d_model, features_out = [seq_len, batch_size, 3*n_head*d_head] (output dims)
        # 3 because we need to generate three separate matrices--queries, keys, and values all at once

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)

        # take concatenated attention outputs from all heads and project back to the original model dimension d_model because residual connections require same dimension as input
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5) 

        self.pre_lnorm = pre_lnorm

    def _parallelogram_mask(self, h, w, left=False):
        """ Create a parallelogram attention mask for relative positional encoding. 
        - different relative distances get different attention patterns;
        - parallelogram shape handles the interaction between current segment and memory;
        - left=False flips the mask vertically for different relative position calculations"""
        mask = torch.ones((h, w)).byte() # start with all ones

        # upper-left triangulation
        m = min(h, w) # to take the top left mxm square
        mask[:m, :m] = torch.triu(mask[:m, :m]) # keep upper triangular part as 1s, set lower triangular part to 0s
        mask[-m:, -m:] = torch.tril(mask[-m:, -m:]) # keep lower triangular part as 1s, set upper triangle to 0s

        if left:
            return mask
        else:
            return mask.flip(0)

    def _rel_shift(self, x, zero_triu=False):
        """ Implements the relative shift operation.
        Input: x is relative positional attention scores, has shape [qlen x klen x batch_size x n_head].
        As a result, position 0 sees realtive positions: [0, -1, -2, -3, ...].
        Position 1 sees realtive positions: [1, 0, -1, -2, ...], etc."""

        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype) # creates padding with shape [qlen x 1 x batch_size x n_head]; adds one column of zeros
        x_padded = torch.cat([zero_pad, x], dim=1) # original x: [qlen x klen x batch_size x n_head], after concat: [qlen x klen+1 x batch_size x n_head]
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:]) # reshape to [klen+1 x qlen x batch_size x n_head]

        x = x_padded[1:].view_as(x) # remove the first row and reshape back to original x [qlen x klen x batch_size x n_head]

        if zero_triu: # optional: upper triangular masking
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None] # xeros out the upper triangle

        return x
    
    def forward(self, w, r, attn_mask=None, mems=None):
        raise NotImplementedError
    
class RelLearnableMultiHeadAttn(RelMultiHeadAttn): # relative position encoding where the positional embeddings (r_emb, r_w_bias, r_bias) are fully learned params rather than fixed sindusoidal functions
    def __init__(self, *args, **kwargs):
        super(RelLearnableMultiHeadAttn, self).__init__(*args, **kwargs)
        # ABLATION: the valence signal no longer becomes a d_head vector added
        # into Q/K. It enters only as the rank-1 pairwise term beta * v_i * v_j
        # on the attention logits, with one learnable beta per head (learnable
        # rather than fixed so a null result cannot be blamed on an untuned beta).
        self.beta = nn.Parameter(torch.ones(self.n_head))

    def forward(self, w, r_emb, r_w_bias, r_bias, v_curr, v_mems=None, attn_mask=None, mems=None):
        # r_emb are relative embeddings: [klen, n_head, d_head], used for term B
        # r_w_bias is content bias: [n_head, d_head], used for term C
        # r_bias is position bias: [klen, n_head], used for term D

        qlen, bsz = w.size(0), w.size(1) # w: [seq_len x batch_size x d_model]

        if mems is not None:
            cat = torch.cat([mems, w], 0) # concatenate memory with current input
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(cat))
            else:
                w_heads = self.qkv_net(cat)
            
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1) # split into three chunks along the last dimension which is 3*n_head*d_head, such that each resulting matrix has dim: seq_len x batch_size x n_head*d_head
            w_head_q = w_head_q[-qlen:] # keep only queries for current sequence
        else:
            if self.pre_lnorm:
                w_heads = self.qkv_net(self.layer_norm(w))
            else:
                w_heads = self.qkv_net(w)
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        # ABLATION: v_curr / v_mems now carry the RAW scalar valences
        if v_curr is None:
            v_curr = torch.zeros((qlen, bsz), device=w.device, dtype=w.dtype)

        # ABLATION: align the query valences to qlen 
        if v_curr.size(0) < qlen:
            padding = torch.zeros((qlen - v_curr.size(0), bsz),
                                  device=v_curr.device, dtype=v_curr.dtype)
            v_q_raw = torch.cat([padding, v_curr], dim=0)
        else:
            v_q_raw = v_curr[-qlen:]

        # ABLATION: key valences span the memory plus the current segment.
        v_cat = torch.cat([v_mems, v_curr], 0) if v_mems is not None else v_curr
        if v_cat.size(0) < klen:
            padding = torch.zeros((klen - v_cat.size(0), bsz),
                                  device=v_cat.device, dtype=v_cat.dtype)
            v_k_raw = torch.cat([padding, v_cat], dim=0)
        else:
            v_k_raw = v_cat[-klen:]

        w_head_q = w_head_q.reshape(w_head_q.size(0), bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.reshape(w_head_k.size(0), bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.reshape(w_head_v.size(0), bsz, self.n_head, self.d_head)

        q_total = w_head_q
        k_total = w_head_k

        # ABLATION: vv[i, j, b] = v_i * v_j
        vv = v_q_raw.float().unsqueeze(1) * v_k_raw.float().unsqueeze(0)  # [qlen, klen, bsz] v_i * v_j
        vmod = self.beta.view(1, 1, 1, -1) * vv.unsqueeze(-1)             # [qlen, klen, bsz, n_head] Beta * v_i * v_j

        # position alignment 
        if klen > r_emb.size(0):
            r_emb = torch.cat([r_emb[0:1].expand(klen - r_emb.size(0), -1, -1), r_emb], 0)
            r_bias = torch.cat([r_bias[0:1].expand(klen - r_bias.size(0), -1), r_bias], 0)
        else:
            r_emb, r_bias = r_emb[-klen:], r_bias[-klen:]

        # attention score

        rw_head_q = q_total + r_w_bias[None, None, :, :] # qlen x bsz x n_head x d_head

        # four-term attention; matmul with specified input and output dims:
        AC = torch.einsum("ibnd, jbnd->ijbn", (rw_head_q, k_total)) # content-to-content attention: qlen x klen x bsz x n_head 
        B_ = torch.einsum("ibnd, jnd->ijbn", (w_head_q, r_emb)) # query-position: qlen x klen x bsz x n_head 
        D_ = r_bias[None, :, None] # position bias: 1 x klen x 1 x n_head 
        BD = self._rel_shift(B_ + D_) # content-position interaction, not incorporating valence embedding since valence and position don't interact

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD # final attention score calculation
        attn_score.mul_(self.scale)

        # ABLATION (additive bias): attn_score <- attn_score + beta * v_i * v_j,
        attn_score = attn_score + vmod

        mask_value = torch.finfo(attn_score.dtype).min # get the smallest value for the tensor's data type, like float32

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            attn_mask = attn_mask == 1

            # broadcast to ensure alignment with attn_score which is [qlen, klen, batch_size, n_head]:
            if attn_mask.dim() == 2:
                attn_score = (attn_score.float().masked_fill(attn_mask[None, :, :, None], mask_value).type_as(attn_score)) # attn_mask[None, :, :, None] adds new dimensions at the beginning and end to [qlen, klen] (original dims of attn_mask)
            elif attn_mask.dim() == 3:
                attn_score = (attn_score.float().masked_fill(attn_mask[:, :, :, None], mask_value).type_as(attn_score)) # attn_mask[:, :, :, None] adds new dimension at the end 
            
        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1) # masked position will have probability of ~0
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head # flatten heads
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            outputs = [w + attn_out]
        else:
            # residual connection + layer normalization
            outputs = [self.layer_norm(w + attn_out)]
        outputs.append(attn_prob)

        return outputs

class RelLearnableDecoderLayer(nn.Module):
    """ single layer decoder that uses learnable relative positional encodings"""
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, **kwargs):
        super(RelLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, **kwargs)
        
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout, pre_lnorm=kwargs.get("pre_lnorm"), alpha=kwargs.get("alpha"), beta=kwargs.get("beta"))

   
    def forward(self, dec_inp, r_emb, r_w_bias, r_bias, v_curr, v_mems=None, dec_attn_mask=None, mems=None):
        # pass valence through attn module
        attn_outputs = self.dec_attn(dec_inp, r_emb, r_w_bias, r_bias, v_curr=v_curr, v_mems=v_mems, attn_mask=dec_attn_mask, mems=mems)
        ff_output, core_output = self.pos_ff(attn_outputs[0])
        outputs = [ff_output, core_output] + attn_outputs[1:] # ff_output is final processed output of FFN, core_output is intermediate hidden activations

        return outputs
    
class TransfoXL(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_token = config.vocab_size
        self.d_embed = config.d_embed
        self.d_model = config.d_model
        self.n_head = config.n_head
        self.d_head = config.d_head
        self.d_inner = config.d_inner

        self.drop = nn.Dropout(config.dropout)

        self.n_layer = config.n_layer

        self.mem_len = config.mem_len
        self.tgt_len = config.tgt_len
        self.ext_len = config.ext_len
        self.max_klen = self.tgt_len + self.ext_len + self.mem_len

        #self.attn_type = config.attn_type

        self.word_emb = nn.Embedding(config.vocab_size + 1, config.d_embed)
        self.grid_emb = nn.Sequential(nn.Linear(config.d_rnn, config.d_embed), nn.ReLU(inplace=True))
        self.rew_emb = nn.Embedding(2, config.d_embed)
        self.valid_seq_emb = nn.Embedding(3, config.d_embed)
        self.ln_f = nn.LayerNorm(config.d_model)

        self.layers = nn.ModuleList()
        #if self.attn_type == 1:
        for _ in range(self.n_layer):
            self.layers.append(RelLearnableDecoderLayer(
                self.n_head, self.d_model, self.d_head, self.d_inner, config.dropout, 
                tgt_len = self.tgt_len, ext_len = self.ext_len, mem_len = self.mem_len,
                dropatt = config.dropatt, pre_lnorm = config.pre_lnorm, alpha = config.alpha, beta = config.beta))
            
        self.same_length = config.same_length
        self.clamp_len = config.clamp_len
        self._create_params()

        # initialize weights and apply final processing
        self.init_weights()

    def _create_params(self):
        self.r_emb = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head, self.d_head))
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_layer, self.n_head, self.d_head))
        self.r_bias = nn.Parameter(torch.Tensor(self.n_layer, self.max_klen, self.n_head))
    
    def init_weights(self):
        self.apply(self._init_weights)
    
    def _init_weight(self, weight):
        if self.config.init == "uniform":
            nn.init.uniform_(weight, -self.config.init_range, self.config.init_range)
        elif self.config.init == "normal":
            nn.init.normal_(weight, 0.0, self.config.init_std)
    
    def _init_bias(self, bias):
        nn.init.constant_(bias, 0.0)
    
    def _init_weights(self, m):
        """ Initialize the weights."""
        classname = m.__class__.__name__
        if classname.find("Linear") != -1:
            if hasattr(m, "weight") and m.weight is not None:
                self._init_weight(m.weight)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        elif classname.find("Embedding") != -1:
            if hasattr(m, "weight"):
                nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find("LayerNorm") != -1:
            if hasattr(m, "weight"):
                nn.init.constant_(m.weight, 1.0)
            if hasattr(m, "bias") and m.bias is not None:
                self._init_bias(m.bias)
        else:
            if hasattr(m, "r_emb"):
                self._init_weight(m.r_emb)
            if hasattr(m, "r_w_bias"):
                self._init_weight(m.r_w_bias)
            if hasattr(m, "r_r_bias"):
                self._init_weight(m.r_r_bias)
            if hasattr(m, "r_bias"):
                self._init_bias(m.r_bias)

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def reset_memory_length(self, mem_len):
        self.mem_len = mem_len

    def init_mems(self):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype = param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None
    
    def _update_mems(self, hids, mems, mlen, qlen): # caching the hidden states from previous segments to be used as memory for the current segment
        # allows the model to process sequences longer than its fixed context window
        # does not deal with None
        if mems is None:
            return None
        
        # mems is not None
        assert len(hids) == len(mems), "len(hids) != len(mems)"
        
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())
        return new_mems
    
    def _update_v_mems(self, v_hids, v_mems, mlen, qlen):
        """
        ABLATION: caches RAW scalar valences rather than 64-D embeddings.
        v_hids: [qlen, bsz]
        v_mems: [mlen, bsz]
        """
        if v_hids is None:
            return v_mems
        if v_mems is None:
            return v_hids.detach()
        with torch.no_grad():
            cat = torch.cat([v_mems, v_hids], dim=0) # concat old valemces with current valence embeddings
            end_idx = mlen + max(0, qlen - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
        return cat[beg_idx:end_idx].detach()
    
    def forward(
            self,
            grid_seq: torch.FloatTensor,
            input_ids: Optional[torch.LongTensor] = None,
            mems: Optional[List[torch.FloatTensor]] = None,
            v_mems: Optional[torch.FloatTensor] = None,
            v_curr: Optional[torch.FloatTensor] = None,
            output_attentions: Optional[bool] = True,
            output_hidden_states: Optional[bool] = True,
            exclude_last: Optional[bool] = True,
            rew_seq: Optional[torch.LongTensor] = None,
            val_seq: Optional[torch.LongTensor] = None
    ) -> TransfoXLModelOutput:
        output_attentions = (output_attentions if output_attentions is not None else self.config.output_attentions)
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)

        if mems is None:
            mems = self.init_mems()
        
        if exclude_last:
            cls_token = torch.ones_like(input_ids[:, :1]) * self.n_token
            input_ids = torch.cat([input_ids, cls_token], dim=1)
        
        bsz, qlen = input_ids.size()
        obs_emb = self.word_emb(input_ids)
        grid_seq = self.grid_emb(grid_seq) # grid_seq is the output from the RNN
        _grid_seq = grid_seq.transpose(0, 1).contiguous()
        if not exclude_last:
            rew_emb = self.rew_emb(rew_seq)
            seq_emb = self.valid_seq_emb(val_seq)
            obs_emb = obs_emb + rew_emb + seq_emb
        obs_emb = obs_emb.transpose(0, 1).contiguous()
        word_emb = torch.cat([_grid_seq, obs_emb], dim=-1) # positional embedding + word embedding before being passed to the transformer layers

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        if self.same_length:
            all_ones = word_emb.new_ones(qlen, klen)
            mask_len = klen - self.mem_len
            if mask_len > 0:
                mask_shift_len = qlen - mask_len
            else:
                mask_shift_len = qlen
            dec_attn_mask = (torch.triu(all_ones, 1 + mlen) + torch.tril(all_ones, -mask_shift_len)).byte()[:, :, None] # -1
        else:
            dec_attn_mask = torch.triu(word_emb.new_ones(qlen, klen), diagonal=1 + mlen).byte()[:, :, None]

        hids = []
        ffn_acts = [] if output_attentions else None
        attentions = [] if output_attentions else None

        core_out = self.drop(word_emb)
        hids.append(core_out)
        for i, layer in enumerate(self.layers):
            if self.clamp_len > 0:
                r_emb = self.r_emb[i][-self.clamp_len :]
                r_bias = self.r_bias[i][-self.clamp_len :]
            else:
                r_emb, r_bias = self.r_emb[i], self.r_bias[i]

            mems_i = None if mems is None else mems[i]
            layer_outputs = layer(core_out, r_emb, self.r_w_bias[i], r_bias, 
                                  v_curr=v_curr, v_mems=v_mems, 
                                  dec_attn_mask=dec_attn_mask, mems = mems_i)
            core_out = layer_outputs[0]
            hids.append(core_out)
            if output_attentions:
                ffn_acts.append(layer_outputs[1])
                attentions.append(layer_outputs[2])
        
        if exclude_last:
            qlen = qlen - 1
        new_mems = self._update_mems(hids, mems, mlen, qlen)
        new_v_mems = self._update_v_mems(v_curr, v_mems, mlen, qlen)

        if output_attentions:
            # transpose to library standard shape [bsz, n_heads, query_seq_len, key_seq_len]
            attentions = tuple(t.permute(2, 3, 0, 1).contiguous() for t in attentions)
            ffn_acts = tuple(t.transpose(0, 1).contiguous() for t in ffn_acts)

        # we transpose back here to shape [bsz, len, hidden_dim]
        core_out = self.ln_f(core_out)
        core_out = core_out.transpose(0, 1).contiguous()

        return TransfoXLModelOutput(last_hidden_state=core_out, mems=new_mems, v_mems=new_v_mems, attentions=attentions, ffn_activations=ffn_acts, grid_activations=grid_seq)


class xlTEM(nn.Module): # integration of a transformer with an RNN
    def __init__(self, config):
        super(xlTEM, self).__init__()
        self.transformer = TransfoXL(config)
        self.rnn = RNN(config)
        self.classifier = nn.Linear(config.d_model, config.vocab_size)
        nn.init.normal_(self.classifier.weight, 0.0, config.init_std)
        nn.init.constant_(self.classifier.bias, 0.0)
        self.valence_embedder = ValenceEmbedder(
            embed_dim=config.valence_embed_dim,
            hidden_dim=config.valence_hidden_dim,
            activation=config.valence_activation
        )
        

    def forward(self, act, obs, prev_hidden, mems, valences=None, v_curr=None, v_mems=None):
        seq_len = act.shape[1]
        g = [prev_hidden] # the first hidden state is the prev_hidden state passed to the function
        for n in range(seq_len):
            prev_hidden = self.rnn(act[:, n], prev_hidden) # generating new hidden states, added to a sequence g
            g.append(prev_hidden)
        g = torch.stack(g, dim=1)
        outputs = self.transformer(g, obs, mems, v_curr=v_curr, v_mems=v_mems)
        hidden = outputs.last_hidden_state[:, -1] # output of the transformer's last layer at the last time step
        outputs.logits = self.classifier(hidden) # prediction of the next token
        outputs.rnn_hidden = prev_hidden.detach()
        return outputs # contains both the last token prediction and the updated hidden state of the RNN

_act_ftns = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}

class RNN(nn.Module): # RNN definition for state updates; next hidden state is a function of the previous hidden state (prev_hidden) and the current action (a)
    def __init__(self, config):
        super(RNN, self).__init__()
        self.d_rnn = config.d_rnn
        self.n_a = config.n_a
        self.side_len = config.side_len
        Wa = torch.zeros(self.n_a, self.d_rnn, self.d_rnn).float()
        self.weight = nn.Parameter(Wa)
        self.act_ftn = _act_ftns[config.rnn_act_ftn]

    def forward(self, a, prev_hidden):
        out = self.act_ftn(prev_hidden + torch.einsum("bi,bij->bj", prev_hidden, self.weight[a])) # update rule; self.weight is indexed by the action a -> the action determines the transformation applied to the previous hidden state
        return out
    
    def init_hidden(self, pos):
        return torch.randn((pos.shape[0], self.d_rnn)).to(pos.device) / math.sqrt(self.d_rnn)