import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from custom_transformer import FMoETransformerMLP, FMoETransformerMLPOpt
from custom_gates import *


# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span
def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """

    def __init__(self, hidden_size, attn_span, dropout, adapt_span_params, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size  # size of a single head
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params["adapt_span_enabled"]
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(
                attn_span=attn_span, **adapt_span_params, **kargs
            )

    def forward(self, query, key, value, key_pe):
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe
            )
            
        ############## RPC ###############
        # B, M, H = query.shape
        # mu = M*H/8/query.norm(p=1,dim=[-1,-2],keepdim=True)
        # lambd = 2.0
        # l = torch.zeros_like(query).to(torch.device("cuda"))
        # y = torch.zeros_like(query).to(torch.device("cuda"))
        # for i in range(0,1):
        #     s = (query-l+y/(mu+1e-8)) 
        #     s_less = s.le(-lambd/(mu+1e-8)).int()
        #     s_more = s.ge(lambd/(mu+1e-8)).int()
        #     s = (s-lambd/(mu+1e-8))*s_more + (s+lambd/(mu+1e-8))*s_less
        #     query = query-s-y/(mu+1e-8)
        #     # compute attention from context
        #     # B x M (dest) x (M+L) (src)
        #     # l = torch.matmul(query2, query2.transpose(-1, -2))
        #     # print(l.shape)
        #     # print(value.shape)
        #     # l = _unskew(l)
        #     # l = l / math.sqrt(self.hidden_size)  # B x M X L_pos
        #     # l = F.softmax(l, dim=-1)
        #     # l = torch.matmul(l, value)
        #     # y = y+mu*(query2-l-s)
        
        # # s = (query-l+y/(mu+1e-8)) 
        # # s_less = s.le(-lambd/(mu+1e-8)).int()
        # # s_more = s.ge(lambd/(mu+1e-8)).int()
        # # s = (s-lambd/(mu+1e-8))*s_more + (s+lambd/(mu+1e-8))*s_less
        # # query = query-s-y/(mu+1e-8)

        #################################

        # compute attention from context
        # B x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2))
        attn_cont = _unskew(attn_cont)  # B x M x L

        # compute the effect of position embedding
        attn_pos = torch.matmul(query, key_pe)  # B x M x L_pos
        attn = attn_cont + attn_pos

        attn = attn / math.sqrt(self.hidden_size)  # B x M X L_pos
        attn = F.softmax(attn, dim=-1)

        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos

        attn_cont = _skew(attn, 0)  # B x M X (L+M)
        out = torch.matmul(attn_cont, value)  # B x M x H
        return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x):
        K = self.nb_heads
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query)
        query = self.head_reshape(query)
        value = self.proj_val(value)
        value = self.head_reshape(value)
        key = self.proj_key(key)
        key = self.head_reshape(key)

        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class CustomizedMoEPositionwiseFF(FMoETransformerMLP):
    def __init__(
        self,
        gate,
        hidden_size,
        inner_hidden_size,
        dropout,
        pre_lnorm=False,
        moe_num_expert=16,
        moe_top_k=2,
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert,
            d_model=hidden_size,
            d_hidden=inner_hidden_size,
            moe_top_k=moe_top_k,
            activation=activation,
            gate=gate,
        )
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class CustomizedMoEPositionwiseFFOpt(FMoETransformerMLPOpt):
    def __init__(
        self,
        gate,
        hidden_size,
        inner_hidden_size,
        dropout,
        pre_lnorm=False,
        moe_num_expert=16,
        moe_top_k=2,
        freq=0.0,
        alpha=0.0,
        act_experts="shuffle",
        g_blance=False,
        opt_blance=False,
        combine_gate=False,
        opt_loss="mse",
    ):
        activation = nn.Sequential(nn.ReLU(), nn.Dropout(dropout))
        super().__init__(
            num_expert=moe_num_expert,
            d_model=hidden_size,
            d_hidden=inner_hidden_size,
            moe_top_k=moe_top_k,
            activation=activation,
            gate=gate,
            freq=freq,
            alpha=alpha,
            act_experts=act_experts,
            g_blance=g_blance,
            opt_blance=opt_blance,
            combine_gate=combine_gate,
            opt_loss=opt_loss,
        )
        self.pre_lnorm = pre_lnorm
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        if self.pre_lnorm:
            ##### layer normalization + positionwise feed-forward
            core_out = super().forward(self.layer_norm(inp))
            core_out = self.dropout(core_out)

            ##### residual connection
            output = core_out + inp
        else:
            ##### positionwise feed-forward
            core_out = super().forward(inp)
            core_out = self.dropout(core_out)

            ##### residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class TransformerSeqLayer(nn.Module):
    def __init__(
        self,
        hidden_size,
        inner_hidden_size,
        dropout,
        s,
        g,
        f,
        gate_name,
        optimal_policy,
        moe_top_k,
        freq,
        alpha,
        act_experts,
        g_blance,
        opt_blance,
        combine_gate,
        opt_loss,
        **kargs,
    ):
        nn.Module.__init__(self)
        if gate_name in ["smoe", "smoe-dropout"]:
            gate = CustomNaiveGate_Balance_SMoE
        elif gate_name == "xmoe":
            gate = CustomNaiveGate_Balance_XMoE
        elif gate_name == "stablemoe":
            gate = CustomNaiveGate_Balance_StableMoE
        else:
            print(f"{gate_name} has not been implemented yet!")

        self.attn = (
            MultiHeadSeqAttention(hidden_size=hidden_size, dropout=dropout, **kargs)
            if s is "s"
            else None
        )
        if optimal_policy:
            self.smoe = (
                CustomizedMoEPositionwiseFFOpt(
                    gate,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    freq=freq,
                    alpha=alpha,
                    act_experts=act_experts,
                    g_blance=g_blance,
                    opt_blance=opt_blance,
                    combine_gate=combine_gate,
                    opt_loss=opt_loss,
                )
                if g is "g"
                else None
            )
        else:
            self.smoe = (
                CustomizedMoEPositionwiseFF(
                    gate,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                )
                if g is "g"
                else None
            )

        self.ff = (
            FeedForwardLayer(
                hidden_size=hidden_size,
                inner_hidden_size=inner_hidden_size,
                dropout=dropout,
            )
            if f is "f"
            else None
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        self.use_attn = s == "s"
        self.use_smoe = g == "g"
        self.use_ff = f == "f"

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        if self.use_attn:
            h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
            attn_out = self.attn(h, h_all, h_all, key_pe)
            h = self.norm1(h + attn_out)  # B x M x H
        if self.use_smoe:
            smoe_out = self.smoe(h)
            h = self.norm2(h + smoe_out)  # B x M x H
        if self.use_ff:
            ff_out = self.ff(h)
            h = self.norm3(h + ff_out)  # B x M x H
        return h


class TransformerSeq(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size,
        inner_hidden_size,
        nb_heads,
        nb_layers,
        attn_span,
        architecture,
        base_arch,
        gate_name,
        optimal_policy,
        dropout,
        moe_top_k,
        freq,
        alpha,
        act_experts,
        g_blance,
        opt_blance,
        combine_gate,
        opt_loss,
        num_classes,
        **kargs,
    ):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, hidden_size)
        self.out_emb = nn.Linear(hidden_size, hidden_size)  # hidden_size
        # position embeddings
        self.key_pe = nn.Parameter(torch.randn(1, hidden_size // nb_heads, attn_span))
        # finetuning parts
        self.project_head = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size),
            # nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, num_classes),
        )

        arch = architecture
        print(arch)
        self.attn_layer_count = arch.count("s")
        self.layers = nn.ModuleList()
        if base_arch == "transformer":
            self.layers.extend(
                TransformerSeqLayer(
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    s=arch[2 * i],
                    g=arch[2 * i + 1],
                    f=None,
                    gate_name=gate_name,
                    optimal_policy=optimal_policy,
                    nb_heads=nb_heads,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    freq=freq,
                    alpha=alpha,
                    act_experts=act_experts,
                    g_blance=g_blance,
                    opt_blance=opt_blance,
                    combine_gate=combine_gate,
                    opt_loss=opt_loss,
                    attn_span=attn_span,
                    **kargs,
                )
                for i in range(nb_layers)
            )
        elif base_arch == "glam":
            for i in range(nb_layers):
                self.layers.extend(
                    [
                        TransformerSeqLayer(
                            hidden_size=hidden_size,
                            inner_hidden_size=inner_hidden_size,
                            s=arch[2 * i],
                            g=arch[2 * i + 1],
                            f=None,
                            gate_name=gate_name,
                            optimal_policy=optimal_policy,
                            nb_heads=nb_heads,
                            dropout=dropout,
                            moe_top_k=moe_top_k,
                            freq=freq,
                            alpha=alpha,
                            act_experts=act_experts,
                            g_blance=g_blance,
                            opt_blance=opt_blance,
                            combine_gate=combine_gate,
                            opt_loss=opt_loss,
                            attn_span=attn_span,
                            **kargs,
                        ),
                        TransformerSeqLayer(
                            hidden_size=hidden_size,
                            inner_hidden_size=inner_hidden_size,
                            s=arch[2 * (i + 1)],
                            g=None,
                            f=arch[2 * (i + 1) + 1],
                            gate_name=gate_name,
                            optimal_policy=optimal_policy,
                            nb_heads=nb_heads,
                            dropout=dropout,
                            moe_top_k=moe_top_k,
                            freq=freq,
                            alpha=alpha,
                            act_experts=act_experts,
                            g_blance=g_blance,
                            opt_blance=opt_blance,
                            combine_gate=combine_gate,
                            opt_loss=opt_loss,
                            attn_span=attn_span,
                            **kargs,
                        ),
                    ]
                )

        else:
            raise RuntimeError(
                "wrong type of base architecture - must be 'transformer' or 'glam'"
            )

    def forward(self, x, h_cache):
        # x size = B x M
        block_size = x.size(1)
        h = self.in_emb(x)  # B x M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):
            if layer.use_attn:
                cache_size = layer.attn.attn.get_cache_size()
                if cache_size > block_size:
                    # print(h_cache[l][:, -cache_size + block_size :, :].shape)
                    # print(h.shape)
                    h_cache_next_l = torch.cat(
                        [h_cache[l][:, -cache_size + block_size :, :], h], dim=1
                    ).detach()
                else:
                    h_cache_next_l = h[:, -cache_size:, :].detach()
                h_cache_next.append(h_cache_next_l)
                h = layer(h, h_cache[l], self.key_pe)  # B x M x H
            else:
                h = layer(h, [], self.key_pe)

        # out = F.log_softmax(self.out_emb(h), dim=-1)
        # project to custom data
        # pdb.set_trace()
        out = self.out_emb(h)
        pre_logits = self.project_head(out[:, -1, :])
        pre_logits = F.softmax(pre_logits, dim=-1)
        return pre_logits, h_cache_next
