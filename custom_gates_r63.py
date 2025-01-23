import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import numpy as np
from fmoe.gates.base_gate import BaseGate

__all__ = [
    "CustomNaiveGate_Balance_SMoE",
    "CustomNaiveGate_Balance_XMoE",
    "CustomNaiveGate_Balance_StableMoE",
]


class CustomNaiveGate_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model // 2, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None
        self.d_model = d_model
        self.weight_control = nn.Linear(self.d_model // 2, 1)
        # self.weight = nn.Parameter(torch.ones([self.d_model, 1]))
        self.capacity = 2 # 0.5, 1
        self.h = 2

    def set_load_balance(self, gate, gate_top_k_idx):
        # import ipdb; ipdb.set_trace()
        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1].long()
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        # print(f'Balancing for expert cluster 1: {fraction_expert=}')
        # print(f'Balancing for expert cluster 1: {fraction_expert=}')
        prob_expert = score.sum(dim=0) / valid_idx.numel() #* 2 # top2 
        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        # self.loss = loss
        return loss
    
    def forward(self, inp, return_all_scores=False):
        # import ipdb; ipdb.set_trace()
        num_token, embed_dim = inp.shape
        num_heads = self.h
        
        inp_h = inp.reshape(num_token, num_heads, embed_dim // num_heads).reshape(num_token * num_heads, embed_dim // num_heads) 
        weight1 = ((self.weight_control(inp_h)) > 0).to(float)
        weight2 = 1 - weight1
        
        non_zero_idx_1 = weight1.sum(dim=-1) != 0  # Rows with non-zero gate_weight1
        non_zero_idx_2 = weight2.sum(dim=-1) != 0  # Rows with non-zero gate_weight2
        
        inp_token_choice = inp_h[non_zero_idx_1]
        inp_expert_choice = inp_h[non_zero_idx_2]
        
        num_token_expert_choice, _ = inp_expert_choice.shape
        gate_token_choice = self.gate(inp_token_choice)
        gate_expert_choice = self.gate(inp_expert_choice)
        
        expert_top_k = num_token_expert_choice * self.capacity // 16
        
        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val_1, gate_top_k_idx_1 = torch.topk(
                gate_token_choice, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k] 

            gate_top_k_val_2, gate_top_k_idx_2 = torch.topk(
                torch.transpose(gate_expert_choice,0,1), k=expert_top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k] 
            gate_top_k_val_1 = gate_top_k_val_1.view(-1, self.top_k)  # (BxL) x 1 x top_k
            # gate_top_k_val_2 = gate_top_k_val_2.view(-1, self.top_k)  # (BxL) x 1 x top_k
            gate_top_k_val_2 = gate_top_k_val_2.view(-1, expert_top_k)  # (BxL) x 1 x top_k
        """
        ipdb> gate_top_k_val.shape
        torch.Size([2048, 2])
        ipdb> gate_top_k_idx.shape
        torch.Size([2048, 2])
        """
        gate_score_1 = F.softmax(gate_top_k_val_1, dim=-1)
        gate_score_2 = F.softmax(gate_top_k_val_2, dim=-1)
        # import ipdb; ipdb.set_trace()
        if self.g_blance:
            self.loss = self.set_load_balance(gate_token_choice, gate_top_k_idx_1)
        if return_all_scores:
            return gate_top_k_idx, gate_score_1, gate_score_2, gate
        ### modify
        return inp_token_choice, inp_expert_choice, gate_top_k_idx_1, gate_score_1, gate_top_k_idx_2, gate_score_2, non_zero_idx_1, non_zero_idx_2

class CustomNaiveGate_Balance_XMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, 8)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

        self.inp_reduction = torch.nn.Linear(d_model, 8, bias=False)

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate_top_k_idx (tokens_number, top-k)
        # gate_top_k_val (tokens_number, top-k)

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        reduced_inp = self.inp_reduction(inp)
        with torch.no_grad():
            expert_embeddings_norm = self.expert_embeddings.norm(
                p=2.0, dim=1, keepdim=True
            )
            self.expert_embeddings.mul_(1.5 / expert_embeddings_norm)

        gate = self._cosine(reduced_inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores


class CustomNaiveGate_Balance_StableMoE(BaseGate):
    r"""
    Naive Gate StableMoE
    """

    def __init__(self, d_model, num_expert, world_size, top_k=2, g_balance=False):
        super().__init__(num_expert, world_size)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_balance = g_balance
        self.loss = 0.0

        expert_embeddings = torch.empty(num_expert, d_model)
        torch.nn.init.orthogonal_(expert_embeddings, gain=0.32)
        self.register_parameter(
            "expert_embeddings", torch.nn.Parameter(expert_embeddings)
        )

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate / 0.3, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert
        self.loss = loss

    def forward(self, inp, return_all_scores=False):

        gate = self._cosine(inp, self.expert_embeddings)
        gate = self._make_finite(gate)

        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)
        # (BxL) x 1 x top_k

        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_balance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        return gate_top_k_idx, gate_score

    def _cosine(self, mat1, mat2, eps=1e-4):
        assert mat1.dim() == 2
        assert mat2.dim() == 2
        # mat1 = F.normalize(mat1, p=2.0, dim=1, eps=eps)
        mat2 = F.normalize(mat2.float(), p=2.0, dim=1, eps=eps)
        return mat1.float().matmul(mat2.transpose(0, 1)).type_as(mat1)

    def _make_finite(self, scores):
        ok = scores.isfinite()
        if not ok.all():
            # NaNs here can break the assignment algorithm
            scores[~ok] = scores[ok].min()
        return scores
