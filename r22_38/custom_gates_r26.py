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
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None
        self.d_model = d_model
        self.weights = nn.Linear(self.d_model, 1)

    def set_load_balance(self, gate, gate_top_k_idx):

        score = F.softmax(gate, dim=-1)
        valid_idx = gate_top_k_idx[gate_top_k_idx > -1]
        fraction_expert = (
            torch.scatter_add(
                torch.zeros(self.tot_expert // 2, device=valid_idx.device),
                0,
                valid_idx,
                torch.ones_like(valid_idx, dtype=torch.float),
            )
            / valid_idx.numel()
        )
        prob_expert = score.sum(dim=0) / valid_idx.numel()

        loss = (fraction_expert * prob_expert).sum() * self.tot_expert / 2
        return loss
            
    def forward(self, inp, return_all_scores=False):
        # import ipdb; ipdb.set_trace()
        device = inp.device
        gate_weights = torch.sigmoid(self.weights(inp)).to(device)
        
        gate_weight1 = (gate_weights > 0.5).float()
        gate_weight2 = 1 - gate_weight1

        # Identify non-zero rows
        non_zero_idx_1 = gate_weight1.sum(dim=-1) != 0  # Rows with non-zero gate_weight1
        non_zero_idx_2 = gate_weight2.sum(dim=-1) != 0  # Rows with non-zero gate_weight2

        # Filter out zero rows for computation
        inp_1 = inp[non_zero_idx_1]
        inp_2 = inp[non_zero_idx_2]

        # Filter out gate weights
        gate = self.gate(inp) # [1024, 16]
        gate_1 = gate[non_zero_idx_1][:, : self.tot_expert//2]
        gate_2 = gate[non_zero_idx_2][:, self.tot_expert//2 :]
        
        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val_1, gate_top_k_idx_1 = torch.topk(
                gate_1, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val_2, gate_top_k_idx_2 = torch.topk(
                gate_2, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val_1 = gate_top_k_val_1.view(-1, self.tot_expert)
            gate_top_k_val_2 = gate_top_k_val_2.view(-1, self.tot_expert)
        else:
            gate_top_k_val_1, gate_top_k_idx_1 = torch.topk(
                gate_1, k=self.top_k, dim=-1, largest=True, sorted=False
            ) 
            gate_top_k_val_2, gate_top_k_idx_2 = torch.topk(
                gate_2, k=self.top_k, dim=-1, largest=True, sorted=False
            ) 
            gate_top_k_val_1 = gate_top_k_val_1.view(-1, self.top_k)  # (BxL) x 1 x top_k
            gate_top_k_val_2 = gate_top_k_val_2.view(-1, self.top_k)  # (BxL) x 1 x top_k
            
        gate_score_1 = F.softmax(gate_top_k_val_1, dim=-1)
        gate_score_2 = F.softmax(gate_top_k_val_2, dim=-1)
        if self.g_blance:
            loss_1 = self.set_load_balance(gate_1, gate_top_k_idx_1)
            loss_2 = self.set_load_balance(gate_2, gate_top_k_idx_2)
            self.loss = loss_1 + loss_2
        if return_all_scores:
            return inp_1, inp_2, gate_top_k_idx_1, gate_score_1, gate_top_k_idx_2, gate_score_2, non_zero_idx_1, non_zero_idx_2
        ### modify
        return inp_1, inp_2, gate_top_k_idx_1, gate_score_1, gate_top_k_idx_2, gate_score_2, non_zero_idx_1, non_zero_idx_2


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
