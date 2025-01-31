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


def sinkhorn_knopp(B, num_iterations=100, tol=1e-2, device="cuda"):
    """
    Solve the entropy-regularized optimal transport problem using Sinkhorn-Knopp algorithm.
    
    Parameters:
    - B: Cost matrix (T x E) (torch.Tensor)
    - num_iterations: Number of iterations for scaling factors
    - tol: Convergence tolerance
    - device: "cuda" or "cpu"
    
    Returns:
    - A: Optimal transport matrix (T x E) (torch.Tensor)
    """
    B = B.to(device)
    T, E = B.shape
    K = torch.exp(B)  # Gibbs kernel
    u = torch.ones(T, device=device)
    v = torch.ones(E, device=device)

    for _ in range(num_iterations):
        u_new = 1.0 / (K @ v)  # Ensure row sum constraint
        v_new = (T/E) / (K.T @ u_new)  # Ensure column sum constraint
        if torch.norm(u_new - u, p=1) < tol and torch.norm(v_new - v, p=1) < tol:
            break
        u, v = u_new, v_new
    
    A = torch.diag(u) @ K @ torch.diag(v)
    return A

class CustomNaiveGate_Balance_SMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, g_blance=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert)
        self.top_k = top_k
        self.dense_moe_flag = False
        self.g_blance = g_blance
        self.loss = None

    def set_load_balance(self, gate, gate_top_k_idx):
        # gate [1024, 16], idx [1024, 2] <-> mỗi token nó đã chắc chắn chọn 2
        # [n_1, 2], idx, [n_1, 1] ; [n_2, 2], idx [n_2, 1]
        # [n, 2], [n, 2]
        score = F.softmax(gate, dim=-1)
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
        # import ipdb; ipdb.set_trace()
        gate = self.gate(inp)
        gate_sinkhorn = sinkhorn_knopp(gate) # r73
        """
        ipdb> self.gate(inp).shape
        torch.Size([2048, 16])      
        """
        if self.dense_moe_flag:
            gate = torch.ones_like(gate)  # average the importance of all experts
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate, k=self.tot_expert, dim=-1, largest=True, sorted=False
            )
            gate_top_k_val = gate_top_k_val.view(-1, self.tot_expert)
        else:
            gate_top_k_val, gate_top_k_idx = torch.topk(
                gate_sinkhorn, k=self.top_k, dim=-1, largest=True, sorted=False
            )  # [.. x top_k] 
            batch_size = gate_top_k_idx.shape[0]  # Number of rows
            gate_top_k_val = gate[torch.arange(batch_size, device=gate.device).unsqueeze(1), gate_top_k_idx]
            gate_top_k_val = gate_top_k_val.view(-1, self.top_k)  # (BxL) x 1 x top_k
        """
        ipdb> gate_top_k_val.shape
        torch.Size([2048, 2])
        ipdb> gate_top_k_idx.shape
        torch.Size([2048, 2])
        """
        gate_score = F.softmax(gate_top_k_val, dim=-1)
        if self.g_blance:
            self.set_load_balance(gate, gate_top_k_idx)

        if return_all_scores:
            return gate_top_k_idx, gate_score, gate
        ### modify
        return gate_top_k_idx, gate_score


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
