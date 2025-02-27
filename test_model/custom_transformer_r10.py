import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
from custom_layers_r10 import FMoE
from custom_layers_r10 import FMoELinear
from custom_layers_opt import FMoEOpt


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        # import ipdb ipdb.set_trace()
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        # inp torch.Size([16384, 128]), fwd_expert_count shape 16
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count) # torch.Size([16384, 128])
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        moe_top_k=2,
        **kwargs
    ):
        super().__init__(
            num_expert=num_expert, d_model=d_model, moe_top_k=moe_top_k, **kwargs
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        # import ipdb; ipdb.set_trace()
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        # input torch.Size([8, 256, 128])
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model) # self.d_model 128
        # torch.Size([2048, 128])
        output = super().forward(inp) 
        return output.reshape(original_shape) # torch.Size([8, 256, 128])


class FMoETransformerMLPOpt(FMoEOpt):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        moe_top_k=2,
        freq=0.0,
        alpha=0.0,
        act_experts="shuffle",
        g_blance=False,
        opt_blance=False,
        combine_gate=False,
        opt_loss="mse",
        **kwargs
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            moe_top_k=moe_top_k,
            freq=freq,
            alpha=alpha,
            act_experts=act_experts,
            g_blance=g_blance,
            opt_blance=opt_blance,
            combine_gate=combine_gate,
            opt_loss=opt_loss,
            **kwargs
        )
        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        # import ipdb ipdb.set_trace()
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        return output.reshape(original_shape)
