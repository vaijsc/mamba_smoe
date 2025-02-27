import os, sys
import argparse
import math, random
import torch
import torch.nn as nn
import tree
from custom_functions_r42 import prepare_forward, prepare_forward_expert_choice ,ensure_comm
from custom_functions_r42 import MOEScatter, MOEGather
from custom_functions_r42 import AllGather, Slice
from gates import NaiveGate
import torch.nn.functional as F
from fastermoe.config import switch_from_env


def mark_module_parallel_comm(module, comm):
    # # import ipdb ipdb.set_trace()
    r"""
    Mark all parameters in `module` as doing data parallel in `comm`, where
    `comm` may be one of `'world', 'dp', 'none'`.
    """
    for p in module.parameters():
        setattr(p, "dp_comm", comm)


def _fmoe_general_global_forward(
    inp, gate, expert_fn, num_expert, world_size, **kwargs
):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    # import ipdb; ipdb.set_trace()
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, num_expert, world_size)
    topk = 1
    # topk 
    if len(gate.shape) == 2:
        topk = gate.shape[1]

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            torch.div(pos, topk, rounding_mode="floor"),
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )

    x = tree.map_structure(scatter_func, inp)
    # import ipdb ipdb.set_trace()
    # x.shape torch.Size([16384, 128])
    # fwd_expert_count torch.Size([16]) 
    # tensor([1101, 1544, 1141,  725,  483, 1337,  787,  868, 1427, 1304, 1118,  904,
    #    970,  896,  837,  942])
    """
        expert_fn
        <bound method FMoE.expert_fn of CustomizedMoEPositionwiseFF(
        (gate): CustomNaiveGate_Balance_SMoE(
            (gate): Linear(in_features=128, out_features=16, bias=True)
        )
        (experts): _Expert(
            (htoh4): FMoELinear(num_expert=16, in_features=128, out_features=128, bias=True, rank=0)
            (h4toh): FMoELinear(num_expert=16, in_features=128, out_features=128, bias=True, rank=0)
            (activation): Sequential(
            (0): ReLU()
            (1): Dropout(p=0.7, inplace=False)
            )
        )
        (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.7, inplace=False)
        )>
    """
    x = expert_fn(x, fwd_expert_count)
    # torch.Size([16384, 128])
    out_batch_size = tree.flatten(inp)[0].shape[0]
    if len(gate.shape) == 2:
        out_batch_size *= gate.shape[1]

    def gather_func(tensor):
        return MOEGather.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            out_batch_size,
            world_size,
        )
    
    outp = tree.map_structure(gather_func, x)
    
    """
    ipdb> fwd_expert_count
    tensor([213, 402, 235, 292, 216, 176, 206, 236, 268, 284, 205, 205, 447, 153,
    297, 261])
    ipdb> x.shape
    torch.Size([4096, 128])
    ipdb> outp.shape
    torch.Size([4096, 128])
    """
    return outp

def _fmoe_expert_choice_general_global_forward(
    inp, gate, expert_fn, num_expert, world_size, **kwargs
):
    r"""
    A private function that performs the following steps to complete the MoE
    computation.
    * Count the number of tokens from each worker to each expert.
    * Send the features to their target position so that input features to each
    expert are contiguous in memory.
    * Perform the forward computation of the experts using `expert_fn`
    * Gather the output features of experts back, and reorder them as sentences.
    Intermediate results like expert counts are hidden from users by this
    function.
    """
    # print(f"gate in general global forward {gate.shape}")
    # num_token = inp.shape[-2]
    # print(f"{num_expert=}")
    # print(f"{world_size=}")
    # print(f"{expert_fn=}")
    # print(f"{gate=}")
    # token_counts = torch.bincount(gate.flatten())
    # print(f"{max(token_counts)=}")
    # print(f"{inp.shape=}")
    # import ipdb; ipdb.set_trace()
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward_expert_choice(gate, num_expert, world_size)
    # tmp = pos[local_expert_count[7]:local_expert_count[8]]
    # tmp = torch.sort(tmp)
    # print(f"{tmp=} {pos.shape=} {max(pos)=}")
    # print(f"{local_expert_count=} {local_expert_count.shape=} {local_expert_count.is_contiguous()=}")
    # print(f"{global_expert_count=} {global_expert_count.shape=} {global_expert_count.is_contiguous()=}")
    # print(f"{fwd_expert_count=} {fwd_expert_count.shape=} {fwd_expert_count.is_contiguous()=}")
    # print(f"{fwd_batch_size=}")
    # haha = gate[0]
    # print(f"{haha=}")

    def scatter_func(tensor):
        return MOEScatter.apply(
            tensor,
            pos,
            local_expert_count,
            global_expert_count,
            fwd_batch_size,
            world_size,
        )
    x = tree.map_structure(scatter_func, inp)
    # print(f"{x.shape=}")
    # print(f"siuuuu {x=}")
    # import ipdb ipdb.set_trace()
    # x.shape torch.Size([16384, 128])
    # fwd_expert_count torch.Size([16]) 
    # tensor([1101, 1544, 1141,  725,  483, 1337,  787,  868, 1427, 1304, 1118,  904,
    #    970,  896,  837,  942])
    """
        expert_fn
        <bound method FMoE.expert_fn of CustomizedMoEPositionwiseFF(
        (gate): CustomNaiveGate_Balance_SMoE(
            (gate): Linear(in_features=128, out_features=16, bias=True)
        )
        (experts): _Expert(
            (htoh4): FMoELinear(num_expert=16, in_features=128, out_features=128, bias=True, rank=0)
            (h4toh): FMoELinear(num_expert=16, in_features=128, out_features=128, bias=True, rank=0)
            (activation): Sequential(
            (0): ReLU()
            (1): Dropout(p=0.7, inplace=False)
            )
        )
        (layer_norm): LayerNorm((128,), eps=1e-05, elementwise_affine=True)
        (dropout): Dropout(p=0.7, inplace=False)
        )>
    """
    x = expert_fn(x, fwd_expert_count)

    return x


fmoe_faster_schedule = False
if switch_from_env("FMOE_FASTER_SCHEDULE_ENABLE", False):
    fmoe_faster_schedule = True
    from .fastermoe.schedule import _fmoe_general_global_forward


class FMoE(nn.Module):
    r"""
    A general moe implementation that supports an arbitrary module as the
    expert.
    * `num_expert` stands for the number of experts on **each** worker.
    * `world_size` stands for the total number of workers that contains
    different experts.
    * `slice_group` can be a torch's communication group, indicating that
    specific model parallel is applied across the group, and workers in the
    group hold the same copy of input feature, and requires the same copy of
    the output. For each worker, FMoE only computes the output of a certain
    slice of the input batch, and will all-gather the outputs after
    computation.
    * `top_k` stands for the number of experts each token is going to.
    * `gate` is a gate class which can found in `fmoe.gates`.
    * `expert` can be specified as a module class, it is used to generate
    `num_expert` expert modules.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        world_size=1,
        mp_group=None,  # being deprecated
        slice_group=None,
        moe_group=None,
        moe_top_k=2,
        gate=NaiveGate,
        expert=None,
        gate_hook=None,
        mask=None,
        mask_dict=None,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.d_model = d_model
        self.world_size = world_size

        self.slice_group = slice_group
        if mp_group is not None:
            print("[Warning] mp_group is being deprecated")
            self.slice_group = mp_group
        if self.slice_group is None:
            self.slice_size = 1
            self.slice_rank = 0
        else:
            self.slice_size = self.slice_group.size()
            self.slice_rank = self.slice_group.rank()

        self.top_k = moe_top_k
        if type(expert) is list:
            self.experts = nn.ModuleList([e(d_model) for e in expert])
            self.experts_fused = False
            self.num_expert = num_expert = len(expert)
        elif expert is not None:
            self.experts = nn.ModuleList([expert(d_model) for _ in range(num_expert)])
            self.experts_fused = False
        else:
            self.experts_fused = True

        self.gate = gate(d_model, num_expert, world_size, moe_top_k)
        self.gate_hook = gate_hook
        self.mask = mask
        self.mask_dict = mask_dict
        self.moe_group = moe_group
        
        # self.weights = nn.Linear(self.d_model, self.d_model)
        # self.weight_in = nn.Parameter(torch.ones([self.d_model, self.d_model]))
        self.weight_out = nn.Parameter(torch.ones([self.d_model, self.d_model]))

        # self.weight = nn.Parameter(torch.ones([self.d_model, 1]))
        # self.weights = nn.Linear(self.d_model, 1)
        
    def expert_fn(self, inp, fwd_expert_count):
        r"""
        The default expert function which either calls the experts as a whole
        or as separate experts.
        """
        # import ipdb; ipdb.set_trace()
        if self.experts_fused:
            return self.experts(inp, fwd_expert_count)
                
        if isinstance(fwd_expert_count, torch.Tensor):
            fwd_expert_count = fwd_expert_count.cpu().numpy()
        outputs = []
        base_idx = 0
        # Get the expert range from kwargs
        start_idx = getattr(self, 'current_expert_range', (0, self.num_expert))[0]
        end_idx = getattr(self, 'current_expert_range', (0, self.num_expert))[1]
        for i in range(start_idx, end_idx):
            batch_size = fwd_expert_count[i - start_idx]
            inp_slice = inp[base_idx : base_idx + batch_size]
            outputs.append(self.experts[i](inp_slice))
            base_idx += batch_size
        return torch.cat(outputs, dim=0)

    def mark_parallel_comm(self, expert_dp_comm="none"):
        # # import ipdb ipdb.set_trace()
        r"""
        Automatically mark the data parallel comms of the parameters within the
        module. This can be typically called at the end of the __init__ function
        in child classes.
        """
        if self.experts is not None:
            comm = expert_dp_comm
            if isinstance(self.experts, list):
                for e in self.experts:
                    mark_module_parallel_comm(e, comm)
            else:
                mark_module_parallel_comm(self.experts, comm)
        mark_module_parallel_comm(self.gate, "gate")

    def forward(self, moe_inp):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """

        """
        ipdb> moe_inp.shape
        torch.Size([2048, 128])
        
        """
        
        num_token, _ = moe_inp.shape
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        ) 
        
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:

            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
        if self.slice_size > 1:

            def slice_func(tensor):
                return Slice.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_inp = tree.map_structure(slice_func, moe_inp)
        
        # import ipdb; ipdb.set_trace()
        """
        ipdb> moe_inp.shape
        torch.Size([2048, 128])
        """
        # moe_inp = torch.matmul(moe_inp, self.weight_in)
        # import ipdb; ipdb.set_trace()
        moe_inp_1, moe_inp_2, gate_top_k_idx_1, gate_score_1, gate_top_k_idx_2, gate_score_2 = self.gate(moe_inp)
        """
        ipdb> gate_top_k_idx.shape
        torch.Size([2048, 2])
        ipdb> gate_score.shape
        torch.Size([2048, 2])
        """
        if hasattr(self.gate, "dynamic_top_k"):
            self.top_k = self.gate.dynamic_top_k

        if self.gate_hook is not None:
            self.gate_hook(gate_top_k_idx_1, gate_score_1, None)
            self.gate_hook(gate_top_k_idx_2, gate_score_2, None)

        # delete masked tensors
        if self.mask is not None and self.mask_dict is not None:
            # TODO: to fix
            def delete_mask_func(tensor):
                # to: (BxL') x d_model
                tensor = tensor[mask == 0, :]
                return tensor

            mask = self.mask.view(-1)
            moe_inp = tree.map_structure(delete_mask_func, moe_inp)
            gate_top_k_idx_1 = gate_top_k_idx_1[mask == 0, :]
            gate_top_k_idx_2 = gate_top_k_idx_2[mask == 0, :]
            
        # self.current_expert_range = (0, 8)
        fwd_1 = _fmoe_general_global_forward(
            moe_inp_1,
            gate_top_k_idx_1,
            self.expert_fn,
            self.num_expert,
            self.world_size,
            experts=self.experts,
        )
        # self.current_expert_range = (8, 16)
        fwd_2 = _fmoe_expert_choice_general_global_forward(
            moe_inp_2,
            gate_top_k_idx_2,
            self.expert_fn,
            self.num_expert,
            self.world_size,
            experts=self.experts,
        )

        # recover deleted tensors
        if self.mask is not None and self.mask_dict is not None:

            def recover_func(tensor):
                # to: (BxL') x top_k x dim
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                # to: (BxL) x top_k x d_model
                x = torch.zeros(
                    mask.shape[0],
                    self.top_k,
                    dim,
                    device=tensor.device,
                    dtype=tensor.dtype,
                )
                # recover
                x[mask == 0] = tensor
                for k, v in self.mask_dict.items():
                    x[mask == k] = v
                return x
            """
            ipdb> moe_outp.shape
            torch.Size([2048, 2, 128])
            """
            moe_outp_1 = tree.map_structure(recover_func, fwd_1)
            moe_outp_2 = tree.map_structure(recover_func, fwd_2)
        else:
            
            def view_func(tensor):
                dim = tensor.shape[-1]
                tensor = tensor.view(-1, self.top_k, dim)
                return tensor

            moe_outp_1 = tree.map_structure(view_func, fwd_1)
            # moe_outp_2 = tree.map_structure(view_func, fwd_2)
            moe_outp_2 = fwd_2
            """
            ipdb> fwd.shape
            torch.Size([4096, 128])
            """
        gate_score_1 = gate_score_1.view(-1, 1, self.top_k)
        # gate_score_2 = gate_score_2.view(-1, 1, self.top_k)
        """
        ipdb> gate_score.shape
        torch.Size([2048, 1, 2])
        """
        def bmm_func(gate_score, tensor):
            # import ipdb; ipdb.set_trace()
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        def expert_combine_func(num_token, gate, gate_score, tensor):
            """
            num_token: Total number of tokens (n)
            gate: Indices of tokens assigned to experts (shape [e, k])
            gate_score: Gating weights (shape [e, k])
            tensor: Expert outputs (shape [e, k, d])
            Returns:
            out: Combined output (shape [n, d])
            """
            # import ipdb; ipdb.set_trace()
            d = tensor.shape[-1]
            flat_I = gate.flatten()  # Shape [e*k]
            flat_G = gate_score.flatten()  # Shape [e*k]
            flat_tensor = tensor.view(-1, d)  # Shape [e*k, d]

            # Initialize output tensor
            out = torch.zeros((num_token, d), device=tensor.device)  # Shape [n, d]

            # Use scatter_add with weighted tensor
            weighted_tensor = flat_G.unsqueeze(-1) * flat_tensor  # Shape [e*k, d]
            out = out.index_add_(0, flat_I, weighted_tensor)  # Scatter-add with index_add_

            return out
        
        # import ipdb; ipdb.set_trace()
        moe_outp_1 = tree.map_structure(bmm_func, gate_score_1, moe_outp_1)
        # moe_outp_2 = tree.map_structure(bmm_func, gate_score_2, moe_outp_2)
        moe_outp_2 = tree.map_structure(expert_combine_func, num_token, gate_top_k_idx_2, gate_score_2, moe_outp_2)
        # moe_outp = torch.concat([moe_outp_1, moe_outp_2], dim = -1)
        # moe_outp = self.weights(moe_outp)
        # moe_outp = torch.sigmoid(self.weights) * moe_outp_1 + (1 - torch.sigmoid(self.weights)) * moe_outp_2
        # g1 = torch.sigmoid(torch.matmul(moe_inp, self.weight)).to(moe_outp_1.device)
        # moe_outp = g1 * moe_outp_1 + (1 - g1) * moe_outp_2
        moe_outp = torch.cat([moe_outp_1, moe_outp_2], dim=-1)
        # moe_outp = self.weights(moe_outp)
        moe_outp = torch.matmul(moe_outp, self.weight_out)
        if self.slice_size > 1:

            def all_gather_func(tensor):
                return AllGather.apply(
                    tensor, self.slice_rank, self.slice_size, self.slice_group
                )

            moe_outp = tree.map_structure(all_gather_func, moe_outp)
        # import ipdb; ipdb.set_trace()
        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        ) # 8, 256, 
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        """
        ipdb> moe_outp_batch_size
        [2048]        
        ipdb> moe_outp.shape
        torch.Size([2048, 128])
        """
        return moe_outp


##############################################################################

import torch
import torch.nn as nn
import math
import fmoe_cuda
from torch.autograd import Function


class MOELinear(Function):
    r"""
    Computes linear operators within one GPU on different experts simutaneously.
    """

    @staticmethod
    def forward(ctx, global_input_buf, fwd_expert_count, weight, bias=None):
        global_output_buf = fmoe_cuda.linear_forward(
            global_input_buf, fwd_expert_count, weight, bias
        )
        # # import ipdb ipdb.set_trace()
        variables = (global_input_buf, fwd_expert_count, weight, bias)
        ctx.save_for_backward(*variables)
        return global_output_buf

    @staticmethod
    def backward(ctx, grad_out):
        (input_buf, fwd_expert_count, weight, bias) = ctx.saved_tensors
        grad_inp_buf, grad_weight, grad_bias = fmoe_cuda.linear_backward(
            grad_out, input_buf, fwd_expert_count, weight, bias
        )
        # # import ipdb ipdb.set_trace()
        if not torch.is_tensor(bias):
            grad_bias = None

        return grad_inp_buf, None, grad_weight, grad_bias


class FMoELinear(nn.Module):
    r"""
    A linear layer that contains multiple experts.
    As multiple experts can be placed on the same worker, the computation can be
    performed in parallel to increase the performance.
    The FMoELinear module provides such function.
    """

    def __init__(
        self,
        num_expert: int,
        in_feat: int,
        out_feat: int,
        bias: bool = True,
        rank: int = 0,
    ):
        super().__init__()
        self.num_expert = num_expert
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.rank = rank
        self.weight = nn.Parameter(torch.Tensor(num_expert, out_feat, in_feat))
        if bias:
            self.bias = nn.Parameter(torch.zeros(num_expert, out_feat))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def forward(self, inp, fwd_expert_count):
        r"""
        Call MOE function
        """
        # # import ipdb ipdb.set_trace()
        x = MOELinear.apply(inp, fwd_expert_count, self.weight, self.bias)
        return x

    def extra_repr(self) -> str:
        # # import ipdb ipdb.set_trace()
        return "num_expert={}, in_features={}, \
        out_features={}, bias={}, rank={}".format(
            self.num_expert,
            self.in_feat,
            self.out_feat,
            self.bias is not None,
            self.rank,
        )

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        # bias is left to zero, similar as megatron
        # # import ipdb ipdb.set_trace()
        # smoe: FMoELinear(num_expert=16, in_features=128, out_features=128, bias=True, rank=0)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))


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
        self.expert_mask = None

    def set_expert_mask(self, mask):
        """Set mask for active/inactive experts"""
        self.expert_mask = mask

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shrink back to h.
        """
        if self.expert_mask is not None:
            # Apply expert mask to the expert counts
            masked_count = fwd_expert_count * self.expert_mask
        else:
            masked_count = fwd_expert_count

        x = self.htoh4(inp, masked_count)
        x = self.activation(x)
        x = self.h4toh(x, masked_count)
        return x
