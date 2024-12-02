import torch
import torch.nn as nn


class TransformerVisionLayer(nn.Module):
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
        gamma1=1.0,
        gamma2=1.0,
        mu=0.7,
        beta1=0.9,
        beta2=0.999,
        layerth=0,
        **kargs,
    ):
        super(TransformerVisionLayer, self).__init__()
        # Define gate mechanism based on gate_name
        if gate_name in ["smoe", "smoe-dropout"]:
            gate = CustomNaiveGate_Balance_SMoE
        elif gate_name == "xmoe":
            gate = CustomNaiveGate_Balance_XMoE
        elif gate_name == "stablemoe":
            gate = CustomNaiveGate_Balance_StableMoE
        else:
            raise NotImplementedError(f"Gate '{gate_name}' is not implemented!")

        # Multi-head self-attention
        self.attn = (
            MultiHeadVisionAttention(hidden_size=hidden_size, dropout=dropout, **kargs)
            if s == "s"
            else None
        )

        # Mixture of Experts (MoE) or Feedforward module
        if optimal_policy:
            self.smoe = (
                CustomizedMoEPositionwiseFFOpt(
                    gate=gate,
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
                if g == "g"
                else None
            )
        else:
            self.smoe = (
                CustomizedMoEPositionwiseFFMoM(
                    gate=gate,
                    hidden_size=hidden_size,
                    inner_hidden_size=inner_hidden_size,
                    dropout=dropout,
                    moe_top_k=moe_top_k,
                    gamma1=gamma1,
                    gamma2=gamma2,
                    mu=mu,
                    beta1=beta1,
                    beta2=beta2,
                    layerth=layerth,
                )
                if g == "m"
                else None
            )

        # Standard feedforward network
        self.ff = (
            FeedForwardLayer(
                hidden_size=hidden_size,
                inner_hidden_size=inner_hidden_size,
                dropout=dropout,
            )
            if f == "f"
            else None
        )

        # Normalization layers
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.norm3 = nn.LayerNorm(hidden_size)

        # Flags for conditional execution
        self.use_attn = s == "s"
        self.use_smoe = g in ["g", "m"]
        self.use_ff = f == "f"

    def forward(self, x, cache=None, moment=None, key_pe=None):
        """
        Forward pass for TransformerVisionLayer.
        
        Args:
            x: Input tensor of shape [B, N, H] (B=batch size, N=num patches, H=hidden size).
            cache: Cached input for attention mechanism (if applicable).
            moment: State for MoE layers (if applicable).
            key_pe: Positional encoding tensor.

        Returns:
            x: Updated tensor after layer operations.
            moment: Updated moment state (if applicable).
        """
        if self.use_attn:
            if cache is not None:
                x_combined = torch.cat([cache, x], dim=1)  # Combine cache with current input
            else:
                x_combined = x

            attn_out = self.attn(x, x_combined, x_combined, key_pe)  # Apply attention
            x = self.norm1(x + attn_out)  # Residual connection + LayerNorm

        if self.use_smoe:
            if self.g in ["m", "a"]:
                smoe_out, moment = self.smoe(x, moment)
            else:
                smoe_out = self.smoe(x)
            x = self.norm2(x + smoe_out)  # Residual connection + LayerNorm

        if self.use_ff:
            ff_out = self.ff(x)
            x = self.norm3(x + ff_out)  # Residual connection + LayerNorm

        return x, moment
