"""Channel attention fusion module.

Given per-channel embeddings for each (batch, time-step) pair, learns
soft attention weights over channels via an MLP and produces a single
fused vector per time step.

Pipeline:
    Input:  [B*T, C, d]
    -> MLP(d -> 1) per channel  -> attention logits [B*T, C, 1]
    -> Softmax over C
    -> Weighted sum   -> [B*T, d]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelFusion(nn.Module):
    """Attention-based channel fusion.

    Parameters
    ----------
    d_model : int
        Dimension of channel embeddings.
    num_channels : int
        Number of EEG/TS channels (C).
    """

    def __init__(self, d_model: int, num_channels: int):
        super().__init__()
        self.d_model = d_model
        self.num_channels = num_channels

        # MLP that maps each channel embedding to a scalar attention logit
        self.attn_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B*T, C, d]
            Per-channel embeddings.

        Returns
        -------
        Tensor, shape [B*T, d]
            Fused embedding via channel attention.
        """
        # Compute attention logits per channel
        attn_logits = self.attn_mlp(x)  # [B*T, C, 1]
        attn_weights = F.softmax(attn_logits, dim=1)  # [B*T, C, 1]

        # Weighted sum over channels
        fused = (attn_weights * x).sum(dim=1)  # [B*T, d]
        return fused
