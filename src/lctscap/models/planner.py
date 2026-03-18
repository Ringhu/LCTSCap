"""Module B: Hierarchical Planner.

The core innovation — processes a sequence of fused time-step embeddings at
two granularities:

1. **Token level**: a Transformer encoder over all T time steps.
2. **Segment level**: pools groups of ``segment_size`` consecutive tokens
   using mean + max concatenation followed by a linear projection.

Pipeline:
    Input:  z [B, T, d]
    -> Token-level Transformer  -> H_token [B, T, d]
    -> SegmentPooling           -> H_seg   [B, S, d]   (S = T // segment_size)
"""

import torch
import torch.nn as nn


class SegmentPooling(nn.Module):
    """Pools every ``segment_size`` tokens via mean+max concatenation + projection.

    Parameters
    ----------
    d_model : int
        Token embedding dimension.
    segment_size : int
        Number of consecutive tokens per segment.
    """

    def __init__(self, d_model: int, segment_size: int = 32):
        super().__init__()
        self.segment_size = segment_size
        # mean and max are concatenated -> 2*d_model, projected back to d_model
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, H_token: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H_token : Tensor, shape [B, T, d]
            Token-level representations.

        Returns
        -------
        Tensor, shape [B, S, d]
            Segment-level representations, where S = T // segment_size.
            If T is not divisible by segment_size, the trailing tokens are
            truncated.
        """
        B, T, d = H_token.shape
        S = T // self.segment_size

        # Truncate to a multiple of segment_size
        H_trunc = H_token[:, : S * self.segment_size, :]  # [B, S*seg, d]
        H_trunc = H_trunc.view(B, S, self.segment_size, d)  # [B, S, seg, d]

        seg_mean = H_trunc.mean(dim=2)  # [B, S, d]
        seg_max = H_trunc.max(dim=2).values  # [B, S, d]

        seg_cat = torch.cat([seg_mean, seg_max], dim=-1)  # [B, S, 2d]
        H_seg = self.proj(seg_cat)  # [B, S, d]
        return H_seg


class HierarchicalPlanner(nn.Module):
    """Hierarchical Planner with token-level Transformer and segment pooling.

    Parameters
    ----------
    d_model : int
        Model dimension.
    num_layers : int
        Number of Transformer encoder layers at the token level.
    num_heads : int
        Number of attention heads.
    segment_size : int
        Segment pooling window.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        segment_size: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )
        self.segment_pool = SegmentPooling(d_model, segment_size)

    def forward(
        self, z: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z : Tensor, shape [B, T, d]
            Fused time-step embeddings.

        Returns
        -------
        H_token : Tensor, shape [B, T, d]
            Token-level representations after the Transformer.
        H_seg : Tensor, shape [B, S, d]
            Segment-level representations.
        """
        H_token = self.token_transformer(z)  # [B, T, d]
        H_seg = self.segment_pool(H_token)  # [B, S, d]
        return H_token, H_seg
