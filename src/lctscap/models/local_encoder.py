"""Module A: Channel-wise Local Encoder.

Converts raw per-channel time-series windows into dense representations via
Conv1d patch embedding, learnable positional encoding, and a Transformer encoder.

Pipeline:
    Input: [B*T*C, 1, L]  (L=500, i.e. 10s @ 50Hz per channel)
    -> Conv1d patch embedding  (kernel=patch_size, stride=patch_size)
    -> Add learnable positional encoding
    -> Transformer encoder (num_layers layers)
    -> Mean pool over patch tokens
    Output: [B*T*C, d_model]
"""

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Conv1d-based patch embedding with learnable positional encoding.

    Splits a 1-D signal of length L into non-overlapping patches of size
    ``patch_size`` using a strided convolution, then adds learnable position
    embeddings.

    Parameters
    ----------
    d_model : int
        Embedding dimension.
    patch_size : int
        Size (and stride) of each patch.
    max_patches : int
        Maximum number of patches (determines the size of the position
        embedding table).
    dropout : float
        Dropout applied after adding positional encoding.
    """

    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 25,
        max_patches: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size

        # Project each patch (1-channel window of length patch_size) to d_model
        self.proj = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Learnable position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B, 1, L]
            Single-channel time-series window.

        Returns
        -------
        Tensor, shape [B, N_patches, d_model]
        """
        # x: [B, 1, L] -> conv -> [B, d_model, N_patches]
        x = self.proj(x)  # [B, d_model, N]
        x = x.transpose(1, 2)  # [B, N, d_model]

        n_patches = x.size(1)
        x = x + self.pos_embed[:, :n_patches, :]
        x = self.dropout(x)
        return x


class LocalEncoder(nn.Module):
    """Channel-wise local encoder.

    Applies ``PatchEmbedding`` followed by a standard Transformer encoder and
    mean-pools the resulting patch tokens into a single vector per channel.

    Parameters
    ----------
    d_model : int
        Model / embedding dimension.
    patch_size : int
        Patch size for the Conv1d embedding.
    num_layers : int
        Number of Transformer encoder layers.
    num_heads : int
        Number of attention heads.
    max_patches : int
        Maximum number of patches (for position embeddings).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        patch_size: int = 25,
        num_layers: int = 4,
        num_heads: int = 8,
        max_patches: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(
            d_model=d_model,
            patch_size=patch_size,
            max_patches=max_patches,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape [B*T*C, 1, L]
            Raw single-channel time-series segments.

        Returns
        -------
        Tensor, shape [B*T*C, d_model]
            Mean-pooled representation per channel.
        """
        # Patch embedding: [B*T*C, 1, L] -> [B*T*C, N_patches, d_model]
        h = self.patch_embed(x)

        # Transformer encoder
        h = self.transformer(h)  # [B*T*C, N_patches, d_model]

        # Mean pool over patch dimension
        out = h.mean(dim=1)  # [B*T*C, d_model]
        return out
