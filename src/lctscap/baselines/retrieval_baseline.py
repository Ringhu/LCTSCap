"""Retrieval baseline: dual-encoder model for TS-text cross-modal retrieval.

Uses average pooling of local encoder output for time-series embedding
and a pretrained sentence transformer for text embedding, trained with
InfoNCE loss.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RetrievalBaseline(nn.Module):
    """Dual-encoder retrieval model for time-series <-> text matching.

    Architecture:
      - Time-series branch: a local encoder followed by average pooling
        and a linear projection.
      - Text branch: a pretrained sentence-transformer encoder with a
        linear projection.
      - Trained with symmetric InfoNCE (contrastive) loss.

    Args:
        encoder: a local time-series encoder module that takes [B, C, L]
                 and returns [B, D_enc] (or [B, T, D_enc] which gets pooled).
        text_encoder_name: name of the sentence-transformers model for text.
        d_align: dimension of the shared alignment space.
        d_ts_in: dimension of the encoder output features.
        logit_scale_init: initial value of the learned logit scale (log scale).
    """

    def __init__(
        self,
        encoder: nn.Module,
        text_encoder_name: str = "all-MiniLM-L6-v2",
        d_align: int = 256,
        d_ts_in: int = 512,
        logit_scale_init: float = 2.6593,  # ln(1/0.07) ~ 2.66
    ):
        super().__init__()
        self.encoder = encoder

        # Text encoder from sentence-transformers
        from sentence_transformers import SentenceTransformer

        self.text_model = SentenceTransformer(text_encoder_name)
        d_text = self.text_model.get_sentence_embedding_dimension()
        # Freeze text encoder by default
        for param in self.text_model.parameters():
            param.requires_grad = False

        # Projection heads
        self.ts_proj = nn.Sequential(
            nn.Linear(d_ts_in, d_align),
            nn.ReLU(),
            nn.Linear(d_align, d_align),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(d_text, d_align),
            nn.ReLU(),
            nn.Linear(d_align, d_align),
        )

        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init))

    def encode_ts(self, x: Tensor) -> Tensor:
        """Encode time-series input into the shared alignment space.

        Args:
            x: time-series tensor.  Accepted shapes:
               - [B, C, L]: single-window input
               - [B, T, C, L]: multi-window (long context) input

        Returns:
            Normalized TS embedding of shape [B, d_align].
        """
        original_shape = x.shape

        if x.dim() == 4:
            # [B, T, C, L] -> reshape to [B*T, C, L], encode, then avg pool
            B, T, C, L = x.shape
            x_flat = x.reshape(B * T, C, L)
            z = self.encoder(x_flat)  # [B*T, D] or [B*T, P, D]
            if z.dim() == 3:
                z = z.mean(dim=1)  # average pool patches: [B*T, D]
            z = z.reshape(B, T, -1).mean(dim=1)  # average pool windows: [B, D]
        elif x.dim() == 3:
            z = self.encoder(x)
            if z.dim() == 3:
                z = z.mean(dim=1)
        else:
            raise ValueError(f"Expected 3D or 4D input, got shape {original_shape}")

        z = self.ts_proj(z)
        z = F.normalize(z, dim=-1)
        return z

    def encode_text(self, captions: List[str]) -> Tensor:
        """Encode text captions into the shared alignment space.

        Args:
            captions: list of caption strings.

        Returns:
            Normalized text embedding of shape [B, d_align].
        """
        with torch.no_grad():
            text_features = self.text_model.encode(
                captions, convert_to_tensor=True, show_progress_bar=False
            )
        text_features = text_features.to(next(self.text_proj.parameters()).device)
        z_text = self.text_proj(text_features.float())
        z_text = F.normalize(z_text, dim=-1)
        return z_text

    def forward(
        self, x: Tensor, captions: List[str]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward pass: encode both modalities and return embeddings + scale.

        Args:
            x: time-series tensor [B, C, L] or [B, T, C, L].
            captions: list of caption strings of length B.

        Returns:
            Tuple of (z_ts, z_text, logit_scale) where:
              - z_ts: [B, d_align] normalized TS embeddings
              - z_text: [B, d_align] normalized text embeddings
              - logit_scale: scalar temperature parameter (clamped exp)
        """
        z_ts = self.encode_ts(x)
        z_text = self.encode_text(captions)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        return z_ts, z_text, logit_scale

    def compute_loss(
        self, z_ts: Tensor, z_text: Tensor, logit_scale: Tensor
    ) -> Tensor:
        """Compute symmetric InfoNCE (CLIP-style) loss.

        Args:
            z_ts: [B, D] normalized TS embeddings.
            z_text: [B, D] normalized text embeddings.
            logit_scale: scalar temperature.

        Returns:
            Scalar loss tensor.
        """
        # Cosine similarity scaled by temperature
        logits = logit_scale * z_ts @ z_text.t()
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_t2s = F.cross_entropy(logits, labels)
        loss_s2t = F.cross_entropy(logits.t(), labels)
        return (loss_t2s + loss_s2t) / 2

    def retrieve(
        self,
        query_text: str,
        ts_bank: Tensor,
        top_k: int = 10,
    ) -> Tensor:
        """Retrieve the top-k most similar time-series from a bank.

        Args:
            query_text: the text query.
            ts_bank: precomputed TS embeddings of shape [N, d_align].
            top_k: number of results to return.

        Returns:
            Tensor of top-k indices ranked by similarity (descending).
        """
        z_text = self.encode_text([query_text])  # [1, D]
        ts_bank_norm = F.normalize(ts_bank, dim=-1)
        similarities = (z_text @ ts_bank_norm.t()).squeeze(0)  # [N]
        top_k = min(top_k, similarities.size(0))
        _, indices = similarities.topk(top_k, largest=True)
        return indices
