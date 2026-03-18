"""Module C: Retrieval Aligner (CLIP-like dual encoder).

Projects time-series segment representations and text embeddings into a
shared alignment space using learned linear projections and a learnable
temperature parameter.

Pipeline:
    TS path:   H_seg [B, S, d] -> mean pool -> Linear+LN -> z_ts  [B, d_align]
    Text path: captions         -> TextEncoder -> Linear+LN -> z_text [B, d_align]
    Output:    (z_ts, z_text, logit_scale)
"""

from typing import List, Tuple

import torch
import torch.nn as nn

from .text_encoder import TextEncoder


class RetrievalAligner(nn.Module):
    """CLIP-style dual-encoder aligner for TS-text retrieval.

    Parameters
    ----------
    d_model : int
        Dimension of the time-series encoder outputs.
    d_align : int
        Dimension of the shared alignment space.
    text_model_name : str
        Name of the sentence-transformers model.
    """

    def __init__(
        self,
        d_model: int,
        d_align: int = 256,
        text_model_name: str = "all-MiniLM-L6-v2",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_align = d_align

        # Time-series projection
        self.ts_proj = nn.Sequential(
            nn.Linear(d_model, d_align),
            nn.LayerNorm(d_align),
        )

        # Text encoder (frozen sentence-transformer)
        self.text_encoder = TextEncoder(model_name=text_model_name, freeze=True)
        text_hidden_dim = self.text_encoder.get_hidden_dim()

        # Text projection
        self.text_proj = nn.Sequential(
            nn.Linear(text_hidden_dim, d_align),
            nn.LayerNorm(d_align),
        )

        # Learnable temperature (log scale, init ~ 1/0.07)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / 0.07))
        )

    def encode_ts(self, H_seg: torch.Tensor) -> torch.Tensor:
        """Encode time-series segment representations.

        Parameters
        ----------
        H_seg : Tensor, shape [B, S, d]
            Segment-level representations from the planner.

        Returns
        -------
        Tensor, shape [B, d_align]
            L2-normalised TS embeddings in the alignment space.
        """
        pooled = H_seg.mean(dim=1)  # [B, d]
        z_ts = self.ts_proj(pooled)  # [B, d_align]
        z_ts = nn.functional.normalize(z_ts, dim=-1)
        return z_ts

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text strings.

        Parameters
        ----------
        texts : List[str]
            Captions or descriptions.

        Returns
        -------
        Tensor, shape [B, d_align]
            L2-normalised text embeddings in the alignment space.
        """
        text_emb = self.text_encoder(texts)  # [B, text_hidden_dim]
        # Move to same device as projection layer (text encoder may stay on CPU)
        text_emb = text_emb.to(self.text_proj[0].weight.device)
        z_text = self.text_proj(text_emb)  # [B, d_align]
        z_text = nn.functional.normalize(z_text, dim=-1)
        return z_text

    def forward(
        self,
        H_seg: torch.Tensor,
        captions: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        H_seg : Tensor, shape [B, S, d]
            Segment-level representations.
        captions : List[str]
            Text captions (one per batch element).

        Returns
        -------
        z_ts : Tensor [B, d_align]
        z_text : Tensor [B, d_align]
        logit_scale : Tensor (scalar, clamped)
        """
        z_ts = self.encode_ts(H_seg)
        z_text = self.encode_text(captions)

        # Clamp logit_scale to prevent instability
        logit_scale = self.logit_scale.exp().clamp(max=100.0)

        return z_ts, z_text, logit_scale
