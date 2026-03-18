"""Wrapper around sentence-transformers for text encoding.

Provides a ``nn.Module``-compatible interface that encodes a list of strings
into dense vectors using a pre-trained sentence-transformer model.
"""

from typing import List

import torch
import torch.nn as nn


class TextEncoder(nn.Module):
    """Sentence-transformer text encoder wrapper.

    Parameters
    ----------
    model_name : str
        Name of the sentence-transformers model to load.
    freeze : bool
        If True, all parameters of the underlying model are frozen.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        freeze: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self._freeze = freeze

        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(model_name, local_files_only=True)
            # Retrieve hidden dim from the model
            self._hidden_dim = self.model.get_sentence_embedding_dimension()
        except Exception as exc:
            # Fallback for missing dependency or offline model loading failure.
            import warnings

            warnings.warn(
                f"Falling back to dummy text encoder for '{model_name}': {exc}"
            )
            self.model = None
            self._hidden_dim = 384

        if freeze and self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = False

    def get_hidden_dim(self) -> int:
        """Return the output embedding dimension."""
        return self._hidden_dim

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of strings into dense vectors.

        Parameters
        ----------
        texts : List[str]
            Input text strings (one per batch element).

        Returns
        -------
        Tensor, shape [B, hidden_dim]
            Sentence embeddings.
        """
        if self.model is not None:
            # sentence-transformers encode returns numpy by default
            embeddings = self.model.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            # Clone to detach from inference_mode context so the
            # downstream projection can be tracked by autograd.
            embeddings = embeddings.clone().detach().float()
            return embeddings
        else:
            # Dummy fallback: hash-based deterministic embeddings
            device = self._get_device()
            out = torch.zeros(len(texts), self._hidden_dim, device=device)
            for i, text in enumerate(texts):
                # Simple deterministic encoding based on character values
                gen = torch.Generator()
                gen.manual_seed(hash(text) % (2**31))
                out[i] = torch.randn(self._hidden_dim, generator=gen)
            return out

    def _get_device(self) -> torch.device:
        """Infer the device this module should produce outputs on."""
        if self.model is not None:
            try:
                return next(self.model.parameters()).device
            except StopIteration:
                return torch.device("cpu")
        return torch.device("cpu")
