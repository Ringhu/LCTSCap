"""Optional Perceiver resampler and LLM bridge (Phase 4).

The ``PerceiverResampler`` compresses a variable-length sequence of
time-series tokens into a fixed set of latent vectors via cross-attention.
These latents can then be projected into an LLM's embedding space and
prepended as soft-prompt tokens.

``LLMBridge`` wraps the resampler and projection, providing a placeholder
forward method that documents the full cross-attention bridge approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    """Perceiver-style resampler that compresses TS tokens to a fixed number
    of latent vectors via iterative cross-attention.

    Parameters
    ----------
    d_model : int
        Dimension of the input time-series tokens.
    num_latents : int
        Number of learnable latent query vectors.
    num_heads : int
        Number of attention heads in cross-attention.
    num_layers : int
        Number of cross-attention + feed-forward layers.
    """

    def __init__(
        self,
        d_model: int,
        num_latents: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_latents = num_latents
        self.num_layers = num_layers

        # Learnable latent queries
        self.latent_queries = nn.Parameter(
            torch.randn(1, num_latents, d_model) * 0.02
        )

        # Cross-attention layers (latents attend to TS tokens)
        self.cross_attn_layers = nn.ModuleList()
        self.cross_attn_norms_q = nn.ModuleList()
        self.cross_attn_norms_kv = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.ffn_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.cross_attn_layers.append(
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    batch_first=True,
                )
            )
            self.cross_attn_norms_q.append(nn.LayerNorm(d_model))
            self.cross_attn_norms_kv.append(nn.LayerNorm(d_model))

            self.ffn_layers.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
            )
            self.ffn_norms.append(nn.LayerNorm(d_model))

    def forward(self, ts_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ts_tokens : Tensor, shape [B, T, d]
            Time-series token representations.

        Returns
        -------
        Tensor, shape [B, num_latents, d]
            Compressed latent representations.
        """
        B = ts_tokens.size(0)
        latents = self.latent_queries.expand(B, -1, -1)  # [B, num_latents, d]

        for i in range(self.num_layers):
            # Pre-norm cross-attention: latents (query) attend to ts_tokens (key/value)
            q = self.cross_attn_norms_q[i](latents)
            kv = self.cross_attn_norms_kv[i](ts_tokens)

            attn_out, _ = self.cross_attn_layers[i](
                query=q, key=kv, value=kv
            )
            latents = latents + attn_out

            # Pre-norm FFN
            ffn_in = self.ffn_norms[i](latents)
            latents = latents + self.ffn_layers[i](ffn_in)

        return latents


class LLMBridge(nn.Module):
    """Bridge from time-series encoder to a frozen LLM.

    Uses a ``PerceiverResampler`` to compress TS token sequences into a fixed
    set of latent vectors, then projects them to the LLM's hidden dimension.

    In practice, these projected latents are prepended to the LLM's input
    embeddings as soft-prompt tokens.  The LLM itself is loaded and managed
    externally; this module only handles the projection.

    Parameters
    ----------
    d_model : int
        Dimension of TS encoder outputs.
    llm_model_name : str
        Name / identifier of the target LLM (used for documentation and to
        infer the hidden dimension when the LLM is available).
    num_latents : int
        Number of Perceiver latent queries.
    llm_hidden_dim : int
        Hidden dimension of the target LLM. If None, defaults to 4096
        (typical for 7B-class models).
    """

    def __init__(
        self,
        d_model: int,
        llm_model_name: str = "Qwen/Qwen3-1.7B",
        num_latents: int = 32,
        llm_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.llm_model_name = llm_model_name
        self.num_latents = num_latents

        if llm_hidden_dim is None:
            llm_hidden_dim = 4096  # Default for 7B-class models

        self.llm_hidden_dim = llm_hidden_dim

        # Perceiver resampler
        self.resampler = PerceiverResampler(
            d_model=d_model,
            num_latents=num_latents,
        )

        # Project from TS encoder dim to LLM hidden dim
        self.proj = nn.Sequential(
            nn.Linear(d_model, llm_hidden_dim),
            nn.LayerNorm(llm_hidden_dim),
        )

    def forward(
        self,
        ts_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Compress and project TS tokens into LLM-compatible soft prompts.

        The returned tensor should be prepended to the LLM's token embeddings
        before the forward pass of the frozen LLM.  Cross-attention between
        the LLM and the TS tokens is achieved implicitly through these
        soft-prompt embeddings; the resampler's cross-attention captures the
        relevant TS information into a compact representation.

        Integration pattern (pseudocode)::

            soft_prompts = llm_bridge(ts_tokens)       # [B, num_latents, llm_d]
            text_embeds = llm.embed_tokens(input_ids)  # [B, S, llm_d]
            combined = cat([soft_prompts, text_embeds], dim=1)
            llm_output = llm(inputs_embeds=combined, ...)

        Parameters
        ----------
        ts_tokens : Tensor, shape [B, T, d]
            Time-series token representations from the encoder.

        Returns
        -------
        Tensor, shape [B, num_latents, llm_hidden_dim]
            Soft-prompt embeddings ready for prepending to LLM input.
        """
        latents = self.resampler(ts_tokens)  # [B, num_latents, d_model]
        soft_prompts = self.proj(latents)  # [B, num_latents, llm_hidden_dim]
        return soft_prompts
