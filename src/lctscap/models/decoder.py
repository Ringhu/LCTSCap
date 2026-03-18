"""Module D: Caption Decoder (small Transformer seq2seq).

A standard Transformer decoder with learned token and position embeddings,
cross-attention to encoder outputs, and a vocabulary projection head.

Supports teacher-forced training via ``forward`` and auto-regressive
greedy/sampling generation via ``generate``.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CaptionDecoder(nn.Module):
    """Transformer caption decoder.

    Parameters
    ----------
    d_model : int
        Model dimension.
    vocab_size : int
        Vocabulary size (default: GPT-2 tokenizer size).
    num_layers : int
        Number of decoder Transformer layers.
    num_heads : int
        Number of attention heads.
    max_seq_len : int
        Maximum decoder sequence length.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        d_model: int = 512,
        vocab_size: int = 50257,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token and position embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_seq_len, d_model) * 0.02
        )
        self.embed_dropout = nn.Dropout(dropout)

        # Project encoder output for cross-attention (in case dims differ)
        self.cross_attn_proj = nn.Linear(d_model, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder_layers = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection to vocabulary
        self.output_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights between token embedding and output head
        self.output_head.weight = self.token_embed.weight

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create an upper-triangular causal mask for the decoder.

        Returns a boolean mask where True means "block attention".
        """
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
        return mask

    def forward(
        self,
        encoder_output: torch.Tensor,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced forward pass.

        Parameters
        ----------
        encoder_output : Tensor, shape [B, S_enc, d]
            Encoder representations (e.g., segment-level from planner).
        target_ids : Tensor, shape [B, S_dec]
            Target token IDs (teacher forcing).

        Returns
        -------
        logits : Tensor, shape [B, S_dec, vocab_size]
        """
        B, S_dec = target_ids.shape

        # Embed target tokens + positional encoding
        tok_emb = self.token_embed(target_ids)  # [B, S_dec, d]
        tok_emb = tok_emb + self.pos_embed[:, :S_dec, :]
        tok_emb = self.embed_dropout(tok_emb)

        # Project encoder output for cross-attention
        memory = self.cross_attn_proj(encoder_output)  # [B, S_enc, d]

        # Causal mask
        tgt_mask = self._make_causal_mask(S_dec, target_ids.device)

        # Decode
        dec_out = self.decoder_layers(
            tgt=tok_emb,
            memory=memory,
            tgt_mask=tgt_mask,
        )  # [B, S_dec, d]

        logits = self.output_head(dec_out)  # [B, S_dec, vocab_size]
        return logits

    @torch.no_grad()
    def generate(
        self,
        encoder_output: torch.Tensor,
        max_len: int = 128,
        temperature: float = 1.0,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
    ) -> torch.Tensor:
        """Auto-regressive greedy / sampling generation.

        Parameters
        ----------
        encoder_output : Tensor, shape [B, S_enc, d]
            Encoder representations.
        max_len : int
            Maximum number of tokens to generate.
        temperature : float
            Sampling temperature; 0 or very small -> greedy.
        bos_token_id : int
            Beginning-of-sequence token ID.
        eos_token_id : int
            End-of-sequence token ID (generation stops here).

        Returns
        -------
        generated : Tensor, shape [B, gen_len]
            Generated token IDs (including BOS).
        """
        B = encoder_output.size(0)
        device = encoder_output.device

        memory = self.cross_attn_proj(encoder_output)

        # Start with BOS token
        generated = torch.full(
            (B, 1), bos_token_id, dtype=torch.long, device=device
        )

        for _ in range(max_len - 1):
            S_cur = generated.size(1)

            # Embed current sequence
            tok_emb = self.token_embed(generated)
            tok_emb = tok_emb + self.pos_embed[:, :S_cur, :]

            # Causal mask
            tgt_mask = self._make_causal_mask(S_cur, device)

            dec_out = self.decoder_layers(
                tgt=tok_emb,
                memory=memory,
                tgt_mask=tgt_mask,
            )

            # Only need last position logits
            next_logits = self.output_head(dec_out[:, -1, :])  # [B, vocab]

            if temperature <= 1e-8:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)  # [B, 1]
            else:
                probs = F.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences have generated EOS
            if (next_token.squeeze(-1) == eos_token_id).all():
                break

        return generated
