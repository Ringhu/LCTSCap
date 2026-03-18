"""Full LCTSCap model combining all modules with ablation flags.

Assembles the complete pipeline:
    local_encoder -> channel_fusion -> planner -> event_head -> aligner -> decoder

Each component can be ablated via configuration flags.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .local_encoder import LocalEncoder
from .channel_fusion import ChannelFusion
from .planner import HierarchicalPlanner
from .event_head import EventProposalHead
from .aligner import RetrievalAligner
from .decoder import CaptionDecoder


@dataclass
class ModelConfig:
    """Configuration for LCTSCapModel.

    Parameters
    ----------
    d_model : int
        Model dimension throughout the pipeline.
    num_channels : int
        Number of input channels (e.g., EEG channels).
    patch_size : int
        Patch size for the local encoder.
    local_encoder_layers : int
        Number of Transformer layers in the local encoder.
    local_encoder_heads : int
        Attention heads in the local encoder.
    planner_layers : int
        Number of Transformer layers in the planner.
    planner_heads : int
        Attention heads in the planner.
    segment_size : int
        Segment pooling window in the planner.
    decoder_layers : int
        Number of Transformer layers in the decoder.
    decoder_heads : int
        Attention heads in the decoder.
    vocab_size : int
        Vocabulary size.
    max_seq_len : int
        Maximum decoder sequence length.
    n_event_types : int
        Number of event categories.
    max_events : int
        Maximum event proposals per sample.
    d_align : int
        Alignment space dimension for the retrieval aligner.
    text_model_name : str
        Sentence-transformer model name.
    dropout : float
        Global dropout rate.
    signal_length : int
        Length of each channel's signal (L).
    max_patches : int
        Maximum patches for position embedding.

    Ablation flags
    --------------
    no_hierarchy : bool
        If True, skip the hierarchical planner (use raw fused embeddings).
    no_event : bool
        If True, skip the event proposal head.
    no_align : bool
        If True, skip the retrieval aligner.
    """

    d_model: int = 512
    num_channels: int = 64
    patch_size: int = 25
    local_encoder_layers: int = 4
    local_encoder_heads: int = 8
    planner_layers: int = 4
    planner_heads: int = 8
    segment_size: int = 32
    decoder_layers: int = 6
    decoder_heads: int = 8
    vocab_size: int = 50257
    max_seq_len: int = 256
    n_event_types: int = 20
    max_events: int = 16
    d_align: int = 256
    text_model_name: str = "all-MiniLM-L6-v2"
    dropout: float = 0.1
    signal_length: int = 500
    max_patches: int = 20

    # Ablation flags
    no_hierarchy: bool = False
    no_event: bool = False
    no_align: bool = False


class LCTSCapModel(nn.Module):
    """Full LCTSCap model with ablation support.

    Parameters
    ----------
    config : ModelConfig or dict
        Model configuration.  If a dict, it will be converted to ModelConfig.
    """

    def __init__(self, config: ModelConfig | dict | None = None):
        super().__init__()

        if config is None:
            config = ModelConfig()
        elif isinstance(config, dict):
            config = ModelConfig(**config)
        self.config = config

        # --- Module A: Local Encoder ---
        self.local_encoder = LocalEncoder(
            d_model=config.d_model,
            patch_size=config.patch_size,
            num_layers=config.local_encoder_layers,
            num_heads=config.local_encoder_heads,
            max_patches=config.max_patches,
            dropout=config.dropout,
        )

        # --- Channel Fusion ---
        self.channel_fusion = ChannelFusion(
            d_model=config.d_model,
            num_channels=config.num_channels,
        )

        # --- Module B: Hierarchical Planner (optional) ---
        if not config.no_hierarchy:
            self.planner = HierarchicalPlanner(
                d_model=config.d_model,
                num_layers=config.planner_layers,
                num_heads=config.planner_heads,
                segment_size=config.segment_size,
                dropout=config.dropout,
            )
        else:
            self.planner = None

        # --- Event Head (optional) ---
        if not config.no_event:
            self.event_head = EventProposalHead(
                d_model=config.d_model,
                n_event_types=config.n_event_types,
                max_events=config.max_events,
            )
        else:
            self.event_head = None

        # --- Module C: Retrieval Aligner (optional) ---
        if not config.no_align:
            self.aligner = RetrievalAligner(
                d_model=config.d_model,
                d_align=config.d_align,
                text_model_name=config.text_model_name,
            )
        else:
            self.aligner = None

        # --- Module D: Caption Decoder ---
        self.decoder = CaptionDecoder(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """Encode time-series input without decoding.

        Parameters
        ----------
        x : Tensor, shape [B, T, C, L]
            Multi-channel time-series. B=batch, T=time steps,
            C=channels, L=signal length per channel.

        Returns
        -------
        H_token : Tensor [B, T, d]
            Token-level representations.
        H_seg : Tensor [B, S, d] or None
            Segment-level representations (None if no_hierarchy).
        event_proposals : dict or None
            Event proposal head output (None if no_event).
        """
        B, T, C, L = x.shape

        # --- Local Encoder: per-channel encoding ---
        # Reshape to [B*T*C, 1, L]
        x_flat = x.reshape(B * T * C, 1, L)
        channel_embeds = self.local_encoder(x_flat)  # [B*T*C, d]

        # --- Channel Fusion ---
        # Reshape to [B*T, C, d]
        d = channel_embeds.size(-1)
        channel_embeds = channel_embeds.view(B * T, C, d)
        fused = self.channel_fusion(channel_embeds)  # [B*T, d]

        # Reshape to [B, T, d]
        fused = fused.view(B, T, d)

        # --- Hierarchical Planner ---
        if self.planner is not None:
            H_token, H_seg = self.planner(fused)
        else:
            H_token = fused
            H_seg = None

        # --- Event Head ---
        event_proposals = None
        if self.event_head is not None:
            event_proposals = self.event_head(H_token)

        return H_token, H_seg, event_proposals

    def forward(
        self,
        x: torch.Tensor,
        captions: Optional[List[str]] = None,
        target_ids: Optional[torch.Tensor] = None,
        events_gt: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """Full forward pass.

        Parameters
        ----------
        x : Tensor, shape [B, T, C, L]
            Multi-channel time-series input.
        captions : List[str] or None
            Text captions for alignment loss (one per batch element).
        target_ids : Tensor [B, S_dec] or None
            Target token IDs for teacher-forced caption generation.
        events_gt : dict or None
            Ground-truth event annotations for event loss.

        Returns
        -------
        dict with keys:
            caption_logits : Tensor [B, S_dec, vocab_size] or None
            z_ts           : Tensor [B, d_align] or None
            z_text         : Tensor [B, d_align] or None
            logit_scale    : Tensor or None
            event_proposals: dict or None
            H_seg          : Tensor [B, S, d] or None
            H_token        : Tensor [B, T, d]
        """
        H_token, H_seg, event_proposals = self.encode(x)

        results: Dict[str, Any] = {
            "H_token": H_token,
            "H_seg": H_seg,
            "event_proposals": event_proposals,
            "caption_logits": None,
            "z_ts": None,
            "z_text": None,
            "logit_scale": None,
        }

        # --- Retrieval Aligner ---
        if self.aligner is not None and captions is not None and H_seg is not None:
            z_ts, z_text, logit_scale = self.aligner(H_seg, captions)
            results["z_ts"] = z_ts
            results["z_text"] = z_text
            results["logit_scale"] = logit_scale

        # --- Caption Decoder ---
        if target_ids is not None:
            # Use segment-level representations as encoder output for cross-attn
            encoder_output = H_seg if H_seg is not None else H_token
            caption_logits = self.decoder(encoder_output, target_ids)
            results["caption_logits"] = caption_logits

        return results
