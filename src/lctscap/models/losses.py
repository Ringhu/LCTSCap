"""All loss functions for LCTSCap training.

Provides:
    - clip_infonce: Symmetric InfoNCE contrastive loss
    - event_loss: Event type cross-entropy + span regression (smooth L1)
    - coverage_loss: Penalises missing top-K event types in generated captions
    - compute_total_loss: Weighted combination of all components
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def clip_infonce(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    logit_scale: torch.Tensor,
) -> torch.Tensor:
    """Symmetric InfoNCE loss (CLIP-style).

    Parameters
    ----------
    z_a : Tensor, shape [B, d]
        Normalised embeddings from modality A (e.g., time-series).
    z_b : Tensor, shape [B, d]
        Normalised embeddings from modality B (e.g., text).
    logit_scale : Tensor (scalar)
        Exponentiated learnable temperature parameter.

    Returns
    -------
    Tensor (scalar)
        Mean of (a->b) and (b->a) cross-entropy losses.
    """
    # Similarity matrix: [B, B]
    logits_ab = logit_scale * z_a @ z_b.t()
    logits_ba = logits_ab.t()

    B = z_a.size(0)
    labels = torch.arange(B, device=z_a.device)

    loss_ab = F.cross_entropy(logits_ab, labels)
    loss_ba = F.cross_entropy(logits_ba, labels)

    return (loss_ab + loss_ba) / 2.0


def event_loss(
    pred_types: torch.Tensor,
    pred_spans: torch.Tensor,
    gt_events: Dict[str, torch.Tensor],
    context_len: int,
) -> torch.Tensor:
    """Combined event type classification + span regression loss.

    Parameters
    ----------
    pred_types : Tensor, shape [B, T, n_event_types]
        Predicted event-type logits per token.
    pred_spans : Tensor, shape [B, T, 2]
        Predicted start/end span logits per token.
    gt_events : dict with keys:
        - "type_labels" : Tensor [B, T] — ground-truth event type per token
          (use -100 or ignore_index for background / no-event tokens).
        - "span_targets" : Tensor [B, T, 2] — ground-truth span offsets.
        - "span_mask"    : Tensor [B, T] — binary mask indicating which
          tokens have valid span annotations.
    context_len : int
        Number of valid context tokens (for masking if needed).

    Returns
    -------
    Tensor (scalar)
        Sum of event-type CE loss and span smooth-L1 loss.
    """
    # --- Event-type cross-entropy ---
    type_labels = gt_events["type_labels"]  # [B, T]
    B, T, n_types = pred_types.shape

    # Flatten for cross-entropy
    type_loss = F.cross_entropy(
        pred_types.reshape(-1, n_types),
        type_labels.reshape(-1),
        ignore_index=-100,
    )

    # --- Span regression (smooth L1) ---
    span_targets = gt_events["span_targets"]  # [B, T, 2]
    span_mask = gt_events["span_mask"]  # [B, T]

    if span_mask.any():
        # Only compute span loss on annotated positions
        mask_expanded = span_mask.bool().unsqueeze(-1).expand_as(pred_spans)  # [B, T, 2]
        span_loss = F.smooth_l1_loss(
            pred_spans[mask_expanded],
            span_targets[mask_expanded],
            reduction="mean",
        )
    else:
        span_loss = torch.tensor(0.0, device=pred_types.device)

    return type_loss + span_loss


def coverage_loss(
    caption_logits: torch.Tensor,
    event_types_gt: torch.Tensor,
    vocab: Optional[Dict[int, str]] = None,
    top_k: int = 3,
) -> torch.Tensor:
    """Coverage loss that penalises missing top-K event types in generated text.

    This is a soft approximation: we check whether the generated token
    distribution assigns probability mass to tokens associated with each
    ground-truth event type.  When no vocab mapping is available, we use
    the predicted event-type distribution directly as a proxy.

    Parameters
    ----------
    caption_logits : Tensor, shape [B, S, vocab_size]
        Decoder output logits (before softmax).
    event_types_gt : Tensor, shape [B, T_events]
        Ground-truth event type indices per batch. Padded with -1.
    vocab : dict, optional
        Mapping from event-type index to a set of vocab token indices that
        represent that event in text.  If None, uses a simpler proxy.
    top_k : int
        Number of top event types to require coverage for.

    Returns
    -------
    Tensor (scalar)
        Coverage penalty.
    """
    B = caption_logits.size(0)
    device = caption_logits.device

    # Softmax over vocab to get token probabilities
    token_probs = F.softmax(caption_logits, dim=-1)  # [B, S, V]

    # Max pool over sequence to get per-token "max probability across time"
    max_token_probs = token_probs.max(dim=1).values  # [B, V]

    total_loss = torch.tensor(0.0, device=device)
    count = 0

    for b in range(B):
        # Get valid (non-padding) event types for this batch element
        valid_mask = event_types_gt[b] >= 0
        valid_types = event_types_gt[b][valid_mask]

        if len(valid_types) == 0:
            continue

        # Take top-K unique event types
        unique_types = valid_types.unique()
        k = min(top_k, len(unique_types))
        selected_types = unique_types[:k]

        if vocab is not None:
            # Use vocab mapping: for each event type, check if its associated
            # tokens appear in the generated distribution
            for etype in selected_types:
                etype_idx = etype.item()
                if etype_idx in vocab:
                    token_indices = vocab[etype_idx]
                    if isinstance(token_indices, str):
                        continue
                    token_indices_t = torch.tensor(
                        token_indices, device=device, dtype=torch.long
                    )
                    # Max prob assigned to any of the event's tokens
                    coverage_prob = max_token_probs[b, token_indices_t].max()
                    # Penalise low coverage
                    total_loss = total_loss - torch.log(coverage_prob + 1e-8)
                    count += 1
        else:
            # Proxy: use the event type index directly as a "token" to look
            # for in the vocabulary.  Penalise if the model doesn't assign
            # probability to those token positions.
            for etype in selected_types:
                etype_idx = etype.item()
                if etype_idx < max_token_probs.size(1):
                    coverage_prob = max_token_probs[b, etype_idx]
                    total_loss = total_loss - torch.log(coverage_prob + 1e-8)
                    count += 1

    if count > 0:
        total_loss = total_loss / count

    return total_loss


def compute_total_loss(
    cap_logits: torch.Tensor,
    cap_targets: torch.Tensor,
    z_ts: Optional[torch.Tensor],
    z_text: Optional[torch.Tensor],
    logit_scale: Optional[torch.Tensor],
    event_preds: Optional[Dict[str, torch.Tensor]],
    event_targets: Optional[Dict[str, torch.Tensor]],
    coverage_score: Optional[torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute weighted total loss from all components.

    Parameters
    ----------
    cap_logits : Tensor [B, S, V]
        Caption decoder logits.
    cap_targets : Tensor [B, S]
        Target token IDs for caption loss.
    z_ts, z_text : Tensor [B, d] or None
        Alignment embeddings (skip if None).
    logit_scale : Tensor (scalar) or None
        CLIP temperature.
    event_preds : dict or None
        With keys "event_type_logits" [B, T, n] and "span_logits" [B, T, 2].
    event_targets : dict or None
        Ground truth for event loss.
    coverage_score : Tensor (scalar) or None
        Pre-computed coverage loss (or None to skip).
    weights : dict
        Loss component weights.  Keys: "caption", "align", "event", "coverage".

    Returns
    -------
    total : Tensor (scalar)
    components : dict mapping component name to float loss value
    """
    if weights is None:
        weights = {
            "caption": 1.0,
            "align": 0.5,
            "event": 0.3,
            "coverage": 0.1,
        }

    device = cap_logits.device
    components: Dict[str, float] = {}

    # --- Caption loss (cross-entropy, shift by 1) ---
    # Shift: predict next token
    shift_logits = cap_logits[:, :-1, :].contiguous()
    shift_targets = cap_targets[:, 1:].contiguous()
    caption_loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_targets.view(-1),
        ignore_index=-100,
    )
    components["caption"] = caption_loss.item()
    total = weights.get("caption", 1.0) * caption_loss

    # --- Alignment loss ---
    if z_ts is not None and z_text is not None and logit_scale is not None:
        align_loss = clip_infonce(z_ts, z_text, logit_scale)
        components["align"] = align_loss.item()
        total = total + weights.get("align", 0.5) * align_loss

    # --- Event loss ---
    if event_preds is not None and event_targets is not None:
        ev_loss = event_loss(
            pred_types=event_preds["event_type_logits"],
            pred_spans=event_preds["span_logits"],
            gt_events=event_targets,
            context_len=event_preds["event_type_logits"].size(1),
        )
        components["event"] = ev_loss.item()
        total = total + weights.get("event", 0.3) * ev_loss

    # --- Coverage loss ---
    if coverage_score is not None:
        components["coverage"] = coverage_score.item()
        total = total + weights.get("coverage", 0.1) * coverage_score

    components["total"] = total.item()
    return total, components
