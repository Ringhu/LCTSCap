"""Event proposal head.

Produces per-token event type classifications and start/end span logits,
then extracts top-K proposals per batch element.

Pipeline:
    Input:  H_token [B, T, d]
    -> event_type_head: Linear(d, n_event_types) -> [B, T, n_event_types]
    -> span_head:       Linear(d, 2)             -> [B, T, 2]  (start/end logits)
    -> top-K extraction per batch item
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EventProposalHead(nn.Module):
    """Event proposal head for token-level event detection.

    Parameters
    ----------
    d_model : int
        Input feature dimension.
    n_event_types : int
        Number of event type categories.
    max_events : int
        Maximum number of event proposals to return per batch element.
    """

    def __init__(
        self,
        d_model: int,
        n_event_types: int = 20,
        max_events: int = 16,
    ):
        super().__init__()
        self.n_event_types = n_event_types
        self.max_events = max_events

        self.event_type_head = nn.Linear(d_model, n_event_types)
        self.span_head = nn.Linear(d_model, 2)

    def forward(self, H_token: torch.Tensor) -> dict:
        """
        Parameters
        ----------
        H_token : Tensor, shape [B, T, d]
            Token-level representations from the planner.

        Returns
        -------
        dict with keys:
            event_type_logits : Tensor [B, T, n_event_types]
            span_logits       : Tensor [B, T, 2]
            top_k_proposals   : List[List[Tuple[int, int, int]]]
                Per batch item, a list of (event_type, start_idx, end_idx).
        """
        event_type_logits = self.event_type_head(H_token)  # [B, T, n_event_types]
        span_logits = self.span_head(H_token)  # [B, T, 2]

        top_k_proposals = self._extract_proposals(event_type_logits, span_logits)

        return {
            "event_type_logits": event_type_logits,
            "span_logits": span_logits,
            "top_k_proposals": top_k_proposals,
        }

    @torch.no_grad()
    def _extract_proposals(
        self,
        event_type_logits: torch.Tensor,
        span_logits: torch.Tensor,
    ) -> List[List[Tuple[int, int, int]]]:
        """Extract top-K event proposals per batch item.

        For each token position, compute the confidence as the max event-type
        probability.  Select the top ``max_events`` positions by confidence and
        return (event_type, start_token, end_token) tuples, where start/end
        come from the span head (clamped to valid range).
        """
        B, T, _ = event_type_logits.shape
        event_probs = F.softmax(event_type_logits, dim=-1)  # [B, T, n]
        max_probs, pred_types = event_probs.max(dim=-1)  # [B, T] each

        # Span logits: interpret as relative offsets in token indices
        # span_logits[..., 0] = start offset (negative = before token)
        # span_logits[..., 1] = end offset   (positive = after token)
        proposals_all: List[List[Tuple[int, int, int]]] = []

        for b in range(B):
            # Pick top-K token positions by confidence
            k = min(self.max_events, T)
            topk_vals, topk_idx = torch.topk(max_probs[b], k)

            proposals: List[Tuple[int, int, int]] = []
            for i in range(k):
                t_idx = topk_idx[i].item()
                etype = pred_types[b, t_idx].item()
                start_off = span_logits[b, t_idx, 0].item()
                end_off = span_logits[b, t_idx, 1].item()

                start = max(0, int(round(t_idx + start_off)))
                end = min(T - 1, int(round(t_idx + end_off)))
                if end < start:
                    end = start

                proposals.append((etype, start, end))
            proposals_all.append(proposals)

        return proposals_all
