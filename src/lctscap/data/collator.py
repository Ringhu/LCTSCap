"""Collation utilities for LCTSCap DataLoader.

Handles batching of variable-length captions, event lists, and
time-series tensors with proper padding and attention masks.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)

# Fixed mapping from activity string to integer index for event_loss.
# Covers both capture24 and harth activity types.
ACTIVITY_TO_IDX: Dict[str, int] = {
    "sleeping": 0,
    "sitting": 1,
    "standing": 2,
    "walking": 3,
    "running": 4,
    "cycling": 5,
    "vehicle": 6,
    "household": 7,
    "stairs_up": 8,
    "stairs_down": 9,
    "shuffling": 10,
    "nordic_walking": 11,
    "eating": 12,
    "self_care": 13,
    "screen_time": 14,
    "socializing": 15,
    "jumping": 16,
    "lying": 17,
    "stairs": 18,
    "other": 19,
}


class LCTSCapCollator:
    """Custom collate function for LCTSCap DataLoader.

    Pads time-series tensors to the batch maximum, optionally tokenizes
    caption strings, and constructs attention masks.

    Args:
        tokenizer: A HuggingFace-compatible tokenizer. If ``None``,
            raw caption strings are passed through without tokenization
            (suitable for encoder-only phases like Phase 1).
        max_caption_len: Maximum token length for captions.
        pad_events_to: Fixed number of events per sample (zero-padded).
        convert_events_to_per_token: If True, convert per-event lists to
            per-token format (type_labels, span_targets, span_mask) needed
            by ``event_loss``.
    """

    def __init__(
        self,
        tokenizer: Any = None,
        max_caption_len: int = 256,
        pad_events_to: Optional[int] = 16,
        convert_events_to_per_token: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_caption_len = max_caption_len
        self.pad_events_to = pad_events_to
        self.convert_events_to_per_token = convert_events_to_per_token

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate a list of dataset items into a batched dictionary."""
        # --- Time-series input ---
        ts_tensors = [item["ts_input"] for item in batch]
        ts_padded, ts_mask = self._pad_ts_tensors(ts_tensors)
        T_max = ts_padded.shape[1]

        # --- Captions ---
        short_captions = [item["caption_short"] for item in batch]
        long_captions = [item["caption_long"] for item in batch]

        result: Dict[str, Any] = {
            "ts_input": ts_padded,
            "ts_mask": ts_mask,
            "caption_short": short_captions,
            "caption_long": long_captions,
            "segment_summaries": [item["segment_summaries"] for item in batch],
            "evidence_bullets": [item["evidence_bullets"] for item in batch],
            "metadata": [item["metadata"] for item in batch],
        }

        # Tokenize captions if tokenizer available
        if self.tokenizer is not None:
            short_enc = self._tokenize_captions(short_captions)
            long_enc = self._tokenize_captions(long_captions)
            result["caption_short_ids"] = short_enc["input_ids"]
            result["caption_short_mask"] = short_enc["attention_mask"]
            result["caption_long_ids"] = long_enc["input_ids"]
            result["caption_long_mask"] = long_enc["attention_mask"]
            decoder_inputs = self._build_decoder_sequences(short_enc)
            result["decoder_input_ids"] = decoder_inputs["decoder_input_ids"]
            result["decoder_attention_mask"] = decoder_inputs["decoder_attention_mask"]
            result["target_ids"] = decoder_inputs["target_ids"]

        # --- Events ---
        events_batch = [item["events"] for item in batch]

        if self.convert_events_to_per_token:
            events_gt = self._events_to_per_token(events_batch, T_max)
            result["events_gt"] = events_gt
        elif self.pad_events_to is not None:
            events_tensor, events_mask = self._pad_events(events_batch)
            result["events"] = events_tensor
            result["events_mask"] = events_mask
        else:
            result["events"] = events_batch

        return result

    def _pad_ts_tensors(
        self,
        tensors: List[torch.Tensor],
    ) -> tuple:
        """Pad time-series tensors to the maximum sequence length in the batch.

        Handles both raw tensors ``(T, C, W)`` and embeddings ``(T, D)``.
        """
        batch_size = len(tensors)
        t_max = max(t.shape[0] for t in tensors)
        rest_shape = tensors[0].shape[1:]

        padded = torch.zeros(batch_size, t_max, *rest_shape, dtype=tensors[0].dtype)
        mask = torch.zeros(batch_size, t_max, dtype=torch.bool)

        for i, t in enumerate(tensors):
            seq_len = t.shape[0]
            padded[i, :seq_len] = t
            mask[i, :seq_len] = True

        return padded, mask

    def _tokenize_captions(
        self,
        captions: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Tokenize and pad a batch of caption strings."""
        captions_clean = [c if c else " " for c in captions]
        encoded = self.tokenizer(
            captions_clean,
            padding=True,
            truncation=True,
            max_length=self.max_caption_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    def _build_decoder_sequences(
        self,
        encoded: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Create decoder inputs and labels with explicit BOS/EOS handling."""
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", pad_token_id)
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            bos_token_id = eos_token_id

        decoder_inputs = []
        decoder_masks = []
        target_ids = []
        max_len = 0

        for ids, mask in zip(input_ids, attention_mask):
            valid_ids = ids[mask.bool()]
            decoder_input = torch.cat(
                [
                    torch.tensor([bos_token_id], dtype=torch.long),
                    valid_ids.to(dtype=torch.long),
                ]
            )
            target = torch.cat(
                [
                    valid_ids.to(dtype=torch.long),
                    torch.tensor([eos_token_id], dtype=torch.long),
                ]
            )
            decoder_inputs.append(decoder_input)
            target_ids.append(target)
            decoder_masks.append(torch.ones_like(decoder_input, dtype=torch.long))
            max_len = max(max_len, decoder_input.size(0))

        batch_size = len(decoder_inputs)
        padded_inputs = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
        padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
        padded_targets = torch.full((batch_size, max_len), -100, dtype=torch.long)

        for i, (decoder_input, decoder_mask, target) in enumerate(
            zip(decoder_inputs, decoder_masks, target_ids)
        ):
            seq_len = decoder_input.size(0)
            padded_inputs[i, :seq_len] = decoder_input
            padded_masks[i, :seq_len] = decoder_mask
            padded_targets[i, :seq_len] = target

        return {
            "decoder_input_ids": padded_inputs,
            "decoder_attention_mask": padded_masks,
            "target_ids": padded_targets,
        }

    def _events_to_per_token(
        self,
        events_batch: List[List[Dict[str, Any]]],
        T_max: int,
    ) -> Dict[str, torch.Tensor]:
        """Convert per-event lists to per-token format for event_loss.

        Returns:
            dict with:
            - ``type_labels``: ``(B, T_max)`` int64, -100 for background
            - ``span_targets``: ``(B, T_max, 2)`` float32, relative start/end offsets
            - ``span_mask``: ``(B, T_max)`` bool, True where event annotations exist
        """
        B = len(events_batch)
        type_labels = torch.full((B, T_max), -100, dtype=torch.long)
        span_targets = torch.zeros(B, T_max, 2, dtype=torch.float32)
        span_mask = torch.zeros(B, T_max, dtype=torch.bool)

        for b, events in enumerate(events_batch):
            for e in events:
                etype_str = e.get("type", "other")
                etype_idx = ACTIVITY_TO_IDX.get(etype_str, ACTIVITY_TO_IDX["other"])
                start = e.get("start_token", 0)
                end = e.get("end_token", 0)

                # Clamp to valid range
                start = max(0, min(start, T_max - 1))
                end = max(start, min(end, T_max))

                # Fill per-token labels for this event's span
                for t in range(start, min(end, T_max)):
                    type_labels[b, t] = etype_idx
                    # Span offsets: relative to current token position
                    span_targets[b, t, 0] = start - t  # start offset (negative)
                    span_targets[b, t, 1] = end - t    # end offset (positive)
                    span_mask[b, t] = True

        return {
            "type_labels": type_labels,
            "span_targets": span_targets,
            "span_mask": span_mask,
        }

    def _pad_events(
        self,
        events_batch: List[List[Dict[str, Any]]],
    ) -> tuple:
        """Convert variable-length event lists into padded tensors.

        Each event is represented as ``[start_token, end_token, duration_sec, is_dominant]``.
        """
        assert self.pad_events_to is not None
        max_events = self.pad_events_to
        batch_size = len(events_batch)

        events_tensor = torch.zeros(batch_size, max_events, 4, dtype=torch.float32)
        events_mask = torch.zeros(batch_size, max_events, dtype=torch.bool)

        for i, events in enumerate(events_batch):
            n = min(len(events), max_events)
            for j in range(n):
                e = events[j]
                events_tensor[i, j, 0] = float(e.get("start_token", 0))
                events_tensor[i, j, 1] = float(e.get("end_token", 0))
                events_tensor[i, j, 2] = float(e.get("duration_sec", 0))
                events_tensor[i, j, 3] = float(e.get("is_dominant", False))
                events_mask[i, j] = True

        return events_tensor, events_mask
