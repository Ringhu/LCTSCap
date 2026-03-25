"""Inference helpers for LCTSCap caption generation and auxiliary alignment."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set

import torch
import torch.nn.functional as F


WHITESPACE_RE = re.compile(r"\s+")
NON_PRINTABLE_RE = re.compile(r"[^\x20-\x7E]+")
REPEATED_PUNCT_RE = re.compile(r"([!\"#$%&'()*+,\-./:;<=>?@[\\\]^_`{|}~])\1{2,}")


def normalize_prediction_text(text: str) -> str:
    """Collapse whitespace and lightly sanitize obvious decoding artifacts."""
    text = NON_PRINTABLE_RE.sub(" ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()


def decode_sequences(tokenizer: Any, token_ids: torch.Tensor) -> List[str]:
    """Decode generated token IDs into cleaned text strings."""
    if hasattr(tokenizer, "batch_decode"):
        decoded = tokenizer.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
    elif hasattr(tokenizer, "decode"):
        decoded = [tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in token_ids]
    else:
        raise TypeError("Tokenizer must provide decode or batch_decode for inference.")

    return [normalize_prediction_text(text) for text in decoded]


def build_prediction_records(
    metadata_batch: Sequence[Dict[str, object]],
    predictions: Iterable[str],
) -> List[Dict[str, object]]:
    """Convert metadata + decoded strings into JSONL prediction records."""
    records: List[Dict[str, object]] = []
    for meta, pred in zip(metadata_batch, predictions):
        records.append(
            {
                "sample_id": meta["sample_id"],
                "dataset": meta.get("dataset"),
                "participant_id": meta.get("participant_id"),
                "split": meta.get("split"),
                "context_len": meta.get("context_len"),
                "prediction": pred,
            }
        )
    return records


def event_proposals_to_records(
    top_k_proposals: Sequence[Sequence[Sequence[int]]],
    idx_to_activity: Dict[int, str],
) -> List[List[Dict[str, object]]]:
    """Convert raw event-head proposals into JSON-friendly evidence records."""
    converted: List[List[Dict[str, object]]] = []
    for sample_proposals in top_k_proposals:
        sample_records: List[Dict[str, object]] = []
        for proposal in sample_proposals:
            event_type, start_token, end_token = proposal
            sample_records.append(
                {
                    "activity": idx_to_activity.get(int(event_type), f"unknown_{event_type}"),
                    "start_token": int(start_token),
                    "end_token": int(end_token),
                }
            )
        converted.append(sample_records)
    return converted


def verbalize_event_evidence_text(
    evidence_records: Sequence[Dict[str, object]],
    *,
    prefix: str = "Evidence:",
) -> str:
    """Render structured evidence into span-explicit text parseable by claim_parser."""
    if not evidence_records:
        return ""

    clauses = []
    for item in evidence_records:
        activity = str(item.get("activity", "unknown")).strip()
        start = item.get("start_token")
        end = item.get("end_token")
        if start is None or end is None:
            clauses.append(activity)
        else:
            clauses.append(f"{activity} windows {int(start)} to {int(end)}")

    return normalize_prediction_text(f"{prefix} {'; '.join(clauses)}.")


def resize_ts_windows(ts_windows: torch.Tensor, target_length: int) -> torch.Tensor:
    """Resize [T, C, L] windows to the target signal length via linear interpolation."""
    if ts_windows.ndim != 3:
        raise ValueError("ts_windows must have shape [T, C, L].")

    time_steps, channels, signal_length = ts_windows.shape
    if signal_length == target_length:
        return ts_windows

    flat = ts_windows.reshape(time_steps * channels, 1, signal_length)
    resized = F.interpolate(flat, size=target_length, mode="linear", align_corners=False)
    return resized.reshape(time_steps, channels, target_length)


@torch.no_grad()
def encode_aux_timeseries(model: Any, ts_batch: torch.Tensor, *, target_length: int = 500) -> torch.Tensor:
    """Encode auxiliary [B, T, C, L] samples into the aligner space.

    Uses the normal hierarchy when enough tokens are present; otherwise falls
    back to mean-pooled token states so short auxiliary samples remain usable.
    """
    if ts_batch.ndim != 4:
        raise ValueError("ts_batch must have shape [B, T, C, L].")
    if getattr(model, "aligner", None) is None:
        raise ValueError("Model must include an aligner for auxiliary retrieval evaluation.")

    batch_size, time_steps, channels, signal_length = ts_batch.shape
    if signal_length != target_length:
        flat = ts_batch.reshape(batch_size * time_steps, channels, signal_length)
        flat = resize_ts_windows(flat, target_length)
        ts_batch = flat.reshape(batch_size, time_steps, channels, target_length)

    x_flat = ts_batch.reshape(batch_size * time_steps * channels, 1, target_length)
    channel_embeds = model.local_encoder(x_flat)
    hidden_dim = channel_embeds.size(-1)
    channel_embeds = channel_embeds.view(batch_size * time_steps, channels, hidden_dim)
    fused = model.channel_fusion(channel_embeds).view(batch_size, time_steps, hidden_dim)

    planner = getattr(model, "planner", None)
    if planner is not None:
        H_token, H_seg = planner(fused)
        if H_seg.size(1) > 0:
            return model.aligner.encode_ts(H_seg)
    else:
        H_token = fused

    pooled = H_token.mean(dim=1)
    z_ts = model.aligner.ts_proj(pooled)
    z_ts = F.normalize(z_ts, dim=-1)
    return z_ts


def _apply_allowed_token_mask(
    logits: torch.Tensor,
    allowed_token_ids: Set[int] | None,
) -> torch.Tensor:
    if not allowed_token_ids:
        return logits
    allowed = torch.tensor(sorted(allowed_token_ids), device=logits.device, dtype=torch.long)
    constrained = torch.full_like(logits, float("-inf"))
    constrained.index_copy_(1, allowed, logits.index_select(1, allowed))
    return constrained


def _apply_repetition_penalty(
    logits: torch.Tensor,
    generated: torch.Tensor,
    repetition_penalty: float,
) -> torch.Tensor:
    if repetition_penalty <= 1.0:
        return logits
    adjusted = logits.clone()
    for row_idx in range(generated.size(0)):
        token_ids = torch.unique(generated[row_idx])
        token_scores = adjusted[row_idx, token_ids]
        token_scores = torch.where(
            token_scores < 0,
            token_scores * repetition_penalty,
            token_scores / repetition_penalty,
        )
        adjusted[row_idx, token_ids] = token_scores
    return adjusted


def _calc_banned_ngram_tokens(generated: torch.Tensor, ngram_size: int) -> List[Set[int]]:
    if ngram_size <= 0:
        return [set() for _ in range(generated.size(0))]

    sequences = generated.tolist()
    banned: List[Set[int]] = []
    for seq in sequences:
        if ngram_size == 1:
            banned.append(set(seq))
            continue
        if len(seq) + 1 < ngram_size:
            banned.append(set())
            continue

        prefix = tuple(seq[-(ngram_size - 1):])
        sample_banned: Set[int] = set()
        for idx in range(len(seq) - ngram_size + 1):
            ngram = tuple(seq[idx : idx + ngram_size])
            if ngram[:-1] == prefix:
                sample_banned.add(int(ngram[-1]))
        banned.append(sample_banned)
    return banned


@torch.no_grad()
def generate_from_prompt(
    decoder: Any,
    encoder_output: torch.Tensor,
    prompt_ids: torch.Tensor,
    *,
    max_len: int,
    temperature: float,
    eos_token_id: int,
    allowed_token_ids: Set[int] | None = None,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    """Run autoregressive decoding from an explicit prompt token prefix."""
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape [B, S] or [1, S].")

    batch_size = encoder_output.size(0)
    if prompt_ids.size(0) == 1 and batch_size > 1:
        generated = prompt_ids.repeat(batch_size, 1)
    elif prompt_ids.size(0) == batch_size:
        generated = prompt_ids.clone()
    else:
        raise ValueError("prompt_ids batch dimension must be 1 or match encoder_output batch size.")

    memory = decoder.cross_attn_proj(encoder_output)

    for _ in range(max_len - generated.size(1)):
        seq_len = generated.size(1)
        tok_emb = decoder.token_embed(generated)
        tok_emb = tok_emb + decoder.pos_embed[:, :seq_len, :]
        tgt_mask = decoder._make_causal_mask(seq_len, generated.device)
        dec_out = decoder.decoder_layers(tgt=tok_emb, memory=memory, tgt_mask=tgt_mask)
        raw_next_logits = decoder.output_head(dec_out[:, -1, :])
        next_logits = raw_next_logits.clone()

        next_logits = _apply_repetition_penalty(next_logits, generated, repetition_penalty)
        allowed_logits = _apply_allowed_token_mask(next_logits, allowed_token_ids)
        next_logits = allowed_logits

        if no_repeat_ngram_size > 0:
            banned_token_ids = _calc_banned_ngram_tokens(generated, no_repeat_ngram_size)
            for row_idx, banned in enumerate(banned_token_ids):
                if banned:
                    next_logits[row_idx, list(banned)] = float("-inf")

        all_blocked = torch.isneginf(next_logits).all(dim=-1)
        if all_blocked.any():
            next_logits[all_blocked] = allowed_logits[all_blocked]
        still_blocked = torch.isneginf(next_logits).all(dim=-1)
        if still_blocked.any():
            next_logits[still_blocked] = raw_next_logits[still_blocked]

        if temperature <= 1e-8:
            next_token = next_logits.argmax(dim=-1, keepdim=True)
        else:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)
        if (next_token.squeeze(-1) == eos_token_id).all():
            break

    return generated
