#!/usr/bin/env python
"""Generate caption predictions from a trained LCTSCap checkpoint."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, List, Set

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.config import load_config
from lctscap.data.collator import ACTIVITY_TO_IDX, LCTSCapCollator
from lctscap.inference import (
    build_prediction_records,
    decode_sequences,
    event_proposals_to_records,
    generate_from_prompt,
    normalize_prediction_text,
    verbalize_event_evidence_text,
)
from train import build_model, build_tokenizer, load_datasets, load_model_weights, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_predictions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LCTSCap predictions from a checkpoint.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt_text", type=str, default="")
    parser.add_argument("--emit_event_evidence", action="store_true")
    parser.add_argument(
        "--evidence_text_mode",
        type=str,
        default="none",
        choices=["none", "append", "replace"],
    )
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--restrict_to_caption_vocab", action="store_true")
    parser.add_argument(
        "--caption_vocab_split",
        type=str,
        default="train",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument(
        "--caption_field",
        type=str,
        default="caption_short",
        choices=["caption_short", "caption_long"],
    )
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    return parser.parse_args()


def build_dataloaders(
    cfg,
    split: str,
    batch_size_override: int | None,
    *,
    device: torch.device,
) -> dict[int, DataLoader]:
    datasets = load_datasets(cfg, cfg.data.context_lens)
    collator = LCTSCapCollator(tokenizer=None, convert_events_to_per_token=False)
    dataloaders: dict[int, DataLoader] = {}

    for ctx_len in sorted(datasets):
        split_ds = datasets[ctx_len].get_split(split)
        if len(split_ds) == 0:
            logger.info("Skipping ctx=%d because split=%s is empty.", ctx_len, split)
            continue
        batch_size = batch_size_override or cfg.train.batch_size_for(ctx_len)
        num_workers = max(1, cfg.num_workers // 2)
        if device.type == "cpu":
            num_workers = 0
        dataloaders[ctx_len] = DataLoader(
            split_ds,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        logger.info(
            "Prepared ctx=%d split=%s samples=%d batch_size=%d",
            ctx_len,
            split,
            len(split_ds),
            batch_size,
        )

    return dataloaders


def default_output_path(cfg, split: str, checkpoint: str) -> Path:
    ckpt_name = Path(checkpoint).stem
    output_dir = Path(cfg.output_dir) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{cfg.data.dataset}_{split}_{ckpt_name}.jsonl"


def _encode_caption_ids(tokenizer, text: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        return list(ids)

    try:
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=False,
            return_tensors="pt",
            add_special_tokens=False,
        )
    except TypeError:
        encoded = tokenizer(
            [text],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
    ids = encoded["input_ids"][0]
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()
    return list(ids)


def build_allowed_token_ids(
    tokenizer,
    annotation_dir: Path,
    *,
    split: str,
    caption_field: str,
) -> Set[int]:
    if not annotation_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")

    unique_captions: Set[str] = set()
    for jsonl_path in sorted(annotation_dir.glob("*.jsonl")):
        with open(jsonl_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if split != "all" and obj.get("split") != split:
                    continue
                caption = obj.get(caption_field, "")
                if caption:
                    unique_captions.add(caption)

    allowed_token_ids: Set[int] = set()
    for token_attr in ("pad_token_id", "bos_token_id", "eos_token_id"):
        token_id = getattr(tokenizer, token_attr, None)
        if token_id is not None:
            allowed_token_ids.add(int(token_id))

    for caption in sorted(unique_captions):
        allowed_token_ids.update(_encode_caption_ids(tokenizer, caption))

    logger.info(
        "Built caption-vocab constraint from %d unique %s texts (%s split): %d token ids",
        len(unique_captions),
        caption_field,
        split,
        len(allowed_token_ids),
    )
    return allowed_token_ids


@torch.no_grad()
def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    if args.data_root:
        cfg.data.data_root = args.data_root

    set_seed(cfg.seed)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info("Using device: %s", device)
    logger.info("Loading checkpoint: %s", args.checkpoint)
    logger.info("Using prompt text: %r", args.prompt_text)
    logger.info("Emit event evidence: %s", args.emit_event_evidence)

    tokenizer = build_tokenizer(cfg)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(tokenizer, "pad_token_id", 0)
    if args.prompt_text:
        prompt_ids = tokenizer(
            args.prompt_text,
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"]
    else:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if bos_token_id is None:
            bos_token_id = eos_token_id
        prompt_ids = torch.tensor([[bos_token_id]], dtype=torch.long)
    if prompt_ids.numel() == 0:
        raise ValueError("prompt_text must produce at least one token.")

    allowed_token_ids = None
    if args.restrict_to_caption_vocab:
        annotation_dir = Path(cfg.data.data_root) / "processed" / cfg.data.dataset / "annotations"
        allowed_token_ids = build_allowed_token_ids(
            tokenizer,
            annotation_dir,
            split=args.caption_vocab_split,
            caption_field=args.caption_field,
        )

    logger.info(
        "Decode controls: restrict_to_caption_vocab=%s repetition_penalty=%.3f no_repeat_ngram_size=%d",
        args.restrict_to_caption_vocab,
        args.repetition_penalty,
        args.no_repeat_ngram_size,
    )

    model = build_model(cfg)
    load_model_weights(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    dataloaders = build_dataloaders(cfg, args.split, args.batch_size, device=device)
    if not dataloaders:
        raise RuntimeError(f"No dataloaders built for split={args.split}.")

    output_path = Path(args.output_path) if args.output_path else default_output_path(
        cfg,
        args.split,
        args.checkpoint,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_records = 0
    idx_to_activity = {idx: activity for activity, idx in ACTIVITY_TO_IDX.items()}
    with open(output_path, "w") as f:
        for ctx_len in sorted(dataloaders):
            dataloader = dataloaders[ctx_len]
            logger.info("Generating predictions for ctx=%d with %d batches", ctx_len, len(dataloader))
            for batch_idx, batch in enumerate(dataloader):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break
                ts_input = batch["ts_input"].to(device, non_blocking=True)
                h_token, h_seg, event_outputs = model.encode(ts_input)
                encoder_output = h_seg if h_seg is not None else h_token
                token_ids = generate_from_prompt(
                    decoder=model.decoder,
                    encoder_output=encoder_output,
                    prompt_ids=prompt_ids.to(device),
                    max_len=args.max_len,
                    temperature=args.temperature,
                    eos_token_id=eos_token_id,
                    allowed_token_ids=allowed_token_ids,
                    repetition_penalty=args.repetition_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                ).cpu()

                predictions = decode_sequences(tokenizer, token_ids)
                records = build_prediction_records(batch["metadata"], predictions)
                evidence_records = None
                need_evidence = args.emit_event_evidence or args.evidence_text_mode != "none"
                if need_evidence and event_outputs is not None:
                    evidence_records = event_proposals_to_records(
                        event_outputs.get("top_k_proposals", []),
                        idx_to_activity,
                    )
                for record in records:
                    record["checkpoint"] = str(args.checkpoint)
                if evidence_records is not None:
                    for record, evidence in zip(records, evidence_records):
                        record["predicted_events"] = evidence
                        record["evidence_mode"] = "event_head_topk"
                        record["event_evidence_count"] = len(evidence)
                        record["event_evidence_order"] = [item["activity"] for item in evidence]
                        record["event_evidence_spans"] = [
                            [item["start_token"], item["end_token"]] for item in evidence
                        ]
                        if args.evidence_text_mode != "none":
                            evidence_text = verbalize_event_evidence_text(evidence)
                            if evidence_text:
                                if args.evidence_text_mode == "replace":
                                    record["prediction"] = evidence_text
                                else:
                                    record["prediction"] = normalize_prediction_text(
                                        f"{record['prediction']} {evidence_text}"
                                    )
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_records += len(records)

                log_every = max(1, len(dataloader) // 10)
                if batch_idx % log_every == 0 or batch_idx == len(dataloader) - 1:
                    logger.info(
                        "  ctx=%d batch=%d/%d total_records=%d",
                        ctx_len,
                        batch_idx + 1,
                        len(dataloader),
                        total_records,
                    )

    logger.info("Saved %d predictions to %s", total_records, output_path)


if __name__ == "__main__":
    main()
