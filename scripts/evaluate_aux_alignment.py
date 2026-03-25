#!/usr/bin/env python
"""Evaluate auxiliary time-series/text alignment on UCR / Sleep-EDF manifests."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.config import load_config
from lctscap.eval.retrieval import compute_grouped_retrieval_metrics, compute_similarity_matrix
from lctscap.inference import encode_aux_timeseries, resize_ts_windows
from lctscap.models.full_model import LCTSCapModel, ModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("evaluate_aux_alignment")


def build_aux_model(cfg, num_channels: int) -> LCTSCapModel:
    model_cfg = ModelConfig(
        d_model=cfg.model.d_model,
        num_channels=num_channels,
        patch_size=cfg.model.patch_size,
        local_encoder_layers=cfg.model.num_layers_local,
        local_encoder_heads=cfg.model.num_heads,
        planner_layers=cfg.model.num_layers_planner,
        planner_heads=cfg.model.num_heads,
        segment_size=cfg.model.segment_size,
        decoder_layers=cfg.model.decoder_layers,
        decoder_heads=cfg.model.num_heads,
        vocab_size=cfg.model.decoder_vocab_size,
        n_event_types=cfg.model.n_event_types,
        max_events=cfg.model.max_events,
        d_align=cfg.model.d_align,
        text_model_name=cfg.model.text_model_name,
        no_hierarchy=cfg.no_hierarchy,
        no_event=cfg.no_event,
        no_align=cfg.no_align,
    )
    return LCTSCapModel(model_cfg)


def load_checkpoint_weights(model: LCTSCapModel, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_state = model.state_dict()
    filtered = {key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape}
    model.load_state_dict(filtered, strict=False)


def load_manifest(path: Path, max_samples: int | None = None) -> list[dict]:
    records = []
    with path.open("r") as handle:
        for line in handle:
            if not line.strip():
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def load_aux_tensor(path: str, target_length: int = 500) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu").float()
    if tensor.ndim == 2:
        tensor = resize_ts_windows(tensor.unsqueeze(0), target_length)
    elif tensor.ndim == 3:
        tensor = resize_ts_windows(tensor, target_length)
    else:
        raise ValueError(f"Unsupported tensor shape at {path}: {tuple(tensor.shape)}")
    return tensor


@torch.no_grad()
def encode_manifest(
    model: LCTSCapModel,
    records: list[dict],
    *,
    device: torch.device,
    batch_size: int,
    target_length: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], dict]:
    ts_embeds = []
    text_embeds = []
    labels = []
    captions = []
    fallback_count = 0
    total_tokens = 0

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        windows = []
        batch_captions = []
        batch_labels = []
        for item in batch:
            ts_tensor = load_aux_tensor(item["tensor_path"], target_length=target_length)
            windows.append(ts_tensor)
            batch_captions.append(item["caption_text"])
            batch_labels.append(str(item["raw_label"]))
            total_tokens += int(ts_tensor.shape[0])
            if int(ts_tensor.shape[0]) < model.config.segment_size:
                fallback_count += 1

        ts_batch = torch.stack(windows, dim=0).to(device)
        z_ts = encode_aux_timeseries(model, ts_batch, target_length=target_length).cpu()
        z_text = model.aligner.encode_text(batch_captions).cpu()
        ts_embeds.append(z_ts)
        text_embeds.append(z_text)
        labels.extend(batch_labels)
        captions.extend(batch_captions)

    return (
        torch.cat(ts_embeds, dim=0),
        torch.cat(text_embeds, dim=0),
        labels,
        {
            "fallback_short_context_count": fallback_count,
            "mean_tokens": total_tokens / max(len(records), 1),
            "unique_captions": len(set(captions)),
        },
    )


def discover_manifest_paths(input_root: Path, split: str) -> list[Path]:
    return sorted(
        path
        for path in input_root.rglob(f"{split}_manifest.jsonl")
        if path.is_file()
    )


def format_markdown(results: list[dict]) -> str:
    lines = ["# Auxiliary Alignment Evaluation", ""]
    for item in results:
        lines.append(f"## {item['dataset']} ({item['split']})")
        lines.append("")
        lines.append(f"- samples: `{item['num_samples']}`")
        lines.append(f"- unique labels: `{item['num_labels']}`")
        lines.append(f"- unique captions: `{item['diagnostics']['unique_captions']}`")
        lines.append(f"- fallback short-context count: `{item['diagnostics']['fallback_short_context_count']}`")
        lines.append(f"- mean tokens per sample: `{item['diagnostics']['mean_tokens']:.2f}`")
        for key, value in sorted(item["metrics"].items()):
            lines.append(f"- {key}: `{value:.4f}`")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate auxiliary retrieval alignment benchmarks.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--target_length", type=int, default=500)
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths = discover_manifest_paths(input_root, args.split)
    if not manifest_paths:
        raise FileNotFoundError(f"No {args.split}_manifest.jsonl files found under {input_root}")

    results = []
    device = torch.device(args.device)

    for manifest_path in manifest_paths:
        records = load_manifest(manifest_path, max_samples=args.max_samples)
        if not records:
            logger.warning("Skipping empty manifest: %s", manifest_path)
            continue

        first_tensor = load_aux_tensor(records[0]["tensor_path"], target_length=args.target_length)
        model = build_aux_model(cfg, num_channels=int(first_tensor.shape[1]))
        if model.aligner is None:
            raise ValueError("Checkpoint/config does not contain an aligner; auxiliary retrieval evaluation is invalid.")
        load_checkpoint_weights(model, args.checkpoint)
        model.to(device)
        model.eval()

        logger.info("Evaluating %s (%d samples)", manifest_path.parent.parent.name, len(records))
        z_ts, z_text, labels, diagnostics = encode_manifest(
            model,
            records,
            device=device,
            batch_size=args.batch_size,
            target_length=args.target_length,
        )
        sim = compute_similarity_matrix(z_ts, z_text)
        metrics = compute_grouped_retrieval_metrics(sim, labels)
        result = {
            "dataset": records[0]["dataset"],
            "split": args.split,
            "manifest_path": str(manifest_path),
            "num_samples": len(records),
            "num_labels": len(set(labels)),
            "metrics": metrics,
            "diagnostics": diagnostics,
        }
        results.append(result)

        result_path = output_dir / f"{records[0]['dataset']}_{args.split}_metrics.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    summary_path = output_dir / f"aux_alignment_{args.split}_summary.json"
    summary_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    markdown_path = output_dir / f"aux_alignment_{args.split}_summary.md"
    markdown_path.write_text(format_markdown(results))
    logger.info("Saved auxiliary alignment summary to %s", markdown_path)


if __name__ == "__main__":
    main()
