#!/usr/bin/env python
"""Preprocess raw datasets into windowed tensors and long-context samples.

Usage:
    python scripts/preprocess.py --dataset capture24 --config configs/data/capture24.yaml
    python scripts/preprocess.py --dataset harth --config configs/data/harth.yaml
    python scripts/preprocess.py --dataset all --config configs/data/capture24.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.config import load_config
from lctscap.data.capture24 import preprocess_all as preprocess_capture24
from lctscap.data.harth import preprocess_all as preprocess_harth
from lctscap.data.long_context import build_all_contexts, compute_statistics
from lctscap.data.splits import make_subject_splits, save_splits, verify_no_leakage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("preprocess")


def assign_splits_to_manifest(manifest, splits):
    """Annotate each window in manifest with its split assignment."""
    pid_to_split = {}
    for split_name, pids in splits.items():
        for pid in pids:
            pid_to_split[pid] = split_name
    for entry in manifest:
        entry["split"] = pid_to_split.get(entry["participant_id"], "train")
    return manifest


def run_preprocessing(dataset, cfg):
    """Run the full preprocessing pipeline for one dataset."""
    data_root = cfg.data.data_root
    raw_dir = Path(data_root) / "raw"
    output_dir = Path(data_root) / "processed"

    logger.info("=== Preprocessing %s ===", dataset)

    # Step 1: Window extraction
    logger.info("[Step 1/4] Extracting windows...")
    if dataset == "capture24":
        manifest = preprocess_capture24(str(raw_dir), str(output_dir), cfg.data)
    elif dataset == "harth":
        manifest = preprocess_harth(str(raw_dir), str(output_dir), cfg.data)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    if not manifest:
        logger.error("No windows extracted for %s. Check raw data.", dataset)
        return

    logger.info("Extracted %d windows.", len(manifest))

    # Step 2: Subject-level splits
    logger.info("[Step 2/4] Creating subject-level splits...")
    subjects = sorted(set(w["participant_id"] for w in manifest))
    logger.info("Found %d unique subjects.", len(subjects))

    if dataset == "harth":
        splits = make_subject_splits(
            subjects, train_ratio=0.6, val_ratio=0.0, test_ratio=0.4, seed=cfg.seed,
        )
    else:
        splits = make_subject_splits(
            subjects, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=cfg.seed,
        )

    splits_path = output_dir / dataset / "splits.json"
    save_splits(splits, str(splits_path))
    is_clean = verify_no_leakage(splits)
    logger.info("Splits saved. Leakage check: %s", "PASS" if is_clean else "FAIL")
    for split_name, pids in splits.items():
        logger.info("  %s: %d subjects", split_name, len(pids))

    manifest = assign_splits_to_manifest(manifest, splits)
    manifest_path = output_dir / dataset / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Step 3: Build long-context samples
    logger.info("[Step 3/4] Building long-context samples...")
    context_output = output_dir / dataset / "contexts"
    all_contexts = build_all_contexts(
        manifest_dir=str(output_dir / dataset),
        output_dir=str(context_output),
        context_lens=cfg.data.context_lens,
        strides=cfg.data.strides,
    )
    total_ctx = sum(len(v) for v in all_contexts.values())
    logger.info("Built %d context samples across %d configs.", total_ctx, len(all_contexts))

    # Step 4: Statistics
    logger.info("[Step 4/4] Computing statistics...")
    stats_all = {}
    for key, samples in all_contexts.items():
        stats = compute_statistics(samples)
        stats_all[key] = stats
        logger.info("  %s: %d samples, %d participants", key, stats["total"], stats["unique_participants"])

    stats_path = output_dir / dataset / "statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats_all, f, indent=2, default=str)

    logger.info("=== %s preprocessing complete ===", dataset)


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for LCTSCap.")
    parser.add_argument("--dataset", type=str, choices=["capture24", "harth", "all"], required=True)
    parser.add_argument("--config", type=str, default="configs/data/capture24.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.dataset == "all":
        for ds in ["capture24", "harth"]:
            run_preprocessing(ds, cfg)
    else:
        run_preprocessing(args.dataset, cfg)
    logger.info("All preprocessing done.")


if __name__ == "__main__":
    main()
