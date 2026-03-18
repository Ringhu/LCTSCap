#!/usr/bin/env python
"""Generate event tables and template captions for long-context samples.

Usage:
    python scripts/generate_annotations.py \
        --manifest_dir /path/to/lctscap_data/processed/capture24
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.data.annotation import annotate_sample
from lctscap.data.schema import ContextSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_annotations")


def load_window_labels(manifest_path: str) -> Dict[str, str]:
    """Load window_id -> activity label mapping from window manifest."""
    with open(manifest_path, "r") as f:
        windows = json.load(f)
    return {w["window_id"]: w["label"] for w in windows}


def annotate_context_file(context_path, window_labels, output_path, segment_size=32):
    """Annotate all context samples in a single JSON file."""
    with open(context_path, "r") as f:
        raw_samples = json.load(f)

    annotated = []
    errors = 0
    for entry in raw_samples:
        sample = ContextSample(**entry)
        labels = [window_labels.get(wid, "unknown") for wid in sample.window_ids]
        try:
            annotated_sample = annotate_sample(sample, labels, segment_size)
            annotated.append(annotated_sample.model_dump())
        except Exception as e:
            logger.warning("Failed to annotate %s: %s", sample.sample_id, e)
            errors += 1

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in annotated:
            f.write(json.dumps(item) + "\n")

    if errors:
        logger.warning("  %d/%d samples had errors.", errors, len(raw_samples))
    return len(annotated)


def main():
    parser = argparse.ArgumentParser(description="Generate annotations for LCTSCap.")
    parser.add_argument("--manifest_dir", type=str, required=True)
    parser.add_argument("--context_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--segment_size", type=int, default=32)
    args = parser.parse_args()

    manifest_dir = Path(args.manifest_dir)
    context_dir = Path(args.context_dir) if args.context_dir else manifest_dir / "contexts"
    output_dir = Path(args.output_dir) if args.output_dir else manifest_dir / "annotations"

    manifest_path = manifest_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error("Window manifest not found: %s", manifest_path)
        sys.exit(1)

    logger.info("Loading window labels from %s", manifest_path)
    window_labels = load_window_labels(str(manifest_path))
    logger.info("Loaded %d window labels.", len(window_labels))

    context_files = sorted(context_dir.glob("*.json"))
    if not context_files:
        logger.error("No context files found in %s", context_dir)
        sys.exit(1)

    logger.info("Found %d context files to annotate.", len(context_files))
    total = 0
    for ctx_file in context_files:
        out_file = output_dir / f"{ctx_file.stem}_annotated.jsonl"
        logger.info("Annotating %s -> %s", ctx_file.name, out_file)
        n = annotate_context_file(str(ctx_file), window_labels, str(out_file), args.segment_size)
        total += n
        logger.info("  Annotated %d samples.", n)

    logger.info("Done. Total: %d samples across %d files.", total, len(context_files))


if __name__ == "__main__":
    main()
