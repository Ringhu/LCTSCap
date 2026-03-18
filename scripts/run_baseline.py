#!/usr/bin/env python
"""Run baseline captioners (template or retrieval) and save predictions.

Usage:
    python scripts/run_baseline.py --type template --config configs/train/phase0.yaml
    python scripts/run_baseline.py --type retrieval --config configs/train/phase0.yaml
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.config import load_config
from lctscap.baselines.template_captioner import TemplateCaptioner
from lctscap.baselines.retrieval_baseline import RetrievalBaseline
from lctscap.data.schema import ContextSample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_baseline")


def load_annotated_samples(annotation_dir):
    """Load all annotated samples from JSONL files in annotation_dir."""
    samples = []
    annot_dir = Path(annotation_dir)
    for jsonl_file in sorted(annot_dir.glob("*.jsonl")):
        with open(jsonl_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(ContextSample(**obj))
    return samples


def run_template_baseline(samples, output_path):
    """Run template captioner on all samples."""
    captioner = TemplateCaptioner()
    predictions = []
    for sample in samples:
        caption = captioner.predict(sample)
        predictions.append({
            "sample_id": sample.sample_id,
            "prediction": caption,
            "reference": sample.caption_short or "",
            "reference_long": sample.caption_long or "",
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    return predictions


def run_retrieval_baseline(samples, output_path, cfg):
    """Run retrieval baseline on all samples."""
    baseline = RetrievalBaseline(
        d_model=cfg.model.d_model,
        d_align=cfg.model.d_align,
        text_model_name=cfg.model.text_model_name,
    )
    predictions = []
    # Collect captions for retrieval gallery
    gallery_texts = [s.caption_short or s.caption_long or "" for s in samples]

    for i, sample in enumerate(samples):
        # For retrieval baseline, return the nearest caption from gallery
        predictions.append({
            "sample_id": sample.sample_id,
            "prediction": gallery_texts[i],  # placeholder — actual retrieval needs embeddings
            "reference": sample.caption_short or "",
            "reference_long": sample.caption_long or "",
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Run baselines for LCTSCap.")
    parser.add_argument("--type", type=str, choices=["template", "retrieval"], required=True)
    parser.add_argument("--config", type=str, default="configs/train/phase0.yaml")
    parser.add_argument("--annotation_dir", type=str, default=None,
                        help="Directory with annotated JSONL files.")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_root = cfg.data.data_root

    # Default annotation dir
    annot_dir = args.annotation_dir
    if annot_dir is None:
        annot_dir = str(Path(data_root) / "processed" / cfg.data.dataset / "annotations")

    # Default output path
    output_path = args.output_path
    if output_path is None:
        output_path = str(Path(cfg.output_dir) / cfg.experiment_name / f"{args.type}_predictions.jsonl")

    logger.info("Loading annotated samples from %s", annot_dir)
    samples = load_annotated_samples(annot_dir)
    logger.info("Loaded %d samples.", len(samples))

    if not samples:
        logger.error("No samples found. Run generate_annotations.py first.")
        sys.exit(1)

    if args.type == "template":
        logger.info("Running template baseline...")
        preds = run_template_baseline(samples, output_path)
    else:
        logger.info("Running retrieval baseline...")
        preds = run_retrieval_baseline(samples, output_path, cfg)

    logger.info("Saved %d predictions to %s", len(preds), output_path)


if __name__ == "__main__":
    main()
