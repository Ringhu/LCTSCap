#!/usr/bin/env python
"""Evaluate predictions against gold references.

Computes classic metrics (BLEU, ROUGE, METEOR, BERTScore),
factuality metrics, grounding metrics, and generates comparison reports.

Usage:
    python scripts/evaluate.py \
        --predictions_path outputs/template_predictions.jsonl \
        --gold_path /path/to/lctscap_data/processed/capture24/annotations \
        --output_dir outputs/eval_results
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.eval.classic_metrics import compute_all_classic
from lctscap.data.schema import Event
from lctscap.eval.claim_parser import ACTIVITY_VOCAB, extract_temporal_order, parse_claims
from lctscap.eval.factuality import (
    activity_mention_f1,
    dominant_activity_accuracy,
    duration_bin_accuracy,
    transition_accuracy,
)
from lctscap.eval.grounding import event_span_iou, order_consistency, unsupported_claim_rate
from lctscap.eval.verifier import verify_claims
from lctscap.eval.report import compare_models, results_to_csv, results_to_markdown

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def load_predictions(path):
    """Load predictions from a JSONL file."""
    preds = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            preds[obj["sample_id"]] = obj
    return preds


def load_gold_samples(path):
    """Load gold samples from a JSONL file or directory of JSONL files."""
    gold = {}
    p = Path(path)
    if p.is_dir():
        files = sorted(p.glob("*.jsonl"))
    else:
        files = [p]

    for f_path in files:
        with open(f_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                gold[obj["sample_id"]] = obj
    return gold


def extract_activities_from_text(text):
    """Extract mentioned activities from caption text (simple keyword matching)."""
    activities = set()
    known = [
        "walking", "running", "sitting", "standing", "sleeping", "lying",
        "cycling", "vehicle", "household", "eating", "self_care", "screen_time",
        "socializing", "stairs", "shuffling", "stairs_up", "stairs_down",
        "nordic_walking", "jumping", "other",
    ]
    text_lower = text.lower()
    for act in known:
        if act.replace("_", " ") in text_lower or act in text_lower:
            activities.add(act)
    return activities


def extract_events_info(sample_dict):
    """Extract structured event info from a gold sample dict."""
    events = sample_dict.get("events", [])
    if not events:
        return [], "unknown", {}, [], []

    # Activity types
    activities = set()
    dominant = "unknown"
    transitions = []
    durations = {}
    spans = []

    max_dur = 0
    for e in events:
        etype = e.get("type", "unknown")
        activities.add(etype)
        dur = e.get("duration_sec", 0)
        durations[etype] = durations.get(etype, 0) + dur
        if dur > max_dur:
            max_dur = dur
            dominant = etype
        if e.get("from_activity"):
            transitions.append((e["from_activity"], etype))
        spans.append((e.get("start_token", 0), e.get("end_token", 0)))

    return list(activities), dominant, durations, transitions, spans


def extract_prediction_info(prediction_text):
    """Extract structured claims from a predicted caption."""
    claims = parse_claims(prediction_text)

    activities = set()
    durations: Dict[str, float] = {}
    transitions: List[Tuple[str, str]] = []
    spans: List[Tuple[int, int]] = []

    for claim in claims:
        if claim.activity:
            activities.add(claim.activity)
        if claim.claim_type == "duration" and claim.duration_sec is not None and claim.activity:
            durations[claim.activity] = max(durations.get(claim.activity, 0.0), claim.duration_sec)
        elif claim.claim_type == "transition" and claim.ordering_ref:
            transitions.append((claim.activity, claim.ordering_ref))
        if claim.span is not None:
            spans.append(claim.span)

    pred_order = extract_temporal_order(prediction_text)
    dominant = pred_order[0] if pred_order else "unknown"

    return {
        "claims": claims,
        "activities": activities,
        "dominant": dominant,
        "durations": durations,
        "transitions": transitions,
        "spans": spans,
        "order": pred_order,
    }


def run_evaluation(predictions, gold, skip_bertscore=False):
    """Run all evaluation metrics."""
    # Align predictions and gold by sample_id
    common_ids = sorted(set(predictions.keys()) & set(gold.keys()))
    if not common_ids:
        logger.error("No matching sample_ids between predictions and gold.")
        return {}

    logger.info("Evaluating %d matched samples.", len(common_ids))

    pred_texts = [predictions[sid].get("prediction", "") for sid in common_ids]
    ref_texts = [gold[sid].get("caption_short", "") or gold[sid].get("caption_long", "") for sid in common_ids]

    results = {}

    # 1. Classic metrics
    logger.info("Computing classic metrics...")
    try:
        if skip_bertscore:
            from lctscap.eval.classic_metrics import compute_bleu, compute_rouge, compute_meteor
            results.update(compute_bleu(pred_texts, ref_texts))
            results.update(compute_rouge(pred_texts, ref_texts))
            results["meteor"] = compute_meteor(pred_texts, ref_texts)
        else:
            results.update(compute_all_classic(pred_texts, ref_texts))
    except Exception as e:
        logger.warning("Classic metrics failed: %s", e)

    # 2. Factuality metrics
    logger.info("Computing factuality metrics...")
    pred_activities: List[Set[str]] = []
    gold_activities: List[Set[str]] = []
    pred_dominant: List[str] = []
    gold_dominant: List[str] = []
    pred_durations: List[Dict[str, float]] = []
    gold_durations: List[Dict[str, float]] = []
    pred_transitions: List[List[Tuple[str, str]]] = []
    gold_transitions: List[List[Tuple[str, str]]] = []

    for sid in common_ids:
        # Pred: extract from text
        pred_info = extract_prediction_info(predictions[sid].get("prediction", ""))
        pred_act = pred_info["activities"] or extract_activities_from_text(predictions[sid].get("prediction", ""))
        pred_activities.append(pred_act)

        # Gold: extract from events
        g_acts, g_dom, g_durs, g_trans, _ = extract_events_info(gold[sid])
        gold_activities.append(set(g_acts))
        gold_dominant.append(g_dom)
        gold_durations.append(g_durs)
        gold_transitions.append(g_trans)

        pred_dominant.append(pred_info["dominant"])
        pred_durations.append(pred_info["durations"])
        pred_transitions.append(pred_info["transitions"])

    try:
        results.update(activity_mention_f1(pred_activities, gold_activities))
    except Exception as e:
        logger.warning("Activity F1 failed: %s", e)

    try:
        results["dominant_accuracy"] = dominant_activity_accuracy(pred_dominant, gold_dominant)
    except Exception as e:
        logger.warning("Dominant accuracy failed: %s", e)

    try:
        results["transition_accuracy"] = transition_accuracy(pred_transitions, gold_transitions)
    except Exception as e:
        logger.warning("Transition accuracy failed: %s", e)

    try:
        results["duration_bin_accuracy"] = duration_bin_accuracy(pred_durations, gold_durations)
    except Exception as e:
        logger.warning("Duration bin accuracy failed: %s", e)

    # 3. Grounding metrics
    logger.info("Computing grounding metrics...")
    iou_scores = []
    oc_scores = []
    unsupported_scores = []
    verification_precisions = []

    for sid in common_ids:
        g_acts, _, _, _, g_spans = extract_events_info(gold[sid])
        pred_info = extract_prediction_info(predictions[sid].get("prediction", ""))
        pred_act_list = pred_info["order"] or list(pred_info["activities"])

        if pred_info["spans"] or g_spans:
            iou_scores.append(event_span_iou(pred_info["spans"], g_spans))

        # Per-sample order consistency
        if len(g_acts) >= 2:
            oc_scores.append(order_consistency(pred_act_list, g_acts))

        claim_dicts = []
        for claim in pred_info["claims"]:
            item = {"activity": claim.activity}
            if claim.span is not None:
                item["start"], item["end"] = claim.span
            claim_dicts.append(item)

        unsupported_scores.append(
            unsupported_claim_rate(claim_dicts, gold[sid].get("events", []), ACTIVITY_VOCAB)
        )
        verification_report = verify_claims(
            pred_info["claims"],
            [
                event if isinstance(event, Event) else Event(**event)
                for event in gold[sid].get("events", [])
            ],
        )
        verification_precisions.append(verification_report["precision"])

    if iou_scores:
        results["event_span_iou"] = sum(iou_scores) / len(iou_scores)
    if oc_scores:
        results["order_consistency"] = sum(oc_scores) / len(oc_scores)
    if unsupported_scores:
        results["unsupported_claim_rate"] = sum(unsupported_scores) / len(unsupported_scores)
    if verification_precisions:
        results["verification_precision"] = sum(verification_precisions) / len(verification_precisions)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate LCTSCap predictions.")
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument("--gold_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/eval_results")
    parser.add_argument("--skip_bertscore", action="store_true",
                        help="Skip BERTScore (faster evaluation).")
    parser.add_argument("--compare_with", type=str, nargs="*", default=[],
                        help="Additional prediction files for comparison.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading predictions from %s", args.predictions_path)
    predictions = load_predictions(args.predictions_path)
    logger.info("Loaded %d predictions.", len(predictions))

    logger.info("Loading gold data from %s", args.gold_path)
    gold = load_gold_samples(args.gold_path)
    logger.info("Loaded %d gold samples.", len(gold))

    # Main evaluation
    results = run_evaluation(predictions, gold, args.skip_bertscore)

    # Save results
    results_to_csv(results, str(output_dir / "results.csv"))
    md_table = results_to_markdown(results)
    with open(output_dir / "results.md", "w") as f:
        f.write("# Evaluation Results\n\n")
        f.write(md_table)
        f.write("\n")

    logger.info("Results:\n%s", md_table)
    logger.info("Saved to %s", output_dir)

    # Comparison (if additional prediction files provided)
    if args.compare_with:
        all_results = [results]
        all_names = [Path(args.predictions_path).stem]

        for comp_path in args.compare_with:
            logger.info("Loading comparison predictions from %s", comp_path)
            comp_preds = load_predictions(comp_path)
            comp_results = run_evaluation(comp_preds, gold, args.skip_bertscore)
            all_results.append(comp_results)
            all_names.append(Path(comp_path).stem)

        comparison = compare_models(all_results, all_names)
        with open(output_dir / "comparison.md", "w") as f:
            f.write("# Model Comparison\n\n")
            f.write(comparison)
            f.write("\n")
        logger.info("Comparison table:\n%s", comparison)


if __name__ == "__main__":
    main()
