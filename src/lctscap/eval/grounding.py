"""Temporal grounding metrics for time series captioning."""

from typing import List, Tuple

from scipy.stats import kendalltau


def _iou(span_a: Tuple[int, int], span_b: Tuple[int, int]) -> float:
    """Compute Intersection over Union of two integer spans.

    Each span is (start, end) where end is exclusive.
    """
    start_a, end_a = span_a
    start_b, end_b = span_b
    inter_start = max(start_a, start_b)
    inter_end = min(end_a, end_b)
    intersection = max(0, inter_end - inter_start)
    union = (end_a - start_a) + (end_b - start_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def event_span_iou(
    pred_spans: List[Tuple[int, int]],
    gold_spans: List[Tuple[int, int]],
) -> float:
    """Compute average IoU between matched event spans.

    Uses greedy matching: for each gold span, find the predicted span
    with the highest IoU, pair them, and remove both from further matching.

    Args:
        pred_spans: list of (start, end) tuples from predictions.
        gold_spans: list of (start, end) tuples from ground truth.

    Returns:
        Average IoU across matched pairs.  Returns 0.0 if either list is empty.
    """
    if not pred_spans or not gold_spans:
        return 0.0

    # Build IoU matrix and do greedy matching
    used_pred = set()
    total_iou = 0.0
    matched = 0

    # Sort gold spans by length (longest first) for greedy matching
    gold_sorted = sorted(range(len(gold_spans)), key=lambda i: -(gold_spans[i][1] - gold_spans[i][0]))

    for gi in gold_sorted:
        best_iou = 0.0
        best_pi = -1
        for pi in range(len(pred_spans)):
            if pi in used_pred:
                continue
            cur_iou = _iou(pred_spans[pi], gold_spans[gi])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_pi = pi
        if best_pi >= 0:
            used_pred.add(best_pi)
            total_iou += best_iou
            matched += 1

    # Denominator is the max of pred and gold counts so that
    # unmatched spans penalize the score.
    denom = max(len(pred_spans), len(gold_spans))
    return total_iou / denom


def unsupported_claim_rate(
    pred_claims: List[dict],
    gold_events: List[dict],
    activity_vocab: List[str],
) -> float:
    """Compute the fraction of predicted claims not grounded in gold events.

    A claim is considered grounded if:
      - Its activity type (key "activity") matches the type of some gold event, AND
      - If both have span information (keys "start" and "end"), their IoU > 0.

    Args:
        pred_claims: list of claim dicts, each with at least "activity" key,
                     and optional "start", "end" integer keys.
        gold_events: list of gold event dicts with "type", "start_token", "end_token" keys.
        activity_vocab: vocabulary of known activity names (used for normalization).

    Returns:
        Fraction of unsupported claims in [0, 1].  Returns 0.0 if no claims.
    """
    if not pred_claims:
        return 0.0

    vocab_lower = {a.lower() for a in activity_vocab}
    unsupported = 0

    for claim in pred_claims:
        claim_act = claim.get("activity", "").lower().strip()

        # A claim mentioning an activity not in the vocabulary is always unsupported
        if claim_act and claim_act not in vocab_lower:
            unsupported += 1
            continue

        grounded = False
        for event in gold_events:
            event_type = event.get("type", "").lower().strip()
            if claim_act != event_type:
                continue

            # If both have spans, require IoU > 0
            if all(k in claim for k in ("start", "end")) and all(
                k in event for k in ("start_token", "end_token")
            ):
                iou = _iou(
                    (claim["start"], claim["end"]),
                    (event["start_token"], event["end_token"]),
                )
                if iou > 0:
                    grounded = True
                    break
            else:
                # Activity-level match is enough when spans are absent
                grounded = True
                break

        if not grounded:
            unsupported += 1

    return unsupported / len(pred_claims)


def order_consistency(pred_order: List[str], gold_order: List[str]) -> float:
    """Compute temporal ordering consistency via Kendall's tau.

    Both lists should contain the same set of activity labels in the order
    they appear.  Only the common elements (preserving first-occurrence order)
    are used for comparison.

    Args:
        pred_order: predicted temporal order of activities.
        gold_order: ground-truth temporal order of activities.

    Returns:
        Kendall's tau correlation in [-1, 1].  Returns 1.0 if fewer than
        2 common elements (trivially ordered).
    """
    if not pred_order or not gold_order:
        return 1.0

    # Find common elements in first-occurrence order
    gold_set = set(gold_order)
    pred_common = []
    seen = set()
    for act in pred_order:
        act_lower = act.lower().strip()
        if act_lower in gold_set and act_lower not in seen:
            pred_common.append(act_lower)
            seen.add(act_lower)

    # Build gold order for common elements
    gold_common = []
    seen2 = set()
    for act in gold_order:
        act_lower = act.lower().strip()
        if act_lower in seen and act_lower not in seen2:
            gold_common.append(act_lower)
            seen2.add(act_lower)

    if len(gold_common) < 2:
        return 1.0

    # Map activities to their gold-order indices
    gold_rank = {act: i for i, act in enumerate(gold_common)}

    # Rank the predicted order relative to gold
    pred_ranks = [gold_rank[act] for act in pred_common if act in gold_rank]
    gold_ranks = list(range(len(pred_ranks)))

    if len(pred_ranks) < 2:
        return 1.0

    tau, _ = kendalltau(pred_ranks, gold_ranks)
    # kendalltau returns nan if all values are the same
    if tau != tau:  # nan check
        return 1.0
    return float(tau)
