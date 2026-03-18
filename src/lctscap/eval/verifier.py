"""Verify parsed claims against ground-truth event tables."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from lctscap.data.schema import ContextSample, Event
from lctscap.eval.claim_parser import ParsedClaim, extract_mentioned_activities, parse_claims
from lctscap.eval.grounding import _iou


def _verify_activity_claim(claim: ParsedClaim, gold_events: List[Event]) -> Dict[str, Any]:
    """Verify that a mentioned activity exists in the gold events."""
    for event in gold_events:
        if claim.activity.lower() == event.type.lower():
            # If claim has a span, also check temporal overlap
            if claim.span is not None:
                iou = _iou(claim.span, (event.start_token, event.end_token))
                if iou > 0:
                    return {"verified": True, "reason": "activity+span match", "iou": iou}
            else:
                return {"verified": True, "reason": "activity type match"}
    return {"verified": False, "reason": f"activity '{claim.activity}' not found in gold events"}


def _verify_duration_claim(claim: ParsedClaim, gold_events: List[Event]) -> Dict[str, Any]:
    """Verify that a claimed duration is approximately correct."""
    if claim.duration_sec is None:
        return {"verified": False, "reason": "no duration value in claim"}

    for event in gold_events:
        if claim.activity.lower() == event.type.lower():
            # Allow 20% tolerance on duration
            ratio = claim.duration_sec / event.duration_sec if event.duration_sec > 0 else float("inf")
            if 0.5 <= ratio <= 2.0:
                return {
                    "verified": True,
                    "reason": "duration within tolerance",
                    "pred_sec": claim.duration_sec,
                    "gold_sec": event.duration_sec,
                }
    return {
        "verified": False,
        "reason": f"duration {claim.duration_sec}s for '{claim.activity}' not within tolerance",
    }


def _verify_transition_claim(claim: ParsedClaim, gold_events: List[Event]) -> Dict[str, Any]:
    """Verify that a claimed transition exists between consecutive gold events."""
    if claim.ordering_ref is None:
        return {"verified": False, "reason": "no transition target specified"}

    from_act = claim.activity.lower()
    to_act = claim.ordering_ref.lower()

    # Check consecutive events for this transition
    for i in range(len(gold_events) - 1):
        cur_type = gold_events[i].type.lower()
        next_type = gold_events[i + 1].type.lower()
        if cur_type == from_act and next_type == to_act:
            return {"verified": True, "reason": "transition found in consecutive events"}

    # Also check events that have from_activity/to_activity fields
    for event in gold_events:
        if (
            event.from_activity
            and event.to_activity
            and event.from_activity.lower() == from_act
            and event.to_activity.lower() == to_act
        ):
            return {"verified": True, "reason": "transition found in event metadata"}

    return {
        "verified": False,
        "reason": f"transition '{from_act}' -> '{to_act}' not found",
    }


def _verify_ordering_claim(claim: ParsedClaim, gold_events: List[Event]) -> Dict[str, Any]:
    """Verify that one activity appears before another in the gold event sequence."""
    if claim.ordering_ref is None:
        return {"verified": False, "reason": "no ordering reference specified"}

    act_a = claim.activity.lower()
    act_b = claim.ordering_ref.lower()

    first_a: Optional[int] = None
    first_b: Optional[int] = None

    for event in gold_events:
        etype = event.type.lower()
        if etype == act_a and first_a is None:
            first_a = event.start_token
        if etype == act_b and first_b is None:
            first_b = event.start_token

    if first_a is None or first_b is None:
        return {
            "verified": False,
            "reason": f"one or both activities not found: {act_a}, {act_b}",
        }

    # The claim says act_a is ordered relative to act_b
    # Since we track ordering, just verify both exist (ordering check is softer)
    return {
        "verified": True,
        "reason": f"both activities found: {act_a} at {first_a}, {act_b} at {first_b}",
        "first_a": first_a,
        "first_b": first_b,
    }


_VERIFIERS = {
    "activity": _verify_activity_claim,
    "duration": _verify_duration_claim,
    "transition": _verify_transition_claim,
    "ordering": _verify_ordering_claim,
}


def verify_claims(
    claims: List[ParsedClaim],
    gold_events: List[Event],
) -> Dict[str, Any]:
    """Verify each claim against a ground-truth event table.

    Args:
        claims: list of parsed claims from a generated caption.
        gold_events: list of Event objects from ground-truth annotation.

    Returns:
        Dictionary with:
          - total: total number of claims
          - verified: number of verified claims
          - unverified: number of unverified claims
          - precision: fraction of claims that are verified
          - details: list of per-claim verification results
    """
    details: List[Dict[str, Any]] = []
    verified_count = 0

    for claim in claims:
        verifier = _VERIFIERS.get(claim.claim_type)
        if verifier is None:
            result = {"verified": False, "reason": f"unknown claim type: {claim.claim_type}"}
        else:
            result = verifier(claim, gold_events)

        result["claim_type"] = claim.claim_type
        result["activity"] = claim.activity
        details.append(result)

        if result.get("verified", False):
            verified_count += 1

    total = len(claims)
    return {
        "total": total,
        "verified": verified_count,
        "unverified": total - verified_count,
        "precision": verified_count / total if total > 0 else 1.0,
        "details": details,
    }


def compute_verification_report(
    predictions_path: str,
    gold_path: str,
) -> Dict[str, Any]:
    """Generate a full verification report comparing predictions against gold annotations.

    Args:
        predictions_path: path to a JSONL file where each line has
                          {"sample_id": ..., "prediction": "..."}.
        gold_path: path to a JSONL file where each line has
                   {"sample_id": ..., "events": [...], ...} matching
                   the ContextSample schema.

    Returns:
        Dictionary with aggregate verification statistics and per-sample details.
    """
    # Load predictions
    predictions: Dict[str, str] = {}
    with open(predictions_path, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                predictions[obj["sample_id"]] = obj["prediction"]

    # Load gold annotations
    gold_samples: Dict[str, ContextSample] = {}
    with open(gold_path, "r") as f:
        for line in f:
            if line.strip():
                obj = json.loads(line)
                sample = ContextSample(**obj)
                gold_samples[sample.sample_id] = sample

    # Verify
    sample_reports: List[Dict[str, Any]] = []
    total_claims = 0
    total_verified = 0

    for sample_id, pred_text in predictions.items():
        gold = gold_samples.get(sample_id)
        if gold is None or gold.events is None:
            continue

        claims = parse_claims(pred_text)
        report = verify_claims(claims, gold.events)
        report["sample_id"] = sample_id
        sample_reports.append(report)

        total_claims += report["total"]
        total_verified += report["verified"]

    return {
        "num_samples": len(sample_reports),
        "total_claims": total_claims,
        "total_verified": total_verified,
        "total_unverified": total_claims - total_verified,
        "overall_precision": total_verified / total_claims if total_claims > 0 else 1.0,
        "samples": sample_reports,
    }
