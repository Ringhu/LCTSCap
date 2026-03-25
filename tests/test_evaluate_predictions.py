import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate import run_evaluation


def test_run_evaluation_uses_structured_event_evidence_metrics():
    predictions = {
        "sample-1": {
            "sample_id": "sample-1",
            "prediction": "walking then sitting",
            "predicted_events": [
                {"activity": "walking", "start_token": 0, "end_token": 10},
                {"activity": "sitting", "start_token": 10, "end_token": 20},
            ],
        }
    }
    gold = {
        "sample-1": {
            "sample_id": "sample-1",
            "caption_short": "walking then sitting",
            "events": [
                {"type": "walking", "start_token": 0, "end_token": 10, "duration_sec": 100},
                {"type": "sitting", "start_token": 10, "end_token": 20, "duration_sec": 100},
            ],
        }
    }

    results = run_evaluation(predictions, gold, skip_bertscore=True)
    assert results["event_evidence_span_iou"] == 1.0
    assert results["event_evidence_precision"] == 1.0


def test_run_evaluation_uses_textual_evidence_spans_for_event_span_iou():
    predictions = {
        "sample-1": {
            "sample_id": "sample-1",
            "prediction": "Evidence: walking windows 0 to 10; sitting windows 10 to 20.",
        }
    }
    gold = {
        "sample-1": {
            "sample_id": "sample-1",
            "caption_short": "walking then sitting",
            "events": [
                {"type": "walking", "start_token": 0, "end_token": 10, "duration_sec": 100},
                {"type": "sitting", "start_token": 10, "end_token": 20, "duration_sec": 100},
            ],
        }
    }

    results = run_evaluation(predictions, gold, skip_bertscore=True)
    assert results["event_span_iou"] == 1.0
