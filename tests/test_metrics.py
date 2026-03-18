"""Tests for evaluation metrics."""
import pytest


# ---------------------------------------------------------------------------
# Classic metrics
# ---------------------------------------------------------------------------

def test_compute_bleu_perfect():
    from lctscap.eval.classic_metrics import compute_bleu
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    result = compute_bleu(preds, refs)
    assert "bleu_1" in result
    assert result["bleu_1"] > 0.9


def test_compute_bleu_mismatch():
    from lctscap.eval.classic_metrics import compute_bleu
    preds = ["hello world"]
    refs = ["the cat sat on the mat"]
    result = compute_bleu(preds, refs)
    assert result["bleu_4"] < 0.1


def test_compute_rouge():
    from lctscap.eval.classic_metrics import compute_rouge
    preds = ["walking for ten minutes then sitting"]
    refs = ["walking for ten minutes then sitting"]
    result = compute_rouge(preds, refs)
    assert "rouge_l" in result
    assert result["rouge_l"] > 0.9


def test_compute_meteor():
    from lctscap.eval.classic_metrics import compute_meteor
    preds = ["the participant walked"]
    refs = ["the participant walked"]
    result = compute_meteor(preds, refs)
    assert isinstance(result, float)
    assert result > 0.9


# ---------------------------------------------------------------------------
# Factuality metrics
# ---------------------------------------------------------------------------

def test_activity_mention_f1_perfect():
    from lctscap.eval.factuality import activity_mention_f1
    pred = [{"walking", "sitting"}]
    gold = [{"walking", "sitting"}]
    result = activity_mention_f1(pred, gold)
    assert result["activity_f1"] == 1.0


def test_activity_mention_f1_partial():
    from lctscap.eval.factuality import activity_mention_f1
    pred = [{"walking"}]
    gold = [{"walking", "sitting"}]
    result = activity_mention_f1(pred, gold)
    assert 0.0 < result["activity_f1"] < 1.0


def test_dominant_activity_accuracy():
    from lctscap.eval.factuality import dominant_activity_accuracy
    preds = ["walking", "sitting", "walking"]
    golds = ["walking", "sitting", "standing"]
    result = dominant_activity_accuracy(preds, golds)
    assert isinstance(result, float)
    assert abs(result - 2.0 / 3.0) < 1e-6


# ---------------------------------------------------------------------------
# Grounding metrics
# ---------------------------------------------------------------------------

def test_event_span_iou_perfect():
    from lctscap.eval.grounding import event_span_iou
    spans = [(0, 10), (10, 20)]
    assert event_span_iou(spans, spans) == 1.0


def test_event_span_iou_partial():
    from lctscap.eval.grounding import event_span_iou
    pred = [(0, 15)]
    gold = [(0, 10)]
    iou = event_span_iou(pred, gold)
    assert 0.0 < iou < 1.0


def test_order_consistency_perfect():
    from lctscap.eval.grounding import order_consistency
    order = ["walking", "sitting", "standing"]
    assert order_consistency(order, order) == 1.0


def test_order_consistency_reversed():
    from lctscap.eval.grounding import order_consistency
    pred = ["standing", "sitting", "walking"]
    gold = ["walking", "sitting", "standing"]
    score = order_consistency(pred, gold)
    assert score < 0.5


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def test_retrieval_recall_identity():
    from lctscap.eval.retrieval import compute_retrieval_metrics
    import torch
    # Identity similarity matrix → perfect retrieval
    n = 10
    sim = torch.eye(n)
    metrics = compute_retrieval_metrics(sim, ks=[1, 5])
    assert metrics["t2s_R@1"] == 1.0
    assert metrics["s2t_R@1"] == 1.0
