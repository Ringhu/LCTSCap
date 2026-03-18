"""Tests for loss functions."""
import pytest
import torch
from lctscap.models.losses import clip_infonce, event_loss, coverage_loss, compute_total_loss


def test_clip_infonce_positive():
    z_a = torch.randn(8, 32)
    z_b = torch.randn(8, 32)
    z_a = z_a / z_a.norm(dim=-1, keepdim=True)
    z_b = z_b / z_b.norm(dim=-1, keepdim=True)
    scale = torch.tensor(1.0).exp()
    loss = clip_infonce(z_a, z_b, scale)
    assert loss.item() > 0


def test_clip_infonce_identical():
    """Identical embeddings should have lower loss than random."""
    z = torch.randn(8, 32)
    z = z / z.norm(dim=-1, keepdim=True)
    scale = torch.tensor(2.0).exp()
    loss_identical = clip_infonce(z, z, scale)

    z_b = torch.randn(8, 32)
    z_b = z_b / z_b.norm(dim=-1, keepdim=True)
    loss_random = clip_infonce(z, z_b, scale)

    assert loss_identical.item() < loss_random.item()


def test_clip_infonce_batch_1():
    z = torch.randn(1, 16)
    z = z / z.norm(dim=-1, keepdim=True)
    scale = torch.tensor(1.0)
    loss = clip_infonce(z, z, scale)
    assert loss.item() >= 0


def test_event_loss_basic():
    B, T, n_types = 2, 8, 5
    pred_types = torch.randn(B, T, n_types)
    pred_spans = torch.randn(B, T, 2)
    gt = {
        "type_labels": torch.randint(0, n_types, (B, T)),
        "span_targets": torch.randn(B, T, 2),
        "span_mask": torch.ones(B, T),
    }
    loss = event_loss(pred_types, pred_spans, gt, context_len=T)
    assert loss.item() > 0


def test_event_loss_no_spans():
    B, T, n_types = 2, 8, 5
    pred_types = torch.randn(B, T, n_types)
    pred_spans = torch.randn(B, T, 2)
    gt = {
        "type_labels": torch.randint(0, n_types, (B, T)),
        "span_targets": torch.randn(B, T, 2),
        "span_mask": torch.zeros(B, T),  # no valid spans
    }
    loss = event_loss(pred_types, pred_spans, gt, context_len=T)
    # Should still compute type loss even without span loss
    assert loss.item() > 0


def test_event_loss_ignore_index():
    B, T, n_types = 2, 8, 5
    pred_types = torch.randn(B, T, n_types)
    pred_spans = torch.randn(B, T, 2)
    labels = torch.full((B, T), -100, dtype=torch.long)
    labels[:, :4] = torch.randint(0, n_types, (B, 4))
    gt = {
        "type_labels": labels,
        "span_targets": torch.randn(B, T, 2),
        "span_mask": torch.zeros(B, T),
    }
    loss = event_loss(pred_types, pred_spans, gt, context_len=T)
    assert not torch.isnan(loss)


def test_coverage_loss_basic():
    B, S, V = 2, 10, 50
    logits = torch.randn(B, S, V)
    event_types = torch.tensor([[0, 1, -1], [2, 3, 1]])
    loss = coverage_loss(logits, event_types, vocab=None, top_k=2)
    assert loss.item() >= 0


def test_coverage_loss_no_events():
    B, S, V = 2, 10, 50
    logits = torch.randn(B, S, V)
    event_types = torch.full((B, 3), -1, dtype=torch.long)
    loss = coverage_loss(logits, event_types, vocab=None, top_k=2)
    assert loss.item() == 0.0


def test_compute_total_loss_caption_only():
    cap_logits = torch.randn(2, 10, 100)
    cap_targets = torch.randint(0, 100, (2, 10))
    total, components = compute_total_loss(
        cap_logits=cap_logits,
        cap_targets=cap_targets,
        z_ts=None, z_text=None, logit_scale=None,
        event_preds=None, event_targets=None,
        coverage_score=None,
    )
    assert total.item() > 0
    assert "caption" in components
    assert "total" in components
    assert "align" not in components
    assert "event" not in components


def test_compute_total_loss_all_components():
    B, T, S, V = 2, 8, 10, 100
    n_types = 5
    cap_logits = torch.randn(B, S, V)
    cap_targets = torch.randint(0, V, (B, S))
    z_ts = torch.randn(B, 32)
    z_text = torch.randn(B, 32)
    z_ts = z_ts / z_ts.norm(dim=-1, keepdim=True)
    z_text = z_text / z_text.norm(dim=-1, keepdim=True)
    logit_scale = torch.tensor(1.0).exp()
    event_preds = {
        "event_type_logits": torch.randn(B, T, n_types),
        "span_logits": torch.randn(B, T, 2),
    }
    event_targets = {
        "type_labels": torch.randint(0, n_types, (B, T)),
        "span_targets": torch.randn(B, T, 2),
        "span_mask": torch.ones(B, T),
    }
    coverage = torch.tensor(0.5)

    total, components = compute_total_loss(
        cap_logits=cap_logits,
        cap_targets=cap_targets,
        z_ts=z_ts, z_text=z_text, logit_scale=logit_scale,
        event_preds=event_preds, event_targets=event_targets,
        coverage_score=coverage,
    )
    assert total.item() > 0
    assert "caption" in components
    assert "align" in components
    assert "event" in components
    assert "coverage" in components
    assert "total" in components


def test_compute_total_loss_custom_weights():
    cap_logits = torch.randn(2, 10, 100)
    cap_targets = torch.randint(0, 100, (2, 10))
    weights = {"caption": 2.0, "align": 0.0, "event": 0.0, "coverage": 0.0}
    total, components = compute_total_loss(
        cap_logits=cap_logits,
        cap_targets=cap_targets,
        z_ts=None, z_text=None, logit_scale=None,
        event_preds=None, event_targets=None,
        coverage_score=None,
        weights=weights,
    )
    # Total should be 2x the caption loss
    assert abs(total.item() - 2.0 * components["caption"]) < 1e-4
