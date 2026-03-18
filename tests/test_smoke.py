"""Smoke tests for LCTSCap: config loading, data schema, model forward pass, losses, annotation."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# 1. Config loading
# ---------------------------------------------------------------------------

def test_default_config():
    """Default LCTSCapConfig should be constructable with sane defaults."""
    from lctscap.config import LCTSCapConfig
    cfg = LCTSCapConfig()
    assert cfg.data.sample_rate == 50
    assert cfg.data.window_sec == 10
    assert cfg.model.d_model == 512
    assert cfg.train.max_epochs == 30
    assert cfg.seed == 42


def test_load_config_missing_file():
    """Loading a non-existent YAML returns default config."""
    from lctscap.config import load_config
    cfg = load_config("/tmp/_nonexistent_config_12345.yaml")
    assert cfg.data.dataset == "capture24"


def test_load_config_from_yaml(tmp_path):
    """Loading a YAML file should override specified values."""
    from lctscap.config import load_config
    yaml_content = "seed: 123\nexperiment_name: test_run\n"
    yaml_file = tmp_path / "test.yaml"
    yaml_file.write_text(yaml_content)
    cfg = load_config(str(yaml_file))
    assert cfg.seed == 123
    assert cfg.experiment_name == "test_run"


def test_data_config_channels():
    """DataConfig.channels_for should return correct channel counts."""
    from lctscap.config import DataConfig
    dc = DataConfig()
    assert dc.channels_for("capture24") == 3
    assert dc.channels_for("harth") == 6
    with pytest.raises(ValueError):
        dc.channels_for("unknown_dataset")


def test_train_config_batch_size():
    """TrainConfig.batch_size_for should handle known and unknown ctx lengths."""
    from lctscap.config import TrainConfig
    tc = TrainConfig()
    assert tc.batch_size_for(128) == 32
    assert tc.batch_size_for(256) == 16
    assert tc.batch_size_for(512) == 8
    assert tc.batch_size_for(1024) == 8  # fallback to min


# ---------------------------------------------------------------------------
# 2. Data schema
# ---------------------------------------------------------------------------

def test_event_schema():
    """Event should be constructable with required fields."""
    from lctscap.data.schema import Event
    e = Event(type="walking", start_token=0, end_token=10, duration_sec=100.0)
    assert e.type == "walking"
    assert e.duration_sec == 100.0
    assert e.is_dominant is False


def test_context_sample_schema():
    """ContextSample should accept all fields."""
    from lctscap.data.schema import ContextSample
    cs = ContextSample(
        sample_id="test_001",
        dataset="capture24",
        participant_id="P001",
        split="train",
        context_len=128,
        stride=32,
        start_window_idx=0,
        end_window_idx=128,
        window_ids=[f"P001_w{i:06d}" for i in range(128)],
        tensor_paths=[f"/tmp/w{i}.pt" for i in range(128)],
    )
    assert cs.context_len == 128
    assert len(cs.window_ids) == 128


# ---------------------------------------------------------------------------
# 3. Annotation pipeline
# ---------------------------------------------------------------------------

def test_extract_events():
    """extract_events should merge consecutive same-label windows."""
    from lctscap.data.annotation import extract_events
    labels = ["walking"] * 5 + ["sitting"] * 3 + ["walking"] * 2
    events = extract_events(labels, context_len=10)
    assert len(events) == 3
    assert events[0].type == "walking"
    assert events[0].end_token == 5
    assert events[1].type == "sitting"
    assert events[2].type == "walking"
    # Check dominant marking
    dominant_events = [e for e in events if e.is_dominant]
    assert len(dominant_events) == 1
    assert dominant_events[0].type == "walking"


def test_generate_short_caption():
    """generate_short_caption should return a non-empty string."""
    from lctscap.data.annotation import extract_events, compute_event_stats, generate_short_caption
    labels = ["sitting"] * 10
    events = extract_events(labels, context_len=10)
    stats = compute_event_stats(events)
    caption = generate_short_caption(events, stats)
    assert isinstance(caption, str)
    assert len(caption) > 0
    assert "sitting" in caption.lower()


def test_annotate_sample():
    """annotate_sample should populate all annotation fields."""
    from lctscap.data.annotation import annotate_sample
    from lctscap.data.schema import ContextSample
    sample = ContextSample(
        sample_id="test_annot",
        dataset="capture24",
        participant_id="P001",
        split="train",
        context_len=64,
        stride=32,
        start_window_idx=0,
        end_window_idx=64,
        window_ids=[f"w{i}" for i in range(64)],
        tensor_paths=[f"/tmp/w{i}.pt" for i in range(64)],
    )
    labels = ["walking"] * 30 + ["sitting"] * 20 + ["standing"] * 14
    annotated = annotate_sample(sample, labels, segment_size=32)
    assert annotated.events is not None
    assert len(annotated.events) == 3
    assert annotated.caption_short is not None
    assert annotated.caption_long is not None
    assert annotated.segment_summaries is not None
    assert annotated.evidence_bullets is not None


# ---------------------------------------------------------------------------
# 4. Splits
# ---------------------------------------------------------------------------

def test_make_subject_splits():
    """make_subject_splits should produce disjoint splits."""
    from lctscap.data.splits import make_subject_splits, verify_no_leakage
    subjects = [f"S{i:03d}" for i in range(20)]
    splits = make_subject_splits(subjects, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    assert verify_no_leakage(splits)
    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    assert total == 20


# ---------------------------------------------------------------------------
# 5. Model forward pass (minimal)
# ---------------------------------------------------------------------------

def test_local_encoder_forward():
    """LocalEncoder should map [B*T*C, 1, L] -> [B*T*C, d_model]."""
    from lctscap.models.local_encoder import LocalEncoder
    enc = LocalEncoder(d_model=64, patch_size=25, num_layers=1, num_heads=4, max_patches=20)
    x = torch.randn(4, 1, 500)  # 4 channel-windows, 1 input channel, 500 samples
    out = enc(x)
    assert out.shape == (4, 64)


def test_channel_fusion_forward():
    """ChannelFusion should map [B*T, C, d] -> [B*T, d]."""
    from lctscap.models.channel_fusion import ChannelFusion
    fusion = ChannelFusion(d_model=64, num_channels=3)
    x = torch.randn(8, 3, 64)  # 8 time-steps, 3 channels, dim=64
    out = fusion(x)
    assert out.shape == (8, 64)


def test_planner_forward():
    """HierarchicalPlanner should produce H_token and H_seg."""
    from lctscap.models.planner import HierarchicalPlanner
    planner = HierarchicalPlanner(d_model=64, num_layers=1, num_heads=4, segment_size=4)
    x = torch.randn(2, 8, 64)  # B=2, T=8, d=64
    h_token, h_seg = planner(x)
    assert h_token.shape == (2, 8, 64)
    assert h_seg.shape == (2, 2, 64)  # 8 / 4 = 2 segments


def test_full_model_forward():
    """LCTSCapModel should accept [B, T, C, L] and return dict with expected keys."""
    from lctscap.models.full_model import LCTSCapModel, ModelConfig
    cfg = ModelConfig(
        d_model=64,
        num_channels=3,
        patch_size=25,
        local_encoder_layers=1,
        local_encoder_heads=2,
        planner_layers=1,
        planner_heads=2,
        segment_size=4,
        decoder_layers=1,
        decoder_heads=2,
        vocab_size=100,
        max_seq_len=16,
        n_event_types=5,
        max_events=4,
        d_align=32,
        text_model_name="all-MiniLM-L6-v2",
        no_align=True,  # skip text encoder to avoid download in CI
    )
    model = LCTSCapModel(cfg)
    model.eval()

    B, T, C, L = 2, 8, 3, 500
    x = torch.randn(B, T, C, L)
    target_ids = torch.randint(0, 100, (B, 10))

    with torch.no_grad():
        out = model(x, target_ids=target_ids)

    assert "H_token" in out
    assert out["H_token"].shape == (B, T, 64)
    assert "H_seg" in out
    assert out["caption_logits"] is not None
    assert out["caption_logits"].shape[0] == B
    assert out["event_proposals"] is not None


# ---------------------------------------------------------------------------
# 6. Loss functions
# ---------------------------------------------------------------------------

def test_clip_infonce():
    """clip_infonce should return a positive scalar."""
    from lctscap.models.losses import clip_infonce
    z_a = torch.randn(4, 32)
    z_b = torch.randn(4, 32)
    z_a = z_a / z_a.norm(dim=-1, keepdim=True)
    z_b = z_b / z_b.norm(dim=-1, keepdim=True)
    scale = torch.tensor(1.0).exp()
    loss = clip_infonce(z_a, z_b, scale)
    assert loss.item() > 0


def test_compute_total_loss():
    """compute_total_loss should combine caption + optional losses."""
    from lctscap.models.losses import compute_total_loss
    cap_logits = torch.randn(2, 10, 100)
    cap_targets = torch.randint(0, 100, (2, 10))
    total, components = compute_total_loss(
        cap_logits=cap_logits,
        cap_targets=cap_targets,
        z_ts=None,
        z_text=None,
        logit_scale=None,
        event_preds=None,
        event_targets=None,
        coverage_score=None,
    )
    assert total.item() > 0
    assert "caption" in components
    assert "total" in components


# ---------------------------------------------------------------------------
# 7. Evaluation metrics
# ---------------------------------------------------------------------------

def test_bleu():
    """compute_bleu should return dict with bleu_1..bleu_4."""
    from lctscap.eval.classic_metrics import compute_bleu
    preds = ["the cat sat on the mat"]
    refs = ["the cat sat on the mat"]
    result = compute_bleu(preds, refs)
    assert "bleu_1" in result
    assert result["bleu_1"] > 0.9  # near-perfect match


def test_rouge():
    """compute_rouge should return dict with rouge_1, rouge_2, rouge_l."""
    from lctscap.eval.classic_metrics import compute_rouge
    preds = ["the participant was walking for ten minutes"]
    refs = ["the participant was walking for ten minutes"]
    result = compute_rouge(preds, refs)
    assert result["rouge_l"] > 0.9


def test_activity_mention_f1():
    """activity_mention_f1 should return perfect score for identical sets."""
    from lctscap.eval.factuality import activity_mention_f1
    pred = [{"walking", "sitting"}]
    gold = [{"walking", "sitting"}]
    result = activity_mention_f1(pred, gold)
    assert result["activity_f1"] == 1.0


def test_event_span_iou():
    """event_span_iou should return 1.0 for identical spans."""
    from lctscap.eval.grounding import event_span_iou
    spans = [(0, 10), (10, 20)]
    assert event_span_iou(spans, spans) == 1.0


def test_order_consistency_perfect():
    """order_consistency should return 1.0 for identical order."""
    from lctscap.eval.grounding import order_consistency
    order = ["walking", "sitting", "standing"]
    assert order_consistency(order, order) == 1.0
