"""Tests for Phase 2/3 framework plumbing."""

from pathlib import Path


def test_load_phase2_config_fields():
    """Phase 2 config should map onto runtime config fields."""
    from lctscap.config import load_config

    cfg = load_config("configs/train/phase2.yaml")

    assert cfg.phase == 2
    assert cfg.model.modules["local_encoder"] == "frozen"
    assert cfg.model.modules["decoder"] == "train"
    assert cfg.model.init_from.endswith("/runs/phase1/checkpoints/best.pt")
    assert cfg.model.decoder_vocab_size == 50257
    assert cfg.model.tokenizer_name == "gpt2"
    assert cfg.train.batch_sizes[128] == 32
    assert cfg.train.batch_sizes[256] == 16
    assert cfg.train.early_stop_metric == "composite"
    assert str(cfg.checkpoint_dir).endswith("/runs/phase2/checkpoints")


def test_paraphrase_pipeline_process_sample(monkeypatch):
    """Paraphrase pipeline should enrich a sample with paraphrase metadata."""
    from lctscap.data.paraphrase import ParaphrasePipeline

    sample = {
        "sample_id": "s1",
        "caption_long": "The participant spent the entire 10 minutes walking.",
        "events": [
            {
                "type": "walking",
                "start_token": 0,
                "end_token": 60,
                "duration_sec": 600.0,
            }
        ],
    }

    monkeypatch.setattr(
        "lctscap.data.paraphrase.paraphrase_caption",
        lambda template, events, model_name="": "The participant was walking throughout the full 10 minutes.",
    )

    pipeline = ParaphrasePipeline()
    enriched = pipeline.process_sample(sample)

    assert enriched["caption_paraphrase"]
    assert enriched["paraphrase_verification"]["is_valid"] is True
    assert enriched["caption_long"] == enriched["caption_paraphrase"]


def test_run_evaluation_uses_prediction_claims():
    """Evaluation should derive grounding/factuality from prediction text, not gold."""
    from scripts.evaluate import run_evaluation

    predictions = {
        "sample-1": {
            "sample_id": "sample-1",
            "prediction": "The participant was sitting for 10 seconds.",
        }
    }
    gold = {
        "sample-1": {
            "sample_id": "sample-1",
            "caption_short": "The participant walked and then ran.",
            "events": [
                {
                    "type": "walking",
                    "start_token": 0,
                    "end_token": 5,
                    "duration_sec": 50.0,
                    "to_activity": "running",
                },
                {
                    "type": "running",
                    "start_token": 5,
                    "end_token": 10,
                    "duration_sec": 50.0,
                    "from_activity": "walking",
                },
            ],
        }
    }

    results = run_evaluation(predictions, gold, skip_bertscore=True)

    assert results["transition_accuracy"] == 0.0
    assert results["duration_bin_accuracy"] == 0.0
    assert results["unsupported_claim_rate"] > 0.0
    assert results["verification_precision"] == 0.0
