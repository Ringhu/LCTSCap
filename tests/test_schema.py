"""Tests for data schema (Pydantic models)."""
import pytest
from lctscap.data.schema import Event, ContextSample, WindowMeta


def test_window_meta_valid():
    w = WindowMeta(
        window_id="P001_w000000",
        participant_id="P001",
        dataset="capture24",
        split="train",
        label="walking",
        start_time_sec=0.0,
        end_time_sec=10.0,
        tensor_path="/tmp/w0.pt",
        channels=3,
    )
    assert w.dataset == "capture24"
    assert w.channels == 3


def test_window_meta_invalid_dataset():
    with pytest.raises(Exception):
        WindowMeta(
            window_id="w0",
            participant_id="P001",
            dataset="invalid",
            split="train",
            label="walking",
            start_time_sec=0.0,
            end_time_sec=10.0,
            tensor_path="/tmp/w0.pt",
            channels=3,
        )


def test_event_defaults():
    e = Event(type="sitting", start_token=0, end_token=5, duration_sec=50.0)
    assert e.from_activity is None
    assert e.to_activity is None
    assert e.is_dominant is False


def test_event_with_transitions():
    e = Event(
        type="walking",
        start_token=5,
        end_token=15,
        duration_sec=100.0,
        from_activity="sitting",
        to_activity="standing",
        is_dominant=True,
    )
    assert e.from_activity == "sitting"
    assert e.to_activity == "standing"
    assert e.is_dominant is True


def test_context_sample_minimal():
    cs = ContextSample(
        sample_id="s001",
        dataset="harth",
        participant_id="S01",
        split="test",
        context_len=128,
        stride=32,
        start_window_idx=0,
        end_window_idx=128,
        window_ids=[f"w{i}" for i in range(128)],
        tensor_paths=[f"/tmp/w{i}.pt" for i in range(128)],
    )
    assert cs.events is None
    assert cs.segment_summaries is None
    assert cs.caption_short is None


def test_context_sample_with_annotations():
    events = [Event(type="walking", start_token=0, end_token=10, duration_sec=100.0)]
    cs = ContextSample(
        sample_id="s002",
        dataset="capture24",
        participant_id="P002",
        split="train",
        context_len=64,
        stride=32,
        start_window_idx=0,
        end_window_idx=64,
        window_ids=[f"w{i}" for i in range(64)],
        tensor_paths=[f"/tmp/w{i}.pt" for i in range(64)],
        events=events,
        segment_summaries=["Walking for the first segment."],
        caption_short="A walking session.",
        caption_long="The participant walked continuously for about 10 minutes.",
        evidence_bullets=["Walking from token 0 to 10 (100s)."],
    )
    assert len(cs.events) == 1
    assert cs.caption_short == "A walking session."


def test_context_sample_serialization():
    cs = ContextSample(
        sample_id="s003",
        dataset="capture24",
        participant_id="P003",
        split="val",
        context_len=32,
        stride=16,
        start_window_idx=10,
        end_window_idx=42,
        window_ids=[f"w{i}" for i in range(32)],
        tensor_paths=[f"/tmp/w{i}.pt" for i in range(32)],
    )
    d = cs.model_dump()
    assert d["sample_id"] == "s003"
    cs2 = ContextSample(**d)
    assert cs2 == cs
