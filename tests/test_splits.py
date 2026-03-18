"""Tests for participant-level data splitting."""
import json
import pytest
from lctscap.data.splits import make_subject_splits, save_splits, load_splits, verify_no_leakage


def test_basic_split():
    subjects = [f"S{i:03d}" for i in range(100)]
    splits = make_subject_splits(subjects)
    assert verify_no_leakage(splits)
    total = sum(len(v) for v in splits.values())
    assert total == 100


def test_split_ratios():
    subjects = [f"S{i:03d}" for i in range(100)]
    splits = make_subject_splits(subjects, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2)
    assert len(splits["train"]) == 70
    assert len(splits["val"]) == 10
    assert len(splits["test"]) == 20


def test_deterministic():
    subjects = [f"S{i:03d}" for i in range(50)]
    s1 = make_subject_splits(subjects, seed=42)
    s2 = make_subject_splits(subjects, seed=42)
    assert s1 == s2


def test_different_seeds():
    subjects = [f"S{i:03d}" for i in range(50)]
    s1 = make_subject_splits(subjects, seed=42)
    s2 = make_subject_splits(subjects, seed=123)
    # Very unlikely to be equal with different seeds
    assert s1["train"] != s2["train"]


def test_input_order_invariant():
    subjects_a = [f"S{i:03d}" for i in range(20)]
    subjects_b = list(reversed(subjects_a))
    s1 = make_subject_splits(subjects_a, seed=42)
    s2 = make_subject_splits(subjects_b, seed=42)
    assert s1 == s2


def test_small_split():
    subjects = ["A", "B", "C"]
    splits = make_subject_splits(subjects)
    assert verify_no_leakage(splits)
    total = sum(len(v) for v in splits.values())
    assert total == 3
    # Each split should have at least 1
    for v in splits.values():
        assert len(v) >= 1


def test_empty_raises():
    with pytest.raises(ValueError):
        make_subject_splits([])


def test_bad_ratios():
    with pytest.raises(ValueError):
        make_subject_splits(["A", "B"], train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)


def test_save_load_roundtrip(tmp_path):
    subjects = [f"S{i:03d}" for i in range(30)]
    splits = make_subject_splits(subjects)
    path = str(tmp_path / "splits.json")
    save_splits(splits, path)
    loaded = load_splits(path)
    assert loaded == splits


def test_verify_leakage_detected():
    bad_splits = {
        "train": ["A", "B", "C"],
        "val": ["C", "D"],  # C overlaps with train
        "test": ["E", "F"],
    }
    assert not verify_no_leakage(bad_splits)


def test_verify_no_leakage_clean():
    clean = {
        "train": ["A", "B"],
        "val": ["C"],
        "test": ["D", "E"],
    }
    assert verify_no_leakage(clean)


def test_harth_split_60_40():
    """HARTH uses 60/40 train/test with no val."""
    subjects = [f"H{i:02d}" for i in range(22)]
    splits = make_subject_splits(subjects, train_ratio=0.6, val_ratio=0.0, test_ratio=0.4)
    total = sum(len(v) for v in splits.values())
    assert total == 22
    assert verify_no_leakage(splits)
