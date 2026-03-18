"""Participant-level data split utilities for LCTSCap.

Ensures deterministic, participant-disjoint splits so that no participant
appears in more than one of train / val / test.
"""

import json
import random
from pathlib import Path
from typing import Dict, List


def make_subject_splits(
    subjects: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """Create deterministic participant-level train/val/test splits.

    Args:
        subjects: List of participant IDs.
        train_ratio: Fraction allocated to training.
        val_ratio: Fraction allocated to validation.
        test_ratio: Fraction allocated to testing.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys ``"train"``, ``"val"``, ``"test"``, each
        mapping to a sorted list of participant IDs.

    Raises:
        ValueError: If ratios do not sum to 1.0 (within tolerance) or
            the subject list is empty.
    """
    if not subjects:
        raise ValueError("subjects list must not be empty.")

    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )

    # Sort first for determinism regardless of input order
    sorted_subjects = sorted(subjects)
    rng = random.Random(seed)
    rng.shuffle(sorted_subjects)

    n = len(sorted_subjects)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    # Remainder goes to test to avoid rounding issues
    n_test = n - n_train - n_val

    # Guarantee at least one subject per split when possible
    if n >= 3:
        if n_train == 0:
            n_train = 1
            n_test -= 1
        if n_val == 0:
            n_val = 1
            n_test -= 1
        if n_test == 0:
            n_test = 1
            n_train -= 1

    train = sorted(sorted_subjects[:n_train])
    val = sorted(sorted_subjects[n_train : n_train + n_val])
    test = sorted(sorted_subjects[n_train + n_val :])

    return {"train": train, "val": val, "test": test}


def save_splits(splits: Dict[str, List[str]], path: str) -> None:
    """Save split assignment to a JSON file.

    Args:
        splits: Dictionary mapping split name to participant IDs.
        path: Output file path (will be created / overwritten).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(splits, f, indent=2, sort_keys=True)


def load_splits(path: str) -> Dict[str, List[str]]:
    """Load split assignment from a JSON file.

    Args:
        path: Path to a JSON file produced by :func:`save_splits`.

    Returns:
        Dictionary mapping split name to participant IDs.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    with open(path, "r") as f:
        splits = json.load(f)
    return splits


def verify_no_leakage(splits: Dict[str, List[str]]) -> bool:
    """Verify that no participant appears in more than one split.

    Args:
        splits: Dictionary mapping split name to participant IDs.

    Returns:
        ``True`` if splits are disjoint, ``False`` otherwise.
    """
    split_names = list(splits.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            set_a = set(splits[split_names[i]])
            set_b = set(splits[split_names[j]])
            overlap = set_a & set_b
            if overlap:
                return False
    return True
