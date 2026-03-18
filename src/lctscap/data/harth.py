"""HARTH dataset preprocessing for LCTSCap.

The Human Activity Recognition Trondheim (HARTH) dataset contains
accelerometer data from two body-worn sensors (back and thigh), each
recording tri-axial acceleration at 50 Hz.  Six channels total
(back_x, back_y, back_z, thigh_x, thigh_y, thigh_z).
Reference: https://github.com/ntnu-ai-lab/harth-ml-experiments
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from lctscap.config import DataConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Activity labels (12 classes as defined in HARTH)
# ---------------------------------------------------------------------------
HARTH_ACTIVITIES: List[str] = [
    "walking",
    "running",
    "shuffling",
    "stairs_up",
    "stairs_down",
    "standing",
    "sitting",
    "lying",
    "cycling",
    "nordic_walking",
    "jumping",
    "other",
]

# Numeric label -> string mapping (HARTH uses 1-indexed integer labels)
_HARTH_LABEL_ID_MAP: Dict[int, str] = {
    1: "walking",
    2: "running",
    3: "shuffling",
    4: "stairs_up",
    5: "stairs_down",
    6: "standing",
    7: "sitting",
    8: "lying",
    9: "cycling",
    10: "nordic_walking",
    11: "jumping",
    12: "other",
    # Some versions use 0 for unknown/transition
    0: "other",
    13: "other",
}


def _map_label(raw_label) -> str:
    """Map a raw HARTH label (int or string) to canonical activity string."""
    if isinstance(raw_label, (int, np.integer)):
        return _HARTH_LABEL_ID_MAP.get(int(raw_label), "other")
    key = str(raw_label).strip().lower()
    if key in HARTH_ACTIVITIES:
        return key
    return "other"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_harth(data_dir: str) -> None:
    """Print instructions for downloading the HARTH dataset.

    Args:
        data_dir: Directory where the data should be placed.
    """
    msg = f"""
    ===================================================================
    HARTH Download Instructions
    ===================================================================
    1. Visit https://github.com/ntnu-ai-lab/harth-ml-experiments
    2. Clone the repository or download the data archive.
    3. Place the CSV files into:
         {data_dir}/harth/raw/
       Expected structure:
         {data_dir}/harth/raw/S001.csv
         {data_dir}/harth/raw/S002.csv
         ...
       Each CSV should contain columns:
         timestamp, back_x, back_y, back_z, thigh_x, thigh_y, thigh_z, label
    4. Run the preprocessing pipeline:
         python -m lctscap.data.harth --data_dir {data_dir}
    ===================================================================
    """
    print(msg)
    logger.info("Printed HARTH download instructions for %s", data_dir)


def load_subject(
    subject_id: str,
    data_dir: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load raw accelerometer data and labels for one HARTH subject.

    Args:
        subject_id: e.g. ``"S001"``.
        data_dir: Root directory containing ``harth/raw/``.

    Returns:
        Tuple of (sensor_data, labels) where:
        - sensor_data: np.ndarray of shape ``(N, 6)`` — back xyz + thigh xyz.
        - labels: pd.DataFrame with a ``label`` column.

    Raises:
        FileNotFoundError: If the data file is missing.
    """
    raw_dir = Path(data_dir) / "harth" / "raw"
    data_path = raw_dir / f"{subject_id}.csv"

    if not data_path.exists():
        raise FileNotFoundError(f"HARTH data file not found: {data_path}")

    df = pd.read_csv(data_path)

    # Expected sensor columns
    sensor_cols = ["back_x", "back_y", "back_z", "thigh_x", "thigh_y", "thigh_z"]
    missing = [c for c in sensor_cols if c not in df.columns]
    if missing:
        # Try alternative column naming
        alt_mapping = {
            "back_x": "bx", "back_y": "by", "back_z": "bz",
            "thigh_x": "tx", "thigh_y": "ty", "thigh_z": "tz",
        }
        for orig, alt in alt_mapping.items():
            if orig not in df.columns and alt in df.columns:
                df = df.rename(columns={alt: orig})
        missing = [c for c in sensor_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing sensor columns in {data_path}: {missing}. "
                f"Available columns: {list(df.columns)}"
            )

    sensor_data = df[sensor_cols].values.astype(np.float32)

    # Labels
    label_col = "label" if "label" in df.columns else "activity"
    if label_col not in df.columns:
        raise ValueError(f"No label column found in {data_path}")
    labels = pd.DataFrame({"label": df[label_col].values})

    logger.info(
        "Loaded HARTH subject %s: %d samples (%.1f sec at 50Hz)",
        subject_id,
        len(sensor_data),
        len(sensor_data) / 50,
    )
    return sensor_data, labels


def cut_windows(
    data: np.ndarray,
    labels: pd.DataFrame,
    window_sec: int = 10,
    sample_rate: int = 50,
) -> List[Dict]:
    """Cut continuous 6-channel data into fixed-length windows.

    HARTH data is already at 50 Hz, so no downsampling is needed.

    Args:
        data: Sensor array of shape ``(N, 6)`` at 50 Hz.
        labels: DataFrame with a ``label`` column (per-sample labels).
        window_sec: Duration of each window in seconds.
        sample_rate: Sampling rate in Hz (should be 50 for HARTH).

    Returns:
        List of dicts, each containing:
        - ``"tensor"``: np.ndarray of shape ``(6, window_samples)``
        - ``"label"``: activity string
        - ``"start_sec"``: start time in seconds
        - ``"end_sec"``: end time in seconds
    """
    window_samples = window_sec * sample_rate
    n_samples = data.shape[0]
    n_windows = n_samples // window_samples

    label_values = labels["label"].values

    windows: List[Dict] = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples

        segment = data[start:end]  # (window_samples, 6)
        tensor = segment.T  # (6, window_samples)

        # Majority label within the window
        window_labels = label_values[start:end]
        unique, counts = np.unique(window_labels, return_counts=True)
        raw_label = unique[np.argmax(counts)]
        activity = _map_label(raw_label)

        start_sec = i * window_sec
        end_sec = start_sec + window_sec

        windows.append(
            {
                "tensor": tensor.astype(np.float32),
                "label": activity,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
        )

    logger.info("Cut %d windows (%.1f hours)", n_windows, n_windows * window_sec / 3600)
    return windows


def preprocess_subject(
    subject_id: str,
    raw_dir: str,
    output_dir: str,
    config: Optional[DataConfig] = None,
) -> List[Dict]:
    """Full preprocessing pipeline for one HARTH subject.

    HARTH is already at 50 Hz, so this simply loads, windows, and saves.

    Args:
        subject_id: e.g. ``"S001"``.
        raw_dir: Root directory containing ``harth/raw/``.
        output_dir: Where to write processed tensors.
        config: Optional data configuration.

    Returns:
        List of window metadata dicts.
    """
    if config is None:
        config = DataConfig()

    logger.info("Preprocessing HARTH subject %s", subject_id)

    # Load (already at 50 Hz — no downsampling needed)
    sensor_data, labels = load_subject(subject_id, raw_dir)

    # Cut windows
    windows = cut_windows(
        sensor_data,
        labels,
        window_sec=config.window_sec,
        sample_rate=config.sample_rate,
    )

    # Save tensors
    out_path = Path(output_dir) / "harth" / "tensors" / subject_id
    out_path.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    for idx, w in enumerate(windows):
        tensor_file = out_path / f"window_{idx:06d}.pt"
        torch.save(torch.from_numpy(w["tensor"]), tensor_file)
        manifest.append(
            {
                "window_id": f"{subject_id}_w{idx:06d}",
                "participant_id": subject_id,
                "dataset": "harth",
                "label": w["label"],
                "start_time_sec": w["start_sec"],
                "end_time_sec": w["end_sec"],
                "tensor_path": str(tensor_file),
                "channels": config.channels_harth,
            }
        )

    logger.info(
        "Saved %d windows for HARTH subject %s to %s",
        len(manifest),
        subject_id,
        out_path,
    )
    return manifest


def preprocess_all(
    raw_dir: str,
    output_dir: str,
    config: Optional[DataConfig] = None,
) -> List[Dict]:
    """Orchestrate preprocessing for all HARTH subjects.

    Discovers subject IDs from CSV files in the raw directory.

    Args:
        raw_dir: Root directory with ``harth/raw/`` subfolder.
        output_dir: Where to save processed outputs.
        config: Optional data config.

    Returns:
        Combined manifest (list of window metadata dicts).
    """
    if config is None:
        config = DataConfig()

    raw_path = Path(raw_dir) / "harth" / "raw"
    if not raw_path.exists():
        logger.error("HARTH raw data directory not found: %s", raw_path)
        download_harth(raw_dir)
        return []

    # Discover subjects
    csv_files = sorted(raw_path.glob("S*.csv"))
    if not csv_files:
        logger.warning("No HARTH subject files found in %s", raw_path)
        return []

    subject_ids = [f.stem for f in csv_files]
    logger.info("Found %d HARTH subjects: %s", len(subject_ids), subject_ids[:5])

    all_manifest: List[Dict] = []
    for sid in subject_ids:
        try:
            manifest = preprocess_subject(sid, raw_dir, output_dir, config)
            all_manifest.extend(manifest)
        except Exception as e:
            logger.error("Failed to preprocess HARTH subject %s: %s", sid, e)
            continue

    # Save combined manifest
    manifest_path = Path(output_dir) / "harth" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)

    logger.info(
        "HARTH preprocessing complete: %d windows from %d subjects. "
        "Manifest saved to %s",
        len(all_manifest),
        len(subject_ids),
        manifest_path,
    )
    return all_manifest
