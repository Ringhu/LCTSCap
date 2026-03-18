"""CAPTURE-24 dataset preprocessing for LCTSCap.

CAPTURE-24 contains wrist-worn tri-axial accelerometer data (Axivity AX3)
collected from ~150 participants over 24 hours, sampled at 100 Hz.
Reference: https://github.com/OxWearables/capture24
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.signal import decimate

from lctscap.config import DataConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label mapping: fine-grained Capture-24 annotations -> coarse activity types
# ---------------------------------------------------------------------------
CAPTURE24_LABEL_MAP: Dict[str, str] = {
    # Walking variants
    "walking": "walking",
    "walking; carrying": "walking",
    "walking; on phone": "walking",
    "walking; talking": "walking",
    "walking; eating": "walking",
    "marching": "walking",
    # Running
    "running": "running",
    "jogging": "running",
    # Sitting
    "sitting": "sitting",
    "sitting; reading": "sitting",
    "sitting; writing": "sitting",
    "sitting; talking": "sitting",
    "sitting; on phone": "sitting",
    "sitting; eating": "sitting",
    "sitting; fidgeting": "sitting",
    "office work": "sitting",
    "desk work": "sitting",
    # Standing
    "standing": "standing",
    "standing; talking": "standing",
    "standing; on phone": "standing",
    "standing still": "standing",
    # Sleeping / lying
    "sleeping": "sleeping",
    "sleep": "sleeping",
    "nap": "sleeping",
    "lying down": "lying",
    "lying": "lying",
    "reclining": "lying",
    # Cycling
    "cycling": "cycling",
    "bicycling": "cycling",
    "biking": "cycling",
    # Vehicle / transport
    "vehicle": "vehicle",
    "driving": "vehicle",
    "riding in a car": "vehicle",
    "riding in a bus": "vehicle",
    "in vehicle": "vehicle",
    "car": "vehicle",
    "bus": "vehicle",
    "train": "vehicle",
    "public transport": "vehicle",
    # Household / chores
    "housework": "household",
    "cleaning": "household",
    "cooking": "household",
    "laundry": "household",
    "washing dishes": "household",
    "ironing": "household",
    "gardening": "household",
    "vacuuming": "household",
    "sweeping": "household",
    # Eating
    "eating": "eating",
    "having a meal": "eating",
    "preparing food": "eating",
    "drinking": "eating",
    # Self-care
    "self-care": "self_care",
    "grooming": "self_care",
    "dressing": "self_care",
    "washing": "self_care",
    "showering": "self_care",
    "bathing": "self_care",
    "brushing teeth": "self_care",
    # Screen time
    "watching tv": "screen_time",
    "watching television": "screen_time",
    "using computer": "screen_time",
    "phone": "screen_time",
    "tablet": "screen_time",
    "screen time": "screen_time",
    # Socializing
    "socializing": "socializing",
    "talking": "socializing",
    "meeting": "socializing",
    "party": "socializing",
    "visiting": "socializing",
    # Stairs
    "stairs": "stairs",
    "going upstairs": "stairs",
    "going downstairs": "stairs",
    "climbing stairs": "stairs",
}

# Canonical coarse labels
CAPTURE24_COARSE_LABELS = sorted(set(CAPTURE24_LABEL_MAP.values()) | {"other"})


# Mapping from WillettsSpecific2018 labels to our coarse categories
_WILLETTS_TO_COARSE: Dict[str, str] = {
    "sleep": "sleeping",
    "sitting": "sitting",
    "standing": "standing",
    "walking": "walking",
    "bicycling": "cycling",
    "vehicle": "vehicle",
    "household-chores": "household",
    "manual-work": "household",
    "mixed-activity": "other",
    "sports": "running",
}

# Raw annotation string → coarse label cache (populated by _load_label_dict)
_RAW_ANNOTATION_MAP: Dict[str, str] = {}


def _load_label_dict(raw_dir: str) -> Dict[str, str]:
    """Load the annotation-label-dictionary.csv and build a raw→coarse map.

    Uses the ``label:WillettsSpecific2018`` column as the intermediate label,
    then maps through ``_WILLETTS_TO_COARSE``.

    If the dictionary is not found, returns an empty dict (fallback to
    keyword-based ``_map_label``).
    """
    global _RAW_ANNOTATION_MAP
    if _RAW_ANNOTATION_MAP:
        return _RAW_ANNOTATION_MAP

    dict_path = Path(raw_dir) / "capture24" / "raw" / "annotation-label-dictionary.csv"
    if not dict_path.exists():
        logger.warning("Label dictionary not found at %s, using keyword fallback", dict_path)
        return {}

    df = pd.read_csv(dict_path)
    annot_col = "annotation"
    label_col = "label:WillettsSpecific2018"
    if label_col not in df.columns:
        # Try the first label column
        label_col = [c for c in df.columns if c.startswith("label:")][0]

    for _, row in df.iterrows():
        raw = str(row[annot_col]).strip()
        willetts = str(row[label_col]).strip()
        coarse = _WILLETTS_TO_COARSE.get(willetts, "other")
        _RAW_ANNOTATION_MAP[raw] = coarse

    logger.info("Loaded label dictionary: %d entries", len(_RAW_ANNOTATION_MAP))
    return _RAW_ANNOTATION_MAP


def _map_label(raw_label: str, annotation_map: Optional[Dict[str, str]] = None) -> str:
    """Map a raw Capture-24 annotation to its coarse activity type.

    Tries the dictionary-based map first, then falls back to keyword matching.
    """
    key = raw_label.strip()

    # Try dictionary lookup (exact match)
    if annotation_map and key in annotation_map:
        return annotation_map[key]

    # Also try global cache
    if key in _RAW_ANNOTATION_MAP:
        return _RAW_ANNOTATION_MAP[key]

    # Keyword-based fallback
    key_lower = key.lower()
    return CAPTURE24_LABEL_MAP.get(key_lower, "other")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_capture24(data_dir: str) -> None:
    """Print instructions for downloading the CAPTURE-24 dataset.

    The dataset must be obtained manually from the OxWearables GitHub
    repository due to licensing requirements.

    Args:
        data_dir: Directory where the data should be placed.
    """
    msg = f"""
    ===================================================================
    CAPTURE-24 Download Instructions
    ===================================================================
    1. Visit https://github.com/OxWearables/capture24
    2. Follow the instructions to request access.
    3. Once approved, download the data files.
    4. Place the raw accelerometer .csv.gz files and annotation files
       into: {data_dir}/capture24/raw/
       Expected structure:
         {data_dir}/capture24/raw/P001.csv.gz
         {data_dir}/capture24/raw/P001-annotation.csv
         ...
    5. Run the preprocessing pipeline:
         python -m lctscap.data.capture24 --data_dir {data_dir}
    ===================================================================
    """
    print(msg)
    logger.info("Printed CAPTURE-24 download instructions for %s", data_dir)


def load_participant(
    participant_id: str,
    data_dir: str,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load raw accelerometer data and annotations for one participant.

    Supports two layouts:
    1. Single file: ``P001.csv.gz`` with columns ``time, x, y, z, annotation``
    2. Separate files: ``P001.csv.gz`` (time,x,y,z) + ``P001-annotation.csv``

    Args:
        participant_id: e.g. ``"P001"``.
        data_dir: Root directory containing ``capture24/raw/``.

    Returns:
        Tuple of (accel_data, annotations) where:
        - accel_data: np.ndarray of shape ``(N, 3)`` with x/y/z acceleration.
        - annotations: pd.DataFrame with a ``label`` column.

    Raises:
        FileNotFoundError: If accelerometer file is missing.
    """
    raw_dir = Path(data_dir) / "capture24" / "raw"
    accel_path = raw_dir / f"{participant_id}.csv.gz"

    if not accel_path.exists():
        raise FileNotFoundError(f"Accelerometer file not found: {accel_path}")

    # Load accelerometer CSV
    accel_df = pd.read_csv(accel_path, parse_dates=["time"], low_memory=False)
    accel_data = accel_df[["x", "y", "z"]].values.astype(np.float32)

    # Check if annotations are embedded (column "annotation" present)
    if "annotation" in accel_df.columns:
        annotations = accel_df[["time", "annotation"]].rename(
            columns={"annotation": "label"}
        )
    else:
        # Try separate annotation file
        annot_path = raw_dir / f"{participant_id}-annotation.csv"
        if not annot_path.exists():
            raise FileNotFoundError(
                f"No 'annotation' column in {accel_path} and "
                f"annotation file not found: {annot_path}"
            )
        annotations = pd.read_csv(annot_path, parse_dates=["time"])

    logger.info(
        "Loaded participant %s: %d samples, %d annotations",
        participant_id,
        len(accel_data),
        len(annotations),
    )
    return accel_data, annotations


def downsample(
    data: np.ndarray,
    from_hz: int = 100,
    to_hz: int = 50,
) -> np.ndarray:
    """Downsample accelerometer data using scipy decimate.

    Args:
        data: Array of shape ``(N, C)`` at ``from_hz``.
        from_hz: Original sampling frequency.
        to_hz: Target sampling frequency.

    Returns:
        Downsampled array of shape ``(N // factor, C)``.

    Raises:
        ValueError: If ``from_hz`` is not a multiple of ``to_hz``.
    """
    if from_hz == to_hz:
        return data
    if from_hz % to_hz != 0:
        raise ValueError(
            f"from_hz ({from_hz}) must be a multiple of to_hz ({to_hz})."
        )
    factor = from_hz // to_hz
    # Decimate each channel independently
    channels = []
    for ch in range(data.shape[1]):
        channels.append(decimate(data[:, ch], factor))
    return np.stack(channels, axis=1).astype(np.float32)


def cut_windows(
    data: np.ndarray,
    labels: pd.DataFrame,
    window_sec: int = 10,
    sample_rate: int = 50,
    annotation_map: Optional[Dict[str, str]] = None,
) -> List[Dict]:
    """Cut continuous data into fixed-length windows.

    Each window covers ``window_sec`` seconds at ``sample_rate`` Hz,
    producing tensors of shape ``(channels, window_sec * sample_rate)``.

    Args:
        data: Accelerometer array of shape ``(N, C)`` at ``sample_rate`` Hz.
        labels: DataFrame with at least a ``label`` column; one row per
            original sample (before windowing) or aligned index.
        window_sec: Duration of each window in seconds.
        sample_rate: Sampling rate in Hz.

    Returns:
        List of dicts, each containing:
        - ``"tensor"``: np.ndarray of shape ``(C, window_samples)``
        - ``"label"``: coarse activity string
        - ``"start_sec"``: start time in seconds
        - ``"end_sec"``: end time in seconds
    """
    window_samples = window_sec * sample_rate
    n_samples = data.shape[0]
    n_windows = n_samples // window_samples

    # Ensure labels aligns: take majority label per window
    label_col = labels["label"].values if "label" in labels.columns else labels.iloc[:, 0].values
    # Convert NaN to "unknown" to avoid mixed-type sort errors
    label_col = np.array([str(x) if pd.notna(x) else "unknown" for x in label_col])

    windows: List[Dict] = []
    for i in range(n_windows):
        start = i * window_samples
        end = start + window_samples

        segment = data[start:end]  # (window_samples, C)
        tensor = segment.T  # (C, window_samples)

        # Majority label within the window
        window_labels = label_col[start:end] if end <= len(label_col) else label_col[start:]
        if len(window_labels) == 0:
            raw_label = "unknown"
        else:
            # Find most common label
            unique, counts = np.unique(window_labels, return_counts=True)
            raw_label = unique[np.argmax(counts)]

        coarse_label = _map_label(str(raw_label), annotation_map)
        start_sec = i * window_sec
        end_sec = start_sec + window_sec

        windows.append(
            {
                "tensor": tensor.astype(np.float32),
                "label": coarse_label,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
            }
        )

    logger.info("Cut %d windows (%.1f hours)", n_windows, n_windows * window_sec / 3600)
    return windows


def preprocess_participant(
    participant_id: str,
    raw_dir: str,
    output_dir: str,
    config: Optional[DataConfig] = None,
) -> List[Dict]:
    """Full preprocessing pipeline for one CAPTURE-24 participant.

    Steps: load -> downsample 100Hz->50Hz -> cut windows -> save tensors.

    Args:
        participant_id: e.g. ``"P001"``.
        raw_dir: Directory containing ``capture24/raw/``.
        output_dir: Directory to write processed tensors and manifest.
        config: Optional data configuration (uses defaults if None).

    Returns:
        List of window metadata dicts (each includes ``tensor_path``).
    """
    if config is None:
        config = DataConfig()

    logger.info("Preprocessing CAPTURE-24 participant %s", participant_id)

    # Load label dictionary for mapping raw annotations
    annotation_map = _load_label_dict(raw_dir)

    # Load
    accel_data, annotations = load_participant(participant_id, raw_dir)

    # Downsample 100 Hz -> target
    accel_ds = downsample(accel_data, from_hz=100, to_hz=config.sample_rate)

    # Align annotations to downsampled rate
    # Create a per-sample label array by nearest-neighbour from annotation
    n_ds = len(accel_ds)
    if "label" not in annotations.columns:
        # Try common alternative column name
        for col_name in ("annotation", "activity", "class"):
            if col_name in annotations.columns:
                annotations = annotations.rename(columns={col_name: "label"})
                break

    # Expand annotations to per-sample at the downsampled rate
    labels_per_sample = _expand_annotations(annotations, n_ds, config.sample_rate)

    # Cut windows
    windows = cut_windows(
        accel_ds,
        labels_per_sample,
        window_sec=config.window_sec,
        sample_rate=config.sample_rate,
        annotation_map=annotation_map,
    )

    # Save tensors
    out_path = Path(output_dir) / "capture24" / "tensors" / participant_id
    out_path.mkdir(parents=True, exist_ok=True)

    manifest = []
    for idx, w in enumerate(windows):
        tensor_file = out_path / f"window_{idx:06d}.pt"
        torch.save(torch.from_numpy(w["tensor"]), tensor_file)
        manifest.append(
            {
                "window_id": f"{participant_id}_w{idx:06d}",
                "participant_id": participant_id,
                "dataset": "capture24",
                "label": w["label"],
                "start_time_sec": w["start_sec"],
                "end_time_sec": w["end_sec"],
                "tensor_path": str(tensor_file),
                "channels": config.channels_capture24,
            }
        )

    logger.info(
        "Saved %d windows for participant %s to %s",
        len(manifest),
        participant_id,
        out_path,
    )
    return manifest


def _expand_annotations(
    annotations: pd.DataFrame,
    n_samples: int,
    sample_rate: int,
) -> pd.DataFrame:
    """Align annotations to the downsampled sample count.

    Handles three cases:
    1. Per-sample annotations (len ~= n_samples or a multiple): subsample.
    2. Sparse timestamp-based annotations: forward-fill.
    3. Short annotation list: distribute evenly.

    Returns:
        DataFrame with a ``label`` column of length ``n_samples``.
    """
    labels_col = (
        annotations["label"].values
        if "label" in annotations.columns
        else annotations.iloc[:, 0].values
    )
    n_annot = len(labels_col)

    # Case 1: Per-sample annotations (embedded in CSV) — subsample to match
    # downsampled length. Ratio should be close to an integer (e.g. 2 for
    # 100Hz→50Hz).
    if n_annot >= n_samples:
        ratio = n_annot / n_samples
        if 0.9 < ratio < 10.0:
            factor = round(ratio)
            if factor >= 2:
                # Take every `factor`-th label (majority would be same)
                subsampled = labels_col[::factor][:n_samples]
                if len(subsampled) < n_samples:
                    subsampled = np.concatenate(
                        [subsampled, np.full(n_samples - len(subsampled), subsampled[-1])]
                    )
                return pd.DataFrame({"label": subsampled})
            else:
                # ratio ~1: just truncate
                return pd.DataFrame({"label": labels_col[:n_samples]})

    # Case 2: Sparse timestamp-based annotations — forward-fill
    if "time" in annotations.columns and "label" in annotations.columns and n_annot < n_samples:
        time_index = pd.to_timedelta(
            np.arange(n_samples) / sample_rate, unit="s"
        )
        annot_times = pd.to_timedelta(
            (annotations["time"] - annotations["time"].iloc[0]).dt.total_seconds(),
            unit="s",
        )
        annot_reindexed = (
            annotations.set_index(annot_times)["label"]
            .reindex(time_index, method="ffill")
            .bfill()
        )
        return pd.DataFrame({"label": annot_reindexed.values})

    # Case 3: Short annotation list — distribute evenly
    samples_per_annot = max(1, n_samples // n_annot)
    expanded = np.repeat(labels_col, samples_per_annot)[:n_samples]
    if len(expanded) < n_samples:
        pad_label = labels_col[-1] if n_annot > 0 else "unknown"
        expanded = np.concatenate(
            [expanded, np.full(n_samples - len(expanded), pad_label)]
        )
    return pd.DataFrame({"label": expanded})


def preprocess_all(
    raw_dir: str,
    output_dir: str,
    config: Optional[DataConfig] = None,
) -> List[Dict]:
    """Orchestrate preprocessing for all CAPTURE-24 participants.

    Discovers participant IDs from .csv.gz files in the raw directory.

    Args:
        raw_dir: Root directory with ``capture24/raw/`` subfolder.
        output_dir: Where to save processed outputs.
        config: Optional data config.

    Returns:
        Combined manifest (list of window metadata dicts).
    """
    if config is None:
        config = DataConfig()

    raw_path = Path(raw_dir) / "capture24" / "raw"
    if not raw_path.exists():
        logger.error("Raw data directory not found: %s", raw_path)
        download_capture24(raw_dir)
        return []

    # Discover participants
    accel_files = sorted(raw_path.glob("P*.csv.gz"))
    if not accel_files:
        logger.warning("No participant files found in %s", raw_path)
        return []

    participant_ids = [f.stem.split(".")[0] for f in accel_files]
    logger.info("Found %d participants: %s", len(participant_ids), participant_ids[:5])

    all_manifest: List[Dict] = []
    for pid in participant_ids:
        try:
            manifest = preprocess_participant(pid, raw_dir, output_dir, config)
            all_manifest.extend(manifest)
        except Exception as e:
            logger.error("Failed to preprocess participant %s: %s", pid, e)
            continue

    # Save combined manifest
    manifest_path = Path(output_dir) / "capture24" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(manifest_path, "w") as f:
        json.dump(all_manifest, f, indent=2)

    logger.info(
        "CAPTURE-24 preprocessing complete: %d windows from %d participants. "
        "Manifest saved to %s",
        len(all_manifest),
        len(participant_ids),
        manifest_path,
    )
    return all_manifest
