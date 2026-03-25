"""Auxiliary cross-domain time-series/text alignment benchmarks."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch


AUX_ALIGNMENT_DATASETS: Dict[str, Dict[str, object]] = {
    "ECG200": {
        "domain": "medical_ecg",
        "label_map": {
            "-1": "normal heartbeat",
            "1": "myocardial infarction heartbeat",
            "0": "normal heartbeat",
            "2": "myocardial infarction heartbeat",
        },
        "template": "This time series records an ECG heartbeat pattern consistent with {label}.",
    },
    "PLAID": {
        "domain": "appliance_power",
        "label_map": {
            "0": "air conditioner",
            "1": "compact fluorescent lamp",
            "2": "fridge",
            "3": "hairdryer",
            "4": "laptop",
            "5": "microwave",
            "6": "washing machine",
            "7": "bulb",
            "8": "vacuum",
            "9": "fan",
            "10": "heater",
        },
        "template": "This electrical load trace matches a {label}.",
    },
    "Chinatown": {
        "domain": "urban_mobility",
        "label_map": {
            "1": "weekend pedestrian traffic",
            "2": "weekday pedestrian traffic",
        },
        "template": "This pedestrian-count time series corresponds to {label}.",
    },
    "EOGHorizontalSignal": {
        "domain": "eye_movement",
        "label_map": {str(i): f"katakana stroke type {i}" for i in range(1, 13)},
        "template": "This horizontal electrooculography signal corresponds to {label}.",
    },
    "SemgHandMovementCh2": {
        "domain": "muscle_activity",
        "label_map": {
            "1": "cylindrical grasp",
            "2": "hook grasp",
            "3": "tip grasp",
            "4": "palmar grasp",
            "5": "spherical grasp",
            "6": "lateral grasp",
        },
        "template": "This surface EMG sequence corresponds to a {label}.",
    },
    "PowerCons": {
        "domain": "household_power",
        "label_map": {
            "1": "warm-season household power consumption",
            "2": "cold-season household power consumption",
        },
        "template": "This household power-consumption profile was recorded during {label}.",
    },
    "SleepEDF": {
        "domain": "sleep_physiology",
        "label_map": {
            "W": "wakefulness",
            "N1": "N1 sleep",
            "N2": "N2 sleep",
            "N3": "deep N3 sleep",
            "REM": "REM sleep",
        },
        "template": "This polysomnography segment corresponds to {label}.",
    },
}


UCR_ALIGNMENT_DATASETS: Dict[str, Dict[str, object]] = {
    key: value for key, value in AUX_ALIGNMENT_DATASETS.items() if key != "SleepEDF"
}


def selected_ucr_datasets() -> List[str]:
    """Return the curated UCR shortlist for auxiliary alignment evaluation."""
    return sorted(UCR_ALIGNMENT_DATASETS.keys())


def make_aux_caption(dataset_name: str, raw_label: str) -> str:
    """Create a short natural-language description for an auxiliary class label."""
    meta = AUX_ALIGNMENT_DATASETS[dataset_name]
    label_map = meta["label_map"]
    label_name = label_map.get(str(raw_label))
    if label_name is None:
        raise KeyError(f"Unknown label '{raw_label}' for dataset '{dataset_name}'.")
    return str(meta["template"]).format(label=label_name)


def _to_2d_signal(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float32)
    return arr.reshape(1, -1)


def load_ucr_tsv(path: str) -> List[Tuple[str, np.ndarray]]:
    """Load a UCR TSV file into (label, signal[C,L]) pairs."""
    records: List[Tuple[str, np.ndarray]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if not row:
                continue
            label = row[0].strip()
            series = []
            for value in row[1:]:
                cell = value.strip()
                if not cell or cell.lower() == "nan":
                    series.append(np.nan)
                else:
                    series.append(float(cell))
            signal = _to_2d_signal(series)
            records.append((label, signal))
    return records


def save_aux_records(
    *,
    dataset_name: str,
    split_name: str,
    records: List[Tuple[str, np.ndarray]],
    output_root: str,
) -> List[Dict[str, object]]:
    """Save UCR samples as tensors plus a JSON manifest."""
    meta = UCR_ALIGNMENT_DATASETS[dataset_name]
    out_dir = Path(output_root) / dataset_name
    tensor_dir = out_dir / "tensors" / split_name
    tensor_dir.mkdir(parents=True, exist_ok=True)

    manifest: List[Dict[str, object]] = []
    for idx, (raw_label, signal) in enumerate(records):
        tensor_path = tensor_dir / f"{dataset_name}_{split_name}_{idx:06d}.pt"
        torch.save(torch.from_numpy(signal), tensor_path)
        caption = make_aux_caption(dataset_name, raw_label)
        manifest.append(
            {
                "sample_id": f"{dataset_name}_{split_name}_{idx:06d}",
                "dataset": dataset_name,
                "domain": meta["domain"],
                "split": split_name,
                "raw_label": str(raw_label),
                "caption_text": caption,
                "tensor_path": str(tensor_path),
                "num_channels": int(signal.shape[0]),
                "length": int(signal.shape[1]),
            }
        )

    manifest_path = out_dir / f"{split_name}_manifest.jsonl"
    with open(manifest_path, "w") as f:
        for item in manifest:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return manifest
