#!/usr/bin/env python
"""Prepare auxiliary cross-domain time-series/text alignment benchmarks."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lctscap.data.auxiliary_alignment import (
    AUX_ALIGNMENT_DATASETS,
    load_ucr_tsv,
    make_aux_caption,
    save_aux_records,
    selected_ucr_datasets,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("prepare_aux_alignment_data")


SLEEP_EDF_STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


def prepare_ucr(ucr_root: str, output_root: str, datasets: list[str]) -> None:
    summary = {}
    for dataset_name in datasets:
        if dataset_name not in AUX_ALIGNMENT_DATASETS:
            raise KeyError(f"Unsupported UCR auxiliary dataset: {dataset_name}")

        dataset_dir = Path(ucr_root) / dataset_name
        train_path = dataset_dir / f"{dataset_name}_TRAIN.tsv"
        test_path = dataset_dir / f"{dataset_name}_TEST.tsv"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"Missing TSV files under {dataset_dir}")

        logger.info("Preparing UCR auxiliary dataset: %s", dataset_name)
        train_records = load_ucr_tsv(str(train_path))
        test_records = load_ucr_tsv(str(test_path))
        train_manifest = save_aux_records(
            dataset_name=dataset_name,
            split_name="train",
            records=train_records,
            output_root=output_root,
        )
        test_manifest = save_aux_records(
            dataset_name=dataset_name,
            split_name="test",
            records=test_records,
            output_root=output_root,
        )
        summary[dataset_name] = {
            "domain": AUX_ALIGNMENT_DATASETS[dataset_name]["domain"],
            "train": len(train_manifest),
            "test": len(test_manifest),
        }
        logger.info(
            "Prepared %s: train=%d test=%d",
            dataset_name,
            len(train_manifest),
            len(test_manifest),
        )

    summary_path = Path(output_root) / "ucr_alignment_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info("Saved summary to %s", summary_path)


def _load_pyedflib(vendor_dir: str | None):
    if vendor_dir:
        sys.path.append(str(Path(vendor_dir).resolve()))
    import pyedflib

    return pyedflib


def _record_key(path: Path) -> str:
    return path.name[:6]


def _downsample_100hz_to_50hz(signal: np.ndarray) -> np.ndarray:
    return signal[..., ::2]


def _epoch_to_windows(epoch_signals: np.ndarray) -> np.ndarray:
    downsampled = _downsample_100hz_to_50hz(epoch_signals)
    return downsampled.reshape(downsampled.shape[0], 3, 500).transpose(1, 0, 2).astype(np.float32)


def _iter_sleep_epochs(pyedflib, psg_path: Path, hyp_path: Path):
    psg = pyedflib.EdfReader(str(psg_path))
    hyp = pyedflib.EdfReader(str(hyp_path))
    try:
        labels = psg.getSignalLabels()
        wanted = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
        indices = []
        for label in wanted:
            if label not in labels:
                raise KeyError(f"Missing channel '{label}' in {psg_path.name}")
            indices.append(labels.index(label))
        freqs = [psg.getSampleFrequency(idx) for idx in indices]
        if any(freq != 100 for freq in freqs):
            raise ValueError(f"Unexpected Sleep-EDF sample frequencies in {psg_path.name}: {freqs}")

        signals = np.stack([psg.readSignal(idx) for idx in indices], axis=0)
        onsets, durations, descriptions = hyp.readAnnotations()
        epoch_counter = 0
        for onset, duration, desc in zip(onsets, durations, descriptions):
            stage = SLEEP_EDF_STAGE_MAP.get(desc)
            if stage is None:
                continue
            if duration < 30:
                continue
            n_epochs = int(duration // 30)
            for i in range(n_epochs):
                start_sec = onset + i * 30
                start = int(round(start_sec * 100))
                end = start + 3000
                if end > signals.shape[1]:
                    break
                yield {
                    "epoch_index": epoch_counter,
                    "raw_label": stage,
                    "windows": _epoch_to_windows(signals[:, start:end]),
                }
                epoch_counter += 1
    finally:
        psg._close()
        hyp._close()


def prepare_sleep_edf(
    *,
    sleep_edf_root: str,
    output_root: str,
    vendor_dir: str | None,
    test_ratio: float,
    max_train_per_label: int,
    max_test_per_label: int,
    seed: int,
) -> None:
    pyedflib = _load_pyedflib(vendor_dir)
    source_root = Path(sleep_edf_root) / "sleep-cassette"
    if not source_root.exists():
        raise FileNotFoundError(f"Missing sleep-cassette directory under {sleep_edf_root}")

    psg_files = { _record_key(path): path for path in source_root.glob("*PSG.edf") }
    hyp_files = { _record_key(path): path for path in source_root.glob("*Hypnogram.edf") }
    record_keys = sorted(set(psg_files) & set(hyp_files))
    if not record_keys:
        raise FileNotFoundError("No matched PSG/Hypnogram EDF pairs found for Sleep-EDF.")

    rng = random.Random(seed)
    rng.shuffle(record_keys)
    split_at = max(1, int(len(record_keys) * (1.0 - test_ratio)))
    split_keys = {
        "train": set(record_keys[:split_at]),
        "test": set(record_keys[split_at:]),
    }

    out_dir = Path(output_root) / "SleepEDF"
    train_dir = out_dir / "tensors" / "train"
    test_dir = out_dir / "tensors" / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    manifests = {"train": [], "test": []}
    counts = {
        "train": {label: 0 for label in AUX_ALIGNMENT_DATASETS["SleepEDF"]["label_map"]},
        "test": {label: 0 for label in AUX_ALIGNMENT_DATASETS["SleepEDF"]["label_map"]},
    }
    limits = {"train": max_train_per_label, "test": max_test_per_label}

    for record_key in record_keys:
        split = "train" if record_key in split_keys["train"] else "test"
        tensor_dir = train_dir if split == "train" else test_dir
        logger.info("Preparing Sleep-EDF record %s (%s)", record_key, split)
        for sample in _iter_sleep_epochs(pyedflib, psg_files[record_key], hyp_files[record_key]):
            label = sample["raw_label"]
            if counts[split][label] >= limits[split]:
                continue
            sample_id = f"SleepEDF_{split}_{record_key}_{sample['epoch_index']:05d}"
            tensor_path = tensor_dir / f"{sample_id}.pt"
            torch.save(torch.from_numpy(sample["windows"]), tensor_path)
            manifests[split].append(
                {
                    "sample_id": sample_id,
                    "dataset": "SleepEDF",
                    "domain": AUX_ALIGNMENT_DATASETS["SleepEDF"]["domain"],
                    "split": split,
                    "raw_label": label,
                    "caption_text": make_aux_caption("SleepEDF", label),
                    "tensor_path": str(tensor_path),
                    "num_channels": int(sample["windows"].shape[1]),
                    "context_len": int(sample["windows"].shape[0]),
                    "length": int(sample["windows"].shape[2]),
                    "recording_id": record_key,
                    "epoch_index": int(sample["epoch_index"]),
                }
            )
            counts[split][label] += 1

    for split in ("train", "test"):
        manifest_path = out_dir / f"{split}_manifest.jsonl"
        with open(manifest_path, "w") as handle:
            for item in manifests[split]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "domain": AUX_ALIGNMENT_DATASETS["SleepEDF"]["domain"],
        "train": len(manifests["train"]),
        "test": len(manifests["test"]),
        "train_per_label": counts["train"],
        "test_per_label": counts["test"],
        "records": len(record_keys),
    }
    summary_path = out_dir / "sleep_edf_alignment_summary.json"
    with open(summary_path, "w") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    logger.info("Saved Sleep-EDF summary to %s", summary_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare auxiliary alignment datasets.")
    parser.add_argument(
        "--source",
        type=str,
        choices=["ucr", "sleep_edf"],
        required=True,
        help="Which source family to prepare.",
    )
    parser.add_argument("--ucr_root", type=str, default=None)
    parser.add_argument("--sleep_edf_root", type=str, default=None)
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--vendor_dir", type=str, default=".vendor")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of dataset names. Defaults to curated shortlist.",
    )
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--max_train_per_label", type=int, default=1000)
    parser.add_argument("--max_test_per_label", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.source == "ucr":
        if not args.ucr_root:
            raise ValueError("--ucr_root is required for source=ucr")
        datasets = args.datasets or selected_ucr_datasets()
        prepare_ucr(args.ucr_root, args.output_root, datasets)
    elif args.source == "sleep_edf":
        if not args.sleep_edf_root:
            raise ValueError("--sleep_edf_root is required for source=sleep_edf")
        prepare_sleep_edf(
            sleep_edf_root=args.sleep_edf_root,
            output_root=args.output_root,
            vendor_dir=args.vendor_dir,
            test_ratio=args.test_ratio,
            max_train_per_label=args.max_train_per_label,
            max_test_per_label=args.max_test_per_label,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
