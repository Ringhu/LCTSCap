"""Long-context sample builder for LCTSCap.

Creates sliding-window context samples by grouping consecutive
per-participant windows into longer sequences of configurable length.
"""

import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from lctscap.data.schema import ContextSample

logger = logging.getLogger(__name__)


def build_context_samples(
    window_manifest_path: str,
    context_len: int,
    stride: int,
) -> List[ContextSample]:
    """Build long-context samples by sliding a window over per-participant data.

    Only contiguous windows from a single participant are grouped together.
    Windows must be pre-sorted by ``start_time_sec`` within each participant.

    Args:
        window_manifest_path: Path to a JSON manifest file containing a list
            of window metadata dicts (as produced by preprocessing).
        context_len: Number of windows per context sample.
        stride: Step size (in windows) between consecutive samples.

    Returns:
        List of :class:`ContextSample` instances.
    """
    with open(window_manifest_path, "r") as f:
        windows = json.load(f)

    # Group windows by participant, preserving order
    by_participant: Dict[str, List[Dict]] = defaultdict(list)
    for w in windows:
        by_participant[w["participant_id"]].append(w)

    # Sort each participant's windows by start time
    for pid in by_participant:
        by_participant[pid].sort(key=lambda w: w["start_time_sec"])

    samples: List[ContextSample] = []
    sample_counter = 0

    for pid, p_windows in sorted(by_participant.items()):
        n = len(p_windows)
        if n < context_len:
            logger.debug(
                "Participant %s has only %d windows (< context_len=%d), skipping.",
                pid,
                n,
                context_len,
            )
            continue

        # Check contiguity: windows must be consecutive in time
        # We verify by checking that end_time of window i == start_time of window i+1
        # Build contiguous runs
        runs = _find_contiguous_runs(p_windows)

        for run_start, run_end in runs:
            run_len = run_end - run_start
            if run_len < context_len:
                continue

            for offset in range(0, run_len - context_len + 1, stride):
                start_idx = run_start + offset
                end_idx = start_idx + context_len

                ctx_windows = p_windows[start_idx:end_idx]
                dataset = ctx_windows[0].get("dataset", "capture24")
                split = ctx_windows[0].get("split", "train")

                sample = ContextSample(
                    sample_id=f"ctx_{dataset}_{pid}_{context_len}_{stride}_{sample_counter:06d}",
                    dataset=dataset,
                    participant_id=pid,
                    split=split,
                    context_len=context_len,
                    stride=stride,
                    start_window_idx=start_idx,
                    end_window_idx=end_idx,
                    window_ids=[w["window_id"] for w in ctx_windows],
                    tensor_paths=[w["tensor_path"] for w in ctx_windows],
                )
                samples.append(sample)
                sample_counter += 1

    logger.info(
        "Built %d context samples (context_len=%d, stride=%d) from %d participants.",
        len(samples),
        context_len,
        stride,
        len(by_participant),
    )
    return samples


def _find_contiguous_runs(
    windows: List[Dict],
    tolerance_sec: float = 0.5,
) -> List[tuple]:
    """Find runs of contiguous windows within a sorted list.

    Two windows are contiguous if the end time of one equals (within tolerance)
    the start time of the next.

    Args:
        windows: Sorted list of window dicts with ``start_time_sec``
            and ``end_time_sec``.
        tolerance_sec: Maximum gap between windows to still be considered
            contiguous.

    Returns:
        List of (start_index, end_index) tuples for each contiguous run.
    """
    if not windows:
        return []

    runs = []
    run_start = 0

    for i in range(1, len(windows)):
        prev_end = windows[i - 1]["end_time_sec"]
        curr_start = windows[i]["start_time_sec"]
        gap = abs(curr_start - prev_end)

        if gap > tolerance_sec:
            runs.append((run_start, i))
            run_start = i

    runs.append((run_start, len(windows)))
    return runs


def build_all_contexts(
    manifest_dir: str,
    output_dir: str,
    context_lens: Optional[List[int]] = None,
    strides: Optional[List[int]] = None,
) -> Dict[str, List[ContextSample]]:
    """Build context samples for all context_len x stride combinations.

    Searches ``manifest_dir`` for dataset manifest files (``manifest.json``).

    Args:
        manifest_dir: Directory containing per-dataset ``manifest.json`` files
            (e.g. ``capture24/manifest.json``, ``harth/manifest.json``).
        output_dir: Directory to write output context sample JSON files.
        context_lens: List of context lengths to use. Defaults to
            ``[128, 256, 512]``.
        strides: List of strides. Defaults to ``[32, 64]``.

    Returns:
        Dictionary mapping ``"<dataset>_<context_len>_<stride>"`` to sample lists.
    """
    if context_lens is None:
        context_lens = [128, 256, 512]
    if strides is None:
        strides = [32, 64]

    manifest_root = Path(manifest_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover manifest files
    manifest_files = list(manifest_root.rglob("manifest.json"))
    if not manifest_files:
        logger.warning("No manifest.json files found in %s", manifest_dir)
        return {}

    all_results: Dict[str, List[ContextSample]] = {}

    for manifest_path in sorted(manifest_files):
        dataset_name = manifest_path.parent.name  # e.g. "capture24"
        logger.info("Processing manifest: %s", manifest_path)

        for ctx_len in context_lens:
            for stride in strides:
                key = f"{dataset_name}_{ctx_len}_{stride}"
                samples = build_context_samples(
                    str(manifest_path), ctx_len, stride
                )
                all_results[key] = samples

                # Save to JSON
                out_file = output_root / f"{key}.json"
                serialized = [s.model_dump() for s in samples]
                with open(out_file, "w") as f:
                    json.dump(serialized, f, indent=2)

                logger.info(
                    "  %s: %d samples -> %s", key, len(samples), out_file
                )

    return all_results


def compute_statistics(samples: List[ContextSample]) -> Dict[str, Any]:
    """Compute summary statistics over a list of context samples.

    Args:
        samples: List of :class:`ContextSample` instances.

    Returns:
        Dictionary with keys:
        - ``total``: total number of samples
        - ``by_split``: count per split
        - ``by_context_len``: count per context length
        - ``by_dataset``: count per dataset
        - ``by_participant``: count per participant
        - ``unique_participants``: number of unique participants
    """
    stats: Dict[str, Any] = {
        "total": len(samples),
        "by_split": dict(Counter(s.split for s in samples)),
        "by_context_len": dict(Counter(s.context_len for s in samples)),
        "by_dataset": dict(Counter(s.dataset for s in samples)),
        "by_participant": dict(Counter(s.participant_id for s in samples)),
        "unique_participants": len(set(s.participant_id for s in samples)),
    }
    return stats
