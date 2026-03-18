"""PyTorch Dataset for LCTSCap long-context time series captioning.

Supports loading either raw window tensors (stacked on-the-fly) or
precomputed embeddings from a local encoder.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from lctscap.data.schema import ContextSample, Event

logger = logging.getLogger(__name__)


class LCTSCapDataset(Dataset):
    """Dataset for long-context time series captioning.

    Each item consists of a time-series input (raw stacked windows or
    precomputed embeddings) paired with caption targets, events, and
    structured metadata.

    Args:
        manifest_path: Path to a JSON file containing serialized
            :class:`ContextSample` objects.
        context_len: Expected context length (used for filtering / validation).
        precomputed_embeddings_dir: If provided, load precomputed embeddings
            of shape ``(T, d)`` instead of raw tensors.  Embedding files are
            expected at ``<dir>/<sample_id>.pt``.
        transform: Optional callable applied to the raw tensor before
            returning.  Ignored when using precomputed embeddings.
    """

    def __init__(
        self,
        manifest_path: str,
        context_len: int,
        precomputed_embeddings_dir: Optional[str] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.context_len = context_len
        self.precomputed_dir = (
            Path(precomputed_embeddings_dir) if precomputed_embeddings_dir else None
        )
        self.transform = transform

        # Load manifest (supports both JSON array and JSONL formats)
        raw_samples = self._load_manifest(manifest_path)

        # Parse into ContextSample objects and filter by context_len
        self.samples: List[ContextSample] = []
        for entry in raw_samples:
            sample = ContextSample(**entry)
            if sample.context_len == context_len:
                self.samples.append(sample)

        logger.info(
            "LCTSCapDataset loaded %d samples (context_len=%d) from %s",
            len(self.samples),
            context_len,
            manifest_path,
        )

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_manifest(path: str) -> list:
        """Load manifest from JSON array or JSONL file."""
        with open(path, "r") as f:
            first_char = f.read(1).strip()

        if first_char == "[":
            # Standard JSON array
            with open(path, "r") as f:
                return json.load(f)
        else:
            # JSONL: one JSON object per line
            items = []
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            return items

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one training example.

        Returns:
            Dictionary with keys:
            - ``ts_input``: torch.Tensor — either stacked raw windows of
              shape ``(T, C, window_samples)`` or precomputed embeddings
              of shape ``(T, d)``.
            - ``caption_short``: str or empty string.
            - ``caption_long``: str or empty string.
            - ``events``: list of event dicts (serialized).
            - ``segment_summaries``: list of strings.
            - ``evidence_bullets``: list of strings.
            - ``metadata``: dict with sample_id, dataset, participant_id,
              split, context_len, stride.
        """
        sample = self.samples[idx]

        # --- Load time-series input ---
        if self.precomputed_dir is not None:
            ts_input = self._load_precomputed(sample)
        else:
            ts_input = self._load_raw_tensors(sample)

        if self.transform is not None and self.precomputed_dir is None:
            ts_input = self.transform(ts_input)

        # --- Targets ---
        events_serialized: List[Dict[str, Any]] = []
        if sample.events:
            events_serialized = [e.model_dump() for e in sample.events]

        return {
            "ts_input": ts_input,
            "caption_short": sample.caption_short or "",
            "caption_long": sample.caption_long or "",
            "events": events_serialized,
            "segment_summaries": sample.segment_summaries or [],
            "evidence_bullets": sample.evidence_bullets or [],
            "metadata": {
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "participant_id": sample.participant_id,
                "split": sample.split,
                "context_len": sample.context_len,
                "stride": sample.stride,
            },
        }

    def _load_raw_tensors(self, sample: ContextSample) -> torch.Tensor:
        """Stack raw window tensors into a single tensor.

        Each window tensor has shape ``(C, window_samples)``.
        The result has shape ``(T, C, window_samples)`` where T is the
        number of windows (context_len).

        Args:
            sample: The context sample with tensor_paths.

        Returns:
            Stacked tensor of shape ``(T, C, window_samples)``.
        """
        tensors: List[torch.Tensor] = []
        for tp in sample.tensor_paths:
            t = torch.load(tp, map_location="cpu", weights_only=True)
            tensors.append(t)

        # Stack: each tensor is (C, window_samples) -> (T, C, window_samples)
        stacked = torch.stack(tensors, dim=0)
        return stacked

    def _load_precomputed(self, sample: ContextSample) -> torch.Tensor:
        """Load precomputed embeddings for a sample.

        Looks for a file named ``<sample_id>.pt`` in the precomputed
        embeddings directory.  Falls back to per-window embeddings if
        the single-file version is not found.

        Args:
            sample: The context sample.

        Returns:
            Tensor of shape ``(T, d)`` where d is the embedding dimension.
        """
        assert self.precomputed_dir is not None

        # Try loading a single pre-concatenated embedding file
        single_path = self.precomputed_dir / f"{sample.sample_id}.pt"
        if single_path.exists():
            return torch.load(single_path, map_location="cpu", weights_only=True)

        # Fall back to loading per-window embeddings and stacking
        embeddings: List[torch.Tensor] = []
        for window_id in sample.window_ids:
            emb_path = self.precomputed_dir / f"{window_id}.pt"
            if not emb_path.exists():
                raise FileNotFoundError(
                    f"Precomputed embedding not found: {emb_path}"
                )
            emb = torch.load(emb_path, map_location="cpu", weights_only=True)
            embeddings.append(emb)

        return torch.stack(embeddings, dim=0)

    def get_split(self, split: str) -> "LCTSCapSubset":
        """Return a subset filtered to the given split.

        Args:
            split: One of ``"train"``, ``"val"``, ``"test"``.

        Returns:
            A :class:`LCTSCapSubset` view over matching samples.
        """
        indices = [i for i, s in enumerate(self.samples) if s.split == split]
        return LCTSCapSubset(self, indices)


class LCTSCapSubset(Dataset):
    """A subset view of :class:`LCTSCapDataset` for a specific split.

    Args:
        dataset: The parent dataset.
        indices: List of indices into the parent dataset.
    """

    def __init__(self, dataset: LCTSCapDataset, indices: List[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.dataset[self.indices[idx]]
