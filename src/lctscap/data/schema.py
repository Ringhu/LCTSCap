"""Pydantic data schemas for LCTSCap samples, events, and metadata."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class WindowMeta(BaseModel):
    """Metadata for a single fixed-length time-series window."""

    window_id: str = Field(..., description="Unique identifier for this window.")
    participant_id: str = Field(..., description="ID of the human participant.")
    dataset: Literal["capture24", "harth"] = Field(
        ..., description="Source dataset name."
    )
    split: Literal["train", "val", "test"] = Field(
        ..., description="Data split this window belongs to."
    )
    label: str = Field(..., description="Activity label for the window.")
    start_time_sec: float = Field(
        ..., description="Start time of the window in seconds from recording start."
    )
    end_time_sec: float = Field(
        ..., description="End time of the window in seconds from recording start."
    )
    tensor_path: str = Field(
        ..., description="Path to the saved tensor file (.pt)."
    )
    channels: int = Field(
        ..., description="Number of sensor channels (e.g. 3 for capture24, 6 for harth)."
    )


class Event(BaseModel):
    """A detected activity event spanning one or more consecutive windows."""

    type: str = Field(..., description="Activity type / label.")
    start_token: int = Field(
        ..., description="Start position in window-token indices."
    )
    end_token: int = Field(
        ..., description="End position in window-token indices (exclusive)."
    )
    duration_sec: float = Field(
        ..., description="Duration of the event in seconds."
    )
    from_activity: Optional[str] = Field(
        None, description="Activity type of the preceding event (for transitions)."
    )
    to_activity: Optional[str] = Field(
        None, description="Activity type of the following event (for transitions)."
    )
    is_dominant: bool = Field(
        False, description="Whether this is the dominant (longest) event."
    )


class ContextSample(BaseModel):
    """A long-context sample composed of multiple consecutive windows."""

    sample_id: str = Field(..., description="Unique identifier for this context sample.")
    dataset: Literal["capture24", "harth"] = Field(
        ..., description="Source dataset name."
    )
    participant_id: str = Field(..., description="Participant who produced the data.")
    split: Literal["train", "val", "test"] = Field(
        ..., description="Data split this sample belongs to."
    )
    context_len: int = Field(
        ..., description="Number of windows in this context."
    )
    stride: int = Field(
        ..., description="Stride (in windows) used when building this sample."
    )
    start_window_idx: int = Field(
        ..., description="Index of the first window in the participant's sequence."
    )
    end_window_idx: int = Field(
        ..., description="Index past the last window (exclusive)."
    )
    window_ids: List[str] = Field(
        ..., description="Ordered list of window IDs in this context."
    )
    tensor_paths: List[str] = Field(
        ..., description="Ordered list of tensor file paths for each window."
    )
    events: Optional[List[Event]] = Field(
        None, description="Extracted activity events."
    )
    segment_summaries: Optional[List[str]] = Field(
        None, description="Template-generated summaries per segment."
    )
    caption_short: Optional[str] = Field(
        None, description="One-sentence summary caption."
    )
    caption_long: Optional[str] = Field(
        None, description="Detailed 2-4 sentence caption."
    )
    evidence_bullets: Optional[List[str]] = Field(
        None, description="Structured evidence bullets supporting the caption."
    )
