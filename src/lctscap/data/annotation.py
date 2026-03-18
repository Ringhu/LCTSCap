"""Event extraction and template caption generation for LCTSCap.

Converts sequences of per-window activity labels into structured events,
then generates template-based captions, segment summaries, and evidence bullets.
"""

import logging
from collections import Counter
from typing import Any, Dict, List, Optional

from lctscap.data.schema import ContextSample, Event

logger = logging.getLogger(__name__)


def extract_events(
    window_labels: List[str],
    context_len: int,
    window_sec: int = 10,
) -> List[Event]:
    """Merge consecutive same-label windows into Event objects.

    Args:
        window_labels: Ordered list of activity labels, one per window.
        context_len: Expected number of windows (used for validation).
        window_sec: Duration of each window in seconds.

    Returns:
        List of :class:`Event` objects, ordered by start position.
    """
    if not window_labels:
        return []

    if len(window_labels) != context_len:
        logger.warning(
            "window_labels length (%d) != context_len (%d). "
            "Using actual label count.",
            len(window_labels),
            context_len,
        )

    events: List[Event] = []
    current_label = window_labels[0]
    start_token = 0

    for i in range(1, len(window_labels)):
        if window_labels[i] != current_label:
            # Close current event
            end_token = i
            duration = (end_token - start_token) * window_sec
            events.append(
                Event(
                    type=current_label,
                    start_token=start_token,
                    end_token=end_token,
                    duration_sec=duration,
                )
            )
            current_label = window_labels[i]
            start_token = i

    # Close the final event
    end_token = len(window_labels)
    duration = (end_token - start_token) * window_sec
    events.append(
        Event(
            type=current_label,
            start_token=start_token,
            end_token=end_token,
            duration_sec=duration,
        )
    )

    # Mark dominant event (longest duration)
    if events:
        max_dur = max(e.duration_sec for e in events)
        for e in events:
            if e.duration_sec == max_dur:
                e.is_dominant = True
                break  # Only mark the first one if tied

    # Fill transition context (from_activity / to_activity)
    for i, event in enumerate(events):
        if i > 0:
            event.from_activity = events[i - 1].type
        if i < len(events) - 1:
            event.to_activity = events[i + 1].type

    return events


def compute_event_stats(events: List[Event]) -> Dict[str, Any]:
    """Compute summary statistics from a list of events.

    Args:
        events: List of :class:`Event` objects.

    Returns:
        Dictionary with:
        - ``dominant_activity``: the activity with longest total duration
        - ``num_transitions``: number of activity changes
        - ``total_duration_sec``: total time span in seconds
        - ``num_events``: number of distinct events
        - ``activity_distribution``: dict mapping activity -> total seconds
        - ``unique_activities``: number of unique activity types
    """
    if not events:
        return {
            "dominant_activity": "unknown",
            "num_transitions": 0,
            "total_duration_sec": 0.0,
            "num_events": 0,
            "activity_distribution": {},
            "unique_activities": 0,
        }

    # Accumulate duration per activity type
    activity_durations: Dict[str, float] = {}
    for e in events:
        activity_durations[e.type] = activity_durations.get(e.type, 0.0) + e.duration_sec

    dominant = max(activity_durations, key=activity_durations.get)  # type: ignore[arg-type]
    total_duration = sum(e.duration_sec for e in events)
    num_transitions = len(events) - 1

    return {
        "dominant_activity": dominant,
        "num_transitions": num_transitions,
        "total_duration_sec": total_duration,
        "num_events": len(events),
        "activity_distribution": activity_durations,
        "unique_activities": len(activity_durations),
    }


def generate_segment_summaries(
    events: List[Event],
    context_len: int,
    segment_size: int = 32,
) -> List[str]:
    """Generate template-based summaries for each segment of the context.

    A segment is a group of ``segment_size`` consecutive windows.

    Args:
        events: List of events spanning the full context.
        context_len: Total number of windows in the context.
        segment_size: Number of windows per segment.

    Returns:
        List of summary strings, one per segment.
    """
    n_segments = (context_len + segment_size - 1) // segment_size
    summaries: List[str] = []

    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_size
        seg_end = min(seg_start + segment_size, context_len)

        # Find events overlapping this segment
        seg_events = []
        for e in events:
            if e.end_token > seg_start and e.start_token < seg_end:
                seg_events.append(e)

        if not seg_events:
            summaries.append(f"Segment {seg_idx + 1}: No activity detected.")
            continue

        # Compute per-activity duration within this segment
        activity_dur: Dict[str, float] = {}
        for e in seg_events:
            overlap_start = max(e.start_token, seg_start)
            overlap_end = min(e.end_token, seg_end)
            overlap_windows = overlap_end - overlap_start
            # Assume 10 sec per window as default
            dur = overlap_windows * 10.0
            activity_dur[e.type] = activity_dur.get(e.type, 0.0) + dur

        # Build summary string
        total_seg_dur = sum(activity_dur.values())
        parts = []
        for activity, dur in sorted(activity_dur.items(), key=lambda x: -x[1]):
            pct = (dur / total_seg_dur * 100) if total_seg_dur > 0 else 0
            parts.append(f"{activity} ({dur:.0f}s, {pct:.0f}%)")

        activities_str = ", ".join(parts)
        n_transitions = max(0, len(seg_events) - 1)

        if n_transitions == 0:
            summary = (
                f"Segment {seg_idx + 1}: The participant performed {activities_str} "
                f"for the entire segment."
            )
        else:
            summary = (
                f"Segment {seg_idx + 1}: The participant engaged in {activities_str} "
                f"with {n_transitions} transition(s)."
            )

        summaries.append(summary)

    return summaries


def generate_short_caption(events: List[Event], stats: Dict[str, Any]) -> str:
    """Generate a one-sentence summary caption.

    Args:
        events: List of events.
        stats: Event statistics from :func:`compute_event_stats`.

    Returns:
        A single sentence summarizing the activity pattern.
    """
    dominant = stats.get("dominant_activity", "unknown")
    total_dur = stats.get("total_duration_sec", 0)
    n_transitions = stats.get("num_transitions", 0)
    n_unique = stats.get("unique_activities", 0)

    # Format duration
    if total_dur >= 3600:
        dur_str = f"{total_dur / 3600:.1f} hours"
    elif total_dur >= 60:
        dur_str = f"{total_dur / 60:.0f} minutes"
    else:
        dur_str = f"{total_dur:.0f} seconds"

    if n_unique == 1:
        return (
            f"The participant spent the entire {dur_str} period {dominant}."
        )
    elif n_transitions <= 2:
        other_activities = [
            a for a in stats.get("activity_distribution", {}) if a != dominant
        ]
        others_str = " and ".join(other_activities[:2]) if other_activities else "other activities"
        return (
            f"Over {dur_str}, the participant was predominantly {dominant}, "
            f"with brief periods of {others_str}."
        )
    else:
        return (
            f"Over {dur_str}, the participant engaged in {n_unique} activities "
            f"with {n_transitions} transitions, predominantly {dominant}."
        )


def generate_long_caption(events: List[Event], stats: Dict[str, Any]) -> str:
    """Generate a detailed 2-4 sentence caption.

    Args:
        events: List of events.
        stats: Event statistics from :func:`compute_event_stats`.

    Returns:
        A multi-sentence description of the activity sequence.
    """
    dominant = stats.get("dominant_activity", "unknown")
    total_dur = stats.get("total_duration_sec", 0)
    n_transitions = stats.get("num_transitions", 0)
    n_events = stats.get("num_events", 0)
    activity_dist = stats.get("activity_distribution", {})

    # Format total duration
    if total_dur >= 3600:
        dur_str = f"{total_dur / 3600:.1f} hours"
    elif total_dur >= 60:
        dur_str = f"{total_dur / 60:.0f} minutes"
    else:
        dur_str = f"{total_dur:.0f} seconds"

    sentences = []

    # Sentence 1: Overview
    sentences.append(
        f"This {dur_str} recording captures {n_events} distinct activity "
        f"events across {n_transitions} transitions."
    )

    # Sentence 2: Dominant activity
    dominant_dur = activity_dist.get(dominant, 0)
    dominant_pct = (dominant_dur / total_dur * 100) if total_dur > 0 else 0
    sentences.append(
        f"The dominant activity is {dominant}, accounting for "
        f"{dominant_pct:.0f}% ({dominant_dur:.0f}s) of the recording."
    )

    # Sentence 3: Sequence description (first few events)
    if len(events) <= 5:
        seq_parts = [f"{e.type} ({e.duration_sec:.0f}s)" for e in events]
        seq_str = " -> ".join(seq_parts)
        sentences.append(f"The activity sequence is: {seq_str}.")
    else:
        # Summarize the first 3 and last 2 events
        first_parts = [f"{e.type} ({e.duration_sec:.0f}s)" for e in events[:3]]
        last_parts = [f"{e.type} ({e.duration_sec:.0f}s)" for e in events[-2:]]
        sentences.append(
            f"The recording begins with {' -> '.join(first_parts)}, "
            f"and concludes with {' -> '.join(last_parts)}."
        )

    # Sentence 4: Minor activities (if any)
    minor = {
        a: d
        for a, d in activity_dist.items()
        if a != dominant and d > 0
    }
    if minor:
        minor_parts = [
            f"{a} ({d:.0f}s)" for a, d in sorted(minor.items(), key=lambda x: -x[1])
        ]
        sentences.append(
            f"Other activities include {', '.join(minor_parts[:4])}."
        )

    return " ".join(sentences)


def generate_evidence_bullets(events: List[Event]) -> List[str]:
    """Generate structured evidence bullets for verifiability.

    Each bullet describes one event with its temporal location and duration.

    Args:
        events: List of events.

    Returns:
        List of evidence strings.
    """
    bullets: List[str] = []
    for i, e in enumerate(events):
        start_min = e.start_token * 10 / 60  # Assuming 10 sec windows
        end_min = e.end_token * 10 / 60

        transition_info = ""
        if e.from_activity:
            transition_info += f" (preceded by {e.from_activity})"
        if e.to_activity:
            transition_info += f" (followed by {e.to_activity})"

        dominant_tag = " [DOMINANT]" if e.is_dominant else ""

        bullet = (
            f"Event {i + 1}: {e.type} from window {e.start_token} to "
            f"{e.end_token} ({e.duration_sec:.0f}s, "
            f"{start_min:.1f}-{end_min:.1f} min)"
            f"{transition_info}{dominant_tag}"
        )
        bullets.append(bullet)

    return bullets


def annotate_sample(
    sample: ContextSample,
    window_labels: List[str],
    segment_size: int = 32,
) -> ContextSample:
    """Full annotation pipeline for one context sample.

    Extracts events, generates summaries, captions, and evidence, then
    returns an updated copy of the sample.

    Args:
        sample: A :class:`ContextSample` to annotate.
        window_labels: Ordered activity labels for each window in the sample.
        segment_size: Number of windows per segment for summaries.

    Returns:
        A new :class:`ContextSample` with all annotation fields populated.
    """
    # Extract events
    events = extract_events(window_labels, sample.context_len)
    stats = compute_event_stats(events)

    # Generate annotations
    segment_summaries = generate_segment_summaries(
        events, sample.context_len, segment_size
    )
    short_caption = generate_short_caption(events, stats)
    long_caption = generate_long_caption(events, stats)
    evidence_bullets = generate_evidence_bullets(events)

    # Return updated sample (pydantic model_copy)
    return sample.model_copy(
        update={
            "events": events,
            "segment_summaries": segment_summaries,
            "caption_short": short_caption,
            "caption_long": long_caption,
            "evidence_bullets": evidence_bullets,
        }
    )
