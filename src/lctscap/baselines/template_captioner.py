"""Template-based captioner baseline.

Generates captions from the structured event annotations
using deterministic templates.  No neural generation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

from lctscap.data.schema import ContextSample, Event


def _format_duration(seconds: float) -> str:
    """Format a duration in seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


def _build_event_summary(event: Event) -> str:
    """Build a single-event description sentence."""
    duration_str = _format_duration(event.duration_sec)
    span_str = f"[{event.start_token},{event.end_token}]"

    if event.from_activity and event.to_activity:
        return (
            f"A transition from {event.from_activity} to {event.to_activity} "
            f"occurs at {span_str}."
        )
    return f"{event.type.replace('_', ' ').capitalize()} lasting {duration_str} {span_str}."


class TemplateCaptioner:
    """Baseline that generates captions from event annotations using templates.

    This captioner uses structured event information already present in the
    ContextSample to produce deterministic, faithful captions.
    """

    def predict(self, sample: ContextSample) -> str:
        """Generate a template caption for a single context sample.

        If the sample already has a short caption in its annotations, return it.
        Otherwise, construct one from the event table.

        Args:
            sample: a ContextSample with events populated.

        Returns:
            Generated caption string.
        """
        # If a template caption already exists, use it
        if sample.caption_short:
            return sample.caption_short

        events = sample.events or []
        if not events:
            return "No activities were detected in this time window."

        # Find dominant activity
        dominant = max(events, key=lambda e: e.duration_sec) if events else None
        total_duration = sum(e.duration_sec for e in events)

        # Build caption from template
        parts: List[str] = []

        # Opening: describe overall duration and number of activities
        activity_types = list(dict.fromkeys(e.type for e in events))  # unique, ordered
        n_types = len(activity_types)

        parts.append(
            f"This {_format_duration(total_duration)} segment contains "
            f"{len(events)} events spanning {n_types} activity type{'s' if n_types != 1 else ''}."
        )

        # Dominant activity
        if dominant:
            pct = (dominant.duration_sec / total_duration * 100) if total_duration > 0 else 0
            parts.append(
                f"The dominant activity is {dominant.type.replace('_', ' ')} "
                f"({pct:.0f}% of total time)."
            )

        # List events in temporal order
        sorted_events = sorted(events, key=lambda e: e.start_token)
        if len(sorted_events) <= 6:
            event_descriptions = [_build_event_summary(e) for e in sorted_events]
            parts.extend(event_descriptions)
        else:
            # Summarize: first, last, and transitions
            parts.append(_build_event_summary(sorted_events[0]))
            transitions = [e for e in sorted_events if e.from_activity and e.to_activity]
            for t in transitions[:3]:
                parts.append(_build_event_summary(t))
            parts.append(_build_event_summary(sorted_events[-1]))

        return " ".join(parts)

    def run_all(
        self,
        manifest_path: str,
        output_path: str,
    ) -> None:
        """Generate template predictions for all samples in a manifest.

        Args:
            manifest_path: path to a JSONL manifest file containing
                           ContextSample records.
            output_path: path to write the predictions JSONL file.
                         Each line: {"sample_id": ..., "prediction": ...}
        """
        predictions: List[Dict[str, str]] = []

        with open(manifest_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                sample = ContextSample(**obj)
                caption = self.predict(sample)
                predictions.append({
                    "sample_id": sample.sample_id,
                    "prediction": caption,
                })

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for pred in predictions:
                f.write(json.dumps(pred) + "\n")
