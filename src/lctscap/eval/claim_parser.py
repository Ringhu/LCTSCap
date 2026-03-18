"""Parse generated captions into structured claims for verification."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

# Canonical activity vocabulary covering both CAPTURE-24 and HARTH
ACTIVITY_VOCAB: List[str] = [
    "walking",
    "running",
    "sitting",
    "standing",
    "sleeping",
    "lying",
    "cycling",
    "vehicle",
    "household",
    "stairs",
    "self_care",
    "eating",
    "shuffling",
    "stairs_up",
    "stairs_down",
    "cycling_sit",
    "cycling_stand",
    "cycling_stand_inactive",
    "other",
]


@dataclass
class ParsedClaim:
    """A structured claim extracted from a generated caption.

    Attributes:
        claim_type: one of "activity", "duration", "transition", "ordering".
        activity: the activity label this claim references.
        span: optional (start_token, end_token) for temporal grounding.
        duration_sec: optional duration mentioned in seconds.
        ordering_ref: optional reference activity for ordering claims
                      (e.g. "walking" in "before walking").
    """

    claim_type: str  # "activity" | "duration" | "transition" | "ordering"
    activity: str
    span: Optional[Tuple[int, int]] = None
    duration_sec: Optional[float] = None
    ordering_ref: Optional[str] = None

    def __post_init__(self) -> None:
        valid_types = {"activity", "duration", "transition", "ordering"}
        if self.claim_type not in valid_types:
            raise ValueError(
                f"claim_type must be one of {valid_types}, got '{self.claim_type}'"
            )


def _build_activity_pattern(activity_vocab: List[str]) -> re.Pattern:
    """Build a regex pattern that matches any activity in the vocabulary."""
    # Sort by length descending so longer names match first
    sorted_vocab = sorted(activity_vocab, key=len, reverse=True)
    # Replace underscores with optional spaces/underscores for flexible matching
    alternatives = []
    for act in sorted_vocab:
        pattern = act.replace("_", r"[\s_]?")
        alternatives.append(pattern)
    return re.compile(r"\b(" + "|".join(alternatives) + r")\b", re.IGNORECASE)


def _normalize_activity(raw: str, activity_vocab: List[str]) -> str:
    """Normalize a matched activity string to canonical vocabulary form."""
    raw_clean = re.sub(r"\s+", "_", raw.strip().lower())
    for act in activity_vocab:
        if raw_clean == act.lower() or raw_clean.replace("_", "") == act.replace("_", "").lower():
            return act
    return raw_clean


# Duration patterns: e.g., "for 120 seconds", "lasting 5 minutes", "about 2 min"
_DURATION_PATTERNS = [
    # "for X seconds/minutes/hours"
    re.compile(
        r"(?:for|lasting|about|approximately|roughly|around)\s+"
        r"(\d+(?:\.\d+)?)\s*(seconds?|secs?|s|minutes?|mins?|m|hours?|hrs?|h)\b",
        re.IGNORECASE,
    ),
    # "X-second/minute/hour" as adjective
    re.compile(
        r"(\d+(?:\.\d+)?)\s*-?\s*(second|sec|minute|min|hour|hr)\b",
        re.IGNORECASE,
    ),
]

# Transition patterns: "from X to Y", "transitions from X to Y", "switches to Y"
_TRANSITION_PATTERNS = [
    re.compile(
        r"(?:transitions?|switches?|changes?|moves?|goes?)\s+from\s+(\w[\w\s]*?)\s+to\s+(\w[\w\s]*?)(?:\.|,|;|\s|$)",
        re.IGNORECASE,
    ),
    re.compile(
        r"from\s+(\w[\w\s]*?)\s+to\s+(\w[\w\s]*?)(?:\.|,|;|\s|$)",
        re.IGNORECASE,
    ),
]

# Ordering patterns: "before X", "after X", "followed by X", "then X"
_ORDERING_PATTERNS = [
    re.compile(r"(?:before|prior\s+to)\s+(\w[\w\s]*?)(?:\.|,|;|\s|$)", re.IGNORECASE),
    re.compile(r"(?:after|following)\s+(\w[\w\s]*?)(?:\.|,|;|\s|$)", re.IGNORECASE),
    re.compile(r"(?:followed\s+by|then)\s+(\w[\w\s]*?)(?:\.|,|;|\s|$)", re.IGNORECASE),
]

# Span patterns: "[18, 25]", "tokens 18-25", "windows 18 to 25"
_SPAN_PATTERNS = [
    re.compile(r"\[(\d+)\s*,\s*(\d+)\]"),
    re.compile(r"(?:tokens?|windows?)\s+(\d+)\s*[-–]\s*(\d+)", re.IGNORECASE),
    re.compile(r"(?:tokens?|windows?)\s+(\d+)\s+to\s+(\d+)", re.IGNORECASE),
]


def _to_seconds(value: float, unit: str) -> float:
    """Convert a duration value + unit string to seconds."""
    unit_lower = unit.lower().rstrip("s")
    if unit_lower in ("second", "sec", ""):
        return value
    elif unit_lower in ("minute", "min", "m"):
        return value * 60
    elif unit_lower in ("hour", "hr", "h"):
        return value * 3600
    return value


def parse_claims(
    caption: str,
    activity_vocab: List[str] | None = None,
) -> List[ParsedClaim]:
    """Parse a generated caption into structured claims.

    Uses regex patterns to extract activity mentions, duration claims,
    transitions, and ordering references.

    Args:
        caption: the generated caption text.
        activity_vocab: list of known activity labels.  Uses default if None.

    Returns:
        List of ParsedClaim objects extracted from the caption.
    """
    if activity_vocab is None:
        activity_vocab = ACTIVITY_VOCAB

    claims: List[ParsedClaim] = []
    act_pattern = _build_activity_pattern(activity_vocab)

    # 1. Extract activity mention claims
    for match in act_pattern.finditer(caption):
        activity = _normalize_activity(match.group(1), activity_vocab)
        # Check if there is a nearby span reference
        span = None
        # Look within 60 characters after the match for span info
        context_after = caption[match.end(): match.end() + 80]
        for sp in _SPAN_PATTERNS:
            span_match = sp.search(context_after)
            if span_match:
                span = (int(span_match.group(1)), int(span_match.group(2)))
                break

        claims.append(ParsedClaim(
            claim_type="activity",
            activity=activity,
            span=span,
        ))

    # 2. Extract duration claims
    for dp in _DURATION_PATTERNS:
        for match in dp.finditer(caption):
            value = float(match.group(1))
            unit = match.group(2)
            duration_sec = _to_seconds(value, unit)

            # Find nearest activity mention before this duration
            preceding = caption[:match.start()]
            act_matches = list(act_pattern.finditer(preceding))
            activity = ""
            if act_matches:
                activity = _normalize_activity(act_matches[-1].group(1), activity_vocab)

            if activity:
                claims.append(ParsedClaim(
                    claim_type="duration",
                    activity=activity,
                    duration_sec=duration_sec,
                ))

    # 3. Extract transition claims
    for tp in _TRANSITION_PATTERNS:
        for match in tp.finditer(caption):
            from_raw = match.group(1).strip()
            to_raw = match.group(2).strip()

            # Check if both are known activities
            from_matches = act_pattern.findall(from_raw)
            to_matches = act_pattern.findall(to_raw)

            if from_matches and to_matches:
                from_act = _normalize_activity(from_matches[0], activity_vocab)
                to_act = _normalize_activity(to_matches[0], activity_vocab)
                claims.append(ParsedClaim(
                    claim_type="transition",
                    activity=from_act,
                    ordering_ref=to_act,
                ))

    # 4. Extract ordering claims
    for op in _ORDERING_PATTERNS:
        for match in op.finditer(caption):
            ref_raw = match.group(1).strip()
            ref_matches = act_pattern.findall(ref_raw)

            # Find the nearest activity mentioned before this ordering keyword
            preceding = caption[:match.start()]
            act_matches = list(act_pattern.finditer(preceding))

            if ref_matches and act_matches:
                activity = _normalize_activity(act_matches[-1].group(1), activity_vocab)
                ref_act = _normalize_activity(ref_matches[0], activity_vocab)
                claims.append(ParsedClaim(
                    claim_type="ordering",
                    activity=activity,
                    ordering_ref=ref_act,
                ))

    return claims


def extract_mentioned_activities(
    caption: str,
    activity_vocab: List[str] | None = None,
) -> Set[str]:
    """Extract the set of activities mentioned in a caption.

    Args:
        caption: the caption text.
        activity_vocab: list of known activity labels.  Uses default if None.

    Returns:
        Set of canonical activity names found in the caption.
    """
    if activity_vocab is None:
        activity_vocab = ACTIVITY_VOCAB

    act_pattern = _build_activity_pattern(activity_vocab)
    found: Set[str] = set()
    for match in act_pattern.finditer(caption):
        found.add(_normalize_activity(match.group(1), activity_vocab))
    return found


def extract_temporal_order(
    caption: str,
    activity_vocab: List[str] | None = None,
) -> List[str]:
    """Extract the temporal order of activities as they are mentioned.

    Returns a list of unique activity names in the order they first appear.

    Args:
        caption: the caption text.
        activity_vocab: list of known activity labels.  Uses default if None.

    Returns:
        Ordered list of unique activity names by first mention position.
    """
    if activity_vocab is None:
        activity_vocab = ACTIVITY_VOCAB

    act_pattern = _build_activity_pattern(activity_vocab)
    order: List[str] = []
    seen: Set[str] = set()
    for match in act_pattern.finditer(caption):
        act = _normalize_activity(match.group(1), activity_vocab)
        if act not in seen:
            order.append(act)
            seen.add(act)
    return order
