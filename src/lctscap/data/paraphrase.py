"""LLM-based paraphrase pipeline for LCTSCap.

Converts template-generated captions into more natural language using
a local LLM (e.g. Qwen3-4B), then verifies the paraphrase preserves
factual consistency with the original events.
"""

import logging
from typing import Any, Dict, List, Optional

from lctscap.data.schema import Event

logger = logging.getLogger(__name__)

# Default activity vocabulary across both datasets
DEFAULT_ACTIVITY_VOCAB: List[str] = [
    "walking",
    "running",
    "sitting",
    "standing",
    "sleeping",
    "lying",
    "cycling",
    "vehicle",
    "household",
    "eating",
    "self_care",
    "screen_time",
    "socializing",
    "stairs",
    "shuffling",
    "stairs_up",
    "stairs_down",
    "nordic_walking",
    "jumping",
    "other",
]


def build_paraphrase_prompt(
    template_caption: str,
    events: List[Event],
) -> str:
    """Construct a prompt for the LLM to paraphrase a template caption.

    The prompt instructs the model to rewrite the caption in natural language
    while preserving all factual content (activities, durations, ordering).

    Args:
        template_caption: The template-generated caption to paraphrase.
        events: The original events for grounding.

    Returns:
        A prompt string ready to send to an LLM.
    """
    # Build event summary for context
    event_lines = []
    for i, e in enumerate(events):
        line = (
            f"  {i + 1}. {e.type}: windows {e.start_token}-{e.end_token}, "
            f"duration {e.duration_sec:.0f}s"
        )
        if e.is_dominant:
            line += " (dominant)"
        event_lines.append(line)
    events_str = "\n".join(event_lines)

    prompt = f"""You are a precise scientific writer. Rewrite the following template-generated caption into natural, fluent English. You MUST preserve ALL factual information:
- Every activity mentioned must appear in your rewrite
- Temporal ordering must be preserved
- Duration information must be accurately reflected
- Do not add activities or details not present in the original
- Do not omit any activities from the original
- Keep the rewrite concise (similar length to original)

## Original Events (ground truth):
{events_str}

## Template Caption:
{template_caption}

## Natural Language Rewrite:
"""
    return prompt


def paraphrase_caption(
    template: str,
    events: List[Event],
    model_name: str = "Qwen/Qwen3-4B",
) -> str:
    """Paraphrase a template caption using a local LLM.

    This function attempts to call a local inference server. If unavailable,
    it falls back to a rule-based paraphrase.

    Args:
        template: Template-generated caption text.
        events: Original events for grounding the prompt.
        model_name: Name/path of the LLM to use.

    Returns:
        Paraphrased caption string.
    """
    prompt = build_paraphrase_prompt(template, events)

    # Try calling local inference server (e.g. vLLM, Ollama, etc.)
    try:
        paraphrased = _call_local_llm(prompt, model_name)
        if paraphrased:
            return paraphrased.strip()
    except Exception as e:
        logger.warning(
            "Local LLM call failed (model=%s): %s. "
            "Falling back to rule-based paraphrase.",
            model_name,
            e,
        )

    # Fallback: rule-based light paraphrase
    return _rule_based_paraphrase(template)


def _call_local_llm(prompt: str, model_name: str) -> Optional[str]:
    """Call a local LLM inference server.

    Supports OpenAI-compatible API endpoints (e.g. vLLM, Ollama, text-generation-inference).

    Args:
        prompt: The full prompt string.
        model_name: Model identifier.

    Returns:
        Generated text, or None if the call fails.
    """
    import requests

    # Try common local endpoints
    endpoints = [
        "http://localhost:11434/api/generate",  # Ollama
        "http://localhost:8000/v1/completions",  # vLLM
    ]

    for endpoint in endpoints:
        try:
            if "ollama" in endpoint or "11434" in endpoint:
                # Ollama API format
                response = requests.post(
                    endpoint,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "max_tokens": 512,
                        },
                    },
                    timeout=60,
                )
                if response.status_code == 200:
                    return response.json().get("response", "")
            else:
                # OpenAI-compatible API format
                response = requests.post(
                    endpoint,
                    json={
                        "model": model_name,
                        "prompt": prompt,
                        "max_tokens": 512,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    },
                    timeout=60,
                )
                if response.status_code == 200:
                    choices = response.json().get("choices", [])
                    if choices:
                        return choices[0].get("text", "")
        except requests.ConnectionError:
            continue
        except Exception as e:
            logger.debug("Endpoint %s failed: %s", endpoint, e)
            continue

    return None


def _rule_based_paraphrase(template: str) -> str:
    """Apply simple rule-based transformations to make the template more natural.

    This is a fallback when no LLM is available. It applies substitutions
    to reduce the mechanical feel of template captions.

    Args:
        template: Template caption string.

    Returns:
        Lightly paraphrased string.
    """
    text = template

    # Replace mechanical phrasing
    replacements = [
        ("The participant spent the entire", "Throughout the entire"),
        ("The participant was predominantly", "The participant primarily engaged in"),
        ("The participant engaged in", "The individual participated in"),
        ("distinct activity events", "separate activity periods"),
        ("accounting for", "comprising"),
        ("The activity sequence is:", "The sequence of activities was:"),
        ("The recording begins with", "Initially, the recording shows"),
        ("and concludes with", "ending with"),
        ("Other activities include", "Additional activities observed were"),
        ("with brief periods of", "interspersed with"),
    ]

    for old, new in replacements:
        text = text.replace(old, new)

    return text


def verify_paraphrase(
    original_events: List[Event],
    paraphrased: str,
    activity_vocab: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Verify that a paraphrased caption is consistent with original events.

    Checks:
    1. **Activity coverage**: all activities from events appear in the text.
    2. **No hallucination**: no activity words in the text that aren't in events.
    3. **Ordering**: activities appear in the text in the same order as events.

    Args:
        original_events: Ground-truth events.
        paraphrased: The paraphrased caption text.
        activity_vocab: Vocabulary of known activity names. Defaults to
            :data:`DEFAULT_ACTIVITY_VOCAB`.

    Returns:
        Dictionary with:
        - ``is_valid``: overall pass/fail
        - ``coverage_score``: fraction of event activities found in text
        - ``missing_activities``: activities in events but not in text
        - ``hallucinated_activities``: activities in text but not in events
        - ``order_preserved``: whether activity ordering is maintained
        - ``details``: human-readable summary
    """
    if activity_vocab is None:
        activity_vocab = DEFAULT_ACTIVITY_VOCAB

    text_lower = paraphrased.lower()

    # Activities present in events
    event_activities = list(dict.fromkeys(e.type for e in original_events))  # ordered, unique

    # Check coverage: which event activities appear in the text
    found = []
    missing = []
    for act in event_activities:
        # Handle underscores -> spaces for matching
        variants = [act, act.replace("_", " "), act.replace("_", "-")]
        if any(v in text_lower for v in variants):
            found.append(act)
        else:
            missing.append(act)

    coverage_score = len(found) / len(event_activities) if event_activities else 1.0

    # Check for hallucinated activities
    event_activity_set = set(event_activities)
    hallucinated = []
    for act in activity_vocab:
        if act in event_activity_set:
            continue
        variants = [act, act.replace("_", " "), act.replace("_", "-")]
        if any(v in text_lower for v in variants):
            hallucinated.append(act)

    # Check ordering: found activities should appear in text in same order as events
    order_preserved = True
    if len(found) >= 2:
        positions = []
        for act in event_activities:
            if act in found:
                variants = [act, act.replace("_", " "), act.replace("_", "-")]
                pos = len(text_lower)
                for v in variants:
                    idx = text_lower.find(v)
                    if idx >= 0:
                        pos = min(pos, idx)
                positions.append(pos)

        # Check that positions are non-decreasing
        for i in range(1, len(positions)):
            if positions[i] < positions[i - 1]:
                order_preserved = False
                break

    # Overall validity
    is_valid = (
        coverage_score >= 0.8
        and len(hallucinated) == 0
        and order_preserved
    )

    details_parts = []
    if missing:
        details_parts.append(f"Missing activities: {missing}")
    if hallucinated:
        details_parts.append(f"Hallucinated activities: {hallucinated}")
    if not order_preserved:
        details_parts.append("Activity ordering not preserved in text")
    if not details_parts:
        details_parts.append("Paraphrase is consistent with events")

    return {
        "is_valid": is_valid,
        "coverage_score": coverage_score,
        "missing_activities": missing,
        "hallucinated_activities": hallucinated,
        "order_preserved": order_preserved,
        "details": "; ".join(details_parts),
    }


class ParaphrasePipeline:
    """Batch paraphrase pipeline for annotation JSONL samples."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-4B",
        batch_size: int = 8,
        keep_invalid: bool = False,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.keep_invalid = keep_invalid

    @staticmethod
    def _coerce_events(events_raw: List[Dict[str, Any]]) -> List[Event]:
        return [e if isinstance(e, Event) else Event(**e) for e in (events_raw or [])]

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Paraphrase one sample and attach verification metadata."""
        events = self._coerce_events(sample.get("events", []))
        template = sample.get("caption_long") or sample.get("caption_short") or ""

        if not template:
            enriched = dict(sample)
            enriched["caption_paraphrase"] = ""
            enriched["paraphrase_verification"] = {
                "is_valid": False,
                "coverage_score": 0.0,
                "missing_activities": [],
                "hallucinated_activities": [],
                "order_preserved": False,
                "details": "No source caption available for paraphrasing",
            }
            return enriched

        paraphrased = paraphrase_caption(template, events, model_name=self.model_name)
        verification = verify_paraphrase(events, paraphrased)

        enriched = dict(sample)
        enriched["caption_paraphrase"] = paraphrased
        enriched["paraphrase_verification"] = verification

        if verification["is_valid"]:
            enriched["caption_long"] = paraphrased
        elif not self.keep_invalid:
            enriched["caption_paraphrase"] = template

        return enriched

    def process(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a list of JSON-like samples."""
        processed: List[Dict[str, Any]] = []
        for sample in samples:
            processed.append(self.process_sample(sample))
        return processed
