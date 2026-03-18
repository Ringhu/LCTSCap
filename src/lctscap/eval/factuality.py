"""Activity-level factuality metrics for time series captioning."""

from typing import Dict, List, Set, Tuple


def activity_mention_f1(
    pred_activities: List[Set[str]],
    gold_activities: List[Set[str]],
) -> Dict[str, float]:
    """Compute precision, recall, and F1 of mentioned activities.

    For each sample pair, we measure set overlap between predicted and
    gold activities, then average across all samples.

    Args:
        pred_activities: list of sets of activity names mentioned in predictions.
        gold_activities: list of sets of activity names from ground truth.

    Returns:
        Dictionary with keys activity_precision, activity_recall, activity_f1.
    """
    if len(pred_activities) != len(gold_activities):
        raise ValueError("pred_activities and gold_activities must have the same length")

    total_precision = 0.0
    total_recall = 0.0
    n = len(pred_activities)

    for pred_set, gold_set in zip(pred_activities, gold_activities):
        if len(pred_set) == 0 and len(gold_set) == 0:
            total_precision += 1.0
            total_recall += 1.0
            continue

        overlap = pred_set & gold_set
        precision = len(overlap) / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = len(overlap) / len(gold_set) if len(gold_set) > 0 else 0.0
        total_precision += precision
        total_recall += recall

    avg_precision = total_precision / n if n > 0 else 0.0
    avg_recall = total_recall / n if n > 0 else 0.0
    f1 = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if (avg_precision + avg_recall) > 0
        else 0.0
    )

    return {
        "activity_precision": avg_precision,
        "activity_recall": avg_recall,
        "activity_f1": f1,
    }


def dominant_activity_accuracy(
    pred_dominant: List[str],
    gold_dominant: List[str],
) -> float:
    """Compute accuracy of predicting the dominant (longest) activity.

    Args:
        pred_dominant: list of predicted dominant activity labels.
        gold_dominant: list of ground-truth dominant activity labels.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if len(pred_dominant) != len(gold_dominant):
        raise ValueError("pred_dominant and gold_dominant must have the same length")
    if len(pred_dominant) == 0:
        return 0.0

    correct = sum(
        1 for p, g in zip(pred_dominant, gold_dominant) if p.lower().strip() == g.lower().strip()
    )
    return correct / len(pred_dominant)


def transition_accuracy(
    pred_transitions: List[List[Tuple[str, str]]],
    gold_transitions: List[List[Tuple[str, str]]],
) -> float:
    """Compute accuracy of detected activity transitions.

    A transition is a (from_activity, to_activity) tuple.  For each sample,
    we compute the F1 of transition sets, then average across samples.

    Args:
        pred_transitions: list of lists of (from, to) transition tuples per sample.
        gold_transitions: list of lists of (from, to) transition tuples per sample.

    Returns:
        Average transition F1 across samples.
    """
    if len(pred_transitions) != len(gold_transitions):
        raise ValueError("pred_transitions and gold_transitions must have the same length")

    total_f1 = 0.0
    n = len(pred_transitions)

    for pred_trans, gold_trans in zip(pred_transitions, gold_transitions):
        pred_set = set((a.lower().strip(), b.lower().strip()) for a, b in pred_trans)
        gold_set = set((a.lower().strip(), b.lower().strip()) for a, b in gold_trans)

        if len(pred_set) == 0 and len(gold_set) == 0:
            total_f1 += 1.0
            continue

        overlap = pred_set & gold_set
        precision = len(overlap) / len(pred_set) if len(pred_set) > 0 else 0.0
        recall = len(overlap) / len(gold_set) if len(gold_set) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        total_f1 += f1

    return total_f1 / n if n > 0 else 0.0


def _assign_duration_bin(
    duration_sec: float,
    bins: Dict[str, float],
) -> str:
    """Assign a duration value to a discrete bin.

    Bins should be ordered from shortest to longest threshold.
    E.g. bins={"short": 60, "medium": 300} means:
       duration < 60  => "short"
       60 <= duration < 300 => "medium"
       duration >= 300 => "long"

    Args:
        duration_sec: duration in seconds.
        bins: mapping from bin name to upper threshold.

    Returns:
        Bin label string.
    """
    sorted_bins = sorted(bins.items(), key=lambda x: x[1])
    for name, threshold in sorted_bins:
        if duration_sec < threshold:
            return name
    return "long"


def duration_bin_accuracy(
    pred_durations: List[Dict[str, float]],
    gold_durations: List[Dict[str, float]],
    bins: Dict[str, float] | None = None,
) -> float:
    """Compute accuracy of duration estimates after discretization into bins.

    Each element is a dict mapping activity type -> duration in seconds.
    We compare the binned duration per activity.

    Args:
        pred_durations: list of dicts {activity: duration_sec} from predictions.
        gold_durations: list of dicts {activity: duration_sec} from ground truth.
        bins: threshold dict for binning. Default: {"short": 60, "medium": 300}.

    Returns:
        Fraction of correct bin assignments across all activity entries.
    """
    if bins is None:
        bins = {"short": 60, "medium": 300}

    if len(pred_durations) != len(gold_durations):
        raise ValueError("pred_durations and gold_durations must have the same length")

    total = 0
    correct = 0

    for pred_dict, gold_dict in zip(pred_durations, gold_durations):
        # Only compare activities present in both
        common_activities = set(pred_dict.keys()) & set(gold_dict.keys())
        for activity in common_activities:
            pred_bin = _assign_duration_bin(pred_dict[activity], bins)
            gold_bin = _assign_duration_bin(gold_dict[activity], bins)
            if pred_bin == gold_bin:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
