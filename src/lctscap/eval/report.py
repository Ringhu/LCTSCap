"""Report generation utilities: CSV, Markdown, LaTeX tables."""

import csv
import io
from typing import Dict, List


def results_to_csv(results: Dict[str, float], path: str) -> None:
    """Write a results dictionary to a CSV file.

    Args:
        results: metric_name -> value mapping.
        path: output file path.
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for metric, value in sorted(results.items()):
            writer.writerow([metric, f"{value:.6f}"])


def results_to_markdown(results: Dict[str, float]) -> str:
    """Format results as a Markdown table.

    Args:
        results: metric_name -> value mapping.

    Returns:
        Markdown-formatted table string.
    """
    lines = ["| Metric | Value |", "|--------|-------|"]
    for metric, value in sorted(results.items()):
        lines.append(f"| {metric} | {value:.4f} |")
    return "\n".join(lines)


def results_to_latex(results: Dict[str, float]) -> str:
    """Format results as a LaTeX tabular environment.

    Args:
        results: metric_name -> value mapping.

    Returns:
        LaTeX tabular string.
    """
    lines = [
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Metric & Value \\",
        r"\midrule",
    ]
    for metric, value in sorted(results.items()):
        # Escape underscores for LaTeX
        safe_metric = metric.replace("_", r"\_")
        lines.append(f"{safe_metric} & {value:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def compare_models(
    results_list: List[Dict[str, float]],
    model_names: List[str],
) -> str:
    """Generate a side-by-side comparison table of multiple models.

    Args:
        results_list: list of results dictionaries (one per model).
        model_names: list of model names corresponding to results_list.

    Returns:
        Markdown-formatted comparison table.
    """
    if len(results_list) != len(model_names):
        raise ValueError("results_list and model_names must have the same length")

    if not results_list:
        return "No results to compare."

    # Collect all metric names across all models
    all_metrics = set()
    for results in results_list:
        all_metrics.update(results.keys())
    sorted_metrics = sorted(all_metrics)

    # Build header
    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|--------|" + "|".join(["-------"] * len(model_names)) + "|"

    lines = [header, separator]
    for metric in sorted_metrics:
        values = []
        best_val = -float("inf")
        best_idx = -1

        # Find the best value for bolding
        for i, results in enumerate(results_list):
            val = results.get(metric)
            if val is not None and val > best_val:
                best_val = val
                best_idx = i

        for i, results in enumerate(results_list):
            val = results.get(metric)
            if val is None:
                values.append("-")
            else:
                formatted = f"{val:.4f}"
                if i == best_idx and len(results_list) > 1:
                    formatted = f"**{formatted}**"
                values.append(formatted)

        lines.append(f"| {metric} | " + " | ".join(values) + " |")

    return "\n".join(lines)
