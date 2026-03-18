"""Visualization utilities for time series samples, token boundaries, and metrics."""

from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# Color palette for activities
_ACTIVITY_COLORS = {
    "walking": "#4CAF50",
    "running": "#F44336",
    "sitting": "#2196F3",
    "standing": "#FFC107",
    "sleeping": "#9C27B0",
    "lying": "#673AB7",
    "cycling": "#00BCD4",
    "vehicle": "#FF9800",
    "household": "#795548",
    "stairs": "#607D8B",
    "self_care": "#E91E63",
    "eating": "#8BC34A",
    "shuffling": "#CDDC39",
    "stairs_up": "#009688",
    "stairs_down": "#3F51B5",
    "other": "#9E9E9E",
}


def _get_color(activity: str) -> str:
    """Get a consistent color for an activity type."""
    return _ACTIVITY_COLORS.get(activity.lower(), "#9E9E9E")


def plot_sample(
    tensor: np.ndarray,
    events: List[dict],
    title: str,
    save_path: str,
    channel_names: Optional[List[str]] = None,
) -> None:
    """Plot a time-series sample with event boundaries overlaid.

    Args:
        tensor: time-series data of shape [T, C, L] or [C, L].
                T = number of tokens/windows, C = channels, L = samples per window.
        events: list of event dicts with keys "type", "start_token", "end_token".
        title: plot title string.
        save_path: file path to save the figure.
        channel_names: optional list of channel names for the legend.
    """
    if tensor.ndim == 3:
        # [T, C, L] -> concatenate along time axis to [C, T*L]
        T, C, L = tensor.shape
        flat = tensor.transpose(1, 0, 2).reshape(C, T * L)
        samples_per_token = L
    elif tensor.ndim == 2:
        C, total_L = tensor.shape
        flat = tensor
        samples_per_token = 1
        T = total_L
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")

    total_samples = flat.shape[1]
    time_axis = np.arange(total_samples)

    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot each channel
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(C)]
    for i in range(min(C, 6)):  # Limit to 6 channels for readability
        ax.plot(time_axis, flat[i], linewidth=0.5, alpha=0.7, label=channel_names[i])

    # Overlay event spans as colored background regions
    legend_entries = {}
    for event in events:
        start = event.get("start_token", 0)
        end = event.get("end_token", start + 1)
        etype = event.get("type", "unknown")
        color = _get_color(etype)

        x_start = start * samples_per_token
        x_end = end * samples_per_token
        ax.axvspan(x_start, x_end, alpha=0.15, color=color)

        if etype not in legend_entries:
            legend_entries[etype] = mpatches.Patch(color=color, alpha=0.3, label=etype)

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title(title)

    # Combine line and event legends
    handles, labels = ax.get_legend_handles_labels()
    handles.extend(legend_entries.values())
    ax.legend(handles=handles, loc="upper right", fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_token_boundaries(
    tensor: np.ndarray,
    context_len: int,
    window_sec: float,
    save_path: str,
) -> None:
    """Visualize token (window) segmentation of a time-series.

    Draws vertical lines at token boundaries and labels each token.

    Args:
        tensor: time-series data of shape [T, C, L] or [C, L].
        context_len: number of tokens in the context.
        window_sec: duration of each window/token in seconds.
        save_path: file path to save the figure.
    """
    if tensor.ndim == 3:
        T, C, L = tensor.shape
        flat = tensor.transpose(1, 0, 2).reshape(C, T * L)
        samples_per_token = L
    elif tensor.ndim == 2:
        C, total_L = tensor.shape
        flat = tensor
        samples_per_token = total_L // context_len if context_len > 0 else 1
        T = context_len
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tensor.shape}")

    total_samples = flat.shape[1]
    time_axis = np.arange(total_samples)

    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot first channel only for clarity
    ax.plot(time_axis, flat[0], linewidth=0.5, color="steelblue")

    # Token boundaries
    for t in range(min(context_len, T) + 1):
        x = t * samples_per_token
        if x <= total_samples:
            ax.axvline(x=x, color="red", linewidth=0.5, alpha=0.6, linestyle="--")

    # Label tokens
    label_interval = max(1, context_len // 16)
    for t in range(0, min(context_len, T), label_interval):
        x_mid = (t + 0.5) * samples_per_token
        t_sec = t * window_sec
        ax.text(x_mid, ax.get_ylim()[1] * 0.9, f"t{t}\n{t_sec:.0f}s",
                ha="center", va="top", fontsize=6, color="red")

    ax.set_xlabel("Sample index")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Token boundaries: {context_len} tokens x {window_sec}s = {context_len * window_sec:.0f}s total")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_comparison(
    results_dict: Dict[str, Dict[str, float]],
    metric_names: List[str],
    save_path: str,
) -> None:
    """Create a grouped bar chart comparing models across selected metrics.

    Args:
        results_dict: mapping model_name -> {metric_name: value}.
        metric_names: list of metric names to include in the comparison.
        save_path: file path to save the figure.
    """
    model_names = list(results_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    if n_models == 0 or n_metrics == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, n_metrics * 1.5), 5))

    x = np.arange(n_metrics)
    width = 0.8 / n_models

    colors = plt.cm.Set2(np.linspace(0, 1, n_models))

    for i, model in enumerate(model_names):
        values = [results_dict[model].get(m, 0.0) for m in metric_names]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model, color=colors[i])

        # Value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=6,
                )

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in metric_names], fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
