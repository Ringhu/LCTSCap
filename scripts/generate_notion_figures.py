#!/usr/bin/env python
"""Generate static figures for Notion pages.

Creates:
    - A workflow overview diagram
    - Sample time-series figures with event timelines and captions
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import textwrap

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import torch

ASSET_DIR = Path("assets/notion")
ASSET_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")

COLORS = {
    "sleeping": "#355C7D",
    "sitting": "#6C5B7B",
    "standing": "#F8B195",
    "walking": "#F67280",
    "running": "#C06C84",
    "cycling": "#45ADA8",
    "vehicle": "#547980",
    "household": "#F4A261",
    "other": "#8D99AE",
}


def load_sample(annotation_path: str, sample_id: str) -> dict:
    with open(annotation_path, "r") as f:
        for line in f:
            sample = json.loads(line)
            if sample["sample_id"] == sample_id:
                return sample
    raise ValueError(f"Sample not found: {sample_id}")


def load_signal(sample: dict) -> tuple[np.ndarray, np.ndarray]:
    windows = []
    for path in sample["tensor_paths"]:
        t = torch.load(path, map_location="cpu").float().numpy()
        windows.append(t)

    stacked = np.concatenate(windows, axis=1)  # [C, T]
    magnitude = np.sqrt((stacked ** 2).sum(axis=0))
    x_minutes = np.arange(magnitude.shape[0]) / 50.0 / 60.0
    return x_minutes, magnitude


def draw_event_timeline(ax, events: list[dict], total_min: float) -> None:
    ax.set_xlim(0, total_min)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_xlabel("Time (minutes)")
    for event in events:
        start = event["start_token"] * 10.0 / 60.0
        end = event["end_token"] * 10.0 / 60.0
        color = COLORS.get(event["type"], "#999999")
        ax.add_patch(Rectangle((start, 0.18), end - start, 0.52, color=color, alpha=0.9))
        mid = (start + end) / 2.0
        if end - start > 0.8:
            ax.text(mid, 0.44, event["type"], ha="center", va="center", fontsize=9, color="white")


def save_sample_figure(sample: dict, out_path: Path, title: str) -> None:
    x_minutes, magnitude = load_signal(sample)
    total_min = x_minutes[-1] if len(x_minutes) else sample["context_len"] * 10.0 / 60.0

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        3,
        1,
        height_ratios=[4.3, 1.0, 1.3],
        left=0.07,
        right=0.98,
        top=0.84,
        bottom=0.08,
        hspace=0.42,
    )
    ax_sig = fig.add_subplot(gs[0, 0])
    ax_evt = fig.add_subplot(gs[1, 0])
    ax_txt = fig.add_subplot(gs[2, 0])

    fig.suptitle(title, x=0.07, y=0.95, ha="left", fontsize=18, fontweight="bold")
    fig.text(0.07, 0.90, sample["caption_short"], fontsize=11, color="#444444", ha="left")

    ax_sig.plot(x_minutes, magnitude, color="#0B3954", linewidth=0.8)
    ax_sig.set_xlim(0, total_min)
    ax_sig.set_ylabel("Signal Magnitude")
    ax_sig.grid(alpha=0.15)

    draw_event_timeline(ax_evt, sample["events"], total_min)
    ax_evt.set_title("Event Timeline", loc="left", fontsize=11, color="#333333")

    event_text = " | ".join(
        f"{e['type']} {e['start_token']}-{e['end_token']} ({int(e['duration_sec'])}s)"
        for e in sample["events"][:6]
    )
    if len(sample["events"]) > 6:
        event_text += " | ..."

    ax_txt.axis("off")
    ax_txt.text(0.0, 0.85, f"sample_id: {sample['sample_id']}", fontsize=10, family="monospace")
    ax_txt.text(
        0.0,
        0.50,
        textwrap.fill(sample["caption_long"], width=145),
        fontsize=10,
        color="#333333",
        va="center",
    )
    ax_txt.text(
        0.0,
        0.08,
        textwrap.fill(event_text, width=165),
        fontsize=9,
        color="#555555",
        va="bottom",
    )

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_workflow_figure(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 5)

    boxes = [
        (0.4, 2.9, 2.0, 1.0, "Phase 0\nData + Annotation"),
        (2.9, 2.9, 2.2, 1.0, "Long Context\n128 / 256 / 512"),
        (5.6, 2.9, 2.2, 1.0, "Template Baseline\n+ Evaluation"),
        (8.3, 2.9, 2.0, 1.0, "Phase 1\nRep. + Align"),
        (10.8, 2.9, 2.0, 1.0, "Phase 2\n+ Decoder"),
        (13.3, 2.9, 2.0, 1.0, "Phase 3\n+ Paraphrase"),
        (9.6, 1.1, 2.0, 1.0, "Phase 4\nLLM Bridge"),
        (12.1, 1.1, 2.0, 1.0, "Phase 5\nExperiments"),
        (14.2, 1.1, 1.4, 1.0, "Phase 6\nPaper"),
    ]

    box_colors = {
        "Phase 0\nData + Annotation": "#D9ED92",
        "Long Context\n128 / 256 / 512": "#B5E48C",
        "Template Baseline\n+ Evaluation": "#99D98C",
        "Phase 1\nRep. + Align": "#76C893",
        "Phase 2\n+ Decoder": "#52B69A",
        "Phase 3\n+ Paraphrase": "#34A0A4",
        "Phase 4\nLLM Bridge": "#168AAD",
        "Phase 5\nExperiments": "#1A759F",
        "Phase 6\nPaper": "#184E77",
    }

    centers = {}
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.2,
            edgecolor="#16324F",
            facecolor=box_colors[text],
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=12, color="white" if "Phase 6" in text or "Phase 5" in text or "Phase 4" in text else "#17324D", fontweight="bold")
        centers[text] = (x + w / 2, y + h / 2)

    def arrow(a: str, b: str) -> None:
        x1, y1 = centers[a]
        x2, y2 = centers[b]
        ax.add_patch(
            FancyArrowPatch(
                (x1 + 1.1, y1),
                (x2 - 1.1, y2),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=2,
                color="#16324F",
            )
        )

    arrow("Phase 0\nData + Annotation", "Long Context\n128 / 256 / 512")
    arrow("Long Context\n128 / 256 / 512", "Template Baseline\n+ Evaluation")
    arrow("Template Baseline\n+ Evaluation", "Phase 1\nRep. + Align")
    arrow("Phase 1\nRep. + Align", "Phase 2\n+ Decoder")
    arrow("Phase 2\n+ Decoder", "Phase 3\n+ Paraphrase")
    ax.add_patch(FancyArrowPatch((14.3, 2.9), (13.2, 2.1), arrowstyle="-|>", mutation_scale=16, linewidth=2, color="#16324F"))
    arrow("Phase 4\nLLM Bridge", "Phase 5\nExperiments")
    ax.add_patch(FancyArrowPatch((14.1, 1.6), (14.8, 1.6), arrowstyle="-|>", mutation_scale=16, linewidth=2, color="#16324F"))

    ax.text(0.4, 4.45, "LCTSCap Workflow", fontsize=22, fontweight="bold", color="#16324F")
    ax.text(0.4, 4.05, "From data and annotation to caption generation, long-context training, and paper-ready experiments", fontsize=11, color="#4B5563")

    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    save_workflow_figure(ASSET_DIR / "workflow_overview.png")

    sample_specs = [
        (
            "/cluster1/user1/lctscap_data/processed/capture24/annotations/capture24_128_32_annotated.jsonl",
            "ctx_capture24_P001_128_32_000000",
            ASSET_DIR / "sample_capture24_sleep.png",
            "CAPTURE-24 Example 1: Single-Activity Sleeping Segment",
        ),
        (
            "/cluster1/user1/lctscap_data/processed/capture24/annotations/capture24_128_32_annotated.jsonl",
            "ctx_capture24_P001_128_32_000061",
            ASSET_DIR / "sample_capture24_transition.png",
            "CAPTURE-24 Example 2: Multi-Event Segment with Transitions",
        ),
        (
            "/cluster1/user1/lctscap_data/processed/harth/annotations/harth_128_32_annotated.jsonl",
            "ctx_harth_S006_128_32_000000",
            ASSET_DIR / "sample_harth_multievent.png",
            "HARTH Example: High-Transition External Validation Sample",
        ),
    ]

    for ann_path, sample_id, out_path, title in sample_specs:
        sample = load_sample(ann_path, sample_id)
        save_sample_figure(sample, out_path, title)


if __name__ == "__main__":
    main()
