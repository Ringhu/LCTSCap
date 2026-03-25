from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_plot_style import COLORS, save_fig

OUT_PATH = Path(__file__).resolve().parent / "task_hierarchy_overview.pdf"

fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

boxes = [
    (0.03, 0.17, 0.28, 0.66, COLORS[0], "CAPTURE24 main task", [
        "Long context, multi-event",
        "Goal: event grounding + caption",
        "Most stable result: GT events -> template = full score",
        "Current gap: R103 still lacks event_type_f1 / span_iou"
    ]),
    (0.37, 0.17, 0.26, 0.66, COLORS[2], "UCR external check", [
        "Univariate, single-label",
        "Task: zero-shot time-series/text retrieval",
        "Strong cases: Wafer, FordA",
        "Meaning: learned features show cross-dataset signal"
    ]),
    (0.69, 0.17, 0.28, 0.66, COLORS[3], "UEA external check", [
        "Multivariate, but still single-label retrieval",
        "Strong cases: Heartbeat, MotorImagery",
        "Weak cases: RacketSports, Cricket",
        "1000-length failed at LocalEncoder patch limit"
    ]),
]

for x, y, w, h, color, title, lines in boxes:
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color,
        alpha=0.12,
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(x + 0.02, y + h - 0.07, title, fontsize=12, weight="bold", va="top")
    yy = y + h - 0.14
    for line in lines:
        ax.text(x + 0.025, yy, f"- {line}", fontsize=10, va="top")
        yy -= 0.11

arrow1 = FancyArrowPatch((0.31, 0.50), (0.37, 0.50), arrowstyle="->", mutation_scale=15, linewidth=1.6, color="black")
arrow2 = FancyArrowPatch((0.63, 0.50), (0.69, 0.50), arrowstyle="->", mutation_scale=15, linewidth=1.6, color="black")
ax.add_patch(arrow1)
ax.add_patch(arrow2)

ax.text(0.34, 0.55, "short-label transfer", fontsize=9, ha="center")
ax.text(0.66, 0.55, "multivariate transfer", fontsize=9, ha="center")
ax.text(
    0.50,
    0.04,
    "What is confirmed: the main-task evaluation pipeline works, and external retrieval shows partial positive transfer.\nWhat is not confirmed yet: a learned long-context caption model is already established.",
    fontsize=9,
    ha="center",
)

save_fig(fig, str(OUT_PATH))
