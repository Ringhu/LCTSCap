from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_plot_style import COLORS, save_fig

OUT_PATH = Path(__file__).resolve().parent / "task_hierarchy_overview.png"

fig, ax = plt.subplots(figsize=(10.8, 4.8), facecolor="white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

boxes = [
    (0.03, 0.18, 0.28, 0.64, COLORS[0], "CAPTURE24 main task", [
        "Long context, multi-event",
        "Goal: event grounding + caption",
        "Most stable result: GT events -> template = full score",
        "Current gap: R103 still lacks event_type_f1 / span_iou",
    ]),
    (0.37, 0.18, 0.26, 0.64, COLORS[2], "UCR external check", [
        "Univariate, single-label",
        "Zero-shot time-series/text retrieval",
        "Strong cases: Wafer, FordA",
        "Meaning: learned features show cross-dataset signal",
    ]),
    (0.69, 0.18, 0.28, 0.64, COLORS[3], "UEA external check", [
        "Multivariate, still single-label retrieval",
        "Strong cases: Heartbeat, MotorImagery",
        "Weak cases: RacketSports, Cricket",
        "1000-length failed at LocalEncoder patch limit",
    ]),
]

for x, y, w, h, color, title, lines in boxes:
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(x + 0.02, y + h - 0.06, title, fontsize=12, weight="bold", va="top")
    yy = y + h - 0.13
    for line in lines:
        ax.text(x + 0.025, yy, f"- {line}", fontsize=9.6, va="top")
        yy -= 0.105

arrow1 = FancyArrowPatch((0.31, 0.50), (0.37, 0.50), arrowstyle="->", mutation_scale=15, linewidth=1.5, color="black")
arrow2 = FancyArrowPatch((0.63, 0.50), (0.69, 0.50), arrowstyle="->", mutation_scale=15, linewidth=1.5, color="black")
ax.add_patch(arrow1)
ax.add_patch(arrow2)
ax.text(0.34, 0.55, "short-label transfer", fontsize=8.5, ha="center")
ax.text(0.66, 0.55, "multivariate transfer", fontsize=8.5, ha="center")
ax.text(
    0.50,
    0.05,
    "Confirmed: the main-task evaluation pipeline works, and external retrieval shows partial positive transfer.\nNot confirmed yet: a learned long-context caption model is already established.",
    fontsize=8.8,
    ha="center",
)

fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.06)
save_fig(fig, str(OUT_PATH))
