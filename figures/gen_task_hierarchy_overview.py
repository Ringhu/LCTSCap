from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from paper_plot_style import COLORS, save_fig

OUT_PATH = Path(__file__).resolve().parent / "task_hierarchy_overview.png"

fig, ax = plt.subplots(figsize=(11.2, 5.8), facecolor="white")
ax.set_facecolor("white")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

boxes = [
    {
        "xy": (0.03, 0.20),
        "wh": (0.27, 0.60),
        "color": COLORS[0],
        "title": "CAPTURE24 main task",
        "lines": [
            "Long context, multi-event",
            "Goal: grounding + caption",
            "Current upper bound: GT events -> template",
            "Missing piece: learned event metrics from R103",
        ],
    },
    {
        "xy": (0.365, 0.20),
        "wh": (0.25, 0.60),
        "color": COLORS[2],
        "title": "UCR external check",
        "lines": [
            "Univariate, single-label",
            "Zero-shot retrieval only",
            "Strong: Wafer, FordA",
            "Meaning: cross-dataset signal exists",
        ],
    },
    {
        "xy": (0.68, 0.20),
        "wh": (0.29, 0.60),
        "color": COLORS[3],
        "title": "UEA external check",
        "lines": [
            "Multivariate, still single-label",
            "Strong: Heartbeat, MotorImagery",
            "Weak: RacketSports, Cricket",
            "1000 length blocked by LocalEncoder limit",
        ],
    },
]

for box in boxes:
    x, y = box["xy"]
    w, h = box["wh"]
    color = box["color"]
    rect = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.02",
        facecolor="white",
        edgecolor=color,
        linewidth=2,
    )
    ax.add_patch(rect)
    ax.text(x + 0.02, y + h - 0.055, box["title"], fontsize=12, weight="bold", va="top")
    yy = y + h - 0.13
    for line in box["lines"]:
        ax.text(x + 0.022, yy, f"• {line}", fontsize=9.4, va="top")
        yy -= 0.105

ax.add_patch(FancyArrowPatch((0.30, 0.50), (0.365, 0.50), arrowstyle="->", mutation_scale=15, linewidth=1.5, color="black"))
ax.add_patch(FancyArrowPatch((0.615, 0.50), (0.68, 0.50), arrowstyle="->", mutation_scale=15, linewidth=1.5, color="black"))

ax.text(0.332, 0.545, "short-label transfer", fontsize=8.5, ha="center")
ax.text(0.648, 0.545, "multivariate transfer", fontsize=8.5, ha="center")

summary = (
    "Confirmed: the main-task evaluation pipeline works, and external retrieval shows partial positive transfer.\n"
    "Not confirmed yet: a learned long-context caption model is already established."
)
ax.text(0.50, 0.08, summary, fontsize=9.0, ha="center", va="center")

fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.04)
save_fig(fig, str(OUT_PATH))
