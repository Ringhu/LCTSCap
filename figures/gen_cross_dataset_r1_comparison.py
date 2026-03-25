from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, save_fig

DATA_PATH = Path(__file__).resolve().parent / "cross_dataset_r1_data.json"
OUT_PATH = Path(__file__).resolve().parent / "cross_dataset_r1_comparison.png"

with open(DATA_PATH) as f:
    raw = json.load(f)

sections = [
    ("CAPTURE24", raw["capture24_main"]),
    ("UCR", raw["ucr"]),
    ("UEA", raw["uea"]),
    ("UEA-long", raw["uea_long"]),
]

labels = []
s2t = []
t2s = []
group_centers = []
y_positions = []
current = 0
for group_name, items in sections:
    start = current
    for item in items:
        labels.append(item["dataset"])
        s2t.append(item["s2t_r1"])
        t2s.append(item["t2s_r1"])
        y_positions.append(current)
        current += 1
    end = current - 1
    group_centers.append((group_name, (start + end) / 2))
    current += 0.9

y = np.array(y_positions, dtype=float)
bar_h = 0.34

fig, ax = plt.subplots(figsize=(10.8, 6.6))
ax.barh(y - bar_h / 2, s2t, height=bar_h, label="s2t_R@1", color=COLORS[0])
ax.barh(y + bar_h / 2, t2s, height=bar_h, label="t2s_R@1", color=COLORS[1])

ax.set_xlim(0, 1.02)
ax.set_xlabel("Top-1 retrieval")
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.invert_yaxis()
ax.legend(frameon=False, loc="lower right")
ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

for group_name, center in group_centers:
    ax.text(1.01, center, group_name, va="center", ha="left", fontsize=9, weight="bold")

for idx, item in enumerate(raw["capture24_main"] + raw["ucr"] + raw["uea"] + raw["uea_long"]):
    if item["dataset"] == "CAPTURE24-template":
        ax.text(0.02, y[idx], "GT template", va="center", ha="left", fontsize=8)
    if item["dataset"] == "AtrialFib":
        ax.text(0.78, y[idx], "n=15", va="center", ha="left", fontsize=8)

fig.subplots_adjust(left=0.24, right=0.86, top=0.98, bottom=0.10)
save_fig(fig, str(OUT_PATH))
