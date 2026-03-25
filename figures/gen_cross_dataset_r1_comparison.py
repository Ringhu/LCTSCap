from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np

from paper_plot_style import COLORS, save_fig

DATA_PATH = Path(__file__).resolve().parent / "cross_dataset_r1_data.json"
OUT_PATH = Path(__file__).resolve().parent / "cross_dataset_r1_comparison.pdf"

with open(DATA_PATH) as f:
    raw = json.load(f)

items = raw["capture24_main"] + raw["ucr"] + raw["uea"] + raw["uea_long"]
labels = [item["dataset"] for item in items]
s2t = [item["s2t_r1"] for item in items]
t2s = [item["t2s_r1"] for item in items]
groups = [item["group"] for item in items]

x = np.arange(len(items))
width = 0.38

fig, ax = plt.subplots(figsize=(11.5, 4.4))
ax.bar(x - width / 2, s2t, width, label="s2t_R@1", color=COLORS[0])
ax.bar(x + width / 2, t2s, width, label="t2s_R@1", color=COLORS[1])

ax.set_ylabel("Top-1 retrieval")
ax.set_ylim(0, 1.08)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.legend(frameon=False, ncol=2, loc="upper right")
ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax.text(0.2, 1.02, "CAPTURE24", fontsize=9)
ax.text(1.8, 1.02, "UCR", fontsize=9)
ax.text(6.1, 1.02, "UEA", fontsize=9)
ax.text(11.2, 1.02, "UEA-long", fontsize=9)

for sep in [0.5, 4.5, 9.5]:
    ax.axvline(sep, color="black", linewidth=0.8, alpha=0.4)

for i, item in enumerate(items):
    if item["dataset"] == "CAPTURE24-template":
        ax.text(i, 0.03, "GT template", rotation=90, ha="center", va="bottom", fontsize=7)
    if item["dataset"] == "AtrialFib":
        ax.text(i, 0.97, "n=15", ha="center", va="bottom", fontsize=7)

save_fig(fig, str(OUT_PATH))
