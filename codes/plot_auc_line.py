"""
plot_auc_line.py — Line plot of AUC per center, matching reference format.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

RESULTS_DIR = "results"
CENTERS = [1, 2, 3, 4]
CENTER_LABELS = ["Center 1", "Center 2", "Center 3", "Center 4"]

METHODS = {
    "none":     {"label": "Baseline (No Norm)", "color": "#1f77b4", "marker": "o", "lw": 2.0},
    "ot":       {"label": "Optimal Transport",  "color": "#d62728", "marker": "o", "lw": 2.0},
    "macenko":  {"label": "Macenko",            "color": "#2ca02c", "marker": "o", "lw": 2.0},
    "reinhard": {"label": "Reinhard",           "color": "#ff7f0e", "marker": "o", "lw": 2.0},
    "vahadane": {"label": "Vahadane",           "color": "#9467bd", "marker": "o", "lw": 2.0},
    "stainnet": {"label": "StainNet",           "color": "#8c564b", "marker": "o", "lw": 2.0},
}

def per_center_aucs(method):
    aucs = []
    for c in CENTERS:
        path = os.path.join(RESULTS_DIR, f"preds_{method}_center{c}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            aucs.append(roc_auc_score(df["label"], df["pred"]))
        else:
            aucs.append(float("nan"))
    return aucs

all_aucs = {m: per_center_aucs(m) for m in METHODS}
x = np.arange(len(CENTERS))

fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Reference line
ax.axhline(y=0.90, color="gray", linestyle="--", lw=1.3, alpha=0.7,
           label="AUC = 0.90 reference")

for method, cfg in METHODS.items():
    vals = all_aucs[method]
    ax.plot(x, vals, color=cfg["color"], lw=cfg["lw"],
            marker=cfg["marker"], markersize=8, label=cfg["label"], zorder=3)

    # Annotate each point
    for xi, v in zip(x, vals):
        if np.isnan(v):
            continue
        # Smart offset: push label up or down based on crowding
        va = "bottom"
        dy = 0.008
        # nudge left/right for first/last center to avoid clipping
        ha = "center"
        if xi == 0:
            ha = "right"
            dx = -0.07
        elif xi == len(CENTERS) - 1:
            ha = "left"
            dx = 0.07
        else:
            dx = 0

        ax.text(xi + dx, v + dy, f"{v:.3f}", ha=ha, va=va,
                fontsize=8, color=cfg["color"], fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CENTER_LABELS, fontsize=11)
ax.set_ylabel("AUC", fontsize=12)
ax.set_xlabel("Test Center (Hospital Scanner)", fontsize=12)
ax.set_title("Cross-Scanner Generalization: AUC by Center and Normalization Method",
             fontsize=12, fontweight="bold")

ax.set_ylim([0.45, 1.02])
ax.set_xlim([-0.3, 3.3])
ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="gray")
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(loc="lower left", fontsize=9.5, framealpha=0.9)

plt.tight_layout()
plt.savefig("auc_per_center_line.png", dpi=300, bbox_inches="tight", facecolor="white")
print("Saved → auc_per_center_line.png")
