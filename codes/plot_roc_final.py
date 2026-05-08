"""
plot_roc_final.py — ROC curves + per-center AUC bar chart for all 6 normalization methods.
All methods use actual prediction CSVs combined across Centers 1-4.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

RESULTS_DIR = "results"
CENTERS = [1, 2, 3, 4]

METHODS = {
    "none":     {"label": "No Normalization", "color": "#d62728", "lw": 2.5, "ls": "--"},
    "ot":       {"label": "Optimal Transport", "color": "#1f77b4", "lw": 2,   "ls": "-"},
    "macenko":  {"label": "Macenko",           "color": "#2ca02c", "lw": 2,   "ls": "-"},
    "reinhard": {"label": "Reinhard",          "color": "#9467bd", "lw": 2,   "ls": "-"},
    "vahadane": {"label": "Vahadane",          "color": "#ff7f0e", "lw": 2,   "ls": "-"},
    "stainnet": {"label": "StainNet",          "color": "#8c564b", "lw": 2,   "ls": "-"},
}

def load_merged(method):
    """Load and concatenate predictions across all centers."""
    dfs = []
    for c in CENTERS:
        path = os.path.join(RESULTS_DIR, f"preds_{method}_center{c}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if not dfs:
        return None, None
    merged = pd.concat(dfs, ignore_index=True)
    return merged["label"].values, merged["pred"].values

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

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Method':<12}", end="")
for c in CENTERS:
    print(f"  Center{c}", end="")
print("    Mean")
print("-" * 56)

all_aucs = {}
for method, cfg in METHODS.items():
    aucs = per_center_aucs(method)
    all_aucs[method] = aucs
    mean = np.nanmean(aucs)
    print(f"{method:<12}", end="")
    for a in aucs:
        print(f"   {a:.3f} " if not np.isnan(a) else "     N/A ", end="")
    print(f"   {mean:.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — ROC Curves (all centers combined)
# ─────────────────────────────────────────────────────────────────────────────
fig1, ax = plt.subplots(figsize=(8, 7))
fig1.patch.set_facecolor('white')

ax.plot([0, 1], [0, 1], color='gray', lw=1.2, linestyle='--',
        label='Random Classifier (AUC = 0.50)')

for method, cfg in METHODS.items():
    labels, preds = load_merged(method)
    if labels is None:
        continue
    fpr, tpr, _ = roc_curve(labels, preds)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr,
            color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
            label=f'{cfg["label"]} (AUC = {roc_auc:.4f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curves by Stain Normalization Method\n'
             '(CAMELYON17 Cross-Scanner, Centers 1–4 Combined)', fontsize=13)
ax.legend(loc='lower right', fontsize=10.5)
ax.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('roc_curves_final.png', dpi=300, bbox_inches='tight', facecolor='white')
print("\nSaved → roc_curves_final.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Per-Center AUC Bar Chart
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(11, 6))
fig2.patch.set_facecolor('white')

method_names = list(METHODS.keys())
n = len(method_names)
x = np.arange(len(CENTERS))
width = 0.13
offsets = np.linspace(-(n - 1) / 2 * width, (n - 1) / 2 * width, n)

for i, method in enumerate(method_names):
    cfg = METHODS[method]
    vals = all_aucs[method]
    bars = ax2.bar(x + offsets[i], vals, width,
                   label=cfg["label"], color=cfg["color"], alpha=0.85,
                   edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.007,
                     f"{v:.2f}", ha='center', va='bottom',
                     fontsize=7, color='black', fontweight='bold')

ax2.set_xlabel('Test Center', fontsize=13)
ax2.set_ylabel('AUC', fontsize=13)
ax2.set_title('AUC per Test Center by Stain Normalization Method\n'
              '(CAMELYON17 Cross-Scanner Evaluation)', fontsize=13)
ax2.set_xticks(x)
ax2.set_xticklabels([f'Center {c}' for c in CENTERS], fontsize=11)
ax2.set_ylim([0.4, 1.08])
ax2.axhline(y=1.0, color='gray', ls='--', lw=0.7, alpha=0.4)
ax2.legend(loc='lower left', fontsize=10, framealpha=0.8)
ax2.grid(True, alpha=0.3, axis='y')
fig2.tight_layout()
fig2.savefig('auc_per_center_final.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Saved → auc_per_center_final.png")
