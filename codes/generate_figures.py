"""
generate_figures.py
Generates:
  1. comparison_final.png  — side-by-side patch visualization of all normalizers
  2. auc_per_center_final.png — per-center AUC bar chart (refreshed)
  3. roc_curves_final.png    — ROC curves (refreshed)
"""

import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import roc_curve, auc, roc_auc_score

from ot_norm import OTSinkhornNormalizer
from baselines import get_baseline_normalizer

# ─── Paths ────────────────────────────────────────────────────────────────────
# Source: Center 4 tumor patch (most challenging scanner)
SOURCE = "patches/patient_080_node_1/patch_patient_080_node_1_x_1024_y_11456.png"
# Target: Center 0 reference
TARGET = "patches/patient_004_node_4/patch_patient_004_node_4_x_3168_y_22176.png"
RESULTS_DIR = "results"
CENTERS = [1, 2, 3, 4]

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Stain Normalizer Comparison
# ═════════════════════════════════════════════════════════════════════════════
print("=== Generating comparison_final.png ===")
source_img = Image.open(SOURCE).convert('RGB')
target_img = Image.open(TARGET).convert('RGB')

results = {}

print("  OT...")
ot = OTSinkhornNormalizer(subsample_size=300)
ot.fit(target_img)
results['Optimal\nTransport'] = ot.transform(source_img)

for name, key in [('Reinhard','reinhard'), ('Macenko','macenko'),
                  ('Vahadane','vahadane'), ('StainNet','stainnet')]:
    print(f"  {name}...")
    try:
        n = get_baseline_normalizer(key)
        n.fit(target_img)
        out = n.transform(source_img)
        # Ensure correct dtype & range for imshow
        if isinstance(out, np.ndarray):
            if out.dtype != np.uint8:
                out = np.clip(out, 0, 255).astype(np.uint8)
            out = Image.fromarray(out)
        results[name] = out
    except Exception as e:
        print(f"    Failed: {e}")

images = [
    ('Source\n(Center 4 — hardest scanner)', source_img),
    ('Target / Reference\n(Center 0 — training scanner)', target_img),
    ('Optimal\nTransport',  results.get('Optimal\nTransport', source_img)),
    ('Reinhard',            results.get('Reinhard',  source_img)),
    ('Macenko',             results.get('Macenko',   source_img)),
    ('Vahadane',            results.get('Vahadane',  source_img)),
    ('StainNet',            results.get('StainNet',  source_img)),
]

fig, axes = plt.subplots(1, 7, figsize=(22, 4))
fig.patch.set_facecolor('white')

for ax, (title, img) in zip(axes, images):
    ax.imshow(img)
    ax.set_title(title, fontsize=10, fontweight='bold' if 'Source' in title or 'Target' in title else 'normal')
    ax.axis('off')
    # Highlight Source and Target with a colored border
    if 'Source' in title:
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('#d62728'); spine.set_linewidth(3)
    elif 'Target' in title:
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color('#2ca02c'); spine.set_linewidth(3)

plt.suptitle('Stain Normalization Visual Comparison — Center 4 Source → Center 0 Reference',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('comparison_final.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved → comparison_final.png")

# ═════════════════════════════════════════════════════════════════════════════
# Shared data for Figures 2 & 3
# ═════════════════════════════════════════════════════════════════════════════
METHODS = {
    "none":     {"label": "No Normalization", "color": "#d62728", "lw": 2.5, "ls": "--"},
    "ot":       {"label": "Optimal Transport", "color": "#1f77b4", "lw": 2,   "ls": "-"},
    "macenko":  {"label": "Macenko",           "color": "#2ca02c", "lw": 2,   "ls": "-"},
    "reinhard": {"label": "Reinhard",          "color": "#9467bd", "lw": 2,   "ls": "-"},
    "vahadane": {"label": "Vahadane",          "color": "#ff7f0e", "lw": 2,   "ls": "-"},
    "stainnet": {"label": "StainNet",          "color": "#8c564b", "lw": 2,   "ls": "-"},
}

def load_merged(method):
    dfs = [pd.read_csv(os.path.join(RESULTS_DIR, f"preds_{method}_center{c}.csv"))
           for c in CENTERS
           if os.path.exists(os.path.join(RESULTS_DIR, f"preds_{method}_center{c}.csv"))]
    if not dfs: return None, None
    m = pd.concat(dfs, ignore_index=True)
    return m["label"].values, m["pred"].values

def center_aucs(method):
    return [roc_auc_score(pd.read_csv(os.path.join(RESULTS_DIR, f"preds_{method}_center{c}.csv"))["label"],
                          pd.read_csv(os.path.join(RESULTS_DIR, f"preds_{method}_center{c}.csv"))["pred"])
            if os.path.exists(os.path.join(RESULTS_DIR, f"preds_{method}_center{c}.csv"))
            else float("nan") for c in CENTERS]

all_aucs = {m: center_aucs(m) for m in METHODS}

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Per-Center AUC Bar Chart
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== Generating auc_per_center_final.png ===")
fig2, ax2 = plt.subplots(figsize=(12, 6))
fig2.patch.set_facecolor('white')

n = len(METHODS)
x = np.arange(len(CENTERS))
width = 0.13
offsets = np.linspace(-(n-1)/2*width, (n-1)/2*width, n)

for i, (method, cfg) in enumerate(METHODS.items()):
    vals = all_aucs[method]
    bars = ax2.bar(x + offsets[i], vals, width, label=cfg["label"],
                   color=cfg["color"], alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, v in zip(bars, vals):
        if not np.isnan(v):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.007,
                     f"{v:.2f}", ha='center', va='bottom',
                     fontsize=6.5, color='black', fontweight='bold')

ax2.set_xlabel('Test Center', fontsize=13)
ax2.set_ylabel('AUC', fontsize=13)
ax2.set_title('AUC per Test Center by Stain Normalization Method\n'
              '(CAMELYON17 Cross-Scanner Evaluation, ResNet18 Trained from Scratch)', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels([f'Center {c}' for c in CENTERS], fontsize=11)
ax2.set_ylim([0.4, 1.08])
ax2.axhline(y=1.0, color='gray', ls='--', lw=0.7, alpha=0.4)
ax2.legend(loc='lower left', fontsize=10, framealpha=0.8)
ax2.grid(True, alpha=0.3, axis='y')
fig2.tight_layout()
fig2.savefig('auc_per_center_final.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved → auc_per_center_final.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — ROC Curves (all centers combined)
# ═════════════════════════════════════════════════════════════════════════════
print("\n=== Generating roc_curves_final.png ===")
fig3, ax3 = plt.subplots(figsize=(8, 7))
fig3.patch.set_facecolor('white')
ax3.plot([0,1],[0,1], color='gray', lw=1.2, ls='--', label='Random Classifier (AUC = 0.50)')

for method, cfg in METHODS.items():
    labels, preds = load_merged(method)
    if labels is None: continue
    fpr, tpr, _ = roc_curve(labels, preds)
    ax3.plot(fpr, tpr, color=cfg["color"], lw=cfg["lw"], ls=cfg["ls"],
             label=f'{cfg["label"]} (AUC = {auc(fpr,tpr):.4f})')

ax3.set_xlim([0,1]); ax3.set_ylim([0,1.05])
ax3.set_xlabel('False Positive Rate', fontsize=13)
ax3.set_ylabel('True Positive Rate', fontsize=13)
ax3.set_title('ROC Curves by Stain Normalization Method\n'
              '(CAMELYON17 Cross-Scanner, Centers 1–4 Combined)', fontsize=13)
ax3.legend(loc='lower right', fontsize=10.5)
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig('roc_curves_final.png', dpi=300, bbox_inches='tight', facecolor='white')
print("  Saved → roc_curves_final.png")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'Method':<12}", end="")
for c in CENTERS: print(f"  Center{c}", end="")
print("    Mean")
print("-"*56)
for method, cfg in METHODS.items():
    aucs = all_aucs[method]
    print(f"{method:<12}", end="")
    for a in aucs: print(f"   {a:.3f} " if not np.isnan(a) else "     N/A ", end="")
    print(f"   {np.nanmean(aucs):.3f}")
