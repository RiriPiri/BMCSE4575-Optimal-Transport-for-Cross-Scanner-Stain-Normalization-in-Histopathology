"""
generate_diagrams.py
Generates:
  1. pipeline_flowchart.png  — end-to-end data pipeline diagram
  2. ot_architecture.png     — OT normalization algorithm diagram
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, label, color='#dce8f7', fontsize=10,
        style='round,pad=0.1', textcolor='#1a1a2e', bold=False):
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                           boxstyle=style, linewidth=1.2,
                           edgecolor='#444', facecolor=color, zorder=3)
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize,
            fontweight=weight, color=textcolor, zorder=4, wrap=True,
            multialignment='center')

def arrow(ax, x1, y1, x2, y2, color='#555'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5),
                zorder=2)

def label_arrow(ax, x, y, text, fontsize=8, color='#666'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=color, style='italic', zorder=5)

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Data Pipeline Flowchart
# ═════════════════════════════════════════════════════════════════════════════
fig1, ax = plt.subplots(figsize=(14, 9))
ax.set_xlim(0, 14); ax.set_ylim(0, 9)
ax.axis('off')
fig1.patch.set_facecolor('white')
ax.set_facecolor('white')

# ── Row 1: Data source ────────────────────────────────────────────────────────
box(ax, 7, 8.3, 5, 0.7, 'CAMELYON17 Dataset\n(H&E Patch Images + metadata.csv)',
    color='#2c3e50', textcolor='white', fontsize=10, bold=True)

arrow(ax, 7, 7.95, 7, 7.55)

# ── Row 2: Split ──────────────────────────────────────────────────────────────
box(ax, 3.5, 7.2, 4.5, 0.6, 'Center 0\nTrain (split=0) / Val (split=1)',
    color='#2980b9', textcolor='white', fontsize=9)
box(ax, 10.5, 7.2, 4.5, 0.6, 'Centers 1–4\nTest (held-out)',
    color='#8e44ad', textcolor='white', fontsize=9)

# branch arrows
ax.annotate('', xy=(3.5, 7.5), xytext=(5.5, 7.95),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.4))
ax.annotate('', xy=(10.5, 7.5), xytext=(8.5, 7.95),
            arrowprops=dict(arrowstyle='->', color='#555', lw=1.4))

# ── Row 3: Normalization transform ───────────────────────────────────────────
box(ax, 3.5, 6.35, 4.5, 0.65,
    'NormalizationTransform\n(none / OT / Macenko / Reinhard / Vahadane / StainNet)',
    color='#27ae60', textcolor='white', fontsize=8.5)
box(ax, 10.5, 6.35, 4.5, 0.65,
    'NormalizationTransform\n(same method as training)',
    color='#27ae60', textcolor='white', fontsize=8.5)

arrow(ax, 3.5, 6.9, 3.5, 6.68)
arrow(ax, 10.5, 6.9, 10.5, 6.68)

# ── Row 4: Augmentation / preprocessing ──────────────────────────────────────
box(ax, 3.5, 5.5, 4.5, 0.65,
    'Augmentation\n(RandomHFlip · RandomVFlip · ToTensor · Normalize)',
    color='#f39c12', textcolor='white', fontsize=8.5)
box(ax, 10.5, 5.5, 4.5, 0.65,
    'Preprocessing\n(ToTensor · Normalize)',
    color='#f39c12', textcolor='white', fontsize=8.5)

arrow(ax, 3.5, 6.02, 3.5, 5.83)
arrow(ax, 10.5, 6.02, 10.5, 5.83)

# ── Row 5: DataLoader ─────────────────────────────────────────────────────────
box(ax, 3.5, 4.65, 4.5, 0.65,
    'DataLoader\n(batch=64, shuffle=True)',
    color='#dce8f7', fontsize=9)
box(ax, 10.5, 4.65, 4.5, 0.65,
    'DataLoader\n(batch=64, shuffle=False)',
    color='#dce8f7', fontsize=9)

arrow(ax, 3.5, 5.17, 3.5, 4.98)
arrow(ax, 10.5, 5.17, 10.5, 4.98)

# ── Row 6: Model ─────────────────────────────────────────────────────────────
box(ax, 3.5, 3.75, 4.5, 0.70,
    'ResNet18 (random init)\nAdam · lr=1e-3 · CosineAnnealingLR · 10 epochs',
    color='#c0392b', textcolor='white', fontsize=8.5, bold=True)

arrow(ax, 3.5, 4.32, 3.5, 4.1)

# checkpoint arrow to eval side
ax.annotate('', xy=(8.2, 3.75), xytext=(5.75, 3.75),
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5,
                            linestyle='dashed'))
label_arrow(ax, 7.0, 3.93, 'best_model_{method}.pth', fontsize=8, color='#c0392b')

# ── Row 6 right: Inference ────────────────────────────────────────────────────
box(ax, 10.5, 3.75, 4.5, 0.70,
    'ResNet18 Inference\n(load best_model_{method}.pth)',
    color='#c0392b', textcolor='white', fontsize=8.5)

arrow(ax, 10.5, 4.32, 10.5, 4.1)

# ── Row 7: Loss / Output ──────────────────────────────────────────────────────
box(ax, 3.5, 2.85, 4.5, 0.65,
    'BCEWithLogitsLoss\nVal AUC monitored → save best checkpoint',
    color='#ecf0f1', fontsize=8.5)
box(ax, 10.5, 2.85, 4.5, 0.65,
    'Sigmoid probabilities\nSaved → results/preds_{method}_center{c}.csv',
    color='#ecf0f1', fontsize=8.5)

arrow(ax, 3.5, 3.4, 3.5, 3.18)
arrow(ax, 10.5, 3.4, 10.5, 3.18)

# ── Row 8: Metrics ────────────────────────────────────────────────────────────
box(ax, 10.5, 2.0, 4.5, 0.65,
    'AUC (sklearn) · ROC Curve\nPer-center + Combined',
    color='#2c3e50', textcolor='white', fontsize=9, bold=True)

arrow(ax, 10.5, 2.52, 10.5, 2.33)

# Labels
ax.text(3.5, 8.72, 'TRAINING', ha='center', fontsize=9,
        color='#2980b9', fontweight='bold')
ax.text(10.5, 8.72, 'EVALUATION', ha='center', fontsize=9,
        color='#8e44ad', fontweight='bold')
ax.text(0.3, 5.5, 'Applied per\nmethod', ha='center', fontsize=7.5,
        color='#27ae60', style='italic')

ax.set_title('End-to-End Data Pipeline — Stain Normalization Benchmark',
             fontsize=13, fontweight='bold', pad=8)
fig1.tight_layout()
fig1.savefig('pipeline_flowchart.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved → pipeline_flowchart.png")

# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — OT Architecture Diagram
# ═════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.set_xlim(0, 14); ax2.set_ylim(-0.3, 7.3)
ax2.axis('off')
fig2.patch.set_facecolor('white')
ax2.set_facecolor('white')

C = {
    'src':    '#2980b9',
    'tgt':    '#27ae60',
    'proc':   '#8e44ad',
    'ot':     '#c0392b',
    'out':    '#2c3e50',
    'light':  '#ecf0f1',
    'note':   '#f39c12',
}

def box2(x, y, w, h, label, color, fontsize=9, textcolor='white', bold=False):
    box(ax2, x, y, w, h, label, color=color, fontsize=fontsize,
        textcolor=textcolor, bold=bold)

def arr2(x1, y1, x2, y2, color='#444'):
    arrow(ax2, x1, y1, x2, y2, color=color)

# ── Source branch (top) ───────────────────────────────────────────────────────
box2(1.8, 6.2, 2.8, 0.65, 'Source Image\n(Center X patch, RGB)', C['src'])
box2(1.8, 5.2, 2.8, 0.65, 'Convert to\nOptical Density (OD)\nOD = –log(I/255 + ε)', C['proc'])
box2(1.8, 4.1, 2.8, 0.65, 'Luminosity Mask\n(background removal\nthreshold = 0.8)', C['proc'])
box2(1.8, 3.05, 2.8, 0.65, 'Subsample Tissue Pixels\n(N = 300 random pixels)', C['proc'])

arr2(1.8, 5.87, 1.8, 5.53)
arr2(1.8, 4.87, 1.8, 4.43)
arr2(1.8, 3.77, 1.8, 3.38)

# ── Target branch (top) ───────────────────────────────────────────────────────
box2(7.0, 6.2, 2.8, 0.65, 'Target / Reference Image\n(Center 0 patch, RGB)', C['tgt'])
box2(7.0, 5.2, 2.8, 0.65, 'Convert to\nOptical Density (OD)\nOD = –log(I/255 + ε)', C['proc'])
box2(7.0, 4.1, 2.8, 0.65, 'Luminosity Mask\n(background removal\nthreshold = 0.8)', C['proc'])
box2(7.0, 3.05, 2.8, 0.65, 'Subsample Tissue Pixels\n(N = 300 random pixels)', C['proc'])

arr2(7.0, 5.87, 7.0, 5.53)
arr2(7.0, 4.87, 7.0, 4.43)
arr2(7.0, 3.77, 7.0, 3.38)

# Labels
ax2.text(1.8, 6.72, 'SOURCE', ha='center', fontsize=9,
         fontweight='bold', color=C['src'])
ax2.text(7.0, 6.72, 'TARGET (Reference)', ha='center', fontsize=9,
         fontweight='bold', color=C['tgt'])

# ── OT Transport ──────────────────────────────────────────────────────────────
box2(4.4, 2.1, 4.2, 0.80,
     'EMD Transport Plan\not.da.EMDTransport()\nFit on subsampled OD distributions\n(Xs_sub → Xt_sub)', C['ot'], bold=True)

# arrows from both subsample boxes to OT
ax2.annotate('', xy=(3.2, 2.1), xytext=(2.7, 2.73),
             arrowprops=dict(arrowstyle='->', color=C['ot'], lw=1.6))
ax2.annotate('', xy=(5.6, 2.1), xytext=(6.3, 2.73),
             arrowprops=dict(arrowstyle='->', color=C['ot'], lw=1.6))

# ── Apply to full tissue ──────────────────────────────────────────────────────
box2(1.8, 1.1, 2.8, 0.65,
     'Apply Transport Map\nto ALL source tissue pixels\n(transform_OT_labels_fn)', C['proc'])
box2(5.5, 1.1, 2.0, 0.65,
     'Background\npixels unchanged', C['note'], textcolor='#333', fontsize=8.5)

arr2(4.4, 1.7, 2.8, 1.43)
ax2.annotate('', xy=(4.5, 1.1), xytext=(3.5, 1.7),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                             linestyle='dashed'))

# ── Reconstruct image ─────────────────────────────────────────────────────────
box2(1.8, 0.22, 5.5, 0.58,
     'Reconstruct Normalized Image\n(OD → RGB: I = 255 × exp(–OD_normalized))', C['out'], bold=True)

arr2(1.8, 0.77, 1.8, 0.51)
ax2.annotate('', xy=(3.3, 0.22), xytext=(5.0, 1.0),
             arrowprops=dict(arrowstyle='->', color='#888', lw=1.2,
                             linestyle='dashed'))

# ── Note box ──────────────────────────────────────────────────────────────────
note = ('Key design choices:\n'
        '• OD space linearizes Beer-Lambert absorption law\n'
        '• Tissue masking avoids transporting background pixels\n'
        '• Subsampling (N=300) makes EMD tractable (~2 min/epoch)\n'
        '• EMD (exact) preferred over Sinkhorn for accuracy')
ax2.text(9.5, 3.5, note, ha='left', va='center', fontsize=8.5,
         color='#333', style='normal',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#fef9e7',
                   edgecolor='#f39c12', lw=1.2))

ax2.set_title('Optimal Transport Stain Normalization — Algorithm Architecture',
              fontsize=13, fontweight='bold', pad=8)
fig2.tight_layout()
fig2.savefig('ot_architecture.png', dpi=200, bbox_inches='tight', facecolor='white')
print("Saved → ot_architecture.png")
