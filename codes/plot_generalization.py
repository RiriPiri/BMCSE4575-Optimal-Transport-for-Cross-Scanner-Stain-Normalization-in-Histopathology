import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import roc_auc_score

METHODS = {
    'none':     'Baseline (No Norm)',
    'ot':       'Optimal Transport',
    'macenko':  'Macenko',
    'reinhard': 'Reinhard',
    'vahadane': 'Vahadane',
    'stainnet': 'StainNet',
}

COLORS = {
    'none':     '#1f77b4',
    'ot':       '#d62728',
    'macenko':  '#2ca02c',
    'reinhard': '#ff7f0e',
    'vahadane': '#9467bd',
    'stainnet': '#8c564b',
}

CENTERS = [1, 2, 3, 4]

def load_auc(method, center):
    path = f'results/preds_{method}_center{center}.csv'
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if df['label'].nunique() < 2:
        return None
    return roc_auc_score(df['label'], df['pred'])

def plot_generalization():
    fig, ax = plt.subplots(figsize=(12, 7))

    found_any = False
    for method, label in METHODS.items():
        aucs = []
        centers_with_data = []
        for c in CENTERS:
            auc = load_auc(method, c)
            if auc is not None:
                aucs.append(auc)
                centers_with_data.append(c)

        if len(aucs) >= 1:
            found_any = True
            ax.plot(centers_with_data, aucs,
                    marker='o', linewidth=2, markersize=8,
                    color=COLORS[method], label=f'{label}')
            for cx, ay in zip(centers_with_data, aucs):
                ax.annotate(f'{ay:.3f}', (cx, ay),
                            textcoords='offset points', xytext=(5, 5),
                            fontsize=8, color=COLORS[method])

    if not found_any:
        print("No per-center CSVs found. Run evaluate.py with --test_center 1/2/3/4 first.")
        return

    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.4, label='AUC = 0.90 reference')
    ax.set_xlabel('Test Center (Hospital Scanner)', fontsize=12)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Cross-Scanner Generalization: AUC by Center and Normalization Method', fontsize=13)
    ax.set_xticks(CENTERS)
    ax.set_xticklabels([f'Center {c}' for c in CENTERS])
    ax.set_ylim([0.5, 1.02])
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('generalization_plot.png', dpi=300)
    print("Saved generalization plot to generalization_plot.png")

if __name__ == '__main__':
    plot_generalization()
