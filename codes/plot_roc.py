import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc_curves():
    csv_files = glob.glob('results/preds_*.csv')
    if not csv_files:
        print("No prediction CSVs found in results/")
        return

    plt.figure(figsize=(10, 8))
    
    # Sort files so Baseline is first, OT is prominent
    csv_files.sort()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for i, file_path in enumerate(csv_files):
        method_name = os.path.basename(file_path).replace('preds_', '').replace('.csv', '')
        
        # Make names nicer
        display_name = method_name.capitalize()
        if display_name == 'Ot': display_name = 'Optimal Transport'
        
        df = pd.read_csv(file_path)
        fpr, tpr, _ = roc_curve(df['label'], df['pred'])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, color=colors[i % len(colors)], label=f'{display_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) by Normalization Method')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    output_path = 'roc_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curves plot to {output_path}")

if __name__ == '__main__':
    plot_roc_curves()
