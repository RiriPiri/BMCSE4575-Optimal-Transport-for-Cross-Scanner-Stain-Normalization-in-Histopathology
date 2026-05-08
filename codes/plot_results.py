import matplotlib.pyplot as plt
import numpy as np

methods = ['Baseline', 'Macenko', 'Reinhard', 'OT (Sinkhorn)', 'Vahadane', 'StainNet']
auc_scores = [0.9648, 0.9421, 0.9286, 0.9037, 0.8968, 0.7499]
acc_scores = [0.8863, 0.8526, 0.8135, 0.7746, 0.8200, 0.4680]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width/2, auc_scores, width, label='AUC', color='#4C72B0')
rects2 = ax.bar(x + width/2, acc_scores, width, label='Accuracy', color='#DD8452')

ax.set_ylabel('Scores')
ax.set_title('Cross-Scanner Performance by Normalization Method')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.1)

ax.bar_label(rects1, padding=3, fmt='%.3f')
ax.bar_label(rects2, padding=3, fmt='%.3f')

fig.tight_layout()
plt.savefig('auc_results.png', dpi=300)
print('Plot saved to auc_results.png')
