import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

df = pd.read_parquet('data/heatmap_grid_results.parquet')

# ── Color by threshold ────────────────────────────────────────────────────────
thresholds = df['threshold'].unique()
thresh_norm = (df['threshold'] - df['threshold'].min()) / (df['threshold'].max() - df['threshold'].min())
colors = cm.plasma(thresh_norm)

fig, ax = plt.subplots(figsize=(10, 7))

sc = ax.scatter(
    df['FPR'], df['AD_recall'],
    c=df['threshold'], cmap='plasma',
    s=60, alpha=0.75, edgecolors='white', linewidths=0.4
)

# Reference diagonal — points above it have recall > FPR
ax.plot([0, 1], [0, 1], color='grey', linestyle='--', linewidth=1, label='Recall = FPR (no benefit)')

# Correlation annotation
corr = df['FPR'].corr(df['AD_recall'])
ax.text(0.97, 0.04, f'r = {corr:.3f}', transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='grey', alpha=0.8))

plt.colorbar(sc, ax=ax, label='Decision Threshold')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('AD Recall', fontsize=12)
ax.set_title(
    'AD Recall vs. False Positive Rate — 15x15 Grid Search\n'
    'scale_pos_weight=12  |  Each point = one (learning rate, threshold) combination\n'
    'Color = decision threshold (darker = lower)',
    fontsize=11
)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(-0.02, 1.02)
ax.legend(fontsize=9)
ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
plt.savefig('fpr_recall_scatter.pdf', format='pdf', bbox_inches='tight')
print("Saved -> fpr_recall_scatter.pdf")
