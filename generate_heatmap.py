import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet('data/all_fall_surveys_combined.parquet')

X_data = df[df.columns[df.columns.str.contains('DEM|HLT|FAL|RSK|ADM|HOU|PUF_ID_NOY')]]
X_data = X_data.drop(columns={'HLT_ALZDEM', 'HLT_DISDECSN'})

y_data = df['HLT_ALZDEM'].copy()
y_data[y_data > 1] = 0

gss = GroupShuffleSplit(test_size=0.2, random_state=4)
train_idx, test_idx = next(gss.split(df, groups=df['PUF_ID_NOY']))

X_train = X_data.iloc[train_idx]
X_test  = X_data.iloc[test_idx]
y_train = y_data.iloc[train_idx]
y_test  = y_data.iloc[test_idx]

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"scale_pos_weight (train split): {scale_pos_weight:.2f}")

# ── Grid ──────────────────────────────────────────────────────────────────────
lr_values     = np.logspace(np.log10(0.01), np.log10(0.30), 15)
thresh_values = np.linspace(0.05, 0.70, 15)

rows = []
for i, lr_val in enumerate(lr_values):
    print(f"  Training model {i+1}/15  (lr={lr_val:.4f})…")
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        max_depth=4,
        learning_rate=float(lr_val),
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    for t in thresh_values:
        preds = (probs >= t).astype(int)
        tn_g, fp_g, fn_g, tp_g = confusion_matrix(y_test, preds).ravel()
        recall_ad = tp_g / (tp_g + fn_g) if (tp_g + fn_g) > 0 else 0.0
        fpr       = fp_g / (fp_g + tn_g) if (fp_g + tn_g) > 0 else 0.0
        rows.append({
            'learning_rate': round(float(lr_val), 4),
            'threshold':     round(float(t), 3),
            'AD_recall':     round(recall_ad, 4),
            'FPR':           round(fpr, 4),
        })

results_df = pd.DataFrame(rows)
results_df.to_parquet('data/heatmap_grid_results.parquet', index=False)
print("Saved -> data/heatmap_grid_results.parquet")

# ── Plot ──────────────────────────────────────────────────────────────────────
def draw_heatmap(ax, pivot, cmap, title, colorbar_label, vmin=0, vmax=1):
    im = ax.imshow(pivot.values, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.4f}" for v in pivot.index], fontsize=8)
    ax.set_xlabel('Decision Threshold', fontsize=10)
    ax.set_ylabel('Learning Rate (log-spaced)', fontsize=10)
    ax.set_title(title, fontsize=11, pad=8)
    for row_i in range(pivot.shape[0]):
        for col_j in range(pivot.shape[1]):
            val = pivot.values[row_i, col_j]
            color = 'black' if 0.25 < val < 0.80 else 'white'
            ax.text(col_j, row_i, f'{val:.2f}', ha='center', va='center',
                    fontsize=6.5, color=color)
    plt.colorbar(im, ax=ax, label=colorbar_label, fraction=0.046, pad=0.04)

recall_pivot = results_df.pivot(index='learning_rate', columns='threshold', values='AD_recall')
fpr_pivot    = results_df.pivot(index='learning_rate', columns='threshold', values='FPR')

diff_pivot = recall_pivot - fpr_pivot

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(32, 8))
fig.suptitle(
    f'Grid Search — scale_pos_weight=12, max_depth=4, min_child_weight=10',
    fontsize=13, y=1.01
)

draw_heatmap(ax1, recall_pivot, 'RdYlGn',
             'AD Recall (Higher is Better)', 'AD Recall')
draw_heatmap(ax2, fpr_pivot,    'RdYlGn_r',
             'False Positive Rate (Lower is Better)', 'FPR')
draw_heatmap(ax3, diff_pivot,   'RdYlGn',
             'Recall - FPR (Higher is Better)', 'Recall - FPR', vmin=-1, vmax=1)

plt.tight_layout()
plt.savefig('heatmap_grid_results.pdf', format='pdf', bbox_inches='tight')
print("Saved -> heatmap_grid_results.pdf")
