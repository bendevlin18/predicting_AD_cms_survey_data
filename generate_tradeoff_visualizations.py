import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from sklearn.model_selection import GroupShuffleSplit
from scipy.interpolate import griddata

# ── Data setup ────────────────────────────────────────────────────────────────
df = pd.read_parquet('data/all_fall_surveys_combined.parquet')
y_data = df['HLT_ALZDEM'].copy()
y_data[y_data > 1] = 0

gss = GroupShuffleSplit(test_size=0.2, random_state=4)
_, test_idx = next(gss.split(df, groups=df['PUF_ID_NOY']))
y_test = y_data.iloc[test_idx]

N         = len(y_test)
total_pos = int((y_test == 1).sum())
total_neg = int((y_test == 0).sum())

grid_df = pd.read_parquet('data/heatmap_grid_results.parquet')

# Derive counts and derived metrics from rates
grid_df['TP']  = (grid_df['AD_recall'] * total_pos).round().astype(int)
grid_df['FP']  = (grid_df['FPR'] * total_neg).round().astype(int)
denom          = (grid_df['TP'] + grid_df['FP']).clip(lower=1)
grid_df['PPV'] = grid_df['TP'] / denom
grid_df['NNS'] = 1 / grid_df['PPV'].replace(0, np.nan)
grid_df['diff'] = grid_df['AD_recall'] - grid_df['FPR']
t = grid_df['threshold']
grid_df['net_benefit'] = (grid_df['TP'] / N) - (grid_df['FP'] / N) * (t / (1 - t))

lr_values   = sorted(grid_df['learning_rate'].unique())
lr_norm     = (np.array(lr_values) - min(lr_values)) / (max(lr_values) - min(lr_values))
lr_colors   = cm.viridis(lr_norm)

DESCRIPTIONS = {
    1: (
        "ROC Curves by Learning Rate\n\n"
        "Each curve traces one learning rate through (FPR, Recall) space as the decision threshold varies.\n"
        "Overlaying all 15 learning rates answers: does the learning rate meaningfully change the available\n"
        "tradeoffs, or are we mostly sliding along a similar curve? Points closer to the top-left corner\n"
        "represent better combined performance. Color progresses light->dark with increasing learning rate."
    ),
    2: (
        "Decision Curve Analysis (Net Benefit)\n\n"
        "Net Benefit = (TP/N) - (FP/N) x (threshold / (1-threshold))\n\n"
        "Compares the model against two reference strategies: 'screen everyone' and 'screen no one'.\n"
        "The region where the model curve sits above both baselines is the range of thresholds where\n"
        "using the model adds clinical value. This is the gold standard in clinical prediction modeling\n"
        "papers (Vickers et al., 2006) and directly encodes the idea that a false positive at a low\n"
        "threshold is less costly than one at a high threshold."
    ),
    3: (
        "Number Needed to Screen (NNS) vs. Recall\n\n"
        "NNS = Total Predicted Positive / True Positives = 1 / PPV\n\n"
        "Answers: 'For every true AD case the model catches, how many total patients does it flag?'\n"
        "NNS is immediately intuitive for clinicians — it translates the abstract FPR into a concrete\n"
        "workload question. The curve typically shows a sharp elbow where pushing for a bit more recall\n"
        "causes NNS to explode. Color = decision threshold (darker = lower threshold)."
    ),
    4: (
        "Contour Plot of Recall - FPR over Parameter Space\n\n"
        "A smooth, interpolated surface of the (Recall - FPR) score over learning rate x threshold space.\n"
        "Contour lines connect regions of equal net performance, creating a topographic map of the\n"
        "parameter landscape. This shows the gradient — how quickly performance degrades as you move\n"
        "away from the sweet spot. Robustness matters in clinical deployment where patient populations\n"
        "shift over time. The X marks the current operating point (lr=0.05, threshold=0.20)."
    ),
    5: (
        "Pareto Frontier on the FPR-Recall Plane\n\n"
        "Pareto-optimal points are combinations where you cannot improve recall without increasing FPR.\n"
        "Every point on the frontier is a defensible clinical choice; every point below it is dominated\n"
        "by something strictly better. This reframes the question from 'which threshold is best?' to\n"
        "'here is the menu of best available tradeoffs — which fits your clinical context?'\n"
        "Color = decision threshold. Frontier points are annotated with their threshold values."
    ),
}

def add_description(fig, text, y=0.02):
    fig.text(0.5, y, text, ha='center', va='bottom', fontsize=8.5,
             color='#333333', wrap=True,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f7f7f7', edgecolor='#cccccc'))

# ── PDF ───────────────────────────────────────────────────────────────────────
with PdfPages('tradeoff_visualizations.pdf') as pdf:

    # ── 1. ROC Curves by Learning Rate ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.30)

    for lr_val, color in zip(lr_values, lr_colors):
        sub = grid_df[grid_df['learning_rate'] == lr_val].sort_values('threshold', ascending=False)
        xs  = [0] + sub['FPR'].tolist()    + [1]
        ys  = [0] + sub['AD_recall'].tolist() + [1]
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.85)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (no skill)')
    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(lr_values), vmax=max(lr_values)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Learning Rate')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('AD Recall', fontsize=11)
    ax.set_title('1. ROC Curves by Learning Rate', fontsize=13, pad=10)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)
    add_description(fig, DESCRIPTIONS[1])
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── 2. Decision Curve Analysis ────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.30)

    thresh_range = np.linspace(0.01, 0.70, 200)
    nb_screen_all = (total_pos / N) - (total_neg / N) * (thresh_range / (1 - thresh_range))
    ax.plot(thresh_range, nb_screen_all, 'b--', linewidth=1.5, label='Screen everyone', alpha=0.7)
    ax.axhline(0, color='orange', linestyle='--', linewidth=1.5, label='Screen no one', alpha=0.7)

    # Plot one curve per LR, highlight default lr=0.05
    default_lr = min(lr_values, key=lambda x: abs(x - 0.05))
    for lr_val, color in zip(lr_values, lr_colors):
        sub = grid_df[grid_df['learning_rate'] == lr_val].sort_values('threshold')
        lw  = 2.5 if lr_val == default_lr else 1.0
        alpha = 1.0 if lr_val == default_lr else 0.4
        ax.plot(sub['threshold'], sub['net_benefit'], color=color, linewidth=lw, alpha=alpha)

    sm = plt.cm.ScalarMappable(cmap='viridis',
                               norm=plt.Normalize(vmin=min(lr_values), vmax=max(lr_values)))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Learning Rate')
    ax.annotate('lr=0.05 (default)', xy=(0.05, grid_df[grid_df['learning_rate'] == default_lr]
                .sort_values('threshold').iloc[0]['net_benefit']),
                fontsize=8, color='black',
                arrowprops=dict(arrowstyle='->', color='black'), xytext=(0.15, 0.03))
    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Net Benefit', fontsize=11)
    ax.set_title('2. Decision Curve Analysis', fontsize=13, pad=10)
    ax.set_xlim(0, 0.72); ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)
    add_description(fig, DESCRIPTIONS[2])
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── 3. NNS vs Recall ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.30)

    plot_df = grid_df.dropna(subset=['NNS']).copy()
    plot_df = plot_df[plot_df['NNS'] < 100]   # clip extreme values for readability
    sc = ax.scatter(plot_df['AD_recall'], plot_df['NNS'],
                    c=plot_df['threshold'], cmap='plasma',
                    s=50, alpha=0.75, edgecolors='white', linewidths=0.4)
    plt.colorbar(sc, ax=ax, label='Decision Threshold')

    # Annotate the elbow region (where NNS starts rising sharply)
    elbow = plot_df.sort_values('AD_recall').iloc[len(plot_df)//2]
    ax.annotate('Elbow region:\nrecall gains become\ncostly in false positives',
                xy=(elbow['AD_recall'], elbow['NNS']),
                xytext=(elbow['AD_recall'] - 0.25, elbow['NNS'] + 10),
                fontsize=8, arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_xlabel('AD Recall', fontsize=11)
    ax.set_ylabel('Number Needed to Screen (NNS)', fontsize=11)
    ax.set_title('3. Number Needed to Screen vs. Recall', fontsize=13, pad=10)
    ax.grid(True, linestyle='--', alpha=0.4)
    add_description(fig, DESCRIPTIONS[3])
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── 4. Contour Plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.30)

    xi = np.linspace(grid_df['threshold'].min(),    grid_df['threshold'].max(),    200)
    yi = np.linspace(grid_df['learning_rate'].min(), grid_df['learning_rate'].max(), 200)
    Xi, Yi = np.meshgrid(xi, yi)
    Zi = griddata(
        (grid_df['threshold'], grid_df['learning_rate']),
        grid_df['diff'],
        (Xi, Yi), method='cubic'
    )

    cf = ax.contourf(Xi, Yi, Zi, levels=20, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.contour(Xi, Yi, Zi, levels=10, colors='black', linewidths=0.4, alpha=0.4)
    plt.colorbar(cf, ax=ax, label='Recall - FPR')

    # Mark current operating point
    ax.scatter([0.20], [0.05], marker='X', s=150, color='black', zorder=5,
               label='Current operating point\n(lr=0.05, thresh=0.20)')
    ax.legend(fontsize=9, loc='upper right')
    ax.set_xlabel('Decision Threshold', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('4. Contour Plot of Recall - FPR', fontsize=13, pad=10)
    add_description(fig, DESCRIPTIONS[4])
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ── 5. Pareto Frontier ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.subplots_adjust(bottom=0.30)

    sc = ax.scatter(grid_df['FPR'], grid_df['AD_recall'],
                    c=grid_df['threshold'], cmap='plasma',
                    s=50, alpha=0.5, edgecolors='white', linewidths=0.3)
    plt.colorbar(sc, ax=ax, label='Decision Threshold')

    # Compute Pareto frontier
    sorted_df  = grid_df.sort_values('FPR').reset_index(drop=True)
    best_recall = -np.inf
    pareto_rows = []
    for _, row in sorted_df.iterrows():
        if row['AD_recall'] > best_recall:
            best_recall = row['AD_recall']
            pareto_rows.append(row)
    pareto_df = pd.DataFrame(pareto_rows).sort_values('FPR')

    ax.plot(pareto_df['FPR'], pareto_df['AD_recall'],
            'k-o', linewidth=2, markersize=6, zorder=5, label='Pareto frontier')

    # Annotate a few frontier points with threshold
    for _, row in pareto_df.iloc[::3].iterrows():
        ax.annotate(f"t={row['threshold']:.2f}",
                    xy=(row['FPR'], row['AD_recall']),
                    xytext=(row['FPR'] + 0.02, row['AD_recall'] - 0.03),
                    fontsize=7, color='black')

    ax.plot([0, 1], [0, 1], 'grey', linestyle='--', linewidth=1, label='Recall = FPR')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('AD Recall', fontsize=11)
    ax.set_title('5. Pareto Frontier on the FPR-Recall Plane', fontsize=13, pad=10)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9); ax.grid(True, linestyle='--', alpha=0.4)
    add_description(fig, DESCRIPTIONS[5])
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

print("Saved -> tradeoff_visualizations.pdf")
