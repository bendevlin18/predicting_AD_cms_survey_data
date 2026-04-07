import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

SAMPLE_N   = 2000   # rows from test set used for SHAP (keeps runtime reasonable)
TOP_N      = 20     # features to show in summary plot
RANDOM_SEED = 4

# ── Data ──────────────────────────────────────────────────────────────────────
df = pd.read_parquet('data/all_fall_surveys_combined.parquet')
full_feature_names = pd.read_parquet('data/full_feature_names.parquet')

X_data = df[df.columns[df.columns.str.contains('DEM|HLT|FAL|RSK|ADM|HOU|PUF_ID_NOY')]]
X_data = X_data.drop(columns={'HLT_ALZDEM', 'HLT_DISDECSN'})

y_data = df['HLT_ALZDEM'].copy()
y_data[y_data > 1] = 0

gss = GroupShuffleSplit(test_size=0.2, random_state=RANDOM_SEED)
train_idx, test_idx = next(gss.split(df, groups=df['PUF_ID_NOY']))

X_train = X_data.iloc[train_idx]
X_test  = X_data.iloc[test_idx]
y_train = y_data.iloc[train_idx]
y_test  = y_data.iloc[test_idx]

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"scale_pos_weight: {scale_pos_weight:.2f}")

# ── Train model ───────────────────────────────────────────────────────────────
print("Training XGBoost model...")
model = XGBClassifier(
    scale_pos_weight=12,
    eval_metric='aucpr',
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
)
model.fit(X_train, y_train)
print("Done training.")

# ── SHAP values ───────────────────────────────────────────────────────────────
rng = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(len(X_test), size=min(SAMPLE_N, len(X_test)), replace=False)
X_sample = X_test.iloc[sample_idx]

print(f"Computing SHAP values on {len(X_sample)} test samples...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
print("Done computing SHAP values.")

# ── Build human-readable feature names ────────────────────────────────────────
name_map = full_feature_names['full_name'].to_dict()

def readable(col):
    if col in name_map:
        return name_map[col]
    base = col.replace('_missing', '')
    if base in name_map:
        return name_map[base] + ' (missing)'
    return col

feat_labels = [readable(c) for c in X_sample.columns]

# ── Identify top-N features by mean |SHAP| ────────────────────────────────────
mean_abs = np.abs(shap_values.values).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1][:TOP_N]

shap_top   = shap_values.values[:, top_idx]
X_top      = X_sample.iloc[:, top_idx]
labels_top = [feat_labels[i] for i in top_idx]

# Build a sliced Explanation object so shap.plots.beeswarm works natively
shap_exp_top = shap.Explanation(
    values      = shap_top,
    base_values = shap_values.base_values,
    data        = X_top.values,
    feature_names = labels_top,
)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))

shap.plots.beeswarm(shap_exp_top, max_display=TOP_N, show=False)

plt.title(
    f'SHAP Summary Plot (Beeswarm) — Top {TOP_N} Features\n'
    f'XGBoost  |  lr=0.05, scale_pos_weight=12  |  '
    f'n={len(X_sample):,} test samples',
    fontsize=12,
    pad=14,
)
plt.tight_layout()
plt.savefig('shap_summary_plot.pdf', format='pdf', bbox_inches='tight')
print("Saved -> shap_summary_plot.pdf")
