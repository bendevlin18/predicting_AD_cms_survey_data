# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Git Workflow

Always run `git status` before making any code changes to check for uncommitted work or unexpected state.

## Environment

Always run Python scripts and the app using the `streamlit-dev` conda environment:

```bash
conda run -n streamlit-dev <command>
```

## Running the App

```bash
conda run -n streamlit-dev streamlit run landing.py
```

## Architecture

This project has two components:

### 1. Data Pipeline (`cleaning_data_creating_parquet.ipynb`)
One-time notebook that ingests raw CMS Medicare Current Beneficiary Survey (MCBS) fall survey CSVs and produces two parquet files in `data/`:
- `all_fall_surveys_combined.parquet` — ~93k rows x 542 cols; Medicare beneficiary survey responses across multiple years, with median imputation and missingness indicator columns appended
- `full_feature_names.parquet` — metadata mapping abbreviated column acronyms to human-readable full names (sourced from `column_labels.txt`)

The notebook strips year prefix digits from `PUF_ID` to create `PUF_ID_NOY`, used as the group key to prevent the same beneficiary appearing in both train and test splits.

### 2. Streamlit Dashboard (`landing.py`)
Loads the parquet files at startup, splits data using `GroupShuffleSplit` on `PUF_ID_NOY` (80/20), then renders four tabs:

- **Tab 2 — Interactive XGBoost**: User-adjustable `learning_rate`, `scale_pos_weight`, and decision threshold. Trains an `XGBClassifier` live and displays confusion matrix + classification report.
- **Tab 3 — Feature Importance**: Bar chart of top-15 XGBoost feature importances joined with human-readable names from the metadata parquet.
- **Tab 4 — SHAP**: SHAP explanations via `streamlit_shap`.
- **Tab 5 — testing**: Placeholder.

### Target & Feature Selection
- **Target**: `HLT_ALZDEM` (Alzheimer's/dementia diagnosis); values recoded so 2→0 (binary: 1=yes, 0=no).
- **Features**: Columns matching `DEM|HLT|FAL|RSK|ADM|HOU|PUF_ID_NOY`, with `HLT_ALZDEM` and `HLT_DISDECSN` dropped to avoid leakage.
- Class imbalance handled via `scale_pos_weight`. The true class ratio from the training split is ~21.6 (negatives/positives), but we use **`scale_pos_weight=12`** for all calculations. This is a deliberate manual choice: full balancing at 21.6 maximizes recall but generates excessive false positives; 12 optimizes for highest recall of the positive (AD) class while keeping false positives manageable.

## Development Plan

Refer to `plan.md` for a prioritized list of known bugs to fix and new panels to implement.

## Key Dependencies
`streamlit`, `xgboost`, `streamlit_shap`, `plotly`, `scikit-learn`, `pandas`, `numpy`
