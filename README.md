# Predicting Alzheimer's Disease from CMS Medicare Survey Data

**Live app:** https://predictingadcmssurveydata.streamlit.app

An interactive Streamlit dashboard that trains an **XGBoost classifier** on the [Medicare Current Beneficiary Survey (MCBS)](https://www.cms.gov/data-research/research/medicare-current-beneficiary-survey) fall survey to predict whether a Medicare beneficiary has an Alzheimer's disease or related dementia (ADRD) diagnosis.

---

## Overview

The MCBS fall survey collects detailed health, demographic, and functional data from Medicare beneficiaries across multiple years. This dashboard uses those survey responses to build a binary classifier predicting `HLT_ALZDEM` (Alzheimer's/dementia diagnosis) and provides a suite of interactive tools for exploring model performance, interpreting predictions, and understanding tradeoffs — with a focus on clinical relevance.

**Class imbalance note:** AD cases represent ~4.4% of the dataset. Rather than using the full class ratio (~21.6) for `scale_pos_weight`, we use **12** — a deliberate choice that maximizes recall of the positive (AD) class while keeping false positives manageable.

---

## Dataset Summary

| | |
|---|---|
| **Survey years** | 2017–2023 |
| **Total observations** | 93,171 |
| **Unique beneficiaries** | 29,502 |
| **AD diagnosis rate** | 4.4% |
| **Total features used** | 370 (after dropping target and leakage variables) |

### Features by Domain

| Prefix | Domain | # Features |
|--------|--------|-----------|
| `DEM` | Demographics (age, sex, race, income, education) | 22 |
| `HLT` | Health conditions and diagnoses | 198 |
| `FAL` | Fall history and circumstances | 24 |
| `RSK` | Fall risk factors | 20 |
| `ADM` | Administrative / insurance enrollment | 44 |
| `HOU` | Housing and living situation | 60 |

Missing values are median-imputed during preprocessing, with binary `_missing` indicator columns appended for each imputed feature.

### Sample Demographics

| Characteristic | Categories |
|----------------|-----------|
| **Age group** | 65–74, 75–84, 85+ |
| **Sex** | Male, Female |
| **Race / Ethnicity** | Non-Hispanic White, Non-Hispanic Black, Hispanic, Other |
| **Education** | Less than high school, High school / GED, Some college or more |
| **Income** | Below poverty line, At or above poverty line |

---

## Dashboard Tabs

| Tab | Description |
|-----|-------------|
| **Landing** | Dataset summary, class balance, feature group overview |
| **Sample Demographics** | Distributions of sex, race, education, income, and age group split by AD diagnosis status |
| **Interactive XGBoost** | Adjust `learning_rate`, `scale_pos_weight`, and decision threshold live; view confusion matrix, precision-recall curve, and classification report |
| **Recall Tradeoffs** | ROC curves by learning rate, Number Needed to Screen vs. Recall scatter, and an interactive NNS explainer graphic |
| **Feature Importance** | Top-15 XGBoost feature importances (by gain) with human-readable feature names |
| **SHAP** | Global SHAP beeswarm plot explaining which survey features most impact predictions |
| **Subgroup Analysis** | Side-by-side model comparison split by income vs. poverty line — separate models, feature importance, SHAP beeswarms, and a diverging importance comparison chart |

---

## Features & Predictors

Features are drawn from MCBS survey modules matching these prefixes:

| Prefix | Domain |
|--------|--------|
| `DEM` | Demographics (age, sex, race, income, education) |
| `HLT` | Health conditions and diagnoses |
| `FAL` | Fall history and circumstances |
| `RSK` | Fall risk factors |
| `ADM` | Administrative / insurance enrollment |
| `HOU` | Housing and living situation |

**Excluded to prevent leakage:** `HLT_ALZDEM` (target), `HLT_DISDECSN` (disease decision flag)

Beneficiaries are split 80/20 into train/test using `PUF_ID_NOY` as a group key, ensuring the same person never appears in both splits across survey years.

---

## Model

- **Algorithm:** XGBoost (`XGBClassifier`)
- **Fixed hyperparameters:** `max_depth=4`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric=aucpr`
- **Tunable:** `learning_rate` (default 0.05), `scale_pos_weight` (default 12), decision threshold (default 0.20)
- **Caching:** Models and SHAP values are cached with `@st.cache_resource` to avoid retraining on every widget interaction

---

## Data Pipeline

Raw CMS MCBS fall survey CSVs → `cleaning_data_creating_parquet.ipynb` → two parquet files in `data/`:

- `all_fall_surveys_combined.parquet` — ~93k rows × 542 cols; survey responses across multiple years with median imputation and missingness indicator columns
- `full_feature_names.parquet` — metadata mapping abbreviated column codes to human-readable names

---

## Standalone Scripts

| Script | Purpose |
|--------|---------|
| `generate_heatmap.py` | Grid search over 15 learning rates × 15 thresholds; saves AD recall and FPR results to `data/heatmap_grid_results.parquet` |
| `generate_shap_summary.py` | Generates a SHAP beeswarm summary PDF for offline review |
| `generate_fpr_recall_scatter.py` | FPR vs. recall scatter plot PDF with Pareto frontier |
| `generate_tradeoff_visualizations.py` | 5-page PDF of recall/FPR tradeoff visualizations (ROC curves, decision curve analysis, NNS, contour plot, Pareto frontier) |

---

## Setup & Running

### Requirements

```
streamlit
xgboost
shap
streamlit-shap
plotly
scikit-learn
pandas==2.2.2
numpy==1.26.4
```

### Install

```bash
pip install -r requirements.txt
```

### Run

```bash
streamlit run landing.py
```

> **Note:** The heatmap tab requires `data/heatmap_grid_results.parquet`. Generate it by running `python generate_heatmap.py` before launching the app.

---

## Repository Structure

```
├── landing.py                          # Main Streamlit dashboard
├── cleaning_data_creating_parquet.ipynb  # Data pipeline notebook
├── generate_heatmap.py                 # Hyperparameter grid search
├── generate_shap_summary.py            # SHAP summary PDF generator
├── generate_fpr_recall_scatter.py      # FPR/recall scatter PDF
├── generate_tradeoff_visualizations.py # Tradeoff visualization suite
├── data/
│   ├── all_fall_surveys_combined.parquet
│   ├── full_feature_names.parquet
│   └── heatmap_grid_results.parquet
├── plan.md                             # Development roadmap
├── visualization_ideas.md              # Notes on future visualizations
└── requirements.txt
```

---

## Data Source

[Medicare Current Beneficiary Survey (MCBS)](https://www.cms.gov/data-research/research/medicare-current-beneficiary-survey) — Centers for Medicare & Medicaid Services (CMS). The MCBS is a continuous, multipurpose survey of a nationally representative sample of Medicare beneficiaries.
