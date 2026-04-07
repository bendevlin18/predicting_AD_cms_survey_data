# Dashboard Fix & Enhancement Plan

## Bugs / Broken Code

- **`lr` parameter silently ignored in XGBClassifier** (`landing.py:51`): The model is called with `lr=lr`, but XGBoost's parameter is `learning_rate`. The user-facing slider has no effect on the trained model. Fix: change to `learning_rate=lr`.

- **`y_data` mutation triggers SettingWithCopyWarning** (`landing.py:27`): `y_data[y_data>1] = 0` modifies a slice of `df` in-place. Fix: use `y_data = df['HLT_ALZDEM'].copy(); y_data[y_data > 1] = 0`.

- **`confusion_matrix` `column_config` key won't match** (`landing.py:61`): `confusion_matrix` returns a numpy array; when converted to a DataFrame the columns are integers (`0`, `1`), but the config key `{0: 'test'}` is ambiguous and may not apply. Fix: convert to a labeled DataFrame explicitly — `pd.DataFrame(confusion_matrix(...), index=['Actual Healthy', 'Actual AD'], columns=['Pred Healthy', 'Pred AD'])` — and drop the `column_config`.

- **`np.unique(y_data, return_counts=True)` result is discarded** (`landing.py:28`): The return value is never stored or displayed. Either remove the line or display the class counts somewhere (e.g., as an informational callout on the Landing tab).

- **Tab 1 (Landing) is empty** (`landing.py:14`): `tab1` is declared but there is no `with tab1:` block anywhere.

- **Tab 4 (SHAP) is empty** (`landing.py:14`): `tab4` is declared and `st_shap` is imported, but no SHAP values are ever computed or rendered.

- **Tab 5 (testing) is a placeholder** (`landing.py:14`): Either populate or remove.

- **Model retrains on every widget interaction**: There is no caching. Every slider or number input triggers a full retrain. Fix: wrap training in `@st.cache_resource` or use `st.session_state` keyed on the hyperparameter values, or at minimum use `@st.cache_data`.

## Suggested New / Improved Panels

- **Tab 1 — Landing / Overview**
  - Dataset summary: number of beneficiaries, years covered, class balance (% with AD diagnosis)
  - Brief plain-language description of the prediction task and feature groups (DEM, HLT, FAL, RSK, ADM, HOU)
  - Display the pre-computed `scale_pos_weight` value as a suggested default for the interactive tab

- **Tab 4 — SHAP (complete the existing stub)**
  - Compute SHAP values using `shap.TreeExplainer(bst)`
  - Beeswarm summary plot (`shap.summary_plot`) for global feature impact
  - Option to select a single observation from the test set and show a waterfall or force plot for individual explanation

- **Tab 2 — Precision-Recall curve on Interactive XGBoost tab**
  - Add an interactive Precision-Recall curve (plotly) that updates with the trained model
  - Mark the currently selected decision threshold as a point on the curve
  - Display AUC-PR in the chart title or as a metric

- **Tab 3 — Model parameter summary on Feature Importance tab**
  - Show a small table or metric cards listing the active hyperparameters: learning_rate, scale_pos_weight, and fixed params (max_depth, min_child_weight, subsample, colsample_bytree)

- **New Tab — Threshold Optimization**
  - Plot precision, recall, and F1 for the positive class (AD) across a range of thresholds (0.05–0.95)
  - Highlight the currently selected threshold as a vertical line
  - Include a Precision-Recall curve and ROC curve with AUC displayed

- **New Tab — Data Explorer**
  - Distribution plots for the top-N most important features, split by AD vs. healthy
  - Missingness heatmap or bar chart showing % missing per feature group before imputation
  - Correlation matrix restricted to top features

- **New Tab — Hyperparameter Heatmap**
  - On the backend, perform a grid search over 15 values of `learning_rate` and 15 values of `decision_threshold` (225 combinations total)
  - For each combination, train the model with fixed `scale_pos_weight` and evaluate AD recall (TPR) on the test set
  - Display results as an interactive Plotly heatmap: x-axis = decision threshold, y-axis = learning rate, color = AD recall
  - Cache the grid search results so the heatmap renders without retraining; allow user to select a `scale_pos_weight` to rerun the grid
  - Optionally overlay a contour or highlight the Pareto-optimal cells (high recall, acceptable precision)

- **Tab 3 — NNS Explainer Graphic**
  - Add an illustrative graphic to the Recall Tradeoffs tab that visually explains what Number Needed to Screen means in plain clinical terms
  - Show a concrete example: e.g. a grid of patient icons where a given NNS value is highlighted — "of these N patients flagged, only 1 is a true AD case"
  - Ideally interactive: updates based on the current NNS value at the user's selected threshold / learning rate
  - Goal is to make the abstract metric tangible for a clinical audience who may not be familiar with PPV/NNS

- **New Tab — Demographic Subgroup Analysis**
  - Split the train/test data into two subgroups by income vs. poverty line (`DEM_INCOME`: 1 = below poverty, 2 = at or above)
  - Train a separate XGBClassifier on each subgroup using the same fixed hyperparameters
  - Side-by-side Feature Importance (Gain) bar charts for each subgroup, joined with full feature names
  - Side-by-side SHAP beeswarm plots for each subgroup using `shap.TreeExplainer`
  - Summary metrics table comparing AD recall, precision, AUC-PR, and class balance across the two subgroups
