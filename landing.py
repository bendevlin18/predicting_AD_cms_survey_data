import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, classification_report, recall_score, precision_recall_curve, auc
import xgboost as xgb
from xgboost import XGBClassifier
from streamlit_shap import st_shap
import plotly.express as px

st.set_page_config(layout = 'wide')
df = pd.read_parquet('data/all_fall_surveys_combined.parquet')
full_feature_names = pd.read_parquet('data/full_feature_names.parquet')
st.title("CMS MCBS Fall Survey — Alzheimer's / Dementia Prediction")
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Landing", "Sample Demographics", "Interactive XGBoost", "Recall Tradeoffs", "Feature Importance", "SHAP", "Subgroup Analysis"])

### i'm excluding some survey group questions because they either have a majority N/A or missing values (e.g. housing), or I don't think that they will be all that useful in prediction (PRV - preventative), OR they will cause leakage (e.g. other hospital admissions data)
### i'm leaving in fall risk data, even though it has lots of missing values, because it is something that may be very predictive
### i could then be convinced to leave in the housing data if it seems like it will make a big difference in performance

X_data = df[df.columns[df.columns.str.contains('DEM|HLT|FAL|RSK|ADM|HOU|PUF_ID_NOY')]]

### dropping any predictors that are super prone to leakage...
X_data = X_data.drop(columns = {'HLT_ALZDEM', 'HLT_DISDECSN'})

### for this data, 1 = yes, 2 = no diagnosis of dementia, changing 2 -> 0 for easier interpretation
y_data = df['HLT_ALZDEM'].copy()
y_data[y_data>1] = 0


from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(test_size=0.2, random_state=4)
train_idx, test_idx = next(gss.split(df, groups=df['PUF_ID_NOY']))

X_train = X_data.iloc[train_idx]
X_test  = X_data.iloc[test_idx]
y_train = y_data.iloc[train_idx]
y_test  = y_data.iloc[test_idx]

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])


@st.cache_resource
def train_model(lr, spw):
    bst = XGBClassifier(scale_pos_weight=spw, eval_metric='aucpr', max_depth=4,
                        learning_rate=lr, min_child_weight=10, subsample=0.8, colsample_bytree=0.8)
    bst.fit(X_train, y_train)
    return bst

with tab1:
    st.markdown(
        """
        This dashboard trains an **XGBoost classifier** on the
        [Medicare Current Beneficiary Survey (MCBS)](https://www.cms.gov/data-research/research/medicare-current-beneficiary-survey)
        fall survey to predict whether a Medicare beneficiary has an Alzheimer's disease or
        related dementia (ADRD) diagnosis (`HLT_ALZDEM`).
        Beneficiaries are split 80/20 into train/test using `PUF_ID_NOY` as the group key,
        ensuring the same person never appears in both splits across survey years.
        """
    )

    st.subheader("Dataset Summary")
    total_rows = len(df)
    unique_beneficiaries = df['PUF_ID_NOY'].nunique()
    years = sorted(df['SURVEYYR'].dropna().unique().astype(int).tolist())
    ad_count = int((y_data == 1).sum())
    healthy_count = int((y_data == 0).sum())
    ad_pct = ad_count / total_rows * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Observations", f"{total_rows:,}")
    c2.metric("Unique Beneficiaries", f"{unique_beneficiaries:,}")
    c3.metric("Survey Years", f"{min(years)}–{max(years)}")
    c4.metric("AD Diagnosis Rate", f"{ad_pct:.1f}%")

    st.subheader("Class Balance")
    balance_df = pd.DataFrame({
        'Class': ['Healthy (0)', 'Alzheimer\'s / Dementia (1)'],
        'Count': [healthy_count, ad_count],
        'Percentage': [f"{healthy_count/total_rows*100:.1f}%", f"{ad_pct:.1f}%"]
    })
    fig_balance = px.bar(
        balance_df, x='Class', y='Count', text='Percentage',
        color='Class', color_discrete_sequence=['#4C78A8', '#E45756'],
        labels={'Count': 'Number of Observations'}
    )
    fig_balance.update_traces(textposition='outside')
    fig_balance.update_layout(showlegend=False)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.plotly_chart(fig_balance, use_container_width=True)

    st.subheader("Feature Groups")
    st.markdown(f"""
    Features are drawn from MCBS survey modules matching these prefixes
    (columns: **{X_data.shape[1]}** features after dropping leakage variables):

    | Prefix | Domain |
    |--------|--------|
    | `DEM` | Demographics (age, sex, race, income, education) |
    | `HLT` | Health conditions and diagnoses |
    | `FAL` | Fall history and circumstances |
    | `RSK` | Fall risk factors |
    | `ADM` | Administrative / insurance enrollment |
    | `HOU` | Housing and living situation |

    **Excluded to prevent leakage:** `HLT_ALZDEM` (target), `HLT_DISDECSN` (disease decision flag)

    > **Class imbalance note:** With ~{ad_pct:.1f}% positive cases, the true class ratio in the
    > training split is **{scale_pos_weight:.1f}** (negatives ÷ positives). Rather than using this
    > value directly, we set `scale_pos_weight = 12` — a deliberate choice that optimizes for the
    > highest recall of the positive (AD) class while reducing false positives relative to full
    > balancing. You can adjust this on the Interactive XGBoost tab.
    """)

    st.subheader("All Features")
    feat_df = pd.DataFrame({'Column': X_data.columns})
    feat_df = feat_df.join(full_feature_names.rename(columns={'full_name': 'Full Name'}), on='Column')
    # _missing indicator columns: look up base column name and append suffix
    is_missing_col = feat_df['Column'].str.endswith('_missing') & feat_df['Full Name'].isna()
    base_cols = feat_df.loc[is_missing_col, 'Column'].str.replace('_missing$', '', regex=True)
    base_names = base_cols.map(full_feature_names['full_name']).fillna(base_cols)
    feat_df.loc[is_missing_col, 'Full Name'] = base_names.values + ' (missing indicator)'
    # remaining unknowns: fall back to column name
    feat_df['Full Name'] = feat_df['Full Name'].fillna(feat_df['Column'])
    feat_df.insert(0, 'Group', feat_df['Column'].str.extract(r'^([A-Z]+)_')[0])
    st.dataframe(feat_df, use_container_width=True, hide_index=True)

with tab3:
    st.markdown("""
    Adjust the three parameters below to explore how they affect model performance,
    particularly **AD recall** (the share of true Alzheimer's/dementia cases the model correctly identifies).

    | Parameter | What it does | Effect on AD recall |
    |-----------|-------------|---------------------|
    | **`learning_rate`** | Controls how much each tree corrects the previous one. Lower values learn more slowly but generalize better; higher values train faster but risk overfitting. | Indirectly affects recall — very high or very low values can degrade overall model quality. |
    | **`scale_pos_weight`** | Tells XGBoost how much extra weight to give the positive (AD) class during training. Set to the negative/positive ratio for full balancing, or lower to reduce false positives. We use **12** (true ratio ≈ 21.6) as a deliberate trade-off. | Higher values push the model to prioritize catching AD cases, increasing recall at the cost of more false positives. |
    | **`decision_threshold`** | The probability cutoff above which a case is predicted as AD. Lowering it flags more cases as positive; raising it makes the model more conservative. | **Most direct lever** — lowering the threshold increases AD recall but also increases false positives. |
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        lr = st.number_input(label = 'learning_rate', value = 0.05)
        spw = st.number_input(label = 'scale_pos_weight', value = 12, key = 2)
        thresh = st.number_input(label = 'decision_threshold (Default = 0.2)', value = .2, key = 3)

    with col2:

        bst = train_model(lr, spw)
        ### the model was having trouble with recall, esp for the actual hospital admissions since they only represent about 5% of the dataset
        ### thus let's try a bunch of different thresholds for prediction based on probability and see how that affects our recall for both groups
        y_probs = bst.predict_proba(X_test)[:, 1]

        ### 0.25 threshold seems to be the best for optimizing + recall while not letting false positives get out of control
        preds = (y_probs >= thresh).astype(int)

        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        ad_recall = tp / (tp + fn)

        # 1. Grouped confusion matrix bar chart
        cm_bar_df = pd.DataFrame({
            'Actual Class': ['Healthy', 'Healthy', 'AD', 'AD'],
            'Outcome':      ['Correct', 'Incorrect', 'Incorrect', 'Correct'],
            'Count':        [tn, fp, fn, tp],
            'Label':        [f'TN: {tn:,}', f'FP: {fp:,}', f'FN: {fn:,}', f'TP: {tp:,}']
        })
        fig_cm_bar = px.bar(
            cm_bar_df, x='Actual Class', y='Count', color='Outcome', barmode='group',
            color_discrete_map={'Correct': '#2ca02c', 'Incorrect': '#E45756'},
            title=f'Confusion Matrix — threshold={thresh:.2f} | AD Recall (TPR): {ad_recall:.1%}',
            text='Label'
        )
        fig_cm_bar.update_traces(textposition='outside')
        fig_cm_bar.update_layout(bargap=0.3, bargroupgap=0.1, yaxis_range=[0, max(tn, fp) + 2000])
        st.plotly_chart(fig_cm_bar, use_container_width=True)

        # 2. Confusion matrix table
        st.dataframe(pd.DataFrame(
            cm,
            index=['Actual Healthy', 'Actual AD'],
            columns=['Pred Healthy', 'Pred AD']
        ))

        # 3. Precision-Recall curve
        precision_vals, recall_vals, thresholds_pr = precision_recall_curve(y_test, y_probs, pos_label=1)
        aucpr = auc(recall_vals, precision_vals)
        pr_df = pd.DataFrame({'Recall': recall_vals[:-1], 'Precision': precision_vals[:-1], 'Threshold': thresholds_pr})
        nearest_idx = (pr_df['Threshold'] - thresh).abs().idxmin()
        fig_pr = px.line(pr_df, x='Recall', y='Precision',
                         title=f'Precision-Recall Curve — AD (positive class) | AUC-PR = {aucpr:.3f}',
                         labels={'Recall': 'Recall (AD)', 'Precision': 'Precision (AD)'})
        fig_pr.add_scatter(
            x=[pr_df.loc[nearest_idx, 'Recall']],
            y=[pr_df.loc[nearest_idx, 'Precision']],
            mode='markers',
            marker=dict(size=12, color='red', symbol='circle'),
            name=f'Threshold = {thresh:.2f}'
        )
        fig_pr.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(fig_pr, use_container_width=True)

        # 4. Classification report table
        target_names = ["Healthy", "AD"]
        st.dataframe(classification_report(y_test, preds, target_names=target_names, output_dict=True))

with tab6:
    import shap
    import matplotlib.pyplot as plt

    @st.cache_resource
    def compute_shap(lr, spw, sample_n=2000):
        _model = XGBClassifier(
            scale_pos_weight=spw, eval_metric='aucpr', max_depth=4,
            learning_rate=lr, min_child_weight=10, subsample=0.8,
            colsample_bytree=0.8, random_state=4,
        )
        _model.fit(X_train, y_train)
        rng = np.random.default_rng(4)
        idx = rng.choice(len(X_test), size=min(sample_n, len(X_test)), replace=False)
        X_sample = X_test.iloc[idx]
        explainer = shap.TreeExplainer(_model)
        sv = explainer(X_sample)
        return sv, X_sample

    name_map = full_feature_names['full_name'].to_dict()

    def readable(col):
        if col in name_map:
            return name_map[col]
        base = col.replace('_missing', '')
        if base in name_map:
            return name_map[base] + ' (missing)'
        return col

    with st.spinner("Computing SHAP values..."):
        shap_vals, X_sample = compute_shap(lr, spw)

    mean_abs = np.abs(shap_vals.values).mean(axis=0)
    top_idx  = np.argsort(mean_abs)[::-1][:20]
    shap_exp_top = shap.Explanation(
        values        = shap_vals.values[:, top_idx],
        base_values   = shap_vals.base_values,
        data          = X_sample.iloc[:, top_idx].values,
        feature_names = [readable(c) for c in X_sample.columns[top_idx]],
    )

    text_col, plot_col = st.columns([2, 1])

    with text_col:
        st.markdown("""
        ### How to Read This Plot

        **SHAP (SHapley Additive exPlanations)** is rooted in cooperative game theory. The core idea:
        treat each feature as a "player" in a game where the outcome is the model's prediction.
        Each feature's SHAP value is its fair share of the credit (or blame) for pushing the prediction
        away from the baseline — the average predicted probability across all observations.

        **Every dot is one observation** from the test set, plotted for a single feature:

        | | Meaning |
        |---|---|
        | **Position (x-axis)** | A positive SHAP value pushes the model toward predicting **AD**; negative pushes toward **Healthy**. |
        | **Color** | The feature's actual value — **red = high**, **blue = low**. |
        | **Feature order** | Features ranked top-to-bottom by mean absolute SHAP value (overall importance). |

        Together, color and position reveal the *direction* of each feature's effect — e.g. a red dot
        (high feature value) on the right side means a high value for that feature
        increases the predicted probability of AD.
        """)

    with plot_col:
        st.subheader("Which features of the survey most significantly impact predictions?")
        shap.plots.beeswarm(shap_exp_top, max_display=20, show=False)
        st.pyplot(plt.gcf(), bbox_inches='tight')
        plt.close()

with tab2:
    st.subheader("Beneficiary Demographics")
    st.markdown("Distributions across all survey observations (2017–2023). Charts are split by AD diagnosis status.")

    dem_df = df[['DEM_SEX', 'DEM_RACE', 'DEM_EDU', 'DEM_INCOME', 'DEM_AGE']].copy()
    dem_df['Diagnosis'] = y_data.map({1.0: 'Alzheimer\'s/Dementia', 0.0: 'Healthy'})

    sex_labels    = {1.0: 'Male', 2.0: 'Female'}
    race_labels   = {1.0: 'Non-Hispanic White', 2.0: 'Non-Hispanic Black', 3.0: 'Hispanic', 4.0: 'Other'}
    edu_labels    = {1.0: 'Less than HS', 2.0: 'HS / GED', 3.0: 'Some college or more'}
    income_labels = {1.0: 'Below poverty line', 2.0: 'At or above poverty line'}
    age_labels    = {1.0: '65–74', 2.0: '75–84', 3.0: '85+'}

    dem_df['Sex']      = dem_df['DEM_SEX'].map(sex_labels)
    dem_df['Race']     = dem_df['DEM_RACE'].map(race_labels)
    dem_df['Education']= dem_df['DEM_EDU'].map(edu_labels)
    dem_df['Income']   = dem_df['DEM_INCOME'].map(income_labels)
    dem_df['Age Group']= dem_df['DEM_AGE'].map(age_labels)

    pie_col1, pie_col2 = st.columns(2)

    with pie_col1:
        fig_sex = px.pie(dem_df.dropna(subset=['Sex']), names='Sex',
                         color='Sex', title='Sex',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_sex.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_sex, use_container_width=True)

        fig_edu = px.pie(dem_df.dropna(subset=['Education']), names='Education',
                         color='Education', title='Highest Education',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        fig_edu.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_edu, use_container_width=True)

    with pie_col2:
        fig_race = px.pie(dem_df.dropna(subset=['Race']), names='Race',
                          color='Race', title='Race / Ethnicity',
                          color_discrete_sequence=px.colors.qualitative.Set2)
        fig_race.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_race, use_container_width=True)

        fig_income = px.pie(dem_df.dropna(subset=['Income']), names='Income',
                            color='Income', title='Income vs. Poverty Line',
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_income.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_income, use_container_width=True)

    st.subheader("Age Group Distribution")
    st.caption("MCBS groups beneficiary ages into three bands. Bars are split by diagnosis status.")
    age_counts = (
        dem_df.dropna(subset=['Age Group'])
        .groupby(['Age Group', 'Diagnosis'])
        .size()
        .reset_index(name='Count')
    )
    age_order = ['65–74', '75–84', '85+']
    fig_age = px.bar(
        age_counts, x='Age Group', y='Count', color='Diagnosis',
        barmode='group', category_orders={'Age Group': age_order},
        color_discrete_map={"Healthy": "#4C78A8", "Alzheimer's/Dementia": "#E45756"},
        labels={'Count': 'Number of Observations'}
    )
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.plotly_chart(fig_age, use_container_width=True)

with tab4:
    import plotly.graph_objects as go

    st.subheader("Hyperparameter Heatmap — AD Recall")
    grid_path = 'data/heatmap_grid_results.parquet'
    if not os.path.exists(grid_path):
        st.info("Grid search results not yet generated. Run `generate_heatmap.py` to produce them.")
    else:
        grid_df = pd.read_parquet(grid_path)

        roc_col, nns_col = st.columns(2)

        # ── ROC Curves by Learning Rate ───────────────────────────────────────
        with roc_col:
            st.subheader("ROC Curves by Learning Rate")
            st.caption("Each curve traces one learning rate through (FPR, Recall) space as the threshold varies. Points closer to the top-left represent better combined performance.")

            lr_values  = sorted(grid_df['learning_rate'].unique())
            n_lrs      = len(lr_values)
            colorscale = px.colors.sample_colorscale('Viridis', [i / (n_lrs - 1) for i in range(n_lrs)])

            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(dash='dash', color='grey', width=1),
                name='Random (no skill)', showlegend=True
            ))
            for lr_val, color in zip(lr_values, colorscale):
                sub = grid_df[grid_df['learning_rate'] == lr_val].sort_values('threshold', ascending=False)
                xs  = [0] + sub['FPR'].tolist()      + [1]
                ys  = [0] + sub['AD_recall'].tolist() + [1]
                fig_roc.add_trace(go.Scatter(
                    x=xs, y=ys, mode='lines',
                    line=dict(color=color, width=1.8),
                    name=f'lr={lr_val:.4f}',
                    hovertemplate='FPR: %{x:.3f}<br>Recall: %{y:.3f}<extra>lr=' + f'{lr_val:.4f}</extra>'
                ))
            fig_roc.update_layout(
                xaxis=dict(title='False Positive Rate', range=[-0.02, 1.02]),
                yaxis=dict(title='AD Recall', range=[-0.02, 1.02]),
                legend=dict(font=dict(size=9), title='Learning Rate'),
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_roc, use_container_width=True)

        # ── Number Needed to Screen vs Recall ─────────────────────────────────
        with nns_col:
            st.subheader("Number Needed to Screen vs. Recall")
            st.caption("NNS = total patients flagged per true AD case caught (1 / PPV). For example, an NNS of 10 means the model flags 10 patients to identify 1 true AD case. The elbow shows where pushing for more recall causes NNS to rise sharply.")

            total_pos = int((y_test == 1).sum())
            total_neg = int((y_test == 0).sum())

            nns_df = grid_df.copy()
            nns_df['TP']  = (nns_df['AD_recall'] * total_pos).round()
            nns_df['FP']  = (nns_df['FPR']       * total_neg).round()
            denom         = (nns_df['TP'] + nns_df['FP']).clip(lower=1)
            nns_df['PPV'] = nns_df['TP'] / denom
            nns_df['NNS'] = (1 / nns_df['PPV'].replace(0, np.nan)).clip(upper=80)
            nns_df = nns_df.dropna(subset=['NNS'])

            fig_nns = px.scatter(
                nns_df, x='AD_recall', y='NNS',
                color='threshold', color_continuous_scale='plasma',
                labels={'AD_recall': 'AD Recall', 'NNS': 'Patients Flagged per True AD Case', 'threshold': 'Threshold'},
                hover_data={'learning_rate': ':.4f', 'threshold': ':.2f', 'NNS': ':.1f', 'AD_recall': ':.3f'}
            )
            fig_nns.update_layout(
                xaxis=dict(title='AD Recall', range=[-0.02, 1.02]),
                yaxis=dict(title='Patients Flagged per True AD Case (NNS)'),
                margin=dict(t=20, b=40),
            )
            st.plotly_chart(fig_nns, use_container_width=True)

        # ── NNS Explainer Graphic ─────────────────────────────────────────────
        st.divider()
        st.subheader("What does this NNS mean in practice?")
        st.markdown("Select a threshold and learning rate to see how many patients the model must flag to find **one true AD case**. Each circle below is one flagged patient.")

        exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 2])

        lr_options     = sorted(nns_df['learning_rate'].unique())
        thresh_options = sorted(nns_df['threshold'].unique())
        default_lr_idx     = min(range(len(lr_options)),     key=lambda i: abs(lr_options[i] - 0.05))
        default_thresh_idx = min(range(len(thresh_options)), key=lambda i: abs(thresh_options[i] - 0.20))

        with exp_col1:
            sel_lr = st.selectbox("Learning Rate", lr_options,
                                  index=default_lr_idx, format_func=lambda x: f"{x:.4f}",
                                  key='nns_lr')
        with exp_col2:
            sel_thresh = st.selectbox("Decision Threshold", thresh_options,
                                      index=default_thresh_idx, format_func=lambda x: f"{x:.2f}",
                                      key='nns_thresh')

        sel_row = nns_df[(nns_df['learning_rate'] == sel_lr) & (nns_df['threshold'] == sel_thresh)]

        if len(sel_row) > 0:
            nns_val    = max(1, int(round(sel_row.iloc[0]['NNS'])))
            recall_val = sel_row.iloc[0]['AD_recall']
            fpr_val    = sel_row.iloc[0]['FPR']

            with exp_col3:
                st.metric("NNS", nns_val)
                st.caption(f"AD Recall: {recall_val:.1%}  |  FPR: {fpr_val:.1%}")

            DISPLAY_CAP = 50
            display_n   = min(nns_val, DISPLAY_CAP)

            rng    = np.random.default_rng(int(sel_thresh * 1000 + sel_lr * 10000))
            ad_pos = int(rng.integers(0, display_n))

            truncation_note = f"<br><small style='color:#888'>Showing {DISPLAY_CAP} of {nns_val} patients</small>" if nns_val > DISPLAY_CAP else ""

            circles = ""
            for i in range(display_n):
                if i == ad_pos:
                    circles += '<div title="True AD Case" style="width:32px;height:32px;border-radius:50%;background:#E45756;border:2px solid #b03030;flex-shrink:0;"></div>'
                else:
                    circles += '<div title="Flagged — No AD" style="width:32px;height:32px;border-radius:50%;background:#d0d0d0;border:2px solid #aaaaaa;flex-shrink:0;"></div>'

            st.markdown(
                f"""
                <p style="font-size:16px;font-weight:600;margin-bottom:8px;">
                    Of every <span style="color:#E45756">{nns_val} patients flagged</span>,
                    <span style="color:#E45756">1</span> has Alzheimer's / Dementia
                    {truncation_note}
                </p>
                <div style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:12px;">
                    {circles}
                </div>
                <div style="display:flex;gap:16px;font-size:13px;color:#ffffff;margin-top:4px;">
                    <span><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#E45756;vertical-align:middle;margin-right:4px;"></span>True AD Case</span>
                    <span><span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#d0d0d0;vertical-align:middle;margin-right:4px;"></span>Flagged — No AD</span>
                </div>
                <p style="font-size:15px;margin-top:12px;">
                    AND by doing so, you have the opportunity to catch
                    <span style="color:#E45756;font-weight:600;">{recall_val:.1%}</span>
                    of all AD patients in this dataset.
                </p>
                <p style="font-size:13px;margin-top:10px;padding:10px 14px;border-left:3px solid #E45756;background:rgba(228,87,86,0.08);">
                    <strong>Key takeaway:</strong> As you increase the decision threshold you are likely able to decrease
                    the number of false positives (i.e. the number of patients flagged without AD), but you capture
                    a smaller percentage of the total AD patients in this dataset.
                </p>
                """,
                unsafe_allow_html=True
            )

with tab5:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.subheader("Model Parameters")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        p1.metric("learning_rate", lr)
        p2.metric("scale_pos_weight", f"{spw:.0f}")
        p3.metric("max_depth", 4)
        p4.metric("min_child_weight", 10)
        p5.metric("subsample", 0.8)
        p6.metric("colsample_bytree", 0.8)

        st.subheader("Feature Importance (Gain)")
        from xgboost import plot_importance
        import matplotlib.pyplot as plt
        importance_df = pd.DataFrame({'feature': X_test.columns, 'importance': bst.feature_importances_})
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        joined_metadata_importance = importance_df.set_index('feature').join(full_feature_names).dropna(axis = 0).head(15)
        ### using gain as it more accurately reflects contribution to loss reduction
        fig2 = px.bar(y = joined_metadata_importance['full_name'], x = joined_metadata_importance.importance , orientation='h', labels={'x': 'XGBoost Feature Importance', 'y': 'Feature Name'})
        fig2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig2, theme="streamlit")
        st.subheader("Top 15 Features")
        st.dataframe(joined_metadata_importance)

with tab7:
    import shap as shap_sg
    import matplotlib.pyplot as plt_sg

    st.subheader("Subgroup Analysis — Income vs. Poverty Line")
    st.markdown("""
    Two separate XGBoost models are trained on subgroup-specific train splits, split by `DEM_INCOME`:
    **below the poverty line** vs. **at or above the poverty line**. Both use `scale_pos_weight=12`.
    This reveals whether the drivers of AD prediction differ meaningfully across income groups.
    """)

    # ── Subgroup splits ───────────────────────────────────────────────────────
    income_train_vals = df.iloc[train_idx]['DEM_INCOME']
    income_test_vals  = df.iloc[test_idx]['DEM_INCOME']

    SUBGROUPS = {
        'Below Poverty Line':       1.0,
        'At or Above Poverty Line': 2.0,
    }

    @st.cache_resource
    def train_subgroup(income_code):
        mask_tr = (income_train_vals == income_code).values
        mask_te = (income_test_vals  == income_code).values
        X_tr = X_train.iloc[mask_tr]; y_tr = y_train.iloc[mask_tr]
        X_te = X_test.iloc[mask_te];  y_te = y_test.iloc[mask_te]
        mdl = XGBClassifier(
            scale_pos_weight=12, eval_metric='aucpr', max_depth=4,
            learning_rate=0.05, min_child_weight=10, subsample=0.8,
            colsample_bytree=0.8, random_state=4,
        )
        mdl.fit(X_tr, y_tr)
        return mdl, X_tr, y_tr, X_te, y_te

    @st.cache_resource
    def subgroup_shap(income_code, sample_n=1000):
        mdl, _, _, X_te, _ = train_subgroup(income_code)
        rng = np.random.default_rng(4)
        idx = rng.choice(len(X_te), size=min(sample_n, len(X_te)), replace=False)
        X_s = X_te.iloc[idx]
        exp = shap_sg.TreeExplainer(mdl)
        return exp(X_s), X_s

    THRESH = 0.20

    def subgroup_metrics(income_code):
        mdl, X_tr, y_tr, X_te, y_te = train_subgroup(income_code)
        probs = mdl.predict_proba(X_te)[:, 1]
        preds = (probs >= THRESH).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, preds).ravel()
        recall   = tp / (tp + fn) if (tp + fn) > 0 else 0
        prec     = tp / (tp + fp) if (tp + fp) > 0 else 0
        fpr      = fp / (fp + tn) if (fp + tn) > 0 else 0
        pr_p, pr_r, _ = precision_recall_curve(y_te, probs)
        aucpr    = auc(pr_r, pr_p)
        ad_pct   = y_te.mean() * 100
        return {
            'Train size':       len(y_tr),
            'Test size':        len(y_te),
            'AD prevalence':    f"{ad_pct:.1f}%",
            'AD Recall':        f"{recall:.1%}",
            'Precision':        f"{prec:.1%}",
            'FPR':              f"{fpr:.1%}",
            'AUC-PR':           f"{aucpr:.3f}",
        }

    with st.spinner("Training subgroup models..."):
        metrics_below  = subgroup_metrics(1.0)
        metrics_above  = subgroup_metrics(2.0)

    # ── Summary metrics ───────────────────────────────────────────────────────
    st.subheader("Model Performance by Subgroup")
    st.caption(f"Decision threshold = {THRESH}")
    metrics_df = pd.DataFrame({
        'Metric':                   list(metrics_below.keys()),
        'Below Poverty Line':       list(metrics_below.values()),
        'At or Above Poverty Line': list(metrics_above.values()),
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.divider()

    # ── Feature importance ────────────────────────────────────────────────────
    st.subheader("Feature Importance (Top 15)")

    fi_col1, fi_col2 = st.columns(2)

    def importance_chart(income_code, label):
        mdl, _, _, X_te, _ = train_subgroup(income_code)
        imp_df = (
            pd.DataFrame({'feature': X_te.columns, 'importance': mdl.feature_importances_})
            .sort_values('importance', ascending=False)
            .set_index('feature')
            .join(full_feature_names)
            .dropna()
            .head(15)
        )
        fig = px.bar(
            imp_df, x='importance', y='full_name', orientation='h',
            title=label,
            labels={'importance': 'Feature Importance', 'full_name': ''},
        )
        fig.update_yaxes(autorange='reversed')
        fig.update_layout(margin=dict(t=40, b=20))
        return fig

    with fi_col1:
        st.plotly_chart(importance_chart(1.0, 'Below Poverty Line'), use_container_width=True)
    with fi_col2:
        st.plotly_chart(importance_chart(2.0, 'At or Above Poverty Line'), use_container_width=True)

    # ── Feature importance comparison ─────────────────────────────────────────
    st.subheader("Feature Importance Comparison")
    st.caption("Difference in feature importance (Below Poverty − At/Above Poverty). Positive = more important for the below poverty model; negative = more important for the at/above poverty model.")

    def get_importance_series(income_code):
        mdl, _, _, X_te, _ = train_subgroup(income_code)
        return pd.Series(mdl.feature_importances_, index=X_te.columns)

    imp_below = get_importance_series(1.0)
    imp_above = get_importance_series(2.0)

    # Union of top 20 features from each group
    top_below = set(imp_below.nlargest(20).index)
    top_above = set(imp_above.nlargest(20).index)
    union_feats = list(top_below | top_above)

    name_map_comp = full_feature_names['full_name'].to_dict()
    def readable_comp(col):
        if col in name_map_comp: return name_map_comp[col]
        base = col.replace('_missing', '')
        return (name_map_comp[base] + ' (missing)') if base in name_map_comp else col

    diff_df = pd.DataFrame({
        'feature':  union_feats,
        'below':    imp_below.reindex(union_feats, fill_value=0).values,
        'above':    imp_above.reindex(union_feats, fill_value=0).values,
    })
    diff_df['diff']      = diff_df['below'] - diff_df['above']
    diff_df['full_name'] = diff_df['feature'].map(readable_comp)
    diff_df = diff_df.sort_values('diff')

    diff_df['color'] = diff_df['diff'].apply(lambda x: 'More important: Below Poverty' if x > 0 else 'More important: At/Above Poverty')

    fig_diff = px.bar(
        diff_df, x='diff', y='full_name', orientation='h',
        color='color',
        color_discrete_map={
            'More important: Below Poverty':       '#E45756',
            'More important: At/Above Poverty':    '#4C78A8',
        },
        labels={'diff': 'Importance Difference (Below − At/Above)', 'full_name': '', 'color': ''},
    )
    fig_diff.add_vline(x=0, line_width=1, line_color='grey')
    fig_diff.update_layout(
        legend=dict(orientation='h', yanchor='bottom', y=1.02),
        margin=dict(t=40, b=20),
        height=600,
    )
    _, comp_col, _ = st.columns([1, 4, 1])
    with comp_col:
        st.plotly_chart(fig_diff, use_container_width=True)

    st.divider()

    # ── SHAP beeswarms ────────────────────────────────────────────────────────
    st.subheader("SHAP Summary (Top 15 Features)")

    shap_col1, shap_col2 = st.columns(2)

    def shap_beeswarm(income_code, label):
        sv, X_s = subgroup_shap(income_code)
        mean_abs = np.abs(sv.values).mean(axis=0)
        top_idx  = np.argsort(mean_abs)[::-1][:15]

        name_map_sg = full_feature_names['full_name'].to_dict()
        def readable_sg(col):
            if col in name_map_sg: return name_map_sg[col]
            base = col.replace('_missing', '')
            return (name_map_sg[base] + ' (missing)') if base in name_map_sg else col

        exp_top = shap_sg.Explanation(
            values        = sv.values[:, top_idx],
            base_values   = sv.base_values,
            data          = X_s.iloc[:, top_idx].values,
            feature_names = [readable_sg(c) for c in X_s.columns[top_idx]],
        )
        shap_sg.plots.beeswarm(exp_top, max_display=15, show=False)
        fig = plt_sg.gcf()
        fig.suptitle(label, fontsize=11, y=1.01)
        return fig

    with st.spinner("Computing SHAP values for both subgroups..."):
        with shap_col1:
            st.pyplot(shap_beeswarm(1.0, 'Below Poverty Line'), bbox_inches='tight')
            plt_sg.close()
        with shap_col2:
            st.pyplot(shap_beeswarm(2.0, 'At or Above Poverty Line'), bbox_inches='tight')
            plt_sg.close()
