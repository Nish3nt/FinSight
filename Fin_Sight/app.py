"""
AutoML Debugger — Streamlit Application v2
==========================================
Three flaws fixed in this version:

  FIX 1 -> No multicollinearity detection
           -> VIF scores + feature correlation heatmap + problematic pairs table + LLM interpretation

  FIX 2 -> No statistical hypothesis testing
           -> Pearson r / t-test / ANOVA / chi-squared per feature + p-values + LLM interpretation

  FIX 3 -> Feature importance only from Random Forest (biased toward high-cardinality features)
           -> Mutual Information scores (model-free) shown alongside RF importance for cross-checking
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif

# must be FIRST Streamlit call
st.set_page_config(
    page_title="AutoML Debugger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.debugger_engine import run_debugger_pipeline

ANTHROPIC_API_KEY = "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"


def call_llm(prompt: str, max_tokens: int = 600) -> list:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        msg = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = "\n".join(raw.split("\n")[1:])
            raw = raw.replace("```", "").strip()
        if raw.lstrip().startswith("["):
            try:
                bullets = json.loads(raw)
                if isinstance(bullets, list):
                    return [str(b) for b in bullets]
            except Exception:
                pass
        lines = [
            ln.strip().lstrip("*-+.1234567890)").strip()
            for ln in raw.split("\n") if ln.strip()
        ]
        return [ln for ln in lines if ln]
    except Exception as e:
        return [f"LLM unavailable: {e}"]


def compute_vif(X_numeric: pd.DataFrame) -> pd.DataFrame:
    cols = X_numeric.columns.tolist()
    if len(cols) < 2:
        return pd.DataFrame({"Feature": cols, "VIF": [1.0] * len(cols)})
    vif_rows = []
    for col in cols:
        others = [c for c in cols if c != col]
        X_other = X_numeric[others].copy()
        y_col   = X_numeric[col].copy()
        mask    = ~(X_other.isna().any(axis=1) | y_col.isna())
        if mask.sum() < 10:
            vif_rows.append({"Feature": col, "VIF": float("nan")})
            continue
        try:
            reg = LinearRegression()
            reg.fit(X_other[mask], y_col[mask])
            r2  = min(max(float(reg.score(X_other[mask], y_col[mask])), 0.0), 0.9999)
            vif = round(1.0 / (1.0 - r2), 2)
        except Exception:
            vif = float("nan")
        vif_rows.append({"Feature": col, "VIF": vif})
    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def get_high_corr_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.75) -> pd.DataFrame:
    pairs = []
    cols  = corr_matrix.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            c = corr_matrix.iloc[i, j]
            if abs(c) >= threshold:
                pairs.append({
                    "Feature A":   cols[i],
                    "Feature B":   cols[j],
                    "Correlation": round(float(c), 4),
                    "Severity":    "🔴 High" if abs(c) >= 0.9 else "🟡 Moderate",
                })
    if not pairs:
        return pd.DataFrame()
    return pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False).reset_index(drop=True)


def run_statistical_tests(df: pd.DataFrame, target_col: str, task_type: str) -> pd.DataFrame:
    results = []
    X = df.drop(columns=[target_col])
    y = df[target_col]
    numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    for col in numeric_cols:
        series = X[col].copy()
        if task_type == "regression":
            y_num  = pd.to_numeric(y, errors="coerce")
            common = series.dropna().index.intersection(y_num.dropna().index)
            if len(common) < 5:
                continue
            try:
                r_val, p_val = stats.pearsonr(series[common].astype(float), y_num[common].astype(float))
                results.append({
                    "Feature": col, "Type": "Numeric", "Test": "Pearson r",
                    "Statistic": round(float(r_val), 4), "p-value": round(float(p_val), 4),
                    "Significant": "Yes" if p_val < 0.05 else "No",
                    "Interpretation": _interp(p_val, col),
                })
            except Exception:
                pass
        else:
            classes = y.dropna().unique()
            groups  = [X.loc[y == c, col].dropna() for c in classes if len(X.loc[y == c, col].dropna()) >= 3]
            if len(groups) < 2:
                continue
            try:
                if len(groups) == 2:
                    stat, p_val = stats.ttest_ind(groups[0], groups[1], equal_var=False)
                    test_name   = "Welch t-test"
                else:
                    stat, p_val = stats.f_oneway(*groups)
                    test_name   = "ANOVA F-test"
                results.append({
                    "Feature": col, "Type": "Numeric", "Test": test_name,
                    "Statistic": round(float(stat), 4), "p-value": round(float(p_val), 4),
                    "Significant": "Yes" if p_val < 0.05 else "No",
                    "Interpretation": _interp(p_val, col),
                })
            except Exception:
                pass

    for col in categorical_cols:
        try:
            ct = pd.crosstab(X[col].fillna("__NA__"), y.fillna("__NA__"))
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                continue
            chi2, p_val, dof, _ = stats.chi2_contingency(ct)
            results.append({
                "Feature": col, "Type": "Categorical", "Test": f"Chi-squared (dof={dof})",
                "Statistic": round(float(chi2), 4), "p-value": round(float(p_val), 4),
                "Significant": "Yes" if p_val < 0.05 else "No",
                "Interpretation": _interp(p_val, col),
            })
        except Exception:
            pass

    return pd.DataFrame(results) if results else pd.DataFrame()


def _interp(p: float, feature: str) -> str:
    if p < 0.001:
        return f"Very strong evidence '{feature}' relates to target (p<0.001)."
    elif p < 0.01:
        return f"Strong evidence '{feature}' is significant (p={p:.4f})."
    elif p < 0.05:
        return f"Moderate evidence '{feature}' is significant (p={p:.4f})."
    return f"No significant relationship for '{feature}' (p={p:.4f})."


def compute_mutual_info(X, y, task_type, numeric_features, categorical_features):
    X_enc = X[numeric_features].copy()
    for col in categorical_features:
        X_enc[col] = X[col].astype("category").cat.codes.astype(float)
    X_enc = X_enc.fillna(X_enc.median(numeric_only=True))
    if X_enc.empty:
        return pd.DataFrame()
    discrete_mask = [False] * len(numeric_features) + [True] * len(categorical_features)
    try:
        if task_type == "regression":
            y_num = pd.to_numeric(y, errors="coerce")
            y_num = y_num.fillna(y_num.median())
            mi    = mutual_info_regression(X_enc, y_num, discrete_features=discrete_mask, random_state=42)
        else:
            from sklearn.preprocessing import LabelEncoder
            le    = LabelEncoder()
            y_enc = le.fit_transform(y.astype(str))
            mi    = mutual_info_classif(X_enc, y_enc, discrete_features=discrete_mask, random_state=42)
    except Exception:
        return pd.DataFrame()
    mi_df = pd.DataFrame({"Feature": X_enc.columns.tolist(), "Mutual Information": mi})
    mi_df = mi_df.sort_values("Mutual Information", ascending=False).head(15).reset_index(drop=True)
    max_mi = mi_df["Mutual Information"].max()
    mi_df["MI_norm"] = (mi_df["Mutual Information"] / max_mi).round(4) if max_mi > 0 else 0.0
    return mi_df


# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"]          { background-color: #161b27; border-right:1px solid #2a2f45; }
h1,h2,h3 { font-family:'Segoe UI',sans-serif; }
.metric-card {
    background: linear-gradient(135deg,#1a1f35 0%,#1e2540 100%);
    border:1px solid #2e3555; border-radius:12px;
    padding:18px 22px; text-align:center;
    box-shadow:0 4px 15px rgba(0,0,0,0.3);
}
.metric-value { font-size:2rem; font-weight:700; color:#7eb6ff; }
.metric-label { font-size:0.8rem; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }
.analysis-bullet {
    background:#161b27; border-left:3px solid #7eb6ff;
    border-radius:6px; padding:12px 16px; margin-bottom:10px;
    font-size:0.95rem; line-height:1.5;
}
.health-bar-bg {
    background:#1a1f35; border-radius:8px; height:16px;
    overflow:hidden; border:1px solid #2e3555;
}
.pill { display:inline-block; padding:3px 12px; border-radius:20px; font-size:0.8rem; font-weight:600; margin:3px; }
.pill-blue  { background:#1a3a5c; color:#7eb6ff; border:1px solid #2a5080; }
.pill-green { background:#1a3a2c; color:#5dbc8a; border:1px solid #2a6040; }
.pill-red   { background:#3a1a1a; color:#e07070; border:1px solid #804040; }
.pill-amber { background:#3a2a1a; color:#e0b070; border:1px solid #806040; }
.section-tag {
    display:inline-block; background:#1a3a5c; color:#7eb6ff;
    border:1px solid #2a5080; border-radius:6px;
    padding:2px 10px; font-size:0.75rem; font-weight:700;
    letter-spacing:1px; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()
    st.markdown("### 📌 About")
    st.markdown("""
AutoML Debugger evaluates your dataset **before** you train any model.

**Diagnostics included:**
- Missing values & duplicates
- Outlier detection (IQR)
- Class imbalance detection
- Baseline + RF model metrics
- 5-fold cross-validation
- **🆕 Multicollinearity (VIF)**
- **🆕 Statistical significance tests**
- **🆕 Mutual Information scores**
- LLM expert commentary (Claude)
    """)
    st.divider()
    st.markdown("Built by **Nishant Diwate**")

# ── Header ────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([1, 8])
with c1:
    st.markdown("<h1 style='font-size:3rem;margin:0'>🧠</h1>", unsafe_allow_html=True)
with c2:
    st.markdown("<h1 style='margin:0;padding-top:10px'>AutoML Debugger</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8892a4;margin:0'>LLM-Assisted Dataset Diagnostics for ML Engineers</p>", unsafe_allow_html=True)
st.divider()

# ── Upload ────────────────────────────────────────────────────────────────────
FALLBACK_DATASET = Path("data/initial_dataset.csv")
cu, ci = st.columns([2, 1])
with cu:
    st.subheader("📂 Upload Dataset")
    uploaded_file = st.file_uploader("Upload CSV or use built-in sample", type=["csv"], label_visibility="collapsed")
with ci:
    st.markdown("**Formats:** CSV  \n**Min:** 10 rows, 2 cols  \n**Tip:** Last col = target by default")

df, source = None, None
if uploaded_file is not None:
    df, source = pd.read_csv(uploaded_file), "uploaded"
elif FALLBACK_DATASET.exists():
    df, source = pd.read_csv(FALLBACK_DATASET), "fallback"

if source == "uploaded":
    st.success(f"✅ Uploaded — {df.shape[0]:,} rows × {df.shape[1]} cols")
elif source == "fallback":
    st.info(f"ℹ️ Using built-in sample — {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── Preview & Target ──────────────────────────────────────────────────────────
target_column = None
if df is not None:
    with st.expander("🔍 Preview Dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    target_column = st.selectbox(
        "🎯 Select Target Column",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
    )
st.divider()

# ── Run ───────────────────────────────────────────────────────────────────────
run_clicked = st.button("🚀 Run AutoML Diagnostics", type="primary", use_container_width=True)

if run_clicked:
    if df is None:
        st.warning("No dataset available. Please upload a CSV file.")
        st.stop()

    with st.spinner("🔬 Running full ML diagnostics…"):
        output = run_debugger_pipeline(df, target_column, api_key=ANTHROPIC_API_KEY)

    metrics      = output.get("metrics", {})
    profile      = output.get("profile", {})
    feature_imp  = output.get("feature_importance", {})
    llm_analysis = output.get("llm_analysis", [])
    task_type    = output.get("task_type", "unknown")
    health_score = output.get("health_score", 0)
    diagnosis    = output.get("diagnosis", "")

    if not metrics:
        st.error(f"❌ {diagnosis}")
        for line in llm_analysis:
            st.warning(line)
        st.stop()

    # Working clean copies
    df_clean     = df.dropna(subset=[target_column]).copy()
    X_all        = df_clean.drop(columns=[target_column])
    y_all        = df_clean[target_column]
    numeric_feats = X_all.select_dtypes(include=[np.number]).columns.tolist()
    cat_feats     = X_all.select_dtypes(exclude=[np.number]).columns.tolist()

    # ── KPI row ───────────────────────────────────────────────────────────────
    st.markdown("---")
    def kpi(col, value, label):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )
    k1,k2,k3,k4,k5 = st.columns(5)
    kpi(k1, f"{metrics.get('rows',0):,}", "Rows")
    kpi(k2, str(metrics.get('columns',0)), "Columns")
    kpi(k3, f"{metrics.get('missing_values',0):,}", "Missing")
    kpi(k4, str(profile.get('duplicate_rows',0)), "Duplicates")
    kpi(k5, task_type.capitalize(), "Task Type")
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Health score ──────────────────────────────────────────────────────────
    if health_score >= 80:
        bar_color, verdict = "#5dbc8a", "🟢 Production-Ready"
    elif health_score >= 60:
        bar_color, verdict = "#e0b070", "🟡 Needs Minor Work"
    else:
        bar_color, verdict = "#e07070", "🔴 Significant Issues"
    st.subheader(f"⭐ Dataset Health Score — {verdict}")
    st.markdown(
        f'<div style="margin-top:12px"><div class="health-bar-bg">'
        f'<div style="width:{health_score}%;background:{bar_color};height:100%;border-radius:8px;"></div>'
        f'</div><p style="color:{bar_color};font-size:1.4rem;font-weight:700;margin-top:8px">'
        f'{health_score} / 100</p></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Pre-compute new sections ──────────────────────────────────────────────
    with st.spinner("Computing multicollinearity scores…"):
        X_num_only      = X_all[numeric_feats].copy() if numeric_feats else pd.DataFrame()
        vif_df          = compute_vif(X_num_only) if len(numeric_feats) >= 2 else pd.DataFrame()
        corr_matrix     = X_num_only.corr() if len(numeric_feats) >= 2 else pd.DataFrame()
        high_corr_pairs = get_high_corr_pairs(corr_matrix, 0.75) if not corr_matrix.empty else pd.DataFrame()

    with st.spinner("Running statistical significance tests…"):
        stat_df = run_statistical_tests(df_clean, target_column, task_type)

    with st.spinner("Computing Mutual Information scores…"):
        mi_df = compute_mutual_info(X_all, y_all, task_type, numeric_feats, cat_feats)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_metrics, tab_analysis, tab_features, tab_dq, tab_mc, tab_st = st.tabs([
        "📊 Model Metrics",
        "🧠 Expert Analysis",
        "🌲 Feature Importance",
        "🔍 Data Quality",
        "🔗 Multicollinearity",
        "📐 Statistical Tests",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — Model Metrics
    # ════════════════════════════════════════════════════════════════════
    with tab_metrics:
        st.subheader("Model Performance Metrics")
        st.caption(f"Auto-detected task: **{task_type}** | {diagnosis}")
        if task_type == "regression":
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("R² (Baseline)",      metrics.get("r2_baseline","—"))
            m2.metric("R² (Random Forest)", metrics.get("r2_rf","—"))
            m3.metric("MAE",                metrics.get("mae","—"))
            m4.metric("RMSE",               metrics.get("rmse","—"))
            m5.metric("CV R² (5-fold)",     f"{metrics.get('cv_r2_mean','—')} ± {metrics.get('cv_r2_std','—')}")
        else:
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Accuracy (Baseline)", metrics.get("accuracy_baseline","—"))
            m2.metric("Accuracy (RF)",       metrics.get("accuracy_rf","—"))
            m3.metric("F1 Score",            metrics.get("f1_score","—"))
            m4.metric("ROC-AUC",             metrics.get("roc_auc","—"))
            m5.metric("CV Acc (5-fold)",     f"{metrics.get('cv_accuracy_mean','—')} ± {metrics.get('cv_accuracy_std','—')}")
        st.divider()
        if task_type == "regression":
            raw = {
                "R² Score":     max(0.0, float(metrics.get("r2_rf", 0))),
                "Low MAE":      max(0.0, 1 - min(1.0, float(metrics.get("mae",0)) / max(float(df_clean[target_column].std()),1))),
                "Data Density": min(1.0, metrics.get("rows",0) / 10000),
                "Completeness": 1 - profile.get("missing_pct",0) / 100,
                "No Dupes":     1 - min(1.0, profile.get("duplicate_rows",0) / max(metrics.get("rows",1),1)),
            }
        else:
            raw = {
                "Accuracy":     float(metrics.get("accuracy_rf",0)),
                "F1 Score":     float(metrics.get("f1_score",0)),
                "ROC-AUC":      float(metrics.get("roc_auc", metrics.get("accuracy_rf",0))),
                "Completeness": 1 - profile.get("missing_pct",0) / 100,
                "No Dupes":     1 - min(1.0, profile.get("duplicate_rows",0) / max(metrics.get("rows",1),1)),
            }
        cats   = list(raw.keys())
        values = [max(0.0, min(1.0, v)) for v in raw.values()]
        fig_radar = go.Figure(go.Scatterpolar(
            r=values+[values[0]], theta=cats+[cats[0]],
            fill="toself", fillcolor="rgba(126,182,255,0.2)",
            line=dict(color="#7eb6ff", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1a1f35",
                radialaxis=dict(visible=True, range=[0,1], gridcolor="#2e3555", color="#8892a4"),
                angularaxis=dict(gridcolor="#2e3555", color="#e0e0e0"),
            ),
            showlegend=False, paper_bgcolor="#0e1117",
            margin=dict(l=60,r=60,t=40,b=40), height=360,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — Expert Analysis
    # ════════════════════════════════════════════════════════════════════
    with tab_analysis:
        st.caption("🤖 Analysis powered by **Claude (Anthropic)**")
        for point in llm_analysis:
            st.markdown(f'<div class="analysis-bullet">{point}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — Feature Importance + FIX 3 (Mutual Information)
    # ════════════════════════════════════════════════════════════════════
    with tab_features:
        if feature_imp:
            fi_df = pd.DataFrame({
                "Feature": list(feature_imp.keys()),
                "RF Importance": list(feature_imp.values()),
            }).sort_values("RF Importance", ascending=True)
            fig_fi = px.bar(
                fi_df, x="RF Importance", y="Feature", orientation="h",
                title="Top Features — Random Forest Importance",
                color="RF Importance", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig_fi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("RF importance could not be computed.")

        # FIX 3 — Mutual Information
        st.markdown('<span class="section-tag">🆕 FIX 3 — Model-Free Signal Check</span>', unsafe_allow_html=True)
        st.subheader("📡 Mutual Information vs RF Importance")
        st.markdown(
            "**RF importance** can be biased toward high-cardinality features. "
            "**Mutual Information** is completely model-free and captures any dependency (linear or non-linear). "
            "Large discrepancies between the two indicate potential over-rating or under-rating by RF."
        )
        if not mi_df.empty:
            if feature_imp:
                rf_norm = pd.DataFrame({
                    "Feature": list(feature_imp.keys()),
                    "RF Importance": list(feature_imp.values()),
                })
                max_rf = rf_norm["RF Importance"].max()
                rf_norm["RF_norm"] = (rf_norm["RF Importance"] / max_rf).round(4) if max_rf > 0 else 0.0
                compare = mi_df[["Feature","MI_norm"]].merge(rf_norm[["Feature","RF_norm"]], on="Feature", how="outer").fillna(0)
                compare = compare.sort_values("MI_norm", ascending=False).head(12)
                clong = compare.melt(id_vars="Feature", value_vars=["MI_norm","RF_norm"], var_name="Method", value_name="Score")
                clong["Method"] = clong["Method"].map({"MI_norm":"Mutual Information","RF_norm":"Random Forest"})
                fig_cmp = px.bar(
                    clong, x="Feature", y="Score", color="Method", barmode="group",
                    title="Feature Importance Comparison (normalised 0-1)",
                    template="plotly_dark",
                    color_discrete_map={"Mutual Information":"#7eb6ff","Random Forest":"#5dbc8a"},
                )
                fig_cmp.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                    xaxis=dict(gridcolor="#2e3555", tickangle=-30),
                    yaxis=dict(gridcolor="#2e3555"),
                    legend=dict(bgcolor="#1a1f35"),
                )
                st.plotly_chart(fig_cmp, use_container_width=True)

                big_disc = compare[(compare["MI_norm"] - compare["RF_norm"]).abs() > 0.25]
                if not big_disc.empty:
                    st.warning(
                        f"⚠️ **{len(big_disc)} feature(s)** show MI vs RF discrepancy > 0.25: "
                        f"**{', '.join(big_disc['Feature'].tolist())}** — RF may be biased here."
                    )
            else:
                fig_mi = px.bar(
                    mi_df.sort_values("MI_norm", ascending=True),
                    x="MI_norm", y="Feature", orientation="h",
                    title="Mutual Information (normalised)",
                    color="MI_norm", color_continuous_scale="Blues",
                    template="plotly_dark",
                )
                fig_mi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", coloraxis_showscale=False)
                st.plotly_chart(fig_mi, use_container_width=True)

            with st.spinner("Claude interpreting MI vs RF discrepancies…"):
                mi_bullets = call_llm(f"""Senior ML engineer reviewing feature importance comparison.
MI scores (top 8): {mi_df[['Feature','MI_norm']].head(8).to_dict('records')}
RF importances: {list(feature_imp.items())[:8] if feature_imp else 'N/A'}
Task: {task_type}
Return ONLY a JSON array of 4-5 bullet strings covering: agreement between methods, discrepancies, 
which features to trust, recommended feature selection actions. No preamble.""", 500)
            st.markdown("**🤖 Claude's Assessment:**")
            for b in mi_bullets:
                st.markdown(f'<div class="analysis-bullet">{b}</div>', unsafe_allow_html=True)
        else:
            st.info("Mutual Information could not be computed.")

        if profile.get("top_correlations"):
            st.subheader("📈 Pearson Correlation with Target")
            corr_df = pd.DataFrame({
                "Feature": list(profile["top_correlations"].keys()),
                "Correlation": list(profile["top_correlations"].values()),
            }).sort_values("Correlation", key=abs, ascending=True)
            fig_corr = px.bar(
                corr_df, x="Correlation", y="Feature", orientation="h",
                color="Correlation", color_continuous_scale="RdBu",
                color_continuous_midpoint=0, template="plotly_dark",
            )
            fig_corr.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", coloraxis_showscale=False)
            st.plotly_chart(fig_corr, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — Data Quality
    # ════════════════════════════════════════════════════════════════════
    with tab_dq:
        qa, qb = st.columns(2)
        with qa:
            st.subheader("🧩 Missing Values per Column")
            miss_s = df.isna().sum()
            miss_s = miss_s[miss_s > 0]
            if not miss_s.empty:
                miss_df2 = miss_s.reset_index()
                miss_df2.columns = ["Column","Missing Count"]
                fig_miss = px.bar(miss_df2, x="Missing Count", y="Column",
                    orientation="h", template="plotly_dark",
                    color="Missing Count", color_continuous_scale="Reds")
                fig_miss.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", coloraxis_showscale=False)
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ No missing values!")
        with qb:
            st.subheader("📊 Target Distribution")
            if task_type == "classification":
                vc = df_clean[target_column].value_counts().reset_index()
                vc.columns = ["Class","Count"]
                fig_dist = px.pie(vc, names="Class", values="Count",
                    template="plotly_dark", color_discrete_sequence=px.colors.sequential.Blues_r)
            else:
                fig_dist = px.histogram(df_clean, x=target_column,
                    template="plotly_dark", color_discrete_sequence=["#7eb6ff"], nbins=40)
            fig_dist.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
            st.plotly_chart(fig_dist, use_container_width=True)

        if profile.get("outlier_counts"):
            st.subheader("⚠️ Outlier Summary (IQR)")
            out_df2 = pd.DataFrame({
                "Feature": list(profile["outlier_counts"].keys()),
                "Outlier Count": list(profile["outlier_counts"].values()),
            }).sort_values("Outlier Count", ascending=False)
            st.dataframe(out_df2, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No significant outliers!")

        st.subheader("🏷️ Data Quality Flags")
        flags = []
        if profile.get("constant_features"):
            flags.append(f'<span class="pill pill-red">⚠️ {len(profile["constant_features"])} constant feature(s)</span>')
        if profile.get("high_cardinality_cols"):
            flags.append(f'<span class="pill pill-amber">⚠️ {len(profile["high_cardinality_cols"])} high-cardinality col(s)</span>')
        if profile.get("duplicate_rows",0) > 0:
            flags.append(f'<span class="pill pill-amber">⚠️ {profile["duplicate_rows"]} duplicate rows</span>')
        if profile.get("missing_pct",0) > 10:
            flags.append(f'<span class="pill pill-red">⚠️ {profile["missing_pct"]}% missing values</span>')
        if profile.get("imbalance_ratio") and profile["imbalance_ratio"] > 3:
            flags.append(f'<span class="pill pill-amber">⚠️ Imbalance {profile["imbalance_ratio"]}:1</span>')
        if flags:
            st.markdown(" ".join(flags), unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-green">✅ No critical flags</span>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 5 — Multicollinearity (FIX 1)
    # ════════════════════════════════════════════════════════════════════
    with tab_mc:
        st.markdown('<span class="section-tag">🆕 FIX 1 — Multicollinearity Detection</span>', unsafe_allow_html=True)
        st.subheader("🔗 Multicollinearity Analysis")
        st.markdown("""
**Why this matters:** When two features are highly correlated *with each other*, the model cannot
reliably determine which one drives the target. This makes coefficients unstable and feature importances
misleading. The previous version only checked feature-*target* correlation. This tab checks
feature-*feature* correlation.

**VIF guide:** < 5 = OK · 5–10 = moderate concern · > 10 = high multicollinearity (consider removing)
        """)

        if len(numeric_feats) < 2:
            st.info("At least 2 numeric features needed for multicollinearity analysis.")
        else:
            cv1, cv2 = st.columns([1, 2])

            with cv1:
                st.subheader("📋 VIF Scores")
                if not vif_df.empty:
                    def color_vif(row):
                        v = row["VIF"]
                        if pd.isna(v):     return ["", "color: #8892a4"]
                        if v > 10:         return ["", "color: #e07070; font-weight:700"]
                        if v > 5:          return ["", "color: #e0b070; font-weight:600"]
                        return ["", "color: #5dbc8a"]
                    st.dataframe(
                        vif_df.style.apply(color_vif, axis=1),
                        use_container_width=True, hide_index=True,
                    )
                    high_vif = vif_df[vif_df["VIF"] > 10]
                    mod_vif  = vif_df[(vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)]
                    if not high_vif.empty:
                        st.error(f"🔴 Critical VIF > 10: {', '.join(high_vif['Feature'].tolist())}")
                    if not mod_vif.empty:
                        st.warning(f"🟡 Moderate VIF 5-10: {', '.join(mod_vif['Feature'].tolist())}")
                    if high_vif.empty and mod_vif.empty:
                        st.success("✅ All VIF < 5 — no multicollinearity concern.")
                else:
                    st.info("VIF could not be computed.")

            with cv2:
                st.subheader("🌡️ Feature Correlation Heatmap")
                if not corr_matrix.empty:
                    show_cols = corr_matrix.columns.tolist()[:20]
                    cm_sub    = corr_matrix.loc[show_cols, show_cols]
                    fig_heat  = go.Figure(go.Heatmap(
                        z=cm_sub.values, x=cm_sub.columns.tolist(), y=cm_sub.index.tolist(),
                        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                        text=np.round(cm_sub.values, 2),
                        texttemplate="%{text}", textfont={"size": 9},
                        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
                    ))
                    fig_heat.update_layout(
                        paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                        font=dict(color="#e0e0e0"),
                        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
                        yaxis=dict(tickfont=dict(size=10)),
                        height=450, margin=dict(l=10,r=10,t=20,b=80),
                    )
                    st.plotly_chart(fig_heat, use_container_width=True)

            st.subheader("⚡ Highly Correlated Pairs (|r| ≥ 0.75)")
            if not high_corr_pairs.empty:
                st.dataframe(high_corr_pairs, use_container_width=True, hide_index=True)
            else:
                st.success("✅ No feature pairs exceed the 0.75 correlation threshold.")

            st.subheader("🤖 Claude's Multicollinearity Assessment")
            with st.spinner("Claude analysing multicollinearity…"):
                mc_bullets = call_llm(f"""Senior ML engineer reviewing multicollinearity report.
VIF scores (top 10): {vif_df.head(10).to_dict('records') if not vif_df.empty else []}
High-correlation pairs (|r|>=0.75): {high_corr_pairs.head(5).to_dict('records') if not high_corr_pairs.empty else []}
Numeric features count: {len(numeric_feats)}
Task type: {task_type}
Return ONLY a JSON array of 5-6 bullet strings covering: which features have dangerous VIF,
which correlated pairs to address, how to fix (drop/PCA/domain knowledge), impact on model.
No preamble.""", 600)
            for b in mc_bullets:
                st.markdown(f'<div class="analysis-bullet">{b}</div>', unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 6 — Statistical Tests (FIX 2)
    # ════════════════════════════════════════════════════════════════════
    with tab_st:
        st.markdown('<span class="section-tag">🆕 FIX 2 — Statistical Hypothesis Testing</span>', unsafe_allow_html=True)
        st.subheader("📐 Per-Feature Statistical Significance Tests")
        st.markdown(f"""
**What this does:** Runs proper hypothesis tests on every feature to determine if it has a
statistically significant relationship with the target variable.

| Scenario | Test Used |
|---|---|
| Numeric feature + Regression target | Pearson correlation + p-value |
| Numeric feature + Binary classification | Welch t-test (unequal variance) |
| Numeric feature + Multi-class | One-way ANOVA F-test |
| Categorical feature (any task) | Chi-squared test of independence |

**Significance threshold:** p < 0.05
        """)

        if stat_df.empty:
            st.info("Statistical tests could not be run on this dataset.")
        else:
            n_sig = (stat_df["Significant"] == "Yes").sum()
            n_not = (stat_df["Significant"] == "No").sum()
            st.markdown(
                f'<span class="pill pill-green">✅ {n_sig} significant features (p < 0.05)</span>'
                f'<span class="pill pill-red">❌ {n_not} non-significant features</span>',
                unsafe_allow_html=True,
            )
            st.markdown("<br>", unsafe_allow_html=True)

            stat_display = stat_df.sort_values(
                ["Significant","p-value"], ascending=[False, True]
            ).reset_index(drop=True)

            # Rename for display
            disp_cols = ["Feature","Type","Test","Statistic","p-value","Significant"]
            st.dataframe(stat_display[disp_cols], use_container_width=True, hide_index=True)

            # -log10(p) bar chart
            st.subheader("📉 Feature Significance Chart (-log10 p-value)")
            st.caption("Bars crossing the dashed orange line are statistically significant (p < 0.05)")
            pvc = stat_display[["Feature","p-value","Significant"]].copy()
            pvc["-log10(p)"] = -np.log10(pvc["p-value"].clip(lower=1e-10))
            fig_pv = px.bar(
                pvc.sort_values("-log10(p)", ascending=True),
                x="-log10(p)", y="Feature", orientation="h",
                color="Significant",
                color_discrete_map={"Yes":"#5dbc8a","No":"#e07070"},
                template="plotly_dark",
            )
            fig_pv.add_vline(
                x=1.301, line_dash="dash", line_color="#e0b070",
                annotation_text="p=0.05", annotation_font_color="#e0b070",
                annotation_position="top right",
            )
            fig_pv.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                xaxis=dict(gridcolor="#2e3555"),
                yaxis=dict(gridcolor="#2e3555"),
                legend=dict(bgcolor="#1a1f35"),
                height=max(300, len(stat_display) * 28),
            )
            st.plotly_chart(fig_pv, use_container_width=True)

            sig_rows = stat_display[stat_display["Significant"] == "Yes"]
            if not sig_rows.empty:
                with st.expander("📋 Interpretation for Significant Features", expanded=False):
                    for _, row in sig_rows.iterrows():
                        st.markdown(
                            f'<div class="analysis-bullet">'
                            f'<b>{row["Feature"]}</b> ({row["Test"]}) — '
                            f'stat={row["Statistic"]}, p={row["p-value"]} · '
                            f'{row.get("Interpretation","")}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.subheader("🤖 Claude's Statistical Assessment")
            with st.spinner("Claude interpreting significance results…"):
                sig_list  = stat_display[stat_display["Significant"]=="Yes"][["Feature","Test","p-value"]].head(10).to_dict("records")
                nsig_list = stat_display[stat_display["Significant"]=="No"]["Feature"].tolist()
                st_bullets = call_llm(f"""Senior ML engineer reviewing hypothesis test results.
Task type: {task_type}
Total features tested: {len(stat_df)}
SIGNIFICANT features (p<0.05): {json.dumps(sig_list)}
NON-significant features: {nsig_list}
Return ONLY a JSON array of 5-6 bullet strings covering: strongest predictors, whether to drop
non-significant features, overfitting risk, feature selection recommendations, statistical caveats.
No preamble.""", 600)
            for b in st_bullets:
                st.markdown(f'<div class="analysis-bullet">{b}</div>', unsafe_allow_html=True)

    st.divider()
    st.caption("AutoML Debugger v2 · Streamlit · scikit-learn · Plotly · Anthropic Claude")
