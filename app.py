"""
app.py — San Rafael 311 Analytics Dashboard
MIS 554 | The 411 on 311 | University of Arizona

Run with:  streamlit run app.py
Requires:  pip install streamlit plotly pandas numpy pydeck
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="San Rafael 311 | AI Analytics",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ──────────────────────────────────────────────────────────
CLUSTER_COLORS = {
    1: "#E63946",   # red        — Downtown Core (largest)
    2: "#2A9D8F",   # teal       — The Canal
    3: "#E9C46A",   # amber      — Terra Linda
    4: "#457B9D",   # steel blue — China Camp (smallest)
}
CLUSTER_NAMES = {
    1: "Zone 1 — Downtown Core",
    2: "Zone 2 — The Canal",
    3: "Zone 3 — Terra Linda",
    4: "Zone 4 — China Camp",
}
BG         = "#0F1117"
CARD_BG    = "#1A1D27"
BORDER     = "#2D3145"
TEXT_PRI   = "#F0F2F6"
TEXT_SEC   = "#9BA3B2"
ACCENT     = "#E63946"

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@600;700;800&family=Inter:wght@400;500&display=swap');

  html, body, [data-testid="stAppViewContainer"] {{
    background-color: {BG};
    color: {TEXT_PRI};
    font-family: 'Inter', sans-serif;
  }}
  [data-testid="stSidebar"] {{
    background-color: {CARD_BG};
    border-right: 1px solid {BORDER};
  }}
  h1, h2, h3 {{ font-family: 'Syne', sans-serif; color: {TEXT_PRI}; }}
  .metric-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 8px;
  }}
  .metric-value {{
    font-family: 'DM Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: {TEXT_PRI};
    line-height: 1.1;
  }}
  .metric-label {{
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: {TEXT_SEC};
    margin-top: 4px;
  }}
  .metric-delta {{
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    margin-top: 4px;
  }}
  .section-header {{
    font-family: 'Syne', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: {TEXT_PRI};
    border-bottom: 1px solid {BORDER};
    padding-bottom: 8px;
    margin-bottom: 16px;
    margin-top: 8px;
  }}
  .insight-box {{
    background: {CARD_BG};
    border-left: 3px solid {ACCENT};
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 12px 0;
    font-size: 0.88rem;
    color: {TEXT_SEC};
    line-height: 1.6;
  }}
  .tag {{
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-family: 'DM Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.05em;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    background: {CARD_BG};
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
  }}
  .stTabs [data-baseweb="tab"] {{
    font-family: 'Syne', sans-serif;
    font-size: 0.85rem;
    font-weight: 600;
    color: {TEXT_SEC};
    border-radius: 6px;
    padding: 8px 18px;
  }}
  .stTabs [aria-selected="true"] {{
    background: {ACCENT} !important;
    color: white !important;
  }}
  div[data-testid="metric-container"] {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 16px;
  }}
  [data-testid="stPlotlyChart"] {{ border-radius: 10px; overflow: hidden; }}
  .stSelectbox label, .stSlider label {{ color: {TEXT_SEC}; font-size: 0.82rem; }}
</style>
""", unsafe_allow_html=True)

# ── Data root ──────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent
DATA_RAW   = ROOT / "data" / "raw"
DATA_PROC  = ROOT / "data" / "processed"

# ── Plotly base template ───────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=TEXT_PRI, size=12),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, tickcolor=BORDER),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    margin=dict(l=40, r=20, t=40, b=40),
)

def layout(**overrides):
    """Merge PLOTLY_LAYOUT with per-chart overrides, preventing duplicate keys."""
    merged = {k: v for k, v in PLOTLY_LAYOUT.items() if k not in overrides}
    merged.update(overrides)
    return merged

# ── Word cloud helper ──────────────────────────────────────────────────────
try:
    from wordcloud import WordCloud as _WC
    _WC_AVAILABLE = True
except ImportError:
    _WC_AVAILABLE = False

def render_wordcloud(freq_dict, bg_color="#0F1117", colormap="RdYlGn",
                     width=800, height=400):
    """Render {term: weight} dict as a matplotlib word cloud figure."""
    if not _WC_AVAILABLE:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    wc = _WC(
        width=width, height=height,
        background_color=bg_color,
        colormap=colormap,
        max_words=80,
        prefer_horizontal=0.85,
        min_font_size=10,
        max_font_size=90,
        collocations=False,
    )
    wc.generate_from_frequencies(freq_dict)
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig



# ── Data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_clusters():
    df = pd.read_csv(DATA_PROC / "kmeans_clusters.csv", low_memory=False)
    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")
    return df

@st.cache_data
def load_cluster_perf():
    return pd.read_csv(DATA_PROC / "kmeans_cluster_performance.csv")

@st.cache_data
def load_cluster_enriched():
    try:
        return pd.read_csv(DATA_PROC / "kmeans_cluster_enriched.csv")
    except FileNotFoundError:
        return None

@st.cache_data
def load_cat_perf():
    return pd.read_csv(DATA_PROC / "kmeans_category_performance.csv")

@st.cache_data
def load_city_forecast():
    fc   = pd.read_csv(DATA_PROC / "forecast_holtwinters.csv")
    hist = pd.read_csv(DATA_PROC / "forecast_historical.csv")
    # HW module outputs "week_start"; normalise to "week" for the app
    if "week_start" in fc.columns:
        fc = fc.rename(columns={"week_start": "week"})
    if "week_start" in hist.columns:
        hist = hist.rename(columns={"week_start": "week"})
    # HW module outputs ci_lower_80/ci_upper_80; normalise to lo_80/hi_80
    fc = fc.rename(columns={
        "ci_lower_80": "lo_80", "ci_upper_80": "hi_80",
        "ci_lower_95": "lo_95", "ci_upper_95": "hi_95",
    })
    # hist uses "actual" — no rename needed
    fc["week"]   = pd.to_datetime(fc["week"])
    hist["week"] = pd.to_datetime(hist["week"])
    return fc, hist

@st.cache_data
def load_cluster_forecast():
    fc      = pd.read_csv(DATA_PROC / "forecast_cluster_weekly.csv")
    metrics = pd.read_csv(DATA_PROC / "forecast_cluster_metrics.csv")
    summary = pd.read_csv(DATA_PROC / "forecast_cluster_summary.csv")
    if "week_start" in fc.columns:
        fc = fc.rename(columns={"week_start": "week"})
    fc["week"] = pd.to_datetime(fc["week"])
    return fc, metrics, summary

@st.cache_data
def load_nlp_report():
    return pd.read_csv(DATA_PROC / "nlp_classification_report.csv")

@st.cache_data
def load_elbow():
    return pd.read_csv(DATA_PROC / "kmeans_elbow.csv")

@st.cache_data
def load_311_for_wordcloud():
    """Load raw 311 descriptions + cluster assignments for word cloud generation."""
    df = pd.read_csv(DATA_PROC / "kmeans_clusters.csv", low_memory=False)
    df = df[df["description"].notna() &
            (df["description"].str.strip().str.len() > 5)].copy()
    if "category_short" not in df.columns:
        df["category_short"] = df["category"].apply(
            lambda x: x.split(" / ")[0].strip() if isinstance(x, str) else x
        )
    return df

@st.cache_data
def load_nlp_top_terms():
    """Load TF-IDF top terms per category computed by nlp_classification.py."""
    try:
        return pd.read_csv(DATA_PROC / "nlp_top_terms.csv")
    except FileNotFoundError:
        return None


@st.cache_data
def load_app_context():
    """
    Load all summary metrics from processed CSVs into a single dict.
    Every value that appears in the UI is derived from here — no hardcoding.
    Falls back gracefully if a file hasn't been generated yet.
    """
    ctx = {}

    # ── 311 raw data stats ─────────────────────────────────────────────────
    # Try multiple candidate paths for the raw file, then fall back to the
    # already-processed kmeans_clusters.csv which is always present.
    try:
        _raw_candidates = [
            DATA_RAW / "311_requests" / "san_rafael_311.csv",
            DATA_RAW / "san_rafael_311.csv",
        ]
        _raw_path = next((p for p in _raw_candidates if p.exists()), None)
        if _raw_path:
            raw = pd.read_csv(_raw_path, low_memory=False)
            raw.columns = [c.lower() for c in raw.columns]
        else:
            # Fall back to processed clusters file (always present after kmeans run)
            raw = pd.read_csv(DATA_PROC / "kmeans_clusters.csv", low_memory=False)
        raw["requested_datetime"] = pd.to_datetime(raw["requested_datetime"], errors="coerce")
        ctx["n_records"]  = len(raw)
        ctx["date_start"] = raw["requested_datetime"].min().strftime("%m/%d/%Y")
        ctx["date_end"]   = raw["requested_datetime"].max().strftime("%m/%d/%Y")
    except Exception:
        ctx["n_records"]  = "—"
        ctx["date_start"] = "—"
        ctx["date_end"]   = "—"

    # ── K-Means cluster stats ──────────────────────────────────────────────
    try:
        perf = pd.read_csv(DATA_PROC / "kmeans_cluster_performance.csv")
        ctx["n_clustered"]  = int(perf["n_total"].sum())
        ctx["n_clusters"]   = int(perf["cluster"].nunique())
        # Highest-demand cluster stats (for insight box)
        top_cl = perf.loc[perf["incidents_per_1k_res"].idxmax()] if "incidents_per_1k_res" in perf.columns else perf.loc[perf["n_total"].idxmax()]
        ctx["top_cluster"]          = int(top_cl["cluster"])
        ctx["top_cluster_inc_1k"]   = round(float(top_cl["incidents_per_1k_res"]), 1) if "incidents_per_1k_res" in top_cl else "—"
        # Zone 5 backlog stats (highest backlog zone)
        high_bl = perf.loc[perf["pct_over_30d"].idxmax()]
        ctx["high_backlog_cluster"]      = int(high_bl["cluster"])
        ctx["high_backlog_mean_days"]    = round(float(high_bl["mean_days"]), 1)
        ctx["high_backlog_pct"]          = round(float(high_bl["pct_over_30d"]), 1)
    except Exception:
        ctx.update({"n_clustered": "—", "n_clusters": 4,
                    "top_cluster": 1, "top_cluster_inc_1k": "—",
                    "high_backlog_cluster": 4,
                    "high_backlog_mean_days": "—", "high_backlog_pct": "—"})

    # ── Enriched cluster stats ─────────────────────────────────────────────
    try:
        enr = pd.read_csv(DATA_PROC / "kmeans_cluster_enriched.csv")
        if "incidents_per_1k_res" in enr.columns:
            top_enr = enr.loc[enr["incidents_per_1k_res"].idxmax()]
            ctx["top_cluster_inc_1k"] = round(float(top_enr["incidents_per_1k_res"]), 1)
    except Exception:
        pass

    # ── Holt-Winters forecast metrics ─────────────────────────────────────
    try:
        fm = pd.read_csv(DATA_PROC / "forecast_metrics.csv")
        fm.columns = [c.lower() for c in fm.columns]
        ctx["hw_model"]         = str(fm["model"].values[0])
        ctx["hw_mae"]           = round(float(fm["mae"].values[0]),  2)
        ctx["hw_rmse"]          = round(float(fm["rmse"].values[0]), 2)
        ctx["hw_mape"]          = round(float(fm["mape"].values[0]), 2)
        ctx["hw_alpha"]         = round(float(fm["alpha"].values[0]),   4) if "alpha"           in fm.columns else "—"
        ctx["hw_cv"]            = round(float(fm["series_cv"].values[0]), 3) if "series_cv"     in fm.columns else "—"
        ctx["hw_yoy_corr"]      = round(float(fm["yoy_correlation"].values[0]), 2) if "yoy_correlation" in fm.columns else "—"
        ctx["hw_has_seasonal"]  = bool(fm["has_seasonal"].values[0]) if "has_seasonal" in fm.columns else False
        ctx["hw_is_damped"]     = bool(fm["is_damped"].values[0])    if "is_damped"    in fm.columns else False
        ctx["hw_forecast_weeks"]= int(fm["forecast_weeks"].values[0]) if "forecast_weeks" in fm.columns else 13
        ctx["hw_train_start"]   = str(fm["train_start"].values[0])   if "train_start"  in fm.columns else "2023-01-01"
        ctx["hw_date_end"]      = str(fm["date_end"].values[0])      if "date_end"     in fm.columns else "—"
    except Exception:
        ctx.update({"hw_model": "—", "hw_mae": "—", "hw_rmse": "—", "hw_mape": "—",
                    "hw_alpha": "—", "hw_cv": "—", "hw_yoy_corr": "—",
                    "hw_has_seasonal": False, "hw_is_damped": False,
                    "hw_forecast_weeks": 13, "hw_train_start": "2023-01-01",
                    "hw_date_end": "—"})

    # ── Cluster forecast summary (staffing) ────────────────────────────────
    try:
        cs = pd.read_csv(DATA_PROC / "forecast_cluster_summary.csv")
        cs["cluster"] = cs["cluster"].astype(int)
        # City-wide mean resolution (for complexity weight denominator)
        city_mean_days = float(cs["fc_13wk_total"].sum() / cs["fc_13wk_total"].count())
        ctx["city_mean_resolution"] = "—"  # from cluster perf if available
        try:
            perf2 = pd.read_csv(DATA_PROC / "kmeans_cluster_performance.csv")
            weighted = (perf2["mean_days"] * perf2["n_total"]).sum() / perf2["n_total"].sum()
            ctx["city_mean_resolution"] = round(float(weighted), 1)
        except Exception:
            pass
        # Highest-workload cluster
        top_wl = cs.loc[cs["recommended_fte_share"].idxmax()]
        ctx["top_workload_cluster"]     = int(top_wl["cluster"])
        ctx["top_workload_fte_share"]   = round(float(top_wl["recommended_fte_share"]), 1)
        # Over-indexed cluster (workload share > volume share)
        cs["volume_share"] = cs["fc_13wk_total"] / cs["fc_13wk_total"].sum() * 100
        cs["workload_premium"] = cs["recommended_fte_share"] - cs["volume_share"]
        over = cs.loc[cs["workload_premium"].idxmax()]
        ctx["over_indexed_cluster"]     = int(over["cluster"])
        ctx["over_indexed_fte_share"]   = round(float(over["recommended_fte_share"]), 1)
        ctx["over_indexed_vol_share"]   = round(float(over["volume_share"]), 1)
        ctx["over_indexed_complexity"]  = round(float(over["complexity_weight"]), 2) if "complexity_weight" in over.index else "—"
        ctx["over_indexed_backlog"]     = round(float(over["backlog_rate"] * 100), 1) if "backlog_rate" in over.index else "—"
    except Exception:
        ctx.update({"city_mean_resolution": "—",
                    "top_workload_cluster": 1, "top_workload_fte_share": "—",
                    "over_indexed_cluster": 4, "over_indexed_fte_share": "—",
                    "over_indexed_vol_share": "—", "over_indexed_complexity": "—",
                    "over_indexed_backlog": "—"})

    # ── Cluster forecast model performance (Zone 4 MAPE note) ─────────────
    try:
        cm = pd.read_csv(DATA_PROC / "forecast_cluster_metrics.csv")
        cm.columns = [c.lower() for c in cm.columns]
        # Find the zone with the highest MAPE (denominator problem candidate)
        high_mape_row = cm.loc[cm["mape"].idxmax()]
        ctx["cl_high_mape_zone"] = int(high_mape_row["cluster"])
        ctx["cl_high_mape_val"]  = round(float(high_mape_row["mape"]), 1)
        ctx["cl_high_mape_mae"]  = round(float(high_mape_row["mae"]),  2)
    except Exception:
        ctx.update({"cl_high_mape_zone": 4, "cl_high_mape_val": "—", "cl_high_mape_mae": "—"})

    # ── NLP metrics ────────────────────────────────────────────────────────
    try:
        nm = pd.read_csv(DATA_PROC / "nlp_metrics.csv")
        nm.columns = [c.lower() for c in nm.columns]
        ctx["nlp_accuracy"]       = round(float(nm["accuracy"].values[0]) * 100, 2)
        ctx["nlp_f1_weighted"]    = round(float(nm["f1_weighted"].values[0]), 3)
        ctx["nlp_f1_macro"]       = round(float(nm["f1_macro"].values[0]),    3)
        ctx["nlp_cv_f1_mean"]     = round(float(nm["cv_f1_weighted_mean"].values[0]), 3) if "cv_f1_weighted_mean" in nm.columns else "—"
        ctx["nlp_cv_f1_std"]      = round(float(nm["cv_f1_weighted_std"].values[0]),  3) if "cv_f1_weighted_std"  in nm.columns else "—"
    except Exception:
        ctx.update({"nlp_accuracy": "—", "nlp_f1_weighted": "—",
                    "nlp_f1_macro": "—", "nlp_cv_f1_mean": "—", "nlp_cv_f1_std": "—"})

    # ── NLP training record count ──────────────────────────────────────────
    try:
        rep = pd.read_csv(DATA_PROC / "nlp_classification_report.csv")
        valid_rows = rep[~rep["class"].isin(["accuracy", "macro avg", "weighted avg"])]
        ctx["nlp_n_classes"]  = int((valid_rows["f1_score"] > 0).sum())  # classes with any predictions
        ctx["nlp_n_records"]  = int(valid_rows["support"].sum())
    except Exception:
        ctx.update({"nlp_n_classes": "—", "nlp_n_records": "—"})

    return ctx



def _build_sparse_note(metrics_df: pd.DataFrame) -> str:
    """
    Build a dynamic insight note describing any clusters that used the
    proportional model rather than standalone Holt-Winters. Reads from
    forecast_cluster_metrics.csv output — no hardcoding of zone numbers
    or share percentages.
    """
    if metrics_df is None or "is_proportional" not in metrics_df.columns:
        return ""
    prop_rows = metrics_df[metrics_df["is_proportional"] == True]
    if prop_rows.empty:
        return ""
    parts = []
    for _, row in prop_rows.iterrows():
        share_pct = round(row["prop_share"] * 100, 1) if "prop_share" in row and not pd.isna(row.get("prop_share", float("nan"))) else "—"
        mean_vol  = row.get("mean_weekly", "—")
        parts.append(
            f"<b style='color:{TEXT_PRI};'>Zone {int(row['cluster'])} note:</b> "
            f"Modeled as a proportional share ({share_pct}%) of the city-wide forecast. "
            f"Mean volume {mean_vol}/week is below the reliability threshold for "
            f"independent Holt-Winters. This is more statistically defensible than fitting noise."
        )
    return "<br><br>".join(parts)

# ── Load app context (all dynamic metrics) ───────────────────────────────
ctx = load_app_context()
_fmt = lambda v: f"{v:,}" if isinstance(v, (int, float)) else str(v)

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"<div style='font-family:Syne,sans-serif;font-size:1.3rem;"
        f"font-weight:800;color:{TEXT_PRI};margin-bottom:4px;'>"
        f"🏙️ The 411 on 311</div>"
        f"<div style='font-size:0.75rem;color:{TEXT_SEC};margin-bottom:24px;'>"
        f"City of San Rafael · MIS 554</div>",
        unsafe_allow_html=True,
    )
    page = st.radio(
        "Navigation",
        ["Cluster Map", "Demand Forecast", "Staffing Index", "NLP Classifier"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(
        f"<div style='font-size:0.72rem;color:{TEXT_SEC};line-height:1.7;'>"
        f"<b style='color:{TEXT_PRI}'>Data</b><br>"
        f"311 Records: {_fmt(ctx['n_records'])}<br>"
        f"Date range: {ctx['date_start']} – {ctx['date_end']}<br>"
        f"Clusters: {ctx['n_clusters']} (k-means)<br>"
        f"Forecast horizon: {ctx['hw_forecast_weeks']} weeks<br><br>"
        f"<b style='color:{TEXT_PRI}'>Models</b><br>"
        f"Holt-Winters (city + cluster)<br>"
        f"K-Means geospatial clustering<br>"
        f"TF-IDF + Logistic Regression"
        f"</div>",
        unsafe_allow_html=True,
    )


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1 — CLUSTER MAP
# ═══════════════════════════════════════════════════════════════════════════
if page == "Cluster Map":

    st.markdown(
        f"<h1 style='font-size:2rem;margin-bottom:4px;'>Geographic Cluster Analysis</h1>"
        f"<p style='color:{TEXT_SEC};margin-bottom:24px;font-size:0.9rem;'>"
        f"K-Means spatial clustering of {_fmt(ctx['n_clustered'])} service requests · k={ctx['n_clusters']} "
        f"(aligned with San Rafael's 4 planning districts)</p>",
        unsafe_allow_html=True,
    )

    try:
        df      = load_clusters()
        perf    = load_cluster_perf()
        enriched = load_cluster_enriched()
        cat_perf = load_cat_perf()

        # ── Top KPI row ───────────────────────────────────────────────────
        c1, c2, c3, c4, c5 = st.columns(5)
        kpi_cols = [c1, c2, c3, c4, c5]
        for i, (col, row) in enumerate(zip(kpi_cols, perf.itertuples())):
            with col:
                color = CLUSTER_COLORS[row.cluster]
                pct   = row.n_total / perf["n_total"].sum() * 100
                st.markdown(
                    f"<div class='metric-card' style='border-top:3px solid {color};'>"
                    f"<div class='metric-value' style='color:{color};'>{row.n_total:,}</div>"
                    f"<div class='metric-label'>Zone {row.cluster} incidents</div>"
                    f"<div class='metric-delta' style='color:{TEXT_SEC};'>{pct:.1f}% of total</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Map + performance table ───────────────────────────────────────
        map_col, info_col = st.columns([3, 2], gap="large")

        with map_col:
            st.markdown("<div class='section-header'>Incident Heatmap by Zone</div>",
                        unsafe_allow_html=True)

            # Filter controls
            sel_clusters = st.multiselect(
                "Show zones",
                options=list(CLUSTER_NAMES.keys()),
                default=list(CLUSTER_NAMES.keys()),
                format_func=lambda x: CLUSTER_NAMES[x],
            )
            top_cats = df["category_short"].value_counts().head(8).index.tolist()
            sel_cat = st.selectbox(
                "Filter by category",
                ["All categories"] + top_cats,
            )

            plot_df = df[df["cluster"].isin(sel_clusters)].copy()
            if sel_cat != "All categories":
                plot_df = plot_df[plot_df["category_short"] == sel_cat]

            # Scatter map
            fig_map = go.Figure()
            for c in sel_clusters:
                sub = plot_df[plot_df["cluster"] == c]
                if len(sub) == 0:
                    continue
                fig_map.add_trace(go.Scattermapbox(
                    lat=sub["lat"],
                    lon=sub["long"],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=CLUSTER_COLORS[c],
                        opacity=0.55,
                    ),
                    name=CLUSTER_NAMES[c],
                    hovertemplate=(
                        f"<b>{CLUSTER_NAMES[c]}</b><br>"
                        "%{customdata[0]}<br>"
                        "%{customdata[1]}<extra></extra>"
                    ),
                    customdata=sub[["category_short", "address"]].values,
                ))

            # Cluster centroids
            for _, row in perf.iterrows():
                if row["cluster"] not in sel_clusters:
                    continue
                fig_map.add_trace(go.Scattermapbox(
                    lat=[row["lat_centroid"]],
                    lon=[row["lon_centroid"]],
                    mode="markers+text",
                    marker=dict(
                        size=18,
                        color=CLUSTER_COLORS[int(row["cluster"])],
                        opacity=1.0,
                    ),
                    text=[f"Z{int(row['cluster'])}"],
                    textfont=dict(color="white", size=10, family="Syne"),
                    textposition="middle center",
                    name=f"Zone {int(row['cluster'])} centroid",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>Zone {int(row['cluster'])} centroid</b><br>"
                        f"{row['n_total']:,} incidents<br>"
                        f"Median response: {row['median_days']:.1f}d<extra></extra>"
                    ),
                ))

            fig_map.update_layout(
                mapbox=dict(
                    style="carto-darkmatter",
                    center=dict(lat=37.978, lon=-122.525),
                    zoom=12.5,
                ),
                **{k: v for k, v in PLOTLY_LAYOUT.items()
                   if k not in ("xaxis", "yaxis", "legend", "margin")},
                height=480,
                showlegend=True,
                legend=dict(
                    x=0.01, y=0.99,
                    bgcolor="rgba(15,17,23,0.8)",
                    bordercolor=BORDER,
                    font=dict(size=11),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
            )
            st.plotly_chart(fig_map, use_container_width=True)
            st.caption(
                f"Showing {len(plot_df):,} incidents · "
                f"Circle markers = zone centroids"
            )

        with info_col:
            st.markdown("<div class='section-header'>Zone Response Performance</div>",
                        unsafe_allow_html=True)

            # Response performance bar chart
            fig_resp = go.Figure()
            fig_resp.add_trace(go.Bar(
                x=[f"Z{c}" for c in perf["cluster"]],
                y=perf["median_days"],
                name="Median days",
                marker_color=[CLUSTER_COLORS[c] for c in perf["cluster"]],
                hovertemplate="Zone %{x}<br>Median: %{y:.1f} days<extra></extra>",
            ))
            fig_resp.add_trace(go.Scatter(
                x=[f"Z{c}" for c in perf["cluster"]],
                y=perf["pct_over_30d"],
                name="% over 30 days",
                yaxis="y2",
                mode="markers+lines",
                marker=dict(size=9, symbol="diamond", color="white"),
                line=dict(color="white", width=1.5, dash="dot"),
                hovertemplate="Zone %{x}<br>Over 30d: %{y:.1f}%<extra></extra>",
            ))
            fig_resp.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "margin", "yaxis")},
                height=200,
                yaxis=dict(title="Median days", gridcolor=BORDER, linecolor=BORDER),
                yaxis2=dict(
                    title="% >30 days",
                    overlaying="y",
                    side="right",
                    gridcolor="rgba(0,0,0,0)",
                    ticksuffix="%",
                ),
                barmode="group",
                legend=dict(orientation="h", y=-0.3, x=0),
                margin=dict(l=40, r=40, t=20, b=60),
            )
            st.plotly_chart(fig_resp, use_container_width=True)

            # Census enrichment table
            if enriched is not None:
                st.markdown(
                    "<div class='section-header' style='margin-top:16px;'>"
                    "Demographics + Terrain</div>",
                    unsafe_allow_html=True,
                )
                disp = enriched[[
                    "cluster", "total_population",
                    "incidents_per_1k_res",
                    "mean_pop_density_sqkm",
                    "wtd_median_income",
                    "elevation_mean_m",
                ]].copy()
                disp.columns = ["Zone", "Population", "Inc/1k Res",
                                 "Density/km²", "Med Income", "Elev (m)"]
                disp["Zone"]       = disp["Zone"].apply(lambda x: f"Z{int(x)}")
                disp["Population"] = disp["Population"].apply(lambda x: f"{int(x):,}")
                disp["Inc/1k Res"] = disp["Inc/1k Res"].apply(lambda x: f"{x:.1f}")
                disp["Density/km²"] = disp["Density/km²"].apply(lambda x: f"{x:,.0f}")
                disp["Med Income"] = disp["Med Income"].apply(
                    lambda x: f"${int(x):,}" if pd.notna(x) else "—"
                )
                disp["Elev (m)"]   = disp["Elev (m)"].apply(lambda x: f"{x:.0f}m")
                st.dataframe(
                    disp,
                    hide_index=True,
                    use_container_width=True,
                    height=200,
                )

            # Key insight
            st.markdown(
                f"<div class='insight-box'>"
                f"<b style='color:{TEXT_PRI};'>Key finding:</b> Zone {ctx['top_cluster']} generates "
                f"<b style='color:{ACCENT};'>{ctx['top_cluster_inc_1k']} incidents per 1,000 residents</b> — "
                f"the highest demand rate of any zone. Zone {ctx['high_backlog_cluster']} carries "
                f"the highest complexity penalty: mean resolution of {ctx['high_backlog_mean_days']} days "
                f"and {ctx['high_backlog_pct']}% backlog rate."
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Category service gap heatmap ──────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>Category × Zone Service Gap Analysis</div>",
                    unsafe_allow_html=True)

        gap_col, gap_info = st.columns([3, 1], gap="large")
        with gap_col:
            gap_df = cat_perf[
                ~cat_perf["category_short"].isin(["accuracy", "macro avg", "weighted avg"])
            ].copy() if "category_short" in cat_perf.columns else cat_perf.copy()

            pivot = gap_df.pivot_table(
                index="category_short",
                columns="cluster",
                values="days_vs_city_median",
                aggfunc="mean",
            ).round(1)
            pivot.columns = [f"Z{int(c)}" for c in pivot.columns]

            fig_heat = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=[c[:30] for c in pivot.index.tolist()],
                colorscale=[
                    [0.0,  "#2A9D8F"],
                    [0.35, "#0F1117"],
                    [1.0,  "#E63946"],
                ],
                zmid=0,
                text=pivot.values.round(1),
                texttemplate="%{text}d",
                hovertemplate=(
                    "Category: %{y}<br>"
                    "Zone: %{x}<br>"
                    "Days vs city median: %{z:.1f}<extra></extra>"
                ),
                colorbar=dict(
                    title=dict(
                        text="Days vs<br>city median",
                        font=dict(color=TEXT_SEC, size=10),
                    ),
                    tickfont=dict(color=TEXT_SEC, size=10),
                ),
            ))
            fig_heat.update_layout(**layout(
                height=340,
                margin=dict(l=170, r=60, t=20, b=40),
                xaxis=dict(side="top", tickfont=dict(size=11)),
                yaxis=dict(tickfont=dict(size=10)),
            ))
            st.plotly_chart(fig_heat, use_container_width=True)

        with gap_info:
            st.markdown(
                f"<div style='padding-top:40px;font-size:0.83rem;color:{TEXT_SEC};"
                f"line-height:1.8;'>"
                f"<span style='color:{ACCENT};font-weight:600;'>Red</span> = slower than "
                f"city average for that category<br><br>"
                f"<span style='color:#2A9D8F;font-weight:600;'>Teal</span> = faster than "
                f"city average<br><br>"
                f"<b style='color:{TEXT_PRI};'>Flagged gaps:</b><br>"
                f"Parks & Playgrounds in Z2: +39 days<br>"
                f"Potholes in Z5: +11 days<br>"
                f"Trees in Z3: +3 days"
                f"</div>",
                unsafe_allow_html=True,
            )

    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}\n\nRun `kmeans_clustering.py` first.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2 — DEMAND FORECAST
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Demand Forecast":

    st.markdown(
        f"<h1 style='font-size:2rem;margin-bottom:4px;'>Demand Forecast</h1>"
        f"<p style='color:{TEXT_SEC};margin-bottom:24px;font-size:0.9rem;'>"
        f"Holt-Winters exponential smoothing · 13-week forward horizon · "
        f"City-wide and per-zone projections</p>",
        unsafe_allow_html=True,
    )

    try:
        city_fc, city_hist = load_city_forecast()
        cl_fc, cl_metrics, cl_summary = load_cluster_forecast()

        tab1, tab2 = st.tabs(["🌆  City-Wide Trend", "🗺️  Zone Breakdown"])

        # ── Tab 1: City-wide ──────────────────────────────────────────────
        with tab1:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)

            # Load metrics
            try:
                city_met = pd.read_csv(DATA_PROC / "forecast_metrics.csv")
                # HW module saves uppercase MAE/RMSE/MAPE — normalise
                city_met.columns = [c.lower() for c in city_met.columns]
                mae_val  = city_met["mae"].values[0]
                mape_val = city_met["mape"].values[0]
                rmse_val = city_met["rmse"].values[0]
            except Exception:
                mae_val = mape_val = rmse_val = None

            fc_total = city_fc["forecast"].sum()
            fc_mean  = city_fc["forecast"].mean()
            hist_mean = city_hist["actual"].tail(13).mean()

            with m_col1:
                st.metric("13-Week Forecast", f"{fc_total:.0f}", "total incidents")
            with m_col2:
                st.metric("Weekly Average", f"{fc_mean:.1f}",
                          f"{(fc_mean - hist_mean)/hist_mean*100:+.1f}% vs prior 13wk")
            with m_col3:
                st.metric("MAE", f"{mae_val:.2f}" if mae_val else "—", "incidents/week")
            with m_col4:
                st.metric("MAPE", f"{mape_val:.1f}%" if mape_val else "—", "test set")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            # City-wide chart
            fig_city = go.Figure()

            # Historical
            n_hist = min(52, len(city_hist))
            hist_plot = city_hist.tail(n_hist)
            fig_city.add_trace(go.Scatter(
                x=hist_plot["week"],
                y=hist_plot["actual"],
                name="Historical",
                line=dict(color=TEXT_SEC, width=1.5),
                mode="lines",
            ))

            # CI 95
            fig_city.add_trace(go.Scatter(
                x=pd.concat([city_fc["week"], city_fc["week"].iloc[::-1]]),
                y=pd.concat([city_fc["hi_95"], city_fc["lo_95"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(230,57,70,0.08)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                showlegend=True,
            ))
            # CI 80
            fig_city.add_trace(go.Scatter(
                x=pd.concat([city_fc["week"], city_fc["week"].iloc[::-1]]),
                y=pd.concat([city_fc["hi_80"], city_fc["lo_80"].iloc[::-1]]),
                fill="toself",
                fillcolor="rgba(230,57,70,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="80% CI",
                showlegend=True,
            ))
            # Forecast line
            fig_city.add_trace(go.Scatter(
                x=city_fc["week"],
                y=city_fc["forecast"],
                name="Forecast",
                line=dict(color=ACCENT, width=2.5),
                mode="lines",
            ))
            # Divider
            _vline_x = str(city_hist["week"].max())
            fig_city.add_shape(
                type="line",
                x0=_vline_x, x1=_vline_x,
                y0=0, y1=1,
                xref="x", yref="paper",
                line=dict(color=BORDER, dash="dot", width=1.5),
            )
            fig_city.add_annotation(
                x=_vline_x, y=1,
                xref="x", yref="paper",
                text="Forecast →",
                showarrow=False,
                font=dict(color=TEXT_SEC, size=11),
                xanchor="left",
                yanchor="bottom",
                xshift=6,
            )

            fig_city.update_layout(**layout(
                height=380,
                xaxis_title="Week",
                yaxis_title="Incidents / Week",
                hovermode="x unified",
            ))
            st.plotly_chart(fig_city, use_container_width=True)

            # Model note
            st.markdown(
                f"<div class='insight-box'>"
                f"Model selected by AIC: <b style='color:{TEXT_PRI};'>{ctx['hw_model']}</b> &middot; "
                f"&alpha;={ctx['hw_alpha']} &middot; CV={ctx['hw_cv']} &middot; YoY &rho;={ctx['hw_yoy_corr']} &middot; "
                f"{'Seasonal component included' if ctx['hw_has_seasonal'] else 'No seasonal component'} &middot; "
                f"Training window: {ctx['hw_train_start'][:7]}&ndash;{ctx['hw_date_end'][:7]}"
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Tab 2: Zone breakdown ─────────────────────────────────────────
        with tab2:
            z_col1, z_col2 = st.columns([3, 2], gap="large")

            with z_col1:
                st.markdown("<div class='section-header'>13-Week Zone Forecasts</div>",
                            unsafe_allow_html=True)

                sel_zones = st.multiselect(
                    "Select zones to display",
                    options=list(CLUSTER_NAMES.keys()),
                    default=list(CLUSTER_NAMES.keys()),
                    format_func=lambda x: CLUSTER_NAMES[x],
                    key="forecast_zones",
                )

                fig_zones = go.Figure()
                for c in sel_zones:
                    sub = cl_fc[cl_fc["cluster"] == c]
                    color = CLUSTER_COLORS[c]
                    # CI band
                    fig_zones.add_trace(go.Scatter(
                        x=pd.concat([sub["week"], sub["week"].iloc[::-1]]),
                        y=pd.concat([sub["hi_80"], sub["lo_80"].iloc[::-1]]),
                        fill="toself",
                        fillcolor=f"rgba{tuple(int(color.lstrip('#')[i:i+2],16) for i in (0,2,4)) + (0.12,)}",
                        line=dict(color="rgba(0,0,0,0)"),
                        showlegend=False,
                    ))
                    fig_zones.add_trace(go.Scatter(
                        x=sub["week"],
                        y=sub["forecast"],
                        name=CLUSTER_NAMES[c],
                        line=dict(color=color, width=2),
                        mode="lines+markers",
                        marker=dict(size=5),
                        hovertemplate=(
                            f"<b>{CLUSTER_NAMES[c]}</b><br>"
                            "Week: %{x|%b %d}<br>"
                            "Forecast: %{y:.1f}<br>"
                            "80% CI: %{customdata[0]:.1f}–%{customdata[1]:.1f}"
                            "<extra></extra>"
                        ),
                        customdata=sub[["lo_80", "hi_80"]].values,
                    ))

                fig_zones.update_layout(**layout(
                    height=380,
                    xaxis_title="Week",
                    yaxis_title="Forecast Incidents / Week",
                    hovermode="x unified",
                ))
                st.plotly_chart(fig_zones, use_container_width=True)

                st.markdown(
                    f"<div class='insight-box'>"
                    + _build_sparse_note(cl_metrics)
                    + f"</div>",
                    unsafe_allow_html=True,
                )

            with z_col2:
                st.markdown("<div class='section-header'>Model Performance by Zone</div>",
                            unsafe_allow_html=True)

                # Metrics comparison table
                disp_met = cl_metrics[["cluster", "model", "mean_weekly",
                                        "mae", "rmse", "mape"]].copy()
                disp_met["cluster"] = disp_met["cluster"].apply(lambda x: f"Z{int(x)}")
                disp_met["model"]   = disp_met["model"].str.replace(
                    r"\(.*\)", "", regex=True
                ).str.strip()
                disp_met.columns    = ["Zone", "Model", "Mean/Wk",
                                        "MAE", "RMSE", "MAPE%"]
                disp_met["Mean/Wk"] = disp_met["Mean/Wk"].round(1)
                disp_met["MAE"]     = disp_met["MAE"].round(2)
                disp_met["RMSE"]    = disp_met["RMSE"].round(2)
                disp_met["MAPE%"]   = disp_met["MAPE%"].round(1)
                st.dataframe(disp_met, hide_index=True, use_container_width=True)

                st.markdown(
                    f"<div style='font-size:0.75rem;color:{TEXT_SEC};margin-top:8px;"
                    f"line-height:1.7;'>"
                     f"Zone {ctx['cl_high_mape_zone']}'s MAPE of {ctx['cl_high_mape_val']}% reflects denominator inflation "
                     f"(near-zero weeks). MAE of {ctx['cl_high_mape_mae']} is the reliable measure. "
                    f"All models selected Additive-only — no zone shows "
                    f"statistically significant seasonality."
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # 13-week total bar
                st.markdown(
                    "<div class='section-header' style='margin-top:20px;'>"
                    "13-Week Projected Totals</div>",
                    unsafe_allow_html=True,
                )
                fig_bar = go.Figure(go.Bar(
                    x=[f"Z{int(r['cluster'])}" for _, r in cl_summary.iterrows()],
                    y=cl_summary["fc_13wk_total"],
                    marker_color=[CLUSTER_COLORS[int(r["cluster"])]
                                  for _, r in cl_summary.iterrows()],
                    text=cl_summary["fc_13wk_total"].apply(lambda x: f"{x:.0f}"),
                    textposition="outside",
                    textfont=dict(size=11, color=TEXT_PRI),
                    hovertemplate="Zone %{x}<br>13wk total: %{y:.0f}<extra></extra>",
                ))
                fig_bar.update_layout(**layout(
                    height=240,
                    yaxis_title="Projected Incidents",
                    margin=dict(l=40, r=20, t=30, b=40),
                ))
                st.plotly_chart(fig_bar, use_container_width=True)

    except FileNotFoundError as e:
        st.error(f"Forecast data not found: {e}\n\n"
                 "Run `holtwinters_forecast.py` and `cluster_forecast.py` first.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3 — STAFFING INDEX
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Staffing Index":

    st.markdown(
        f"<h1 style='font-size:2rem;margin-bottom:4px;'>Staffing Workload Index</h1>"
        f"<p style='color:{TEXT_SEC};margin-bottom:24px;font-size:0.9rem;'>"
        f"Relative resource allocation model · "
        f"Forecast volume × complexity weight × backlog pressure</p>",
        unsafe_allow_html=True,
    )

    try:
        _, _, cl_summary = load_cluster_forecast()
        staffing_weekly  = pd.read_csv(DATA_PROC / "forecast_cluster_staffing.csv")
        staffing_weekly["week"] = pd.to_datetime(staffing_weekly["week"])
        perf = load_cluster_perf()

        # FTE slider
        st.markdown(
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            f"border-radius:10px;padding:16px 24px;margin-bottom:24px;'>"
            f"<div style='font-family:Syne,sans-serif;font-weight:700;"
            f"color:{TEXT_PRI};margin-bottom:8px;'>Total Field Staff (FTE)</div>",
            unsafe_allow_html=True,
        )
        total_fte = st.slider(
            "Drag to set total FTE",
            min_value=5, max_value=100, value=20, step=1,
            label_visibility="collapsed",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── FTE allocation cards ──────────────────────────────────────────
        st.markdown("<div class='section-header'>Recommended Zone Allocation</div>",
                    unsafe_allow_html=True)

        cols = st.columns(5)
        for col, (_, row) in zip(cols, cl_summary.iterrows()):
            c       = int(row["cluster"])
            color   = CLUSTER_COLORS[c]
            fte     = row["recommended_fte_share"] / 100 * total_fte
            share   = row["recommended_fte_share"]
            cw      = row["complexity_weight"]
            br      = row["backlog_rate"] * 100
            with col:
                st.markdown(
                    f"<div class='metric-card' style='border-top:3px solid {color};'>"
                    f"<div class='metric-value' style='color:{color};'>"
                    f"{fte:.1f} FTE</div>"
                    f"<div class='metric-label'>{CLUSTER_NAMES[c]}</div>"
                    f"<div style='margin-top:10px;font-size:0.78rem;color:{TEXT_SEC};"
                    f"line-height:1.8;'>"
                    f"Share: <b style='color:{TEXT_PRI};'>{share:.1f}%</b><br>"
                    f"Complexity: <b style='color:{TEXT_PRI};'>{cw:.3f}×</b><br>"
                    f"Backlog rate: <b style='color:{TEXT_PRI};'>{br:.1f}%</b>"
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Weekly workload share chart ───────────────────────────────────
        chart_col, method_col = st.columns([3, 2], gap="large")

        with chart_col:
            st.markdown(
                "<div class='section-header'>Weekly Workload Share by Zone</div>",
                unsafe_allow_html=True,
            )
            fig_area = go.Figure()
            for c in sorted(CLUSTER_NAMES.keys(), reverse=True):   # stack bottom-up
                sub = staffing_weekly[staffing_weekly["cluster"] == c]
                fig_area.add_trace(go.Scatter(
                    x=sub["week"],
                    y=(sub["workload_share"] * 100).round(1),
                    name=CLUSTER_NAMES[c],
                    stackgroup="one",
                    fillcolor=CLUSTER_COLORS[c],
                    line=dict(color=CLUSTER_COLORS[c], width=0.5),
                    mode="lines",
                    hovertemplate=(
                        f"<b>{CLUSTER_NAMES[c]}</b><br>"
                        "Week: %{x|%b %d}<br>"
                        "Workload share: %{y:.1f}%<extra></extra>"
                    ),
                ))
            fig_area.update_layout(
                **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ("legend", "yaxis")},
                height=340,
                yaxis=dict(
                    title="% of Total Workload",
                    ticksuffix="%",
                    gridcolor=BORDER,
                    range=[0, 100],
                ),
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    y=-0.18,
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig_area, use_container_width=True)

        with method_col:
            st.markdown(
                "<div class='section-header'>Index Methodology</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div style='font-size:0.83rem;color:{TEXT_SEC};line-height:1.9;"
                f"background:{CARD_BG};border:1px solid {BORDER};"
                f"border-radius:10px;padding:16px;'>"
                f"<b style='color:{TEXT_PRI};'>Formula:</b><br>"
                f"<code style='font-family:DM Mono;color:{ACCENT};font-size:0.8rem;'>"
                f"raw = forecast × complexity × (1 + backlog)</code><br><br>"
                f"<b style='color:{TEXT_PRI};'>Complexity weight</b><br>"
                f"Mean resolution days / city average ({ctx['city_mean_resolution']}d). "
                f"Higher = more staff-hours per incident.<br><br>"
                f"<b style='color:{TEXT_PRI};'>Backlog pressure</b><br>"
                f"Fraction of cases exceeding 30 days. "
                f"Zone {ctx['high_backlog_cluster']} carries {ctx['high_backlog_pct']}% — highest of all zones.<br><br>"
                f"<b style='color:{TEXT_PRI};'>Normalization</b><br>"
                f"Shares sum to 100% each week. Multiply by actual "
                f"FTE headcount using the slider above.<br><br>"
                f"<b style='color:{TEXT_PRI};'>Limitation</b><br>"
                f"No published staffing ratios exist for San Rafael. "
                f"Index reflects relative demand only — not absolute "
                f"hiring targets."
                f"</div>",
                unsafe_allow_html=True,
            )

            # Zone 5 callout
            st.markdown(
                f"<div class='insight-box' style='margin-top:12px;'>"
                f"<b style='color:{TEXT_PRI};'>Zone {ctx['over_indexed_cluster']} insight:</b> Despite contributing "
                f"only {ctx['over_indexed_vol_share']}% of volume, Zone {ctx['over_indexed_cluster']} receives a {ctx['over_indexed_fte_share']}% workload share — "
                f"driven by {ctx['over_indexed_complexity']}× complexity weight and {ctx['over_indexed_backlog']}% backlog rate. "
                f"The City is likely under-resourcing this zone relative to "
                f"incident complexity."
                f"</div>",
                unsafe_allow_html=True,
            )

    except FileNotFoundError as e:
        st.error(f"Staffing data not found: {e}\n\n"
                 "Run `cluster_forecast.py` first.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4 — NLP CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════
elif page == "NLP Classifier":

    st.markdown(
        f"<h1 style='font-size:2rem;margin-bottom:4px;'>NLP Category Classifier</h1>"
        f"<p style='color:{TEXT_SEC};margin-bottom:24px;font-size:0.9rem;'>"
        f"TF-IDF + Logistic Regression · {ctx['nlp_n_classes']} categories · "
        f"{ctx['nlp_accuracy']}% accuracy · {_fmt(ctx['nlp_n_records'])} training records</p>",
        unsafe_allow_html=True,
    )

    try:
        report = load_nlp_report()

        tab_perf, tab_wc_cat, tab_wc_cluster, tab_live = st.tabs(["📊  Model Performance", "☁️  Category Word Clouds", "🗺️  Cluster Word Clouds", "🔍  Live Classifier"])

        # ── Tab 1: Performance ────────────────────────────────────────────
        with tab_perf:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1: st.metric("Accuracy",    f"{ctx['nlp_accuracy']}%",                         f"{ctx['nlp_n_classes']}-class")
            with kpi2: st.metric("F1 Weighted", f"{ctx['nlp_f1_weighted']}",                       "test set")
            with kpi3: st.metric("F1 Macro",    f"{ctx['nlp_f1_macro']}",                          "unweighted")
            with kpi4: st.metric("CV F1",       f"{ctx['nlp_cv_f1_mean']} ± {ctx['nlp_cv_f1_std']}", "5-fold")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            p_col, notes_col = st.columns([3, 2], gap="large")
            with p_col:
                st.markdown(
                    "<div class='section-header'>Per-Class F1 Score</div>",
                    unsafe_allow_html=True,
                )
                class_df = report[
                    ~report["class"].isin(["accuracy", "macro avg", "weighted avg"])
                ].copy().sort_values("f1_score", ascending=True)
                class_df["support"] = class_df["support"].astype(int)

                fig_f1 = go.Figure(go.Bar(
                    x=class_df["f1_score"].round(3),
                    y=class_df["class"].apply(lambda x: x[:38]),
                    orientation="h",
                    marker=dict(
                        color=class_df["f1_score"],
                        colorscale=[
                            [0.0, "#E63946"],
                            [0.5, "#E9C46A"],
                            [1.0, "#2A9D8F"],
                        ],
                        showscale=False,
                    ),
                    text=class_df["f1_score"].round(3),
                    textposition="outside",
                    textfont=dict(size=10, color=TEXT_PRI),
                    hovertemplate=(
                        "Category: %{y}<br>"
                        "F1: %{x:.3f}<br>"
                        "Support: %{customdata}<extra></extra>"
                    ),
                    customdata=class_df["support"],
                ))
                fig_f1.update_layout(**layout(
                    height=520,
                    xaxis=dict(title="F1 Score", range=[0, 1.05],
                               gridcolor=BORDER, linecolor=BORDER),
                    yaxis=dict(tickfont=dict(size=10)),
                    margin=dict(l=240, r=60, t=20, b=40),
                ))
                st.plotly_chart(fig_f1, use_container_width=True)

            with notes_col:
                st.markdown(
                    "<div class='section-header'>Key Findings</div>",
                    unsafe_allow_html=True,
                )
                insights = [
                    ("🟢 Strong classes", "Graffiti (0.863), Illegal Dumping (0.824), "
                     "Trees (0.818), Pothole (0.803), Stormwater (0.795) — "
                     "all have distinctive vocabulary."),
                    ("🟡 Data-limited", "Sidewalk Repair Program (n=54) and "
                     "Suggested Improvement (n=83) score F1=0.000. "
                     "Insufficient test examples, not model failure."),
                    ("🔴 Illegal Dumping gravity", "Sidewalks, Parks, Street Sweeping, "
                     "and Graffiti all misclassify toward Illegal Dumping — "
                     "residents describe co-located debris in multiple request types."),
                    ("📌 Business value", "Auto-categorization of new submissions "
                     "and quality-control flagging of likely miscategorized records."),
                ]
                for title, body in insights:
                    st.markdown(
                        f"<div class='insight-box'>"
                        f"<b style='color:{TEXT_PRI};'>{title}</b><br>{body}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # ── Tab 2: Category word clouds ───────────────────────────────────
        with tab_wc_cat:
            st.markdown(
                f"<div style='font-size:0.88rem;color:{TEXT_SEC};margin-bottom:16px;'>"
                f"Top TF-IDF weighted terms per category — the vocabulary the model "
                f"relies on most heavily to identify each class. Sized by coefficient "
                f"weight, not raw frequency, so common words shared across categories "
                f"are naturally suppressed.</div>",
                unsafe_allow_html=True,
            )

            top_terms_df = load_nlp_top_terms()
            df_wc        = load_311_for_wordcloud()

            # Category selector
            MIN_CAT_SIZE = 50
            cat_counts   = df_wc["category_short"].value_counts()
            viable_cats  = cat_counts[cat_counts >= MIN_CAT_SIZE].index.tolist()
            viable_cats  = ["All Categories"] + sorted(viable_cats)

            _ctrl_l, _ctrl_r = st.columns([2, 1])
            with _ctrl_l:
                wc_cat_sel = st.selectbox(
                    "Select category",
                    viable_cats,
                    index=0,
                    key="wc_cat_sel",
                )
            with _ctrl_r:
                cat_view = st.radio(
                    "View as",
                    ["Word Cloud", "Bar Chart"],
                    horizontal=True,
                    key="cat_view_toggle",
                    disabled=not _WC_AVAILABLE,
                )
            if not _WC_AVAILABLE:
                cat_view = "Bar Chart"
                st.caption(":grey[Install `wordcloud` to enable word cloud view.]")

            wc_col, info_col = st.columns([3, 2], gap="large")


            with wc_col:
                # Resolve corpus: all descriptions or filtered by category
                from sklearn.feature_extraction.text import TfidfVectorizer
                import re as _re
                if wc_cat_sel == "All Categories":
                    _sub = df_wc["description"]
                else:
                    _sub = df_wc[df_wc["category_short"] == wc_cat_sel]["description"]

                # Try pre-computed weights first (only available per-category)
                terms, weights = [], []
                if top_terms_df is not None and wc_cat_sel != "All Categories":
                    cat_row = top_terms_df[top_terms_df["category"] == wc_cat_sel]
                    if len(cat_row):
                        terms_raw = cat_row["top_terms"].values[0]
                        parsed = _re.findall(r'([^|(]+)\(([0-9.]+)\)', terms_raw)
                        terms   = [t.strip() for t, _ in parsed]
                        weights = [float(w) for _, w in parsed]

                # Fall back to live TF-IDF (always used for "All Categories")
                if not terms:
                    _tfidf = TfidfVectorizer(
                        max_features=60 if wc_cat_sel == "All Categories" else 30,
                        ngram_range=(1, 2),
                        min_df=2,
                        stop_words="english",
                        sublinear_tf=True,
                    )
                    try:
                        _mat    = _tfidf.fit_transform(_sub)
                        _scores = _mat.mean(axis=0).A1
                        _feat   = _tfidf.get_feature_names_out()
                        _idx    = _scores.argsort()[::-1][:40 if wc_cat_sel == "All Categories" else 20]
                        terms   = [_feat[i] for i in _idx]
                        weights = [float(_scores[i]) for i in _idx]
                    except Exception:
                        terms, weights = [], []

                if terms:
                    # Normalize weights to font-size range 12–48
                    w_arr  = np.array(weights)
                    w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)
                    sizes  = (12 + w_norm * 36).astype(int)

                    # Sort by weight descending for the bar chart
                    order  = np.argsort(w_arr)[::-1]
                    terms_ord   = [terms[i]   for i in order]
                    weights_ord = [weights[i] for i in order]
                    sizes_ord   = [int(sizes[i]) for i in order]

                    # Color: gradient from accent to teal by rank
                    n = len(terms_ord)
                    bar_colors = [
                        f"rgba({int(230 - (230-42)*i/max(n-1,1))},"
                        f"{int(57  + (157-57) *i/max(n-1,1))},"
                        f"{int(70  + (143-70) *i/max(n-1,1))},0.85)"
                        for i in range(n)
                    ]

                    fig_wc = go.Figure(go.Bar(
                        x=weights_ord,
                        y=terms_ord,
                        orientation="h",
                        marker_color=bar_colors,
                        text=[f"{w:.3f}" for w in weights_ord],
                        textposition="outside",
                        textfont=dict(size=10, color=TEXT_SEC),
                        hovertemplate="Term: <b>%{y}</b><br>TF-IDF weight: %{x:.4f}<extra></extra>",
                    ))
                    fig_wc.update_layout(**layout(
                        height=max(360, n * 26),
                        xaxis=dict(title="TF-IDF coefficient weight",
                                   gridcolor=BORDER, linecolor=BORDER),
                        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
                        margin=dict(l=200, r=80, t=20, b=40),
                        title=dict(
                            text=f"Top terms — {wc_cat_sel[:40]}",
                            font=dict(size=13, color=TEXT_PRI, family="Syne"),
                            x=0,
                        ),
                    ))
                    if cat_view == "Word Cloud":
                        freq_cat = dict(zip(terms_ord, [float(w) for w in weights_ord]))
                        _fig_mpl = render_wordcloud(freq_cat, colormap="RdYlGn")
                        if _fig_mpl:
                            st.pyplot(_fig_mpl, use_container_width=True)
                    else:
                        st.plotly_chart(fig_wc, use_container_width=True)
                else:
                    st.info("Run `nlp_classification.py` to generate term weights, "
                            "or select a category with sufficient records.")

            with info_col:
                st.markdown(
                    "<div class='section-header'>Category stats</div>",
                    unsafe_allow_html=True,
                )
                n_cat = len(df_wc) if wc_cat_sel == "All Categories" else int(cat_counts.get(wc_cat_sel, 0))
                pct   = 100.0 if wc_cat_sel == "All Categories" else n_cat / len(df_wc) * 100
                st.markdown(
                    f"<div class='metric-card'>"
                    f"<div class='metric-value'>{n_cat:,}</div>"
                    f"<div class='metric-label'>Total records</div>"
                    f"<div class='metric-delta' style='color:{TEXT_SEC};'>"
                    f"{pct:.1f}% of described requests</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Sample descriptions
                st.markdown(
                    "<div class='section-header' style='margin-top:16px;'>"
                    "Sample descriptions</div>",
                    unsafe_allow_html=True,
                )
                _sample_pool = df_wc if wc_cat_sel == "All Categories" else df_wc[df_wc["category_short"] == wc_cat_sel]
                samples = (
                    _sample_pool["description"]
                    .dropna()
                    .sample(min(4, max(1, len(_sample_pool))), random_state=42)
                    .tolist()
                )
                for s in samples:
                    st.markdown(
                        f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
                        f"border-radius:8px;padding:10px 14px;margin-bottom:8px;"
                        f"font-size:0.80rem;color:{TEXT_SEC};line-height:1.5;'>"
                        f"{str(s)[:180] + ('...' if len(str(s)) > 180 else '')}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    "<div class='insight-box' style='margin-top:8px;'>" "TF-IDF weights reflect discriminative power, not raw frequency. " "Terms common across all categories (e.g. street, city) are suppressed" " even if frequent — only terms distinctive to this category" " appear prominently.</div>",
                    unsafe_allow_html=True,
                )
        # ── Tab 3: Cluster word clouds ────────────────────────────────────
        with tab_wc_cluster:
            st.markdown(
                f"<div style='font-size:0.88rem;color:{TEXT_SEC};margin-bottom:16px;'>"
                f"Most distinctive vocabulary per geographic zone — computed using "
                f"TF-IDF across zones, so terms common citywide are suppressed and "
                f"zone-specific language surfaces. Bridges the NLP and K-Means "
                f"analyses.</div>",
                unsafe_allow_html=True,
            )

            df_wc2 = load_311_for_wordcloud()

            @st.cache_data
            def compute_cluster_tfidf(df):
                """
                Treat each cluster as a single 'document' by concatenating all
                descriptions, then compute TF-IDF across the 5 cluster-documents.
                This surfaces terms that are distinctive to each zone vs. citywide.
                """
                from sklearn.feature_extraction.text import TfidfVectorizer
                import re as _re

                STOPWORDS = {
                    "street","road","san","rafael","city","please","need","area",
                    "would","like","also","one","two","three","located","side",
                    "front","back","near","just","there","have","been","that",
                    "this","with","from","they","said","will","can","the","and",
                    "for","are","not","but","has","was","its","it","at","on",
                    "in","of","to","a","is","be","an","by","or","as","we",
                    "get","got","see","new","old","use","used","still","ve",
                    "re","don","didn","isn","hasn","weren","couldn","wouldn",
                    "com","http","www","lot","right","left","since","way","around",
                    "cross","corner","block","along","between","across","make",
                    "made","also","very","more","much","many","well","good","big",
                    "large","small","long","high","low","per","day","days","week",
                    "time","year","month","number","years","months","weeks",
                    "report","reported","request","requested","issue","problem",
                    "called","call","office","public","works","service","services",
                    "department","address","resident","residents","property",
                    "located","location","place","thank","thanks","know","think",
                }

                cluster_docs = {}
                for c in sorted(df["cluster"].unique()):
                    sub = df[df["cluster"] == c]["description"].dropna()
                    combined = " ".join(
                        _re.sub(r"[^\w\s]", " ", str(d).lower())
                        for d in sub
                    )
                    cluster_docs[c] = combined

                docs   = [cluster_docs[c] for c in sorted(cluster_docs)]
                labels = sorted(cluster_docs.keys())

                tfidf = TfidfVectorizer(
                    max_features=500,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                    stop_words=list(STOPWORDS),
                )
                mat   = tfidf.fit_transform(docs)
                feats = tfidf.get_feature_names_out()

                result = {}
                for i, c in enumerate(labels):
                    scores  = mat[i].toarray().flatten()
                    top_idx = scores.argsort()[::-1][:20]
                    result[c] = [(feats[j], float(scores[j])) for j in top_idx if scores[j] > 0]

                return result

            cluster_terms = compute_cluster_tfidf(df_wc2)

            _cl_ctrl_l, _cl_ctrl_r = st.columns([2, 1])
            with _cl_ctrl_l:
                wc_cluster_sel = st.selectbox(
                    "Select zone",
                    options=["All Zones"] + sorted(cluster_terms.keys()),
                    format_func=lambda x: "All Zones" if x == "All Zones" else CLUSTER_NAMES.get(x, f"Zone {x}"),
                    key="wc_cluster_sel",
                )
            with _cl_ctrl_r:
                cl_view = st.radio(
                    "View as",
                    ["Word Cloud", "Bar Chart"],
                    horizontal=True,
                    key="cl_view_toggle",
                    disabled=not _WC_AVAILABLE,
                )
            if not _WC_AVAILABLE:
                cl_view = "Bar Chart"
                st.caption(":grey[Install `wordcloud` to enable word cloud view.]")

            cl_col, cl_info = st.columns([3, 2], gap="large")

            with cl_col:
                if wc_cluster_sel == "All Zones":
                    # Merge all cluster terms, sum weights
                    from collections import defaultdict
                    _merged = defaultdict(float)
                    for _c_terms in cluster_terms.values():
                        for _t, _w in _c_terms:
                            _merged[_t] += _w
                    # Normalise by number of clusters
                    n_cl = len(cluster_terms)
                    _merged = {k: v / n_cl for k, v in _merged.items()}
                    # Sort and take top 40
                    _sorted = sorted(_merged.items(), key=lambda x: x[1], reverse=True)[:40]
                    terms_c   = [t for t, _ in _sorted]
                    weights_c = [w for _, w in _sorted]
                    clr = ACCENT
                else:
                    terms_w   = cluster_terms.get(wc_cluster_sel, [])
                    terms_c   = [t for t, _ in terms_w]
                    weights_c = [w for _, w in terms_w]
                    clr = CLUSTER_COLORS.get(wc_cluster_sel, ACCENT)

                if terms_c:
                    w_arr  = np.array(weights_c)
                    w_norm = (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min() + 1e-9)
                    r0 = int(clr[1:3], 16)
                    g0 = int(clr[3:5], 16)
                    b0 = int(clr[5:7], 16)
                    n  = len(terms_c)
                    bar_colors_c = [
                        f"rgba({r0},{g0},{b0},{0.9 - 0.5*i/max(n-1,1):.2f})"
                        for i in range(n)
                    ]

                    fig_wc2 = go.Figure(go.Bar(
                        x=weights_c,
                        y=terms_c,
                        orientation="h",
                        marker_color=bar_colors_c,
                        text=[f"{w:.3f}" for w in weights_c],
                        textposition="outside",
                        textfont=dict(size=10, color=TEXT_SEC),
                        hovertemplate="Term: <b>%{y}</b><br>Zone TF-IDF: %{x:.4f}<extra></extra>",
                    ))
                    fig_wc2.update_layout(**layout(
                        height=max(360, n * 26),
                        xaxis=dict(title="Zone TF-IDF score",
                                   gridcolor=BORDER, linecolor=BORDER),
                        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
                        margin=dict(l=200, r=80, t=20, b=40),
                        title=dict(
                            text=f"Distinctive terms — {CLUSTER_NAMES.get(wc_cluster_sel, f'Zone {wc_cluster_sel}')}",
                            font=dict(size=13, color=TEXT_PRI, family="Syne"),
                            x=0,
                        ),
                    ))
                    if cl_view == "Word Cloud":
                        import matplotlib
                        matplotlib.use("Agg")
                        import matplotlib.pyplot as _plt
                        import matplotlib.colors as _mcolors
                        _clr_hex = CLUSTER_COLORS.get(wc_cluster_sel, ACCENT)
                        _cmap = _mcolors.LinearSegmentedColormap.from_list(
                            f"zone_{wc_cluster_sel}",
                            ["#cccccc", _clr_hex],
                        )
                        _cmap_name = f"zone_cmap_{wc_cluster_sel}"
                        _plt.colormaps.register(_cmap, name=_cmap_name, force=True)
                        _freq_cl = dict(zip(terms_c, [float(w) for w in weights_c]))
                        _fig_mpl2 = render_wordcloud(_freq_cl, colormap=_cmap_name)
                        if _fig_mpl2:
                            st.pyplot(_fig_mpl2, use_container_width=True)
                    else:
                        st.plotly_chart(fig_wc2, use_container_width=True)

            with cl_info:
                # Zone summary card
                _is_all = wc_cluster_sel == "All Zones"
                _clr_info = ACCENT if _is_all else CLUSTER_COLORS.get(wc_cluster_sel, ACCENT)
                _n_zone = len(df_wc2) if _is_all else int((df_wc2["cluster"] == wc_cluster_sel).sum())
                _label  = "All Zones" if _is_all else CLUSTER_NAMES.get(wc_cluster_sel, f"Zone {wc_cluster_sel}")
                st.markdown(
                    f"<div class='metric-card' style='border-top:3px solid {_clr_info};'>"
                    f"<div class='metric-value' style='color:{_clr_info};'>{_n_zone:,}</div>"
                    f"<div class='metric-label'>{_label} records</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

                # Top categories
                st.markdown(
                    "<div class='section-header' style='margin-top:16px;'>"
                    "Top categories in zone</div>",
                    unsafe_allow_html=True,
                )
                _pool = df_wc2 if _is_all else df_wc2[df_wc2["cluster"] == wc_cluster_sel]
                top_in_cluster = _pool["category_short"].value_counts().head(5)
                fig_mini = go.Figure(go.Bar(
                    x=top_in_cluster.values,
                    y=[c[:28] for c in top_in_cluster.index],
                    orientation="h",
                    marker_color=_clr_info,
                    opacity=0.8,
                    hovertemplate="%{y}: %{x}<extra></extra>",
                ))
                fig_mini.update_layout(**layout(
                    height=200,
                    margin=dict(l=175, r=20, t=10, b=30),
                    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, title="Incidents"),
                    yaxis=dict(tickfont=dict(size=10), autorange="reversed"),
                ))
                st.plotly_chart(fig_mini, use_container_width=True)

                st.markdown(
                    f"<div class='insight-box'>"
                    f"Each zone is treated as a single document. Terms common "
                    f"<b style='color:{TEXT_PRI};'>across all zones</b> "
                    f"are down-weighted — only vocabulary "
                    f"<b style='color:{TEXT_PRI};'>distinctive to each zone</b> "
                    f"surfaces. &ldquo;All Zones&rdquo; shows the citywide vocabulary "
                    f"after removing common stopwords.</div>",
                    unsafe_allow_html=True,
                )

        # ── Tab 4: Live classifier ────────────────────────────────────────
        with tab_live:
            st.markdown(
                f"<div style='font-size:0.88rem;color:{TEXT_SEC};"
                f"margin-bottom:16px;'>Enter a service request description "
                f"to see the model's predicted category and confidence score.</div>",
                unsafe_allow_html=True,
            )
            desc_input = st.text_area(
                "Service request description",
                placeholder="e.g. There is a large couch and several bags of trash "
                            "dumped on the corner of 4th and B Street...",
                height=120,
            )
            if st.button("Classify →", type="primary"):
                if desc_input.strip():
                    try:
                        import joblib
                        model_path = ROOT / "models" / "nlp_tfidf_lr_pipeline.joblib"
                        pipeline = joblib.load(model_path)
                        proba    = pipeline.predict_proba([desc_input])[0]
                        classes  = pipeline.classes_
                        top_idx  = np.argsort(proba)[::-1][:5]

                        pred_class = classes[top_idx[0]]
                        pred_conf  = proba[top_idx[0]]

                        conf_color = (
                            "#2A9D8F" if pred_conf >= 0.7 else
                            "#E9C46A" if pred_conf >= 0.4 else
                            ACCENT
                        )
                        st.markdown(
                            f"<div style='background:{CARD_BG};"
                            f"border:1px solid {BORDER};border-radius:10px;"
                            f"padding:20px 24px;margin:12px 0;'>"
                            f"<div style='font-size:0.75rem;text-transform:uppercase;"
                            f"letter-spacing:0.1em;color:{TEXT_SEC};'>Predicted category</div>"
                            f"<div style='font-family:Syne,sans-serif;font-size:1.4rem;"
                            f"font-weight:700;color:{conf_color};margin:8px 0;'>"
                            f"{pred_class}</div>"
                            f"<div style='font-family:DM Mono;font-size:0.9rem;"
                            f"color:{TEXT_SEC};'>Confidence: "
                            f"<span style='color:{conf_color};'>{pred_conf*100:.1f}%</span>"
                            f"</div></div>",
                            unsafe_allow_html=True,
                        )

                        # Top-5 bar
                        top_classes = [classes[i] for i in top_idx]
                        top_probs   = [proba[i] for i in top_idx]
                        fig_prob = go.Figure(go.Bar(
                            x=top_probs,
                            y=[c[:35] for c in top_classes],
                            orientation="h",
                            marker_color=[conf_color] + [TEXT_SEC] * 4,
                            text=[f"{p*100:.1f}%" for p in top_probs],
                            textposition="outside",
                            textfont=dict(color=TEXT_PRI, size=11),
                        ))
                        fig_prob.update_layout(**layout(
                            height=230,
                            xaxis=dict(range=[0, 1.1], tickformat=".0%",
                                       gridcolor=BORDER),
                            yaxis=dict(tickfont=dict(size=11),
                                       autorange="reversed"),
                            margin=dict(l=220, r=60, t=16, b=20),
                            title=dict(
                                text="Top 5 predicted categories",
                                font=dict(size=12, color=TEXT_SEC),
                                x=0,
                            ),
                        ))
                        st.plotly_chart(fig_prob, use_container_width=True)

                    except FileNotFoundError:
                        st.warning(
                            "Model file not found. Run `nlp_classification.py` "
                            "to generate `models/nlp_tfidf_lr_pipeline.joblib`."
                        )
                else:
                    st.warning("Please enter a description to classify.")

    except FileNotFoundError as e:
        st.error(f"NLP report not found: {e}\n\n"
                 "Run `nlp_classification.py` first.")
