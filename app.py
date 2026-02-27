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
    1: "#E63946",   # red      — largest cluster
    2: "#2A9D8F",   # teal
    3: "#E9C46A",   # amber
    4: "#457B9D",   # steel blue
    5: "#A8DADC",   # light teal — smallest
}
CLUSTER_NAMES = {
    1: "Zone 1 — West Central",
    2: "Zone 2 — Central",
    3: "Zone 3 — North",
    4: "Zone 4 — East",
    5: "Zone 5 — Downtown Core",
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
        f"311 Records: 11,822<br>"
        f"Date range: Jan 2023 – Sep 2025<br>"
        f"Clusters: 5 (k-means)<br>"
        f"Forecast horizon: 13 weeks<br><br>"
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
        f"K-Means spatial clustering of 11,698 service requests · k=5 "
        f"(aligned with San Rafael's 5-district planning structure)</p>",
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
                options=[1, 2, 3, 4, 5],
                default=[1, 2, 3, 4, 5],
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
                f"<b style='color:{TEXT_PRI};'>Key finding:</b> Zone 1 generates "
                f"<b style='color:{ACCENT};'>120.8 incidents per 1,000 residents</b> — "
                f"2.4× Zone 3's rate, despite similar populations. Zone 5 carries "
                f"the highest complexity penalty: mean resolution of 66.8 days "
                f"and 32.7% backlog rate."
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
                f"Model selected by AIC: <b style='color:{TEXT_PRI};'>Damped Additive</b> · "
                f"α=0.1333 (low smoothing reflects high week-to-week noise, CV=0.310) · "
                f"Year-over-year correlation ρ=−0.20 (weak seasonality — model "
                f"correctly omits seasonal component) · "
                f"Training window: Jan 2023–present (excludes 2022 ramp-up)"
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
                    options=[1, 2, 3, 4, 5],
                    default=[1, 2, 3],
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
                    f"<b style='color:{TEXT_PRI};'>Zone 5 note:</b> Modeled as a "
                    f"proportional share (5.6%) of the city-wide forecast. "
                    f"Too sparse for independent HW (mean 3.8/week, CV=0.65). "
                    f"This is more statistically defensible than fitting noise."
                    f"</div>",
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
                    f"Zone 4's MAPE of 231% reflects denominator inflation "
                    f"(near-zero weeks). MAE of 5.1 is the reliable measure. "
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
            for c in [5, 4, 3, 2, 1]:   # stack bottom-up
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
                f"Mean resolution days / city average (50.0d). "
                f"Higher = more staff-hours per incident.<br><br>"
                f"<b style='color:{TEXT_PRI};'>Backlog pressure</b><br>"
                f"Fraction of cases exceeding 30 days. "
                f"Zone 5 carries 32.7% — highest of all zones.<br><br>"
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
                f"<b style='color:{TEXT_PRI};'>Zone 5 insight:</b> Despite contributing "
                f"only 5.8% of volume, Zone 5 receives a 9.0% workload share — "
                f"driven by 1.34× complexity weight and 32.7% backlog rate. "
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
        f"TF-IDF + Logistic Regression · 20 categories · "
        f"73.5% accuracy · 11,161 training records</p>",
        unsafe_allow_html=True,
    )

    try:
        report = load_nlp_report()

        tab_perf, tab_live = st.tabs(["📊  Model Performance", "🔍  Live Classifier"])

        # ── Tab 1: Performance ────────────────────────────────────────────
        with tab_perf:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1: st.metric("Accuracy",    "73.53%", "20-class")
            with kpi2: st.metric("F1 Weighted", "0.711",  "test set")
            with kpi3: st.metric("F1 Macro",    "0.548",  "unweighted")
            with kpi4: st.metric("CV F1",       "~0.71",  "5-fold")

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

        # ── Tab 2: Live classifier ────────────────────────────────────────
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
