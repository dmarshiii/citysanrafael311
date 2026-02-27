"""
cluster_forecast.py
--------------------
Per-cluster Holt-Winters time-series forecasting of weekly 311 demand,
with a derived staffing workload index for prescriptive resource allocation.

Design
------
Extends the city-wide Holt-Winters model (holtwinters_forecast.py) by
running independent forecasts for each of the 5 K-Means geographic clusters,
then producing a relative staffing recommendation layer.

Per-cluster modeling decisions
-------------------------------
Clusters 1–2 (mean ~24 and ~18 incidents/week): Full AIC-selected
  Holt-Winters, same approach as city-wide model.

Clusters 3–4 (mean ~10/week): Same AIC selection, but seasonal component
  is unlikely to win given low volume. Damped or simple trend expected.

Cluster 5 (mean ~3.7/week): Too sparse for reliable standalone time series
  (CV=0.65, 9 zero-weeks). Modeled as a proportional share of the city-wide
  forecast instead (consistent ~5.6% of total demand). This is more
  statistically defensible than fitting noise.

Staffing workload index
------------------------
Without published staffing ratios, we derive a RELATIVE workload index
per cluster per week. The index accounts for:
  1. Forecast volume     : expected incidents/week
  2. Complexity weight   : mean resolution days normalized city-wide
                           (longer cases = higher per-incident workload)
  3. Backlog pressure    : % of cases exceeding 30 days (from cluster
                           performance profile)

  workload_index = forecast_volume × complexity_weight × (1 + backlog_pressure)

This is normalized so that the city-wide total = 1.0, giving each cluster
a share of total staffing capacity. The City can multiply this by their
actual FTE count to get a suggested per-district allocation.

Example: If the City has 20 field staff total and Cluster 1 index = 0.38,
the model suggests ~7.6 FTE equivalent for that cluster's area.

Outputs
-------
  forecast_cluster_weekly.csv     : Forecast + CI per cluster per week
  forecast_cluster_metrics.csv    : MAE/RMSE/MAPE per cluster
  forecast_cluster_staffing.csv   : Staffing workload index per cluster/week
  forecast_cluster_summary.csv    : 13-week totals + staffing share per cluster
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
CLEAN_311     = ROOT / "data" / "processed" / "san_rafael_311_clean.csv"
CLUSTER_PERF  = ROOT / "data" / "processed" / "kmeans_cluster_performance.csv"
OUT_DIR       = ROOT / "data" / "processed"
OUT_WEEKLY    = OUT_DIR / "forecast_cluster_weekly.csv"
OUT_METRICS   = OUT_DIR / "forecast_cluster_metrics.csv"
OUT_STAFFING  = OUT_DIR / "forecast_cluster_staffing.csv"
OUT_SUMMARY   = OUT_DIR / "forecast_cluster_summary.csv"

# ---------------------------------------------------------------------------
# Constants (must match holtwinters_forecast.py)
# ---------------------------------------------------------------------------
TRAIN_START     = "2023-01-01"
FORECAST_WEEKS  = 13
TEST_WEEKS      = 13
SEASONAL_PERIOD = 52
K               = 5
RANDOM_STATE    = 42

# Cluster 5 proportion — derived from full dataset cluster sizes
# (consistent across runs due to fixed random_state)
CLUSTER5_SHARE  = 0.056

# Lat/lon bounding box
LAT_MIN, LAT_MAX = 37.85, 38.10
LON_MIN, LON_MAX = -122.65, -122.40


# ---------------------------------------------------------------------------
# 1. Load data and assign clusters
# ---------------------------------------------------------------------------

def load_and_cluster(path: Path = CLEAN_311) -> pd.DataFrame:
    """
    Load 311 data, assign K-Means cluster labels, and attach week column.
    Uses identical clustering parameters to kmeans_clustering.py to ensure
    cluster labels are consistent.
    """
    df = pd.read_csv(path, low_memory=False)
    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")

    # Spatial filter
    df = df[
        (df["lat"]  > LAT_MIN) & (df["lat"]  < LAT_MAX) &
        (df["long"] > LON_MIN) & (df["long"] < LON_MAX)
    ].copy()

    # Training window
    df = df[df["requested_datetime"] >= TRAIN_START].copy()

    # K-Means clustering (identical to kmeans_clustering.py)
    coords = df[["lat", "long"]].values
    scaler = StandardScaler()
    scaled = scaler.fit_transform(coords)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
        df["cluster"] = km.fit_predict(scaled)

    # Relabel by descending size (Cluster 1 = largest)
    size_order = (
        df["cluster"].value_counts()
          .reset_index()
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )
    size_order["new_label"] = size_order.index + 1
    remap = dict(zip(size_order["cluster"], size_order["new_label"]))
    df["cluster"] = df["cluster"].map(remap)

    # Week column
    df["week"] = df["requested_datetime"].dt.to_period("W")

    print(f"[CF] Records loaded (2023+, valid coords): {len(df):,}")
    print(f"[CF] Cluster sizes:")
    for c in sorted(df["cluster"].unique()):
        n = (df["cluster"] == c).sum()
        pct = n / len(df) * 100
        print(f"     Cluster {c}: {n:,} records ({pct:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 2. Build weekly series per cluster
# ---------------------------------------------------------------------------

def build_cluster_series(df: pd.DataFrame) -> dict:
    """
    Aggregate to weekly incident counts per cluster.
    Returns dict: {cluster_id: pd.Series(weekly counts, freq='W-MON')}
    """
    series_dict = {}

    # City-wide series (for Cluster 5 proportional model)
    weekly_all = (
        df.groupby("week").size()
          .reset_index(name="count")
          .sort_values("week")
    )
    weekly_all = weekly_all.iloc[:-1].copy()   # drop final partial week
    weekly_all["week_start"] = weekly_all["week"].dt.start_time
    city_series = weekly_all.set_index("week_start")["count"].astype(float)
    city_series.index.freq = pd.tseries.frequencies.to_offset("W-MON")
    series_dict["city"] = city_series

    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        weekly = (
            sub.groupby("week").size()
               .reset_index(name="count")
               .sort_values("week")
        )
        # Reindex to full city date range (fills weeks with zero incidents)
        weekly = weekly.set_index("week").reindex(
            weekly_all["week"].values, fill_value=0
        ).reset_index()
        weekly = weekly.iloc[:-1].copy()
        weekly["week_start"] = weekly["week"].dt.start_time
        s = weekly.set_index("week_start")["count"].astype(float)
        s.index.freq = pd.tseries.frequencies.to_offset("W-MON")
        series_dict[cluster_id] = s

        print(f"[CF] Cluster {cluster_id}: {len(s)} weeks  "
              f"mean={s.mean():.1f}/wk  std={s.std():.1f}  "
              f"zeros={( s==0).sum()}")

    return series_dict


# ---------------------------------------------------------------------------
# 3. Fit model for a single series
# ---------------------------------------------------------------------------

def fit_hw(train: pd.Series,
           label: str = "") -> tuple:
    """
    AIC-selected Holt-Winters fit. Same candidate set as holtwinters_forecast.py.
    Returns (fitted_model, model_label, aic).
    """
    candidates = [
        {"trend": "add", "seasonal": "add",  "damped_trend": False, "label": "Add+Seasonal"},
        {"trend": "add", "seasonal": None,   "damped_trend": False, "label": "Add only"},
        {"trend": "add", "seasonal": None,   "damped_trend": True,  "label": "Damped Add"},
    ]
    best_model, best_aic, best_label = None, np.inf, ""

    for cfg in candidates:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kwargs = dict(
                    trend=cfg["trend"],
                    seasonal=cfg["seasonal"],
                    damped_trend=cfg["damped_trend"],
                    initialization_method="estimated",
                )
                if cfg["seasonal"] is not None:
                    kwargs["seasonal_periods"] = SEASONAL_PERIOD
                m = ExponentialSmoothing(train, **kwargs).fit(
                    optimized=True, use_brute=True
                )
            if m.aic < best_aic:
                best_aic, best_model, best_label = m.aic, m, cfg["label"]
        except Exception:
            pass

    return best_model, best_label, best_aic


# ---------------------------------------------------------------------------
# 4. Evaluate + forecast per cluster
# ---------------------------------------------------------------------------

def bootstrap_ci(residuals: np.ndarray,
                 forecast: np.ndarray,
                 n_boot: int = 500,
                 level_80: float = 0.80,
                 level_95: float = 0.95,
                 seed: int = 42) -> dict:
    """Residual bootstrap confidence intervals, same as city-wide model."""
    rng = np.random.default_rng(seed)
    horizons = len(forecast)
    boot_paths = np.zeros((n_boot, horizons))
    for i in range(n_boot):
        sampled = rng.choice(residuals, size=horizons, replace=True)
        boot_paths[i] = forecast + sampled.cumsum()

    lo80 = np.percentile(boot_paths, (1-level_80)/2*100, axis=0)
    hi80 = np.percentile(boot_paths, (1-(1-level_80)/2)*100, axis=0)
    lo95 = np.percentile(boot_paths, (1-level_95)/2*100, axis=0)
    hi95 = np.percentile(boot_paths, (1-(1-level_95)/2)*100, axis=0)

    return {"lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95}


def forecast_cluster(cluster_id,
                     series: pd.Series,
                     city_series: pd.Series = None) -> tuple:
    """
    Forecast a single cluster's weekly demand for FORECAST_WEEKS ahead.

    Cluster 5 uses proportional share of city-wide forecast rather than
    an independent model (see module docstring for rationale).

    Returns (forecast_df, metrics_dict).
    """
    print(f"\n[CF] --- Cluster {cluster_id} ---")

    # Cluster 5: proportional model
    if cluster_id == 5 and city_series is not None:
        print(f"[CF]   Using proportional model "
              f"({CLUSTER5_SHARE*100:.1f}% of city-wide forecast)")
        train      = series.iloc[:-TEST_WEEKS]
        test       = series.iloc[-TEST_WEEKS:]

        # Fit city-wide model on same training window
        city_train = city_series.iloc[:-TEST_WEEKS]
        city_model, city_label, _ = fit_hw(city_train, label="city")
        city_fc    = city_model.forecast(FORECAST_WEEKS)
        fc_vals    = city_fc.values * CLUSTER5_SHARE

        # Test evaluation: compare proportional prediction vs actuals
        city_test_pred = city_model.forecast(TEST_WEEKS).values * CLUSTER5_SHARE
        test_vals      = test.values
        valid          = test_vals > 0
        mae  = float(np.abs(city_test_pred - test_vals).mean())
        rmse = float(np.sqrt(((city_test_pred - test_vals)**2).mean()))
        mape = float(np.mean(np.abs((city_test_pred[valid] - test_vals[valid]) / test_vals[valid])) * 100)
        print(f"[CF]   Test MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")

        # CI from city-wide residuals scaled
        city_resid = city_model.resid.values * CLUSTER5_SHARE
        ci         = bootstrap_ci(city_resid, fc_vals)
        model_label = f"Proportional ({city_label})"

    else:
        train = series.iloc[:-TEST_WEEKS]
        test  = series.iloc[-TEST_WEEKS:]

        model, model_label, aic = fit_hw(train, label=f"Cluster {cluster_id}")
        if model is None:
            print(f"[CF]   All models failed — skipping")
            return None, None

        print(f"[CF]   Best model: {model_label} (AIC={aic:.2f})")
        print(f"[CF]   alpha={model.params['smoothing_level']:.4f}")

        # Test evaluation
        test_pred = model.forecast(TEST_WEEKS).values
        test_vals = test.values
        valid     = test_vals > 0
        mae  = float(np.abs(test_pred - test_vals).mean())
        rmse = float(np.sqrt(((test_pred - test_vals)**2).mean()))
        mape = float(np.mean(np.abs((test_pred[valid] - test_vals[valid]) / test_vals[valid])) * 100) if valid.any() else np.nan
        print(f"[CF]   Test MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape:.1f}%")

        # Refit on full series for forward forecast
        model_full, _, _ = fit_hw(series, label=f"Cluster {cluster_id} (full)")
        fc_vals    = model_full.forecast(FORECAST_WEEKS).values
        ci         = bootstrap_ci(model_full.resid.values, fc_vals)

    # Build forecast date index
    last_date  = series.index[-1]
    fc_dates   = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=FORECAST_WEEKS,
        freq="W-MON"
    )

    fc_vals = np.maximum(fc_vals, 0)   # no negative incident counts

    forecast_df = pd.DataFrame({
        "cluster":      cluster_id,
        "week":         fc_dates,
        "forecast":     np.round(fc_vals, 1),
        "lo_80":        np.round(np.maximum(ci["lo80"], 0), 1),
        "hi_80":        np.round(ci["hi80"], 1),
        "lo_95":        np.round(np.maximum(ci["lo95"], 0), 1),
        "hi_95":        np.round(ci["hi95"], 1),
        "model":        model_label,
    })

    metrics = {
        "cluster":      cluster_id,
        "model":        model_label,
        "n_train_weeks": len(train),
        "mean_weekly":  round(series.mean(), 2),
        "mae":          round(mae, 2),
        "rmse":         round(rmse, 2),
        "mape":         round(mape, 1) if not np.isnan(mape) else np.nan,
        "fc_13wk_total": round(fc_vals.sum(), 1),
    }

    return forecast_df, metrics


# ---------------------------------------------------------------------------
# 5. Build staffing workload index
# ---------------------------------------------------------------------------

def build_staffing_index(forecast_dfs: list,
                         perf_path: Path = CLUSTER_PERF) -> pd.DataFrame:
    """
    Derive a relative staffing workload index per cluster per week.

    Index formula (see module docstring for full rationale):
      raw_index = forecast × complexity_weight × (1 + backlog_rate)
      normalized so city total = 1.0 each week

    Also produces a 13-week summary with recommended FTE share per cluster.
    """
    all_fc = pd.concat(forecast_dfs, ignore_index=True)

    # Load cluster performance metrics for complexity weights
    try:
        perf = pd.read_csv(perf_path)
        has_perf = True
    except FileNotFoundError:
        has_perf = False
        print("[CF] Warning: cluster performance file not found — "
              "using equal complexity weights")

    if has_perf:
        city_mean_days = perf["mean_days"].mean()
        # Complexity weight: cluster mean resolution relative to city average
        # Higher mean = more staff time per incident
        perf["complexity_weight"] = (perf["mean_days"] / city_mean_days).round(4)
        # Backlog rate: fraction of cases over 30 days
        perf["backlog_rate"] = (perf["pct_over_30d"] / 100).round(4)
        weights = perf.set_index("cluster")[["complexity_weight", "backlog_rate"]]
    else:
        weights = pd.DataFrame({
            "complexity_weight": {c: 1.0 for c in range(1, K+1)},
            "backlog_rate":      {c: 0.0 for c in range(1, K+1)},
        })

    print("\n[CF] Workload index weights:")
    print(f"     {'Cluster':8} {'complexity_wt':>15} {'backlog_rate':>12}")
    for c in sorted(weights.index):
        print(f"     {c:<8} {weights.loc[c,'complexity_weight']:>15.4f} "
              f"{weights.loc[c,'backlog_rate']:>12.4f}")

    # Apply weights to forecast
    all_fc = all_fc.merge(weights, left_on="cluster", right_index=True, how="left")
    all_fc["raw_workload"] = (
        all_fc["forecast"] *
        all_fc["complexity_weight"] *
        (1 + all_fc["backlog_rate"])
    ).round(3)

    # Normalize: each cluster's share of total city workload per week
    week_totals = all_fc.groupby("week")["raw_workload"].sum().rename("city_total")
    all_fc = all_fc.merge(week_totals, on="week")
    all_fc["workload_share"] = (
        all_fc["raw_workload"] / all_fc["city_total"]
    ).round(4)

    # 13-week summary
    summary = (
        all_fc.groupby("cluster")
          .agg(
              fc_13wk_total    = ("forecast",       "sum"),
              mean_weekly_fc   = ("forecast",       "mean"),
              mean_workload_share = ("workload_share", "mean"),
              complexity_weight   = ("complexity_weight", "first"),
              backlog_rate        = ("backlog_rate",      "first"),
          )
          .round(3)
          .reset_index()
    )
    summary["fc_13wk_total"]   = summary["fc_13wk_total"].round(1)
    summary["mean_weekly_fc"]  = summary["mean_weekly_fc"].round(1)
    summary["recommended_fte_share"] = (summary["mean_workload_share"] * 100).round(1)

    print("\n[CF] 13-week staffing workload summary:")
    print(f"     {'Cluster':8} {'FC Total':>10} {'Mean/Wk':>10} "
          f"{'Workload Share':>16} {'Rec FTE %':>10}")
    for _, row in summary.iterrows():
        print(f"     {int(row['cluster']):<8} "
              f"{row['fc_13wk_total']:>10.1f} "
              f"{row['mean_weekly_fc']:>10.1f} "
              f"{row['mean_workload_share']:>16.4f} "
              f"{row['recommended_fte_share']:>9.1f}%")

    print(f"\n[CF] Interpretation: Cluster 1 should receive "
          f"~{summary.loc[summary['cluster']==1,'recommended_fte_share'].values[0]:.1f}% "
          f"of total field capacity over the next 13 weeks.")
    print(f"[CF] Multiply by actual FTE count for concrete headcount targets.")

    return all_fc, summary


# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(all_fc:   pd.DataFrame,
                 metrics:  list,
                 staffing: pd.DataFrame,
                 summary:  pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_fc.to_csv(OUT_WEEKLY, index=False)
    print(f"\n[CF] Cluster forecasts saved  → {OUT_WEEKLY}")

    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False)
    print(f"[CF] Cluster metrics saved    → {OUT_METRICS}")

    staffing.to_csv(OUT_STAFFING, index=False)
    print(f"[CF] Staffing index saved     → {OUT_STAFFING}")

    summary.to_csv(OUT_SUMMARY, index=False)
    print(f"[CF] 13-week summary saved    → {OUT_SUMMARY}")


# ---------------------------------------------------------------------------
# 7. Full pipeline runner
# ---------------------------------------------------------------------------

def run(data_path: Path = CLEAN_311,
        perf_path: Path = CLUSTER_PERF) -> pd.DataFrame:
    """
    Full cluster-level forecast + staffing index pipeline.
    Returns combined forecast DataFrame.
    """
    print("=" * 60)
    print("Cluster-Level Forecast + Staffing Workload Index")
    print("=" * 60)

    df           = load_and_cluster(data_path)
    series_dict  = build_cluster_series(df)
    city_series  = series_dict["city"]

    forecast_dfs = []
    all_metrics  = []

    for cluster_id in sorted(k for k in series_dict if k != "city"):
        fc_df, metrics = forecast_cluster(
            cluster_id,
            series_dict[cluster_id],
            city_series=city_series,
        )
        if fc_df is not None:
            forecast_dfs.append(fc_df)
            all_metrics.append(metrics)

    staffing_df, summary = build_staffing_index(forecast_dfs, perf_path)
    save_outputs(
        pd.concat(forecast_dfs, ignore_index=True),
        all_metrics,
        staffing_df,
        summary,
    )

    print("\n✓ Cluster forecast pipeline complete.")
    return staffing_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Per-cluster Holt-Winters forecast + staffing index."
    )
    parser.add_argument("--input", type=Path, default=CLEAN_311)
    parser.add_argument("--perf",  type=Path, default=CLUSTER_PERF)
    args = parser.parse_args()
    run(data_path=args.input, perf_path=args.perf)
