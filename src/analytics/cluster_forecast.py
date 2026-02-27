"""
cluster_forecast.py
--------------------
Per-cluster Holt-Winters time-series forecasting of weekly 311 demand,
with a derived staffing workload index for prescriptive resource allocation.

Design
------
Extends the city-wide Holt-Winters model (holtwinters_forecast.py) by
running independent forecasts for each of the K-Means geographic clusters,
then producing a relative staffing recommendation layer.

This module is a pure CONSUMER of kmeans_clustering.py outputs. It reads
cluster assignments directly from kmeans_clusters.csv rather than re-running
K-Means independently. This ensures:
  - Cluster labels are identical to those used in the Cluster Map page
  - Changing K in kmeans_clustering.py propagates automatically here
  - No risk of numbering drift between a different training window slice

Per-cluster modeling decisions
-------------------------------
Clusters with sufficient volume (mean >= SPARSE_MEAN_THRESHOLD incidents/week
AND <= MAX_ZERO_WEEK_PCT zero-value weeks): Full AIC-selected Holt-Winters,
same approach as city-wide model.

Sparse clusters (below threshold): Too thin for reliable standalone time
series. Modeled as a proportional share of the city-wide forecast. The
share is computed from the cluster's actual volume in the training window,
not hardcoded. This is more statistically defensible than fitting noise
and is documented clearly in the output metrics.

Staffing workload index
------------------------
Without published staffing ratios, we derive a RELATIVE workload index
per cluster per week. The index accounts for:
  1. Forecast volume     : expected incidents/week
  2. Complexity weight   : mean resolution days normalized city-wide
                           (longer cases = higher per-incident workload)
  3. Backlog pressure    : % of cases exceeding 30 days (from cluster
                           performance profile)

  workload_index = forecast_volume x complexity_weight x (1 + backlog_pressure)

Normalized so that the city-wide total = 1.0, giving each cluster a share
of total staffing capacity. Multiply by actual FTE count for headcount targets.

Zones (k=4)
-----------
  Zone 1 - Downtown Core   (largest volume)
  Zone 2 - The Canal
  Zone 3 - Terra Linda
  Zone 4 - China Camp      (lowest volume; likely uses proportional model)

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
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]
CLUSTERS_CSV  = ROOT / "data" / "processed" / "kmeans_clusters.csv"
CLUSTER_PERF  = ROOT / "data" / "processed" / "kmeans_cluster_performance.csv"
OUT_DIR       = ROOT / "data" / "processed"
OUT_WEEKLY    = OUT_DIR / "forecast_cluster_weekly.csv"
OUT_METRICS   = OUT_DIR / "forecast_cluster_metrics.csv"
OUT_STAFFING  = OUT_DIR / "forecast_cluster_staffing.csv"
OUT_SUMMARY   = OUT_DIR / "forecast_cluster_summary.csv"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_START      = "2023-01-01"   # exclude pre-operational ramp-up
FORECAST_WEEKS   = 13
TEST_WEEKS       = 13
SEASONAL_PERIOD  = 52
RANDOM_STATE     = 42

# Sparse cluster detection thresholds.
# A cluster uses the proportional model if EITHER condition is met:
#   - mean weekly volume < SPARSE_MEAN_THRESHOLD incidents/week
#   - proportion of zero-volume weeks > SPARSE_ZERO_PCT_THRESHOLD
# These thresholds are data-driven and apply regardless of k or zone numbering.
SPARSE_MEAN_THRESHOLD     = 8.0   # incidents/week
SPARSE_ZERO_PCT_THRESHOLD = 0.20  # 20% zero weeks


# ---------------------------------------------------------------------------
# 1. Load cluster assignments from kmeans_clusters.csv
# ---------------------------------------------------------------------------

def load_clustered_data(path: Path = CLUSTERS_CSV) -> pd.DataFrame:
    """
    Load the cluster-labeled 311 dataset produced by kmeans_clustering.py.
    Applies training window filter and derives the week column.

    Reads cluster assignments from disk rather than re-running K-Means,
    ensuring label consistency with the Cluster Map page.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"[CF] kmeans_clusters.csv not found at {path}\n"
            f"     Run kmeans_clustering.py first."
        )

    df = pd.read_csv(path, low_memory=False)
    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")

    df = df[df["requested_datetime"] >= TRAIN_START].copy()
    df["week"] = df["requested_datetime"].dt.to_period("W")

    k = df["cluster"].nunique()
    print(f"[CF] Clusters loaded from kmeans_clusters.csv: k={k}")
    print(f"[CF] Training window: {TRAIN_START} to present")
    print(f"[CF] Records in training window: {len(df):,}")
    print(f"[CF] Cluster distribution:")
    for c in sorted(df["cluster"].unique()):
        n   = (df["cluster"] == c).sum()
        pct = n / len(df) * 100
        print(f"     Cluster {c}: {n:,} records ({pct:.1f}%)")

    return df


# ---------------------------------------------------------------------------
# 2. Build weekly series per cluster
# ---------------------------------------------------------------------------

def build_cluster_series(df: pd.DataFrame) -> dict:
    """
    Aggregate to weekly incident counts per cluster.
    Fills weeks with zero incidents (full date-range reindex).
    Returns dict: {cluster_id (int): pd.Series, "city": pd.Series}
    """
    weekly_all = (
        df.groupby("week").size()
          .reset_index(name="count")
          .sort_values("week")
    )
    weekly_all = weekly_all.iloc[:-1].copy()   # drop final partial week
    weekly_all["week_start"] = weekly_all["week"].dt.start_time
    city_series = weekly_all.set_index("week_start")["count"].astype(float)
    city_series.index.freq = pd.tseries.frequencies.to_offset("W-MON")

    series_dict = {"city": city_series}

    print(f"\n[CF] Weekly series statistics:")
    for cluster_id in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == cluster_id]
        weekly = (
            sub.groupby("week").size()
               .reset_index(name="count")
               .sort_values("week")
        )
        weekly = weekly.set_index("week").reindex(
            weekly_all["week"].values, fill_value=0
        ).reset_index()
        weekly = weekly.iloc[:-1].copy()
        weekly["week_start"] = weekly["week"].dt.start_time
        s = weekly.set_index("week_start")["count"].astype(float)
        s.index.freq = pd.tseries.frequencies.to_offset("W-MON")
        series_dict[cluster_id] = s

        zero_pct  = (s == 0).mean()
        is_sparse = (s.mean() < SPARSE_MEAN_THRESHOLD or
                     zero_pct > SPARSE_ZERO_PCT_THRESHOLD)
        flag = " <- SPARSE (proportional model)" if is_sparse else ""
        print(f"     Cluster {cluster_id}: {len(s)} weeks  "
              f"mean={s.mean():.1f}/wk  zeros={zero_pct:.1%}{flag}")

    return series_dict


# ---------------------------------------------------------------------------
# 3. Classify sparse clusters and compute proportional shares
# ---------------------------------------------------------------------------

def classify_clusters(series_dict: dict) -> dict:
    """
    Determine modeling approach for each cluster based on volume diagnostics.
    Returns dict: {cluster_id: {"sparse": bool, "prop_share": float, ...}}

    prop_share is derived from the cluster's actual training-window volume
    as a fraction of the city-wide total. Never hardcoded.
    """
    city       = series_dict["city"]
    city_total = city.sum()

    classification = {}
    for key, s in series_dict.items():
        if key == "city":
            continue
        zero_pct   = (s == 0).mean()
        mean_vol   = s.mean()
        is_sparse  = (mean_vol < SPARSE_MEAN_THRESHOLD or
                      zero_pct > SPARSE_ZERO_PCT_THRESHOLD)
        prop_share = float(s.sum() / city_total) if city_total > 0 else 0.0
        classification[key] = {
            "sparse":     is_sparse,
            "prop_share": round(prop_share, 4),
            "mean_vol":   round(mean_vol, 2),
            "zero_pct":   round(zero_pct, 4),
        }

    print(f"\n[CF] Cluster classification (sparse threshold: "
          f"mean<{SPARSE_MEAN_THRESHOLD}/wk or zeros>{SPARSE_ZERO_PCT_THRESHOLD:.0%}):")
    for c, info in sorted(classification.items()):
        model = (f"Proportional ({info['prop_share']*100:.2f}% of city)"
                 if info["sparse"] else "Holt-Winters (AIC-selected)")
        print(f"     Cluster {c}: mean={info['mean_vol']:.1f}/wk  "
              f"zeros={info['zero_pct']:.1%}  -> {model}")

    return classification


# ---------------------------------------------------------------------------
# 4. Fit Holt-Winters model (AIC selection)
# ---------------------------------------------------------------------------

def fit_hw(train: pd.Series, label: str = "") -> tuple:
    """
    AIC-selected Holt-Winters. Same candidate set as holtwinters_forecast.py.
    Returns (fitted_model, model_label, aic) or (None, None, inf) if all fail.
    """
    candidates = [
        {"trend": "add", "seasonal": "add",  "damped_trend": False,
         "label": "Add+Seasonal"},
        {"trend": "add", "seasonal": None,   "damped_trend": False,
         "label": "Add only"},
        {"trend": "add", "seasonal": None,   "damped_trend": True,
         "label": "Damped Add"},
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
# 5. Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(residuals: np.ndarray,
                 forecast:  np.ndarray,
                 n_boot:    int   = 500,
                 level_80:  float = 0.80,
                 level_95:  float = 0.95,
                 seed:      int   = 42) -> dict:
    """Residual bootstrap CIs, same methodology as holtwinters_forecast.py."""
    rng        = np.random.default_rng(seed)
    horizons   = len(forecast)
    boot_paths = np.zeros((n_boot, horizons))
    for i in range(n_boot):
        sampled        = rng.choice(residuals, size=horizons, replace=True)
        boot_paths[i]  = forecast + sampled.cumsum()

    lo80 = np.percentile(boot_paths, (1 - level_80) / 2 * 100,       axis=0)
    hi80 = np.percentile(boot_paths, (1 - (1 - level_80) / 2) * 100, axis=0)
    lo95 = np.percentile(boot_paths, (1 - level_95) / 2 * 100,       axis=0)
    hi95 = np.percentile(boot_paths, (1 - (1 - level_95) / 2) * 100, axis=0)

    return {"lo80": lo80, "hi80": hi80, "lo95": lo95, "hi95": hi95}


# ---------------------------------------------------------------------------
# 6. Forecast a single cluster
# ---------------------------------------------------------------------------

def forecast_cluster(cluster_id:     int,
                     series:         pd.Series,
                     classification: dict,
                     city_series:    pd.Series) -> tuple:
    """
    Forecast a single cluster's weekly demand for FORECAST_WEEKS ahead.

    Modeling approach is determined by the classification dict:
      - Sparse clusters  : proportional share of city-wide forecast
      - Normal clusters  : AIC-selected Holt-Winters on the cluster series

    Returns (forecast_df, metrics_dict) or (None, None) on failure.
    """
    info  = classification[cluster_id]
    train = series.iloc[:-TEST_WEEKS]
    test  = series.iloc[-TEST_WEEKS:]

    print(f"\n[CF] --- Cluster {cluster_id} ---")

    # ── Proportional model ────────────────────────────────────────────────
    if info["sparse"]:
        prop_share = info["prop_share"]
        print(f"[CF]   Proportional model: {prop_share*100:.2f}% of city-wide forecast")
        print(f"[CF]   Rationale: mean={info['mean_vol']:.1f}/wk, "
              f"zero-weeks={info['zero_pct']:.1%}")

        city_train              = city_series.iloc[:-TEST_WEEKS]
        city_model, city_label, _ = fit_hw(city_train, label="city")
        if city_model is None:
            print(f"[CF]   City-wide model failed -- skipping cluster {cluster_id}")
            return None, None

        city_full, _, _  = fit_hw(city_series, label="city (full)")
        fc_vals          = city_full.forecast(FORECAST_WEEKS).values * prop_share

        city_test_pred   = city_model.forecast(TEST_WEEKS).values * prop_share
        test_vals        = test.values
        valid            = test_vals > 0
        mae  = float(np.abs(city_test_pred - test_vals).mean())
        rmse = float(np.sqrt(((city_test_pred - test_vals) ** 2).mean()))
        mape = (float(np.mean(np.abs(
                    (city_test_pred[valid] - test_vals[valid]) / test_vals[valid]
                )) * 100) if valid.any() else np.nan)

        ci          = bootstrap_ci(city_full.resid.values * prop_share, fc_vals)
        model_label = f"Proportional ({city_label})"

    # ── Full Holt-Winters ─────────────────────────────────────────────────
    else:
        model, model_label, aic = fit_hw(train, label=f"Cluster {cluster_id}")
        if model is None:
            print(f"[CF]   All HW models failed -- skipping cluster {cluster_id}")
            return None, None

        print(f"[CF]   Best model : {model_label}  (AIC={aic:.2f})")
        print(f"[CF]   alpha      : {model.params['smoothing_level']:.4f}")

        test_pred = model.forecast(TEST_WEEKS).values
        test_vals = test.values
        valid     = test_vals > 0
        mae  = float(np.abs(test_pred - test_vals).mean())
        rmse = float(np.sqrt(((test_pred - test_vals) ** 2).mean()))
        mape = (float(np.mean(np.abs(
                    (test_pred[valid] - test_vals[valid]) / test_vals[valid]
                )) * 100) if valid.any() else np.nan)

        model_full, _, _ = fit_hw(series, label=f"Cluster {cluster_id} (full)")
        fc_vals          = model_full.forecast(FORECAST_WEEKS).values
        ci               = bootstrap_ci(model_full.resid.values, fc_vals)

    mape_str = f"{mape:.1f}%" if not np.isnan(mape) else "n/a"
    print(f"[CF]   MAE={mae:.2f}  RMSE={rmse:.2f}  MAPE={mape_str}")

    # ── Output DataFrames ─────────────────────────────────────────────────
    last_date = series.index[-1]
    fc_dates  = pd.date_range(
        start   = last_date + pd.Timedelta(weeks=1),
        periods = FORECAST_WEEKS,
        freq    = "W-MON",
    )
    fc_vals = np.maximum(fc_vals, 0)

    forecast_df = pd.DataFrame({
        "cluster":         cluster_id,
        "week":            fc_dates,
        "forecast":        np.round(fc_vals, 1),
        "lo_80":           np.round(np.maximum(ci["lo80"], 0), 1),
        "hi_80":           np.round(ci["hi80"], 1),
        "lo_95":           np.round(np.maximum(ci["lo95"], 0), 1),
        "hi_95":           np.round(ci["hi95"], 1),
        "model":           model_label,
        "is_proportional": info["sparse"],
    })

    metrics = {
        "cluster":         cluster_id,
        "model":           model_label,
        "is_proportional": info["sparse"],
        "prop_share":      info["prop_share"] if info["sparse"] else np.nan,
        "n_train_weeks":   len(train),
        "mean_weekly":     round(series.mean(), 2),
        "zero_week_pct":   round(info["zero_pct"] * 100, 1),
        "mae":             round(mae,  2),
        "rmse":            round(rmse, 2),
        "mape":            round(mape, 1) if not np.isnan(mape) else np.nan,
        "fc_13wk_total":   round(float(fc_vals.sum()), 1),
    }

    return forecast_df, metrics


# ---------------------------------------------------------------------------
# 7. Build staffing workload index
# ---------------------------------------------------------------------------

def build_staffing_index(forecast_dfs: list,
                         perf_path:    Path = CLUSTER_PERF) -> tuple:
    """
    Derive a relative staffing workload index per cluster per week.

    complexity_weight and backlog_rate are read from kmeans_cluster_performance.csv
    produced by kmeans_clustering.py, ensuring the staffing index uses the same
    cluster definitions as the map and forecast pages.
    """
    all_fc   = pd.concat(forecast_dfs, ignore_index=True)
    clusters = sorted(all_fc["cluster"].unique())

    try:
        perf = pd.read_csv(perf_path)
        perf["cluster"]           = perf["cluster"].astype(int)
        city_mean_days            = perf["mean_days"].mean()
        perf["complexity_weight"] = (perf["mean_days"] / city_mean_days).round(4)
        perf["backlog_rate"]      = (perf["pct_over_30d"] / 100).round(4)
        weights = perf.set_index("cluster")[["complexity_weight", "backlog_rate"]]
        print(f"\n[CF] Complexity weights (city mean resolution: {city_mean_days:.1f}d):")
    except FileNotFoundError:
        print("[CF] Warning: kmeans_cluster_performance.csv not found -- "
              "using equal complexity weights")
        weights = pd.DataFrame({
            "complexity_weight": {c: 1.0 for c in clusters},
            "backlog_rate":      {c: 0.0 for c in clusters},
        })

    print(f"     {'Cluster':8} {'complexity_wt':>15} {'backlog_rate':>13}")
    for c in sorted(weights.index):
        if c in clusters:
            print(f"     {c:<8} {weights.loc[c, 'complexity_weight']:>15.4f} "
                  f"{weights.loc[c, 'backlog_rate']:>13.4f}")

    all_fc = all_fc.merge(weights, left_on="cluster", right_index=True, how="left")
    all_fc["complexity_weight"] = all_fc["complexity_weight"].fillna(1.0)
    all_fc["backlog_rate"]      = all_fc["backlog_rate"].fillna(0.0)
    all_fc["raw_workload"]      = (
        all_fc["forecast"] *
        all_fc["complexity_weight"] *
        (1 + all_fc["backlog_rate"])
    ).round(3)

    week_totals            = all_fc.groupby("week")["raw_workload"].sum().rename("city_total")
    all_fc                 = all_fc.merge(week_totals, on="week")
    all_fc["workload_share"] = (all_fc["raw_workload"] / all_fc["city_total"]).round(4)

    summary = (
        all_fc.groupby("cluster")
          .agg(
              fc_13wk_total       = ("forecast",          "sum"),
              mean_weekly_fc      = ("forecast",          "mean"),
              mean_workload_share = ("workload_share",    "mean"),
              complexity_weight   = ("complexity_weight", "first"),
              backlog_rate        = ("backlog_rate",      "first"),
              is_proportional     = ("is_proportional",   "first"),
          )
          .round(3)
          .reset_index()
    )
    summary["fc_13wk_total"]         = summary["fc_13wk_total"].round(1)
    summary["mean_weekly_fc"]        = summary["mean_weekly_fc"].round(1)
    summary["recommended_fte_share"] = (summary["mean_workload_share"] * 100).round(1)

    print(f"\n[CF] 13-week staffing workload summary:")
    print(f"     {'Cluster':8} {'FC Total':>10} {'Mean/Wk':>9} "
          f"{'Wkld Share':>11} {'Rec FTE%':>9} {'Model':>14}")
    for _, row in summary.iterrows():
        mtype = "Proportional" if row["is_proportional"] else "Holt-Winters"
        print(f"     {int(row['cluster']):<8} "
              f"{row['fc_13wk_total']:>10.1f} "
              f"{row['mean_weekly_fc']:>9.1f} "
              f"{row['mean_workload_share']:>11.4f} "
              f"{row['recommended_fte_share']:>8.1f}% "
              f"{mtype:>14}")

    top = summary.loc[summary["recommended_fte_share"].idxmax()]
    print(f"\n[CF] Cluster {int(top['cluster'])} should receive "
          f"~{top['recommended_fte_share']:.1f}% of total field capacity.")

    return all_fc, summary


# ---------------------------------------------------------------------------
# 8. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(all_fc:   pd.DataFrame,
                 metrics:  list,
                 staffing: pd.DataFrame,
                 summary:  pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_fc.to_csv(OUT_WEEKLY,   index=False)
    print(f"\n[CF] Cluster forecasts saved  -> {OUT_WEEKLY}")
    pd.DataFrame(metrics).to_csv(OUT_METRICS, index=False)
    print(f"[CF] Cluster metrics saved    -> {OUT_METRICS}")
    staffing.to_csv(OUT_STAFFING, index=False)
    print(f"[CF] Staffing index saved     -> {OUT_STAFFING}")
    summary.to_csv(OUT_SUMMARY,   index=False)
    print(f"[CF] 13-week summary saved    -> {OUT_SUMMARY}")


# ---------------------------------------------------------------------------
# 9. Full pipeline runner
# ---------------------------------------------------------------------------

def run(clusters_path: Path = CLUSTERS_CSV,
        perf_path:     Path = CLUSTER_PERF) -> pd.DataFrame:
    """
    Full cluster-level forecast + staffing index pipeline.
    Reads cluster assignments from kmeans_clusters.csv.
    Returns the combined staffing DataFrame.
    """
    print("=" * 60)
    print("Cluster-Level Forecast + Staffing Workload Index")
    print("=" * 60)

    df             = load_clustered_data(clusters_path)
    series_dict    = build_cluster_series(df)
    classification = classify_clusters(series_dict)
    city_series    = series_dict["city"]

    forecast_dfs = []
    all_metrics  = []

    for cluster_id in sorted(k for k in series_dict if k != "city"):
        fc_df, metrics = forecast_cluster(
            cluster_id,
            series_dict[cluster_id],
            classification,
            city_series,
        )
        if fc_df is not None:
            forecast_dfs.append(fc_df)
            all_metrics.append(metrics)

    if not forecast_dfs:
        raise RuntimeError("[CF] No cluster forecasts produced -- check input data.")

    staffing_df, summary = build_staffing_index(forecast_dfs, perf_path)
    save_outputs(
        pd.concat(forecast_dfs, ignore_index=True),
        all_metrics,
        staffing_df,
        summary,
    )

    print("\n Cluster forecast pipeline complete.")
    return staffing_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Per-cluster Holt-Winters forecast + staffing index. "
                    "Reads cluster assignments from kmeans_clusters.csv."
    )
    parser.add_argument("--clusters", type=Path, default=CLUSTERS_CSV,
                        help="Path to kmeans_clusters.csv")
    parser.add_argument("--perf",     type=Path, default=CLUSTER_PERF,
                        help="Path to kmeans_cluster_performance.csv")
    args = parser.parse_args()
    run(clusters_path=args.clusters, perf_path=args.perf)
