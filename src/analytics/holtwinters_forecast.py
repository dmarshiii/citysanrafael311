"""
holtwinters_forecast.py
-----------------------
Holt-Winters (Triple Exponential Smoothing) time-series forecasting of
weekly 311 service request demand for the City of San Rafael.

Model Design
------------
- Granularity  : Weekly (ISO week periods)
- Series length: ~194 usable weeks (May 2022 – Jan 2026)
- Seasonality  : Annual (period = 52 weeks)
- Horizon      : 13 weeks (~90 days forward)
- Trend        : Additive
- Seasonal     : Additive
- Optimization : statsmodels auto-optimizes smoothing parameters (alpha,
                 beta, gamma) by minimizing SSE

Justification over alternatives
--------------------------------
- ARIMA requires stationarity transformations and manual order selection;
  Holt-Winters is better suited to series with clear trend + seasonality.
- Prophet (Facebook) is powerful but heavy; Holt-Winters is transparent,
  lightweight, and directly interpretable for a government stakeholder.
- Simple moving average lacks trend/seasonal decomposition.

Business Value
--------------
Provides the City of San Rafael a 90-day forward view of expected 311
request volume, enabling proactive staffing, budget allocation, and
resource deployment before demand peaks materialize.

Evaluation Metrics
------------------
- MAE  : Mean Absolute Error (raw request count units — intuitive)
- RMSE : Root Mean Squared Error (penalizes large misses more heavily)
- MAPE : Mean Absolute Percentage Error (scale-independent, % terms)
- Walk-forward cross-validation (last 13 weeks held out as test set)
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
CLEAN_311    = ROOT / "data" / "processed" / "san_rafael_311_clean.csv"
OUT_DIR      = ROOT / "data" / "processed"
OUT_FORECAST = OUT_DIR / "forecast_holtwinters.csv"
OUT_METRICS  = OUT_DIR / "forecast_metrics.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEASONAL_PERIOD  = 52   # annual weekly seasonality
FORECAST_WEEKS   = 13   # ~90-day horizon
TEST_WEEKS       = 13   # hold-out for evaluation

# Training window: 2023-01-01 onward.
# Rationale: 2022 data represents a system ramp-up period in which the City
# of San Rafael was still growing 311 platform adoption (avg ~36 req/week vs
# ~76 req/week in 2024). Including this low-volume period suppresses the
# seasonal signal, causing the optimizer to collapse gamma (seasonal) and
# beta (trend) smoothing parameters to zero. Restricting to 2023+ gives
# three full seasonal cycles of steady-state operations, which is sufficient
# for Holt-Winters to learn reliable annual patterns.
TRAIN_START = "2023-01-01"


# ---------------------------------------------------------------------------
# 1. Build weekly time series
# ---------------------------------------------------------------------------

def build_weekly_series(path: Path = CLEAN_311) -> pd.Series:
    """
    Aggregate cleaned 311 data to weekly request counts.
    Drops the final partial week to avoid biasing the model.

    Returns a pd.Series indexed by week-start Timestamp, freq='W-SUN'.
    """
    df = pd.read_csv(path, low_memory=False)
    df["requested_datetime"] = pd.to_datetime(df["requested_datetime"], errors="coerce")

    # Aggregate to ISO week periods
    df["week"] = df["requested_datetime"].dt.to_period("W")
    weekly = (
        df.groupby("week")
          .size()
          .reset_index(name="count")
          .sort_values("week")
    )

    # Drop the final partial week (almost always incomplete)
    weekly = weekly.iloc[:-1].copy()

    # Convert PeriodIndex to Timestamp for statsmodels compatibility
    weekly["week_start"] = weekly["week"].dt.start_time
    series = weekly.set_index("week_start")["count"].astype(float)

    # Apply training window — exclude 2022 ramp-up period
    series = series[series.index >= TRAIN_START]

    series.index.freq = pd.tseries.frequencies.to_offset("W-MON")

    print(f"[HW] Weekly series built: {len(series)} weeks")
    print(f"[HW] Range : {series.index[0].date()} → {series.index[-1].date()}")
    print(f"[HW] Mean  : {series.mean():.1f}  |  Std: {series.std():.1f}")
    print(f"[HW] Min   : {series.min():.0f}   |  Max: {series.max():.0f}")

    return series


# ---------------------------------------------------------------------------
# 2. Train / test split
# ---------------------------------------------------------------------------

def train_test_split(series: pd.Series,
                     test_weeks: int = TEST_WEEKS) -> tuple[pd.Series, pd.Series]:
    """Split series into train (all but last N weeks) and test (last N weeks)."""
    train = series.iloc[:-test_weeks]
    test  = series.iloc[-test_weeks:]
    print(f"[HW] Train: {len(train)} weeks  |  Test: {len(test)} weeks")
    return train, test


# ---------------------------------------------------------------------------
# 3. Fit Holt-Winters model
# ---------------------------------------------------------------------------

def fit_model(train: pd.Series,
              seasonal_period: int = SEASONAL_PERIOD) -> ExponentialSmoothing:
    """
    Fit Holt-Winters models and select the best by AIC.

    Candidates evaluated:
      1. Additive trend + additive seasonal (full Holt-Winters)
      2. Additive trend only (double exponential smoothing — no seasonal)
      3. Damped additive trend only (conservative trend dampening)

    Year-over-year correlation in this dataset is low (~-0.20), indicating
    that weekly seasonal patterns are weak relative to noise. Model selection
    via AIC lets the data determine whether the seasonal component genuinely
    improves fit or simply adds parameters without benefit.
    """
    candidates = [
        {"trend": "add", "seasonal": "add",  "damped_trend": False, "label": "Add+Seasonal"},
        {"trend": "add", "seasonal": None,   "damped_trend": False, "label": "Add only"},
        {"trend": "add", "seasonal": None,   "damped_trend": True,  "label": "Damped Add"},
    ]

    best_model = None
    best_aic   = np.inf
    best_label = ""

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
                    kwargs["seasonal_periods"] = seasonal_period

                m = ExponentialSmoothing(train, **kwargs).fit(
                    optimized=True, use_brute=True
                )
            print(f"[HW]   {cfg['label']:20s} AIC={m.aic:.2f}")
            if m.aic < best_aic:
                best_aic   = m.aic
                best_model = m
                best_label = cfg["label"]
        except Exception as e:
            print(f"[HW]   {cfg['label']:20s} FAILED: {e}")

    print(f"\n[HW] Best model: {best_label} (AIC={best_aic:.2f})")
    print(f"[HW]   alpha (level)    : {best_model.params['smoothing_level']:.4f}")
    if "smoothing_trend" in best_model.params:
        print(f"[HW]   beta  (trend)    : {best_model.params['smoothing_trend']:.4f}")
    if "smoothing_seasonal" in best_model.params:
        print(f"[HW]   gamma (seasonal) : {best_model.params['smoothing_seasonal']:.4f}")

    return best_model, best_label


# ---------------------------------------------------------------------------
# 4. Evaluate on held-out test set
# ---------------------------------------------------------------------------

def evaluate(model_tuple: tuple,
             test: pd.Series) -> tuple[dict, pd.Series]:
    """
    Predict over the test window and compute MAE, RMSE, MAPE.

    Why these metrics for this business problem:
    - MAE  : Directly interpretable as average weekly request-count error.
    - RMSE : Highlights weeks with large misses (important for staffing spikes).
    - MAPE : % error provides a scale-free benchmark across categories.
    """
    model, label = model_tuple
    preds = model.forecast(len(test))
    preds.index = test.index

    errors = test.values - preds.values
    mae    = np.mean(np.abs(errors))
    rmse   = np.sqrt(np.mean(errors ** 2))
    mape   = np.mean(np.abs(errors / test.values.clip(min=1))) * 100

    metrics = {
        "model": label,
        "MAE":   round(mae,  2),
        "RMSE":  round(rmse, 2),
        "MAPE":  round(mape, 2),
    }

    print(f"\n[HW] Evaluation on {len(test)}-week hold-out:")
    print(f"[HW]   MAE  : {metrics['MAE']:.2f} requests/week")
    print(f"[HW]   RMSE : {metrics['RMSE']:.2f} requests/week")
    print(f"[HW]   MAPE : {metrics['MAPE']:.2f}%")

    return metrics, preds


# ---------------------------------------------------------------------------
# 5. Produce 90-day forward forecast
# ---------------------------------------------------------------------------

def forecast_future(model_tuple: tuple,
                    series: pd.Series,
                    n_weeks: int = FORECAST_WEEKS) -> pd.DataFrame:
    """
    Refit the winning model configuration on the FULL series and forecast
    n_weeks ahead with 80% and 95% confidence intervals via residual bootstrap.
    """
    model, label = model_tuple
    has_seasonal = "smoothing_seasonal" in model.params
    is_damped    = getattr(model.model, "damped_trend", False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kwargs = dict(
            trend="add",
            seasonal="add" if has_seasonal else None,
            damped_trend=is_damped,
            initialization_method="estimated",
        )
        if has_seasonal:
            kwargs["seasonal_periods"] = SEASONAL_PERIOD

        full_model = ExponentialSmoothing(series, **kwargs).fit(
            optimized=True, use_brute=True
        )

    forecast_vals = full_model.forecast(n_weeks)
    residuals     = full_model.resid
    rng           = np.random.default_rng(42)

    ci_records = []
    for step in range(n_weeks):
        boot = [
            forecast_vals.iloc[step] + rng.choice(residuals, size=step + 1, replace=True).sum()
            for _ in range(500)
        ]
        ci_records.append({
            "ci_lower_80": np.percentile(boot, 10),
            "ci_upper_80": np.percentile(boot, 90),
            "ci_lower_95": np.percentile(boot, 2.5),
            "ci_upper_95": np.percentile(boot, 97.5),
        })

    ci_df = pd.DataFrame(ci_records, index=forecast_vals.index)

    forecast_df = pd.DataFrame({
        "week_start":  forecast_vals.index,
        "forecast":    forecast_vals.values.round(1),
        "ci_lower_80": ci_df["ci_lower_80"].round(1),
        "ci_upper_80": ci_df["ci_upper_80"].round(1),
        "ci_lower_95": ci_df["ci_lower_95"].round(1),
        "ci_upper_95": ci_df["ci_upper_95"].round(1),
    }).reset_index(drop=True)

    for col in ["ci_lower_80", "ci_lower_95"]:
        forecast_df[col] = forecast_df[col].clip(lower=0)

    print(f"\n[HW] 13-week forward forecast:")
    print(forecast_df[["week_start", "forecast", "ci_lower_80", "ci_upper_80"]].to_string(index=False))

    return forecast_df


# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(forecast_df: pd.DataFrame,
                 metrics: dict,
                 train: pd.Series,
                 test: pd.Series,
                 test_preds: pd.Series,
                 model_tuple=None,
                 series=None,
                 yoy_corr: float = float("nan"),
                 cv: float = float("nan")) -> None:
    """Persist forecast, historical series, and metrics to processed/ folder."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    hist_df = pd.DataFrame({
        "week_start": train.index.tolist() + test.index.tolist(),
        "actual":     train.values.tolist() + test.values.tolist(),
        "type":       ["train"] * len(train) + ["test"] * len(test),
    })
    test_pred_df = pd.DataFrame({
        "week_start": test_preds.index,
        "fitted":     test_preds.values.round(1),
    })
    hist_df = hist_df.merge(test_pred_df, on="week_start", how="left")
    forecast_df["type"] = "forecast"

    forecast_df.to_csv(OUT_FORECAST, index=False)
    print(f"[HW] Forecast saved        → {OUT_FORECAST}")

    hist_path = OUT_DIR / "forecast_historical.csv"
    hist_df.to_csv(hist_path, index=False)
    print(f"[HW] Historical series saved → {hist_path}")

    metrics_df = pd.DataFrame([metrics])
    metrics_df["seasonal_period"] = SEASONAL_PERIOD
    metrics_df["test_weeks"]      = TEST_WEEKS
    metrics_df["forecast_weeks"]  = FORECAST_WEEKS
    metrics_df["train_start"]     = TRAIN_START
    # Series diagnostics
    metrics_df["yoy_correlation"] = round(yoy_corr, 4)
    metrics_df["series_cv"]       = round(cv, 4)
    if series is not None:
        metrics_df["n_weeks_total"] = len(series)
        metrics_df["series_mean"]   = round(float(series.mean()), 2)
        metrics_df["date_start"]    = str(series.index.min().date())
        metrics_df["date_end"]      = str(series.index.max().date())
    # Model parameters
    if model_tuple is not None:
        m, label = model_tuple
        params = m.params
        metrics_df["alpha"] = round(float(params.get("smoothing_level", float("nan"))), 4)
        metrics_df["beta"]  = round(float(params.get("smoothing_trend",    float("nan"))), 4)
        metrics_df["gamma"] = round(float(params.get("smoothing_seasonal", float("nan"))), 4)
        metrics_df["has_seasonal"] = "smoothing_seasonal" in params
        metrics_df["is_damped"]    = getattr(m.model, "damped_trend", False)
        metrics_df["aic"]          = round(float(m.aic), 2)
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"[HW] Metrics saved         → {OUT_METRICS}")


# ---------------------------------------------------------------------------
# 7. Full pipeline runner
# ---------------------------------------------------------------------------

def run(data_path: Path = CLEAN_311) -> tuple[pd.DataFrame, dict]:
    """Execute the full Holt-Winters pipeline. Returns (forecast_df, metrics)."""
    print("=" * 60)
    print("Holt-Winters Time-Series Forecasting — 311 Demand")
    print("=" * 60)

    series              = build_weekly_series(data_path)
    train, test         = train_test_split(series)
    model_tuple         = fit_model(train)
    metrics, test_preds = evaluate(model_tuple, test)
    forecast_df         = forecast_future(model_tuple, series)
    # Compute series-level diagnostics for the metrics file
    yoy_corr = float(series.autocorr(lag=52)) if len(series) >= 104 else float("nan")
    cv       = float(series.std() / series.mean()) if series.mean() != 0 else float("nan")
    save_outputs(forecast_df, metrics, train, test, test_preds,
                 model_tuple=model_tuple, series=series,
                 yoy_corr=yoy_corr, cv=cv)

    print("\n✓ Holt-Winters pipeline complete.")
    return forecast_df, metrics


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Holt-Winters 311 demand forecast.")
    parser.add_argument("--input", type=Path, default=CLEAN_311)
    args = parser.parse_args()
    run(data_path=args.input)
