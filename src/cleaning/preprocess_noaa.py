"""
preprocess_noaa.py
------------------
Cleans and aggregates raw NOAA multi-station climate data into a single
daily summary row per date, ready for merging with 311 service request data.

Preprocessing rules applied:
  - AWND  : boolean flag (1 = any station recorded wind speed that day, 0 = none)
  - PRCP  : null → 0 (absence of precip report treated as no precipitation)
  - TMAX/TMIN : daily average across all reporting stations (fills gaps cleanly)
  - WT*   : weather type flags collapsed to a single severe_weather_event boolean

Output columns:
  date, tmax_avg, tmin_avg, prcp_avg, severe_wind, severe_weather_event
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths  (adjust ROOT if running from a different working directory)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_NOAA   = ROOT / "data" / "raw"  / "noaa"  / "noaa_climate_latest.csv"
OUT_DIR    = ROOT / "data" / "processed"
OUT_CSV    = OUT_DIR / "noaa_daily_clean.csv"
OUT_PARQUET = OUT_DIR / "noaa_daily_clean.parquet"


# ---------------------------------------------------------------------------
# Weather-type flag columns that indicate a notable weather event
# WT01 = Fog/Ice fog, WT02 = Heavy fog, WT03 = Thunder, WT08 = Smoke/haze
# ---------------------------------------------------------------------------
WT_COLS = ["WT01", "WT02", "WT03", "WT08"]


def load_raw(path: Path = RAW_NOAA) -> pd.DataFrame:
    """Load raw NOAA CSV from disk."""
    df = pd.read_csv(path, low_memory=False)
    print(f"[NOAA] Loaded {len(df):,} station-day records from {path.name}")
    return df


def clean_and_aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing rules and aggregate to one row per calendar date.

    Steps
    -----
    1. Parse DATE to datetime.
    2. AWND  → boolean (1 if any numeric value present, else 0).
    3. PRCP  → fill nulls with 0 before aggregation.
    4. WT*   → each flag: 1 if present, else 0; collapse to severe_weather_event.
    5. Group by DATE → mean(TMAX), mean(TMIN), mean(PRCP), max(AWND_bool),
                        max(severe_weather_event).
    """

    df = df.copy()

    # -- 1. Parse dates --------------------------------------------------
    df["DATE"] = pd.to_datetime(df["DATE"])

    # -- 2. AWND: boolean ------------------------------------------------
    df["AWND_bool"] = df["AWND"].notna().astype(int)

    # -- 3. PRCP: null → 0 -----------------------------------------------
    df["PRCP"] = df["PRCP"].fillna(0.0)

    # -- 4. Weather-type flags -------------------------------------------
    for col in WT_COLS:
        if col in df.columns:
            df[col] = df[col].notna().astype(int)
        else:
            df[col] = 0

    df["severe_weather_event"] = df[WT_COLS].max(axis=1)

    # -- 5. Aggregate to daily -------------------------------------------
    agg_dict = {
        "TMAX": "mean",
        "TMIN": "mean",
        "PRCP": "mean",          # regional average precipitation
        "AWND_bool": "max",      # 1 if ANY station recorded wind
        "severe_weather_event": "max",
    }

    daily = (
        df.groupby("DATE", as_index=False)
          .agg(agg_dict)
          .rename(columns={
              "TMAX": "tmax_avg",
              "TMIN": "tmin_avg",
              "PRCP": "prcp_avg",
              "AWND_bool": "severe_wind",
          })
    )

    # -- 6. Sort chronologically -----------------------------------------
    daily = daily.sort_values("DATE").reset_index(drop=True)

    # -- 7. Sanity check -------------------------------------------------
    null_pct = daily.isnull().sum() / len(daily) * 100
    print("[NOAA] Null % after preprocessing:")
    print(null_pct.to_string())
    print(f"\n[NOAA] Daily records: {len(daily):,}")
    print(f"[NOAA] Date range   : {daily['DATE'].min().date()} → {daily['DATE'].max().date()}")
    print(f"[NOAA] Severe wind days : {daily['severe_wind'].sum()}")
    print(f"[NOAA] Severe weather events : {daily['severe_weather_event'].sum()}")

    return daily


def save(daily: pd.DataFrame,
         csv_path: Path = OUT_CSV,
         parquet_path: Path = OUT_PARQUET) -> None:
    """Persist processed data to CSV and Parquet."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(csv_path, index=False)
    daily.to_parquet(parquet_path, index=False)
    print(f"\n[NOAA] Saved → {csv_path}")
    print(f"[NOAA] Saved → {parquet_path}")


def run(raw_path: Path = RAW_NOAA) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns the cleaned daily DataFrame."""
    raw  = load_raw(raw_path)
    daily = clean_and_aggregate(raw)
    save(daily)
    return daily


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess NOAA climate data.")
    parser.add_argument(
        "--input", type=Path, default=RAW_NOAA,
        help="Path to raw noaa_climate_latest.csv"
    )
    args = parser.parse_args()
    run(raw_path=args.input)
