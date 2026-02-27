"""
run_preprocessing.py
--------------------
Orchestrates the full preprocessing pipeline:
  1. Clean NOAA climate data  → data/processed/noaa_daily_clean.{csv,parquet}
  2. Clean 311 service data   → data/processed/san_rafael_311_clean.{csv,parquet}
  3. Merge on calendar date   → data/processed/merged_311_noaa.{csv,parquet}

The merged dataset is the primary input for all downstream analytics:
  - Holt-Winters time-series forecasting
  - K-Means geospatial clustering
  - TF-IDF / Logistic Regression NLP classification

Usage
-----
  python src/cleaning/run_preprocessing.py
  python src/cleaning/run_preprocessing.py --311 path/to/311.csv --noaa path/to/noaa.csv
"""

import argparse
import pandas as pd
from pathlib import Path

# Allow running from repo root or src/cleaning/
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cleaning.preprocess_noaa import run as run_noaa, RAW_NOAA
from src.cleaning.preprocess_311  import run as run_311,  RAW_311

ROOT    = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "processed"
MERGED_CSV     = OUT_DIR / "merged_311_noaa.csv"
MERGED_PARQUET = OUT_DIR / "merged_311_noaa.parquet"


def merge(df_311: pd.DataFrame, df_noaa: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join 311 records onto the daily NOAA summary.

    Join key: 311.requested_date == NOAA.DATE (both as date, not datetime).
    All 311 records are preserved; weather columns are NaN for any date
    outside the NOAA range (should be none given overlapping windows).
    """
    # Ensure join keys are the same type (date)
    df_noaa = df_noaa.copy()
    df_noaa["requested_date"] = pd.to_datetime(df_noaa["DATE"]).dt.date
    df_noaa = df_noaa.drop(columns=["DATE"])

    df_311 = df_311.copy()
    df_311["requested_date"] = pd.to_datetime(df_311["requested_date"]).dt.date if \
        not isinstance(df_311["requested_date"].iloc[0], type(pd.Timestamp("today").date())) \
        else df_311["requested_date"]

    merged = df_311.merge(df_noaa, on="requested_date", how="left")

    # Report join quality
    weather_fill = merged["tmax_avg"].notna().mean() * 100
    print(f"\n[MERGE] Records        : {len(merged):,}")
    print(f"[MERGE] Weather coverage: {weather_fill:.1f}% of 311 records matched a NOAA date")

    return merged


def save_merged(df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGED_CSV, index=False)
    df.to_parquet(MERGED_PARQUET, index=False)
    print(f"[MERGE] Saved → {MERGED_CSV}")
    print(f"[MERGE] Saved → {MERGED_PARQUET}")


def run_all(path_311: Path = RAW_311, path_noaa: Path = RAW_NOAA) -> pd.DataFrame:
    print("=" * 60)
    print("STEP 1: Preprocessing NOAA climate data")
    print("=" * 60)
    df_noaa = run_noaa(raw_path=path_noaa)

    print("\n" + "=" * 60)
    print("STEP 2: Preprocessing San Rafael 311 data")
    print("=" * 60)
    df_311 = run_311(raw_path=path_311)

    print("\n" + "=" * 60)
    print("STEP 3: Merging datasets")
    print("=" * 60)
    merged = merge(df_311, df_noaa)
    save_merged(merged)

    print("\n✓ Preprocessing complete. Output files in data/processed/")
    return merged


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full preprocessing pipeline.")
    parser.add_argument("--311",  dest="path_311",  type=Path, default=RAW_311)
    parser.add_argument("--noaa", dest="path_noaa", type=Path, default=RAW_NOAA)
    args = parser.parse_args()
    run_all(path_311=args.path_311, path_noaa=args.path_noaa)
