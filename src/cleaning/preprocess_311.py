"""
preprocess_311.py
-----------------
Cleans and enriches the San Rafael 311 service request data, producing a
feature-ready DataFrame for time-series forecasting, geospatial clustering,
and NLP text classification.

Preprocessing steps
-------------------
1.  Parse all datetime columns; handle the sentinel value 1/1/1900 (= not closed).
2.  Derive calendar features: date, year, month, week, day_of_week, hour.
3.  Compute resolution_days (days from open to close) where applicable.
4.  Shorten category labels (strip Spanish translation suffix).
5.  Encode status as binary: closed=1, open=0.
6.  Drop rows missing lat/long (critical for spatial analysis).
7.  Flag rows with usable free-text descriptions.
8.  Output a clean CSV and Parquet ready for merging with NOAA data.

Output columns (key)
--------------------
  object_id, service_request_id, category, category_short, status,
  status_binary, requested_date, requested_year, requested_month,
  requested_week, requested_day_of_week, requested_hour,
  closed_date, resolution_days, lat, long, district, zipcode, description,
  has_description
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT        = Path(__file__).resolve().parents[2]
RAW_311     = ROOT / "data" / "raw" / "311_requests" / "san_rafael_311.csv"
OUT_DIR     = ROOT / "data" / "processed"
OUT_CSV     = OUT_DIR / "san_rafael_311_clean.csv"
OUT_PARQUET = OUT_DIR / "san_rafael_311_clean.parquet"

# Sentinel used by the source system for "not yet closed"
SENTINEL_DATE = pd.Timestamp("1900-01-01")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shorten_category(cat: str) -> str:
    """
    Strip the Spanish translation portion from bilingual category labels.
    E.g. 'Illegal Dumping / Desecho ilegal' → 'Illegal Dumping'
    """
    if not isinstance(cat, str):
        return cat
    # Split on ' / ' and take the first (English) segment
    return cat.split(" / ")[0].strip()


def _parse_dt(series: pd.Series) -> pd.Series:
    """Flexibly parse mixed datetime strings."""
    return pd.to_datetime(series, errors="coerce")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def load_raw(path: Path = RAW_311) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"[311] Loaded {len(df):,} raw records from {path.name}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ------------------------------------------------------------------ #
    # 1. Rename columns to snake_case for consistency                     #
    # ------------------------------------------------------------------ #
    df.columns = [c.lower().strip() for c in df.columns]
    df = df.rename(columns={
        "objectid": "object_id",
        "long":     "long",
        "lat":      "lat",
    })

    # ------------------------------------------------------------------ #
    # 2. Parse datetime columns                                            #
    # ------------------------------------------------------------------ #
    df["requested_datetime"] = _parse_dt(df["requested_datetime"])
    df["updated_datetime"]   = _parse_dt(df["updated_datetime"])
    df["date_closed"]        = _parse_dt(df["date_closed"])

    # Treat sentinel 1900-01-01 as NaT (request not yet closed)
    df.loc[df["date_closed"].dt.year == 1900, "date_closed"] = pd.NaT

    # ------------------------------------------------------------------ #
    # 3. Calendar features from requested_datetime                        #
    # ------------------------------------------------------------------ #
    dt = df["requested_datetime"]
    df["requested_date"]        = dt.dt.date
    df["requested_year"]        = dt.dt.year
    df["requested_month"]       = dt.dt.month
    df["requested_week"]        = dt.dt.isocalendar().week.astype("Int64")
    df["requested_day_of_week"] = dt.dt.day_name()
    df["requested_hour"]        = dt.dt.hour

    # ------------------------------------------------------------------ #
    # 4. Resolution time (days open → closed)                             #
    # ------------------------------------------------------------------ #
    df["resolution_days"] = (
        (df["date_closed"] - df["requested_datetime"])
        .dt.total_seconds() / 86_400
    ).round(2)
    # Negative or implausibly large values → NaN
    df.loc[df["resolution_days"] < 0,    "resolution_days"] = np.nan
    df.loc[df["resolution_days"] > 1_825, "resolution_days"] = np.nan  # > 5 yrs

    # ------------------------------------------------------------------ #
    # 5. Shorten category labels                                          #
    # ------------------------------------------------------------------ #
    df["category_short"] = df["category"].apply(_shorten_category)

    # ------------------------------------------------------------------ #
    # 6. Binary status                                                     #
    # ------------------------------------------------------------------ #
    df["status_binary"] = (df["status"].str.lower() == "closed").astype(int)

    # ------------------------------------------------------------------ #
    # 7. Drop rows missing coordinates (needed for spatial analysis)      #
    # ------------------------------------------------------------------ #
    before = len(df)
    df = df.dropna(subset=["lat", "long"])
    dropped = before - len(df)
    if dropped:
        print(f"[311] Dropped {dropped} rows missing lat/long")

    # ------------------------------------------------------------------ #
    # 8. Description quality flag                                          #
    # ------------------------------------------------------------------ #
    df["has_description"] = (
        df["description"].notna() &
        (df["description"].str.strip().str.len() > 5)
    ).astype(int)

    # ------------------------------------------------------------------ #
    # 9. Clean up column order                                             #
    # ------------------------------------------------------------------ #
    ordered_cols = [
        "object_id", "service_request_id",
        "category", "category_short",
        "status", "status_binary",
        "requested_datetime", "requested_date",
        "requested_year", "requested_month", "requested_week",
        "requested_day_of_week", "requested_hour",
        "date_closed", "resolution_days",
        "lat", "long", "address", "zipcode", "district",
        "description", "has_description",
    ]
    # Keep any extra columns at the end
    extra = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra]

    # ------------------------------------------------------------------ #
    # 10. Summary                                                          #
    # ------------------------------------------------------------------ #
    print(f"[311] Clean records  : {len(df):,}")
    print(f"[311] Date range     : {df['requested_date'].min()} → {df['requested_date'].max()}")
    print(f"[311] Categories     : {df['category_short'].nunique()}")
    print(f"[311] With description: {df['has_description'].sum():,} ({df['has_description'].mean()*100:.1f}%)")
    print(f"[311] Closed         : {df['status_binary'].sum():,} ({df['status_binary'].mean()*100:.1f}%)")
    null_pct = df[["lat","long","requested_datetime","category"]].isnull().sum()
    print(f"[311] Nulls in key columns:\n{null_pct.to_string()}")

    return df


def save(df: pd.DataFrame,
         csv_path: Path = OUT_CSV,
         parquet_path: Path = OUT_PARQUET) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    print(f"\n[311] Saved → {csv_path}")
    print(f"[311] Saved → {parquet_path}")


def run(raw_path: Path = RAW_311) -> pd.DataFrame:
    """Full preprocessing pipeline. Returns the cleaned DataFrame."""
    raw   = load_raw(raw_path)
    clean_df = clean(raw)
    save(clean_df)
    return clean_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess San Rafael 311 data.")
    parser.add_argument(
        "--input", type=Path, default=RAW_311,
        help="Path to raw san_rafael_311.csv"
    )
    args = parser.parse_args()
    run(raw_path=args.input)
