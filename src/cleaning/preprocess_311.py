"""
preprocess_311.py
-----------------
Cleans and enriches the San Rafael 311 service request data, producing a
feature-ready DataFrame for time-series forecasting, geospatial clustering,
and NLP text classification.

Data source
-----------
Primary: ArcGIS FeatureServer (SeeClickFix production export)
  Endpoint : ARCGIS_ENDPOINT constant below
  Auth     : Public — no API key required
  Paging   : 2,000 records per request (server maxRecordCount)
  Dates    : Returned as Unix millisecond timestamps; converted to datetime
  Fallback : If API is unreachable, falls back to local CSV at RAW_311

Preprocessing steps
-------------------
1.  Fetch all records from ArcGIS FeatureServer via paginated REST queries.
2.  Convert Unix ms timestamps to datetime; handle sentinel -2208988800000
    (= 1900-01-01, meaning request not yet closed).
3.  Derive calendar features: date, year, month, week, day_of_week, hour.
4.  Compute resolution_days (days from open to close) where applicable.
5.  Shorten category labels (strip Spanish translation suffix).
6.  Encode status as binary: closed=1, open=0.
7.  Drop rows missing lat/long (critical for spatial analysis).
8.  Flag rows with usable free-text descriptions.
9.  Output a clean CSV and Parquet ready for merging with NOAA data.

Output columns (key)
--------------------
  object_id, service_request_id, category, category_short, status,
  status_binary, requested_date, requested_year, requested_month,
  requested_week, requested_day_of_week, requested_hour,
  closed_date, resolution_days, lat, long, district, zipcode, description,
  has_description
"""

import re
import time
import requests
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

# ArcGIS FeatureServer endpoint — San Rafael SeeClickFix 311 data (public, no auth)
ARCGIS_ENDPOINT = (
    "https://services5.arcgis.com/sruoiBDPu8SihcGN/arcgis/rest/services"
    "/seeclickfix_production_public_gdb/FeatureServer/0/query"
)
ARCGIS_PAGE_SIZE = 2000   # server maxRecordCount

# Sentinel used by the source system for "not yet closed"
# CSV source  : date string parsed to 1900-01-01
# API source  : Unix ms timestamp -2208988800000 (= 1900-01-01 00:00:00 UTC)
SENTINEL_DATE    = pd.Timestamp("1900-01-01")
SENTINEL_DATE_MS = -2208988800000


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

def fetch_from_api(endpoint: str = ARCGIS_ENDPOINT,
                   page_size: int = ARCGIS_PAGE_SIZE) -> pd.DataFrame:
    """
    Fetch all 311 records from the ArcGIS FeatureServer via paginated REST queries.

    ArcGIS returns dates as Unix millisecond timestamps. This function leaves
    them as raw integers; clean() handles the conversion so the pipeline
    remains agnostic to the source.

    Paging strategy: increment resultOffset by page_size until the response
    no longer sets exceededTransferLimit=true, or returns fewer records than
    page_size.
    """
    params = {
        "outFields": "*",
        "where":     "1=1",
        "f":         "json",
        "orderByFields": "OBJECTID",
        "resultRecordCount": page_size,
        "resultOffset": 0,
    }

    all_records = []
    page = 0

    while True:
        params["resultOffset"] = page * page_size
        try:
            resp = requests.get(endpoint, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            raise RuntimeError(f"[311] ArcGIS API request failed (page {page}): {e}")

        if "error" in data:
            raise RuntimeError(f"[311] ArcGIS API error: {data['error']}")

        features = data.get("features", [])
        if not features:
            break

        records = [f["attributes"] for f in features]
        all_records.extend(records)

        n_fetched = len(all_records)
        exceeded  = data.get("exceededTransferLimit", False)
        print(f"[311] Page {page+1}: +{len(records):,} records  "
              f"(total so far: {n_fetched:,})"
              + ("  [more pages...]" if exceeded else "  [complete]"))

        if not exceeded or len(records) < page_size:
            break

        page += 1
        time.sleep(0.1)   # polite pacing — avoid hammering the server

    df = pd.DataFrame(all_records)
    # Normalise DISTRICT column name (API returns uppercase)
    if "DISTRICT" in df.columns and "district" not in df.columns:
        df = df.rename(columns={"DISTRICT": "district"})
    print(f"[311] API fetch complete: {len(df):,} total records")
    return df


def load_raw(path: Path = RAW_311) -> pd.DataFrame:
    """Load raw 311 data from local CSV (fallback / offline use)."""
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
    # Handle both sources:
    #   CSV source : string datetimes  → pd.to_datetime() parses directly
    #   API source : Unix ms integers  → pd.to_datetime(..., unit="ms")
    for col in ["requested_datetime", "updated_datetime", "date_closed"]:
        if pd.api.types.is_numeric_dtype(df[col]):
            # API source: Unix milliseconds; sentinel -2208988800000 → 1900-01-01
            df[col] = pd.to_datetime(df[col], unit="ms", errors="coerce")
        else:
            df[col] = _parse_dt(df[col])

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


def run(raw_path: Path = RAW_311,
        use_api: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns the cleaned DataFrame.

    Parameters
    ----------
    raw_path : Path
        Path to local CSV fallback (used when use_api=False or API unreachable).
    use_api  : bool
        If True (default), fetch live data from the ArcGIS FeatureServer.
        If False, load from local CSV at raw_path.
    """
    if use_api:
        try:
            raw = fetch_from_api()
        except Exception as e:
            print(f"[311] API fetch failed: {e}")
            print(f"[311] Falling back to local CSV: {raw_path}")
            raw = load_raw(raw_path)
    else:
        raw = load_raw(raw_path)

    clean_df = clean(raw)
    save(clean_df)
    return clean_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess San Rafael 311 data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python preprocess_311.py                  # fetch from ArcGIS API\n"
            "  python preprocess_311.py --no-api         # use local CSV only\n"
            "  python preprocess_311.py --no-api --input path/to/file.csv"
        )
    )
    parser.add_argument(
        "--input", type=Path, default=RAW_311,
        help="Path to local CSV fallback (default: data/raw/311_requests/san_rafael_311.csv)"
    )
    parser.add_argument(
        "--no-api", dest="no_api", action="store_true",
        help="Skip API fetch and use local CSV only"
    )
    args = parser.parse_args()
    run(raw_path=args.input, use_api=not args.no_api)
