"""
extract_terrain_census.py
--------------------------
Enriches San Rafael 311 service request records with two spatial context
layers:

  1. TERRAIN  (USGS DEM elevation raster)
     - elevation_m    : Elevation in meters at the request location
     - terrain_class  : Categorical zone derived from elevation bands:
                          Lowland  < 15m   (bay flats, Canal district)
                          Valley   15-60m  (mid-elevation residential)
                          Hillside 60-150m (Terra Linda, Dominican hills)
                          Ridge    >= 150m (Marin headlands, open space)

     Note on slope raster: The pre-computed slope raster was inspected and
     found to contain a processing artifact — only 1,373 genuinely valid
     pixels exist across the San Rafael area (out of ~464,000 DEM-valid
     pixels), with nodata encoded as floating-point values climbing toward
     90°. It is excluded. Terrain classification is derived from elevation
     bands instead, which correlate meaningfully with SR's geography.

  2. DEMOGRAPHICS  (U.S. Census ACS 2022 x TIGER/Line tract boundaries)
     - tract_geoid          : Census tract GEOID
     - population           : Tract total population (B01003_001E)
     - pop_density_per_sqkm : Population per sq km of land area
     - median_income        : Median household income; NaN if suppressed
     - median_home_value    : Median home value (B25077_001E)

Dependencies: pip install rasterio geopandas
"""

import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from pathlib import Path
from shapely.geometry import Point


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
CLEAN_311    = ROOT / "data" / "processed" / "san_rafael_311_clean.csv"
CENSUS_CSV   = ROOT / "data" / "raw"       / "census" / "census_tracts_2022.csv"
TIGER_GPKG   = ROOT / "data" / "raw"       / "tiger"  / "tiger_tracts_2023.gpkg"
DEM_TIF      = ROOT / "data" / "raw"       / "usgs"   / "elevation_dem_20260215_144458.tif"
SLOPE_TIF    = ROOT / "data" / "raw"       / "usgs"   / "slope_elevation_dem_20260215_144458.tif"
OUT_DIR      = ROOT / "data" / "processed"
OUT_ENRICHED = OUT_DIR / "san_rafael_311_enriched.csv"
OUT_SUMMARY  = OUT_DIR / "terrain_census_summary.csv"

LAT_MIN, LAT_MAX = 37.85, 38.10
LON_MIN, LON_MAX = -122.65, -122.40

ELEV_BINS   = [-np.inf, 15, 60, 150, np.inf]
ELEV_LABELS = ["Lowland", "Valley", "Hillside", "Ridge"]


# ---------------------------------------------------------------------------
# 1. Load and filter 311 data
# ---------------------------------------------------------------------------

def load_311(path: Path = CLEAN_311) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    before = len(df)
    df = df[
        (df["lat"]  > LAT_MIN) & (df["lat"]  < LAT_MAX) &
        (df["long"] > LON_MIN) & (df["long"] < LON_MAX)
    ].copy()
    print(f"[TC] Loaded {before:,} records, {len(df):,} with valid coordinates")
    return df


# ---------------------------------------------------------------------------
# 2. Extract elevation at each 311 point
# ---------------------------------------------------------------------------

def extract_terrain(df: pd.DataFrame,
                    dem_path:   Path = DEM_TIF,
                    slope_path: Path = SLOPE_TIF) -> pd.DataFrame:
    df   = df.copy()
    lats = df["lat"].values
    lons = df["long"].values
    n    = len(df)

    elevations = np.full(n, np.nan)
    print(f"\n[TC] Extracting elevation for {n:,} points...")

    with rasterio.open(dem_path) as src:
        data   = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data < -10]  = np.nan
        data[data > 1000] = np.nan
        height, width = data.shape

        for i, (lat, lon) in enumerate(zip(lats, lons)):
            try:
                row, col = src.index(lon, lat)
                if 0 <= row < height and 0 <= col < width:
                    val = data[row, col]
                    if np.isfinite(val):
                        elevations[i] = val
                    else:
                        patch = data[max(0,row-1):row+2, max(0,col-1):col+2]
                        valid = patch[np.isfinite(patch)]
                        if len(valid):
                            elevations[i] = valid.mean()
            except Exception:
                pass

    df["elevation_m"] = np.round(elevations, 1)
    df["slope_deg"]   = np.nan  # excluded — see docstring

    df["terrain_class"] = pd.cut(
        df["elevation_m"],
        bins=ELEV_BINS,
        labels=ELEV_LABELS,
        right=False
    ).astype(str)
    df.loc[df["elevation_m"].isna(), "terrain_class"] = np.nan

    valid = df["elevation_m"].notna().sum()
    print(f"[TC] Elevation extracted   : {valid:,} of {n:,} ({valid/n*100:.1f}%)")
    print(f"[TC] Range                 : "
          f"{df['elevation_m'].min():.1f} → {df['elevation_m'].max():.1f} m")
    print(f"[TC] Terrain class distribution:")
    print(df["terrain_class"].value_counts().to_string())

    return df


# ---------------------------------------------------------------------------
# 3. Assign census tract + demographics
# ---------------------------------------------------------------------------

def extract_census(df: pd.DataFrame,
                   census_path: Path = CENSUS_CSV,
                   tiger_path:  Path = TIGER_GPKG) -> pd.DataFrame:
    df = df.copy()

    census = pd.read_csv(census_path)
    census["tract_geoid"] = (
        census["state"].astype(str).str.zfill(2) +
        census["county"].astype(str).str.zfill(3) +
        census["tract"].astype(str).str.zfill(6)
    )
    census.loc[census["B19013_001E"] < 0, "B19013_001E"] = np.nan
    census = census.rename(columns={
        "B01003_001E": "population",
        "B19013_001E": "median_income",
        "B25077_001E": "median_home_value",
    })[["tract_geoid", "population", "median_income", "median_home_value"]]
    print(f"\n[TC] Census tracts loaded  : {len(census):,}")
    print(f"[TC] Sample GEOIDs         : {census['tract_geoid'].head(3).tolist()}")

    print(f"[TC] Loading TIGER shapefile...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tiger = gpd.read_file(tiger_path)

    geoid_col = next(
        (c for c in tiger.columns if c.upper() in ("GEOID", "GEOID20", "GEO_ID")),
        None
    )
    if geoid_col is None:
        raise ValueError(f"No GEOID column found. Columns: {tiger.columns.tolist()}")

    tiger = tiger.rename(columns={geoid_col: "tract_geoid"})
    tiger = tiger[["tract_geoid", "geometry"]].copy()
    if tiger.crs != CRS.from_epsg(4326):
        tiger = tiger.to_crs(epsg=4326)

    tiger_marin = tiger[tiger["tract_geoid"].str.startswith("06041")].copy()
    print(f"[TC] Marin County tracts   : {len(tiger_marin):,}")

    tiger_ea = tiger_marin.to_crs(epsg=3310)
    tiger_marin = tiger_marin.copy()
    tiger_marin["land_area_sqkm"] = (tiger_ea.geometry.area / 1_000_000).round(4)

    geometry = [Point(lon, lat) for lon, lat in zip(df["long"], df["lat"])]
    gdf_311  = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

    print(f"[TC] Running spatial join  ({len(gdf_311):,} points × {len(tiger_marin):,} tracts)...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        joined = gpd.sjoin(
            gdf_311,
            tiger_marin[["tract_geoid", "land_area_sqkm", "geometry"]],
            how="left",
            predicate="within"
        )
    joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right"], errors="ignore"))
    joined = joined.merge(census, on="tract_geoid", how="left")
    joined["pop_density_per_sqkm"] = (
        joined["population"] / joined["land_area_sqkm"]
    ).round(1)

    matched = joined["tract_geoid"].notna().sum()
    print(f"[TC] Points matched        : {matched:,} ({matched/len(joined)*100:.1f}%)")
    print(f"[TC] Pop density range     : "
          f"{joined['pop_density_per_sqkm'].min():.0f} → "
          f"{joined['pop_density_per_sqkm'].max():.0f} /sq km")
    print(f"[TC] Income range          : "
          f"${joined['median_income'].min():,.0f} → "
          f"${joined['median_income'].max():,.0f}")

    return joined


# ---------------------------------------------------------------------------
# 4. Per-cluster summary
# ---------------------------------------------------------------------------

def cluster_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate terrain + census to cluster level.
    Deduplicates by census tract before summing population to avoid
    counting each tract's population once per incident.
    """
    if "cluster" not in df.columns:
        print("[TC] No 'cluster' column — skipping summary")
        return pd.DataFrame()

    records = []

    for cluster_id, grp in df.groupby("cluster"):
        elev = grp["elevation_m"].dropna()
        tc   = grp["terrain_class"].value_counts()
        dominant_terrain = tc.index[0] if len(tc) else np.nan

        if "tract_geoid" in grp.columns:
            ut = (
                grp[["tract_geoid", "population", "median_income",
                      "median_home_value", "pop_density_per_sqkm"]]
                .dropna(subset=["tract_geoid"])
                .drop_duplicates(subset="tract_geoid")
                .copy()
            )
            total_pop    = int(ut["population"].sum())
            mean_density = float(ut["pop_density_per_sqkm"].mean())
            w = ut["population"].values.astype(float)

            def wavg(col):
                v = ut[col].values.astype(float)
                m = np.isfinite(v) & np.isfinite(w)
                return float(np.average(v[m], weights=w[m])) if m.any() else np.nan

            inc_wavg  = wavg("median_income")
            home_wavg = wavg("median_home_value")
        else:
            total_pop = 0
            mean_density = inc_wavg = home_wavg = np.nan

        records.append({
            "cluster":               cluster_id,
            "n_incidents":           len(grp),
            "elevation_mean_m":      round(float(elev.mean()), 1) if len(elev) else np.nan,
            "elevation_std_m":       round(float(elev.std()),  1) if len(elev) else np.nan,
            "elevation_min_m":       round(float(elev.min()),  1) if len(elev) else np.nan,
            "elevation_max_m":       round(float(elev.max()),  1) if len(elev) else np.nan,
            "dominant_terrain":      dominant_terrain,
            "lat_centroid":          round(grp["lat"].mean(), 6),
            "lon_centroid":          round(grp["long"].mean(), 6),
            "total_population":      total_pop,
            "incidents_per_1k_res":  round(len(grp) / total_pop * 1000, 2) if total_pop > 0 else np.nan,
            "mean_pop_density_sqkm": round(mean_density, 1),
            "wtd_median_income":     round(inc_wavg, 0)  if np.isfinite(inc_wavg)  else np.nan,
            "wtd_median_home_value": round(home_wavg, 0) if np.isfinite(home_wavg) else np.nan,
        })

    summary = pd.DataFrame(records)

    print(f"\n[TC] Cluster terrain + census summary:")
    cols = ["cluster", "n_incidents", "elevation_mean_m", "dominant_terrain",
            "total_population", "incidents_per_1k_res",
            "mean_pop_density_sqkm", "wtd_median_income"]
    print(summary[cols].to_string(index=False))

    return summary


# ---------------------------------------------------------------------------
# 5. Save
# ---------------------------------------------------------------------------

def save_outputs(df_enriched: pd.DataFrame, summary: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_enriched.drop(columns=["geometry"], errors="ignore").to_csv(OUT_ENRICHED, index=False)
    print(f"\n[TC] Enriched records saved  → {OUT_ENRICHED}")
    if len(summary):
        summary.to_csv(OUT_SUMMARY, index=False)
        print(f"[TC] Cluster summary saved   → {OUT_SUMMARY}")


# ---------------------------------------------------------------------------
# 6. Runner
# ---------------------------------------------------------------------------

def run(data_path=CLEAN_311, dem_path=DEM_TIF, slope_path=SLOPE_TIF,
        census_path=CENSUS_CSV, tiger_path=TIGER_GPKG):
    print("=" * 60)
    print("Terrain + Census Enrichment — 311 Service Requests")
    print("=" * 60)
    df      = load_311(data_path)
    df      = extract_terrain(df, dem_path, slope_path)
    df      = extract_census(df, census_path, tiger_path)
    summary = cluster_summary(df) if "cluster" in df.columns else pd.DataFrame()
    save_outputs(df, summary)
    print("\n✓ Enrichment complete.")
    return df


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path, default=CLEAN_311)
    p.add_argument("--dem",    type=Path, default=DEM_TIF)
    p.add_argument("--slope",  type=Path, default=SLOPE_TIF)
    p.add_argument("--census", type=Path, default=CENSUS_CSV)
    p.add_argument("--tiger",  type=Path, default=TIGER_GPKG)
    a = p.parse_args()
    run(a.input, a.dem, a.slope, a.census, a.tiger)
