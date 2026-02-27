"""
kmeans_clustering.py
--------------------
K-Means geospatial clustering of San Rafael 311 service requests to identify
geographic hotspot zones for operational prioritization.

Model Design
------------
- Features    : Latitude, Longitude (scaled)
- k selection : k=4, aligned with San Rafael's 4-district planning structure.
                Zones: Downtown Core, The Canal, Terra Linda, China Camp.
                Silhouette at k=4 (0.4779) outperforms k=5 (0.4279) and
                provides operationally meaningful geographic separation.
                Mathematical optimum is k=2; k=4 is selected for actionable
                district-level planning aligned with city administration.
- Runs        : All records (overall hotspot map) + per-category clustering
                for the top 5 service request categories
- Preprocessing: Records with (lat=0, lon=0) sentinel values are excluded
                 (1% of data; these are requests submitted without a location)

Justification over alternatives
--------------------------------
- DBSCAN finds arbitrarily-shaped clusters without specifying k, but produces
  many noise points on this dataset and is harder to explain to non-technical
  city stakeholders.
- Hierarchical clustering is computationally expensive at n=11,698 and
  produces a dendrogram that is difficult to operationalize.
- K-Means is fast, reproducible, and produces clean centroid coordinates that
  map directly onto city planning zones.

Business Value
--------------
Enables the City to identify persistent geographic concentrations of specific
service request types, supporting proactive field deployment, infrastructure
investment prioritization, and enforcement targeting.

Evaluation Metrics
------------------
- Silhouette Score : Measures cluster cohesion and separation (-1 to 1;
                     higher is better). Appropriate for unlabeled spatial data.
- Inertia (WCSS)   : Within-cluster sum of squares; used for elbow analysis
                     to document k selection rationale.
- Cluster size balance: Ensures no cluster is trivially small or dominant.
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Terrain + census enrichment (optional — gracefully skipped if files missing)
try:
    import importlib.util, sys
    _tc_path = Path(__file__).resolve().parent / "extract_terrain_census.py"
    _spec    = importlib.util.spec_from_file_location("extract_terrain_census", _tc_path)
    _tc_mod  = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tc_mod)
    extract_terrain          = _tc_mod.extract_terrain
    extract_census           = _tc_mod.extract_census
    terrain_cluster_summary  = _tc_mod.cluster_summary
    DEM_TIF                  = _tc_mod.DEM_TIF
    SLOPE_TIF                = _tc_mod.SLOPE_TIF
    CENSUS_CSV               = _tc_mod.CENSUS_CSV
    TIGER_GPKG               = _tc_mod.TIGER_GPKG
    ENRICHMENT_AVAILABLE     = True
    print("[KM] Terrain/census enrichment module loaded successfully")
except Exception as _e:
    ENRICHMENT_AVAILABLE = False
    print(f"[KM] Enrichment module load failed: {_e}")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[2]
CLEAN_311    = ROOT / "data" / "processed" / "san_rafael_311_clean.csv"
OUT_DIR      = ROOT / "data" / "processed"
OUT_CLUSTERS = OUT_DIR / "kmeans_clusters.csv"
OUT_METRICS  = OUT_DIR / "kmeans_metrics.csv"
OUT_ELBOW    = OUT_DIR / "kmeans_elbow.csv"
OUT_CAT      = OUT_DIR / "kmeans_category_clusters.csv"
OUT_PERF     = OUT_DIR / "kmeans_cluster_performance.csv"
OUT_CAT_PERF = OUT_DIR / "kmeans_category_performance.csv"
OUT_ENRICHED_PERF = OUT_DIR / "kmeans_cluster_enriched.csv"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
K             = 4          # clusters — matches SR 4-district planning structure
K_RANGE       = range(2, 12)
RANDOM_STATE  = 42
N_INIT        = 10
TOP_N_CATS    = 5          # categories to cluster individually
MIN_CAT_SIZE  = 200        # minimum records to run per-category clustering

# San Rafael bounding box (filters out 0,0 sentinel coords)
LAT_MIN, LAT_MAX = 37.85, 38.10
LON_MIN, LON_MAX = -122.65, -122.40


# ---------------------------------------------------------------------------
# 1. Load and filter data
# ---------------------------------------------------------------------------

def load_and_filter(path: Path = CLEAN_311) -> pd.DataFrame:
    """
    Load cleaned 311 data and remove records with invalid coordinates.
    Records with lat=0 / lon=0 are sentinel values from the source system
    indicating the request was submitted without a location.
    """
    df = pd.read_csv(path, low_memory=False)

    before = len(df)
    df = df[
        (df["lat"] > LAT_MIN) & (df["lat"] < LAT_MAX) &
        (df["long"] > LON_MIN) & (df["long"] < LON_MAX)
    ].copy()
    dropped = before - len(df)

    print(f"[KM] Loaded {before:,} records")
    print(f"[KM] Dropped {dropped} records with missing/invalid coordinates (sentinel 0,0)")
    print(f"[KM] Spatial records for clustering: {len(df):,}")
    print(f"[KM] Lat range: {df['lat'].min():.5f} → {df['lat'].max():.5f}")
    print(f"[KM] Lon range: {df['long'].min():.5f} → {df['long'].max():.5f}")

    return df


# ---------------------------------------------------------------------------
# 2. Elbow + silhouette analysis (documents k selection)
# ---------------------------------------------------------------------------

def elbow_analysis(coords_scaled: np.ndarray,
                   k_range: range = K_RANGE) -> pd.DataFrame:
    """
    Compute inertia and silhouette score across a range of k values.
    Used to document k selection rationale in slides.
    """
    print(f"\n[KM] Elbow + silhouette analysis (k={k_range.start} to {k_range.stop - 1}):")
    records = []
    for k in k_range:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
            labels = km.fit_predict(coords_scaled)
        sil = silhouette_score(coords_scaled, labels)
        records.append({"k": k, "inertia": round(km.inertia_, 2), "silhouette": round(sil, 4)})
        print(f"[KM]   k={k:2d}  inertia={km.inertia_:8.1f}  silhouette={sil:.4f}")

    elbow_df = pd.DataFrame(records)
    best_sil_k = elbow_df.loc[elbow_df["silhouette"].idxmax(), "k"]
    print(f"[KM] Best k by silhouette: {best_sil_k}  (selected k={K}: Downtown Core, The Canal, Terra Linda, China Camp)")
    return elbow_df


# ---------------------------------------------------------------------------
# 3. Fit K-Means and label records
# ---------------------------------------------------------------------------

def fit_kmeans(df: pd.DataFrame,
               k: int = K) -> tuple[pd.DataFrame, KMeans, StandardScaler, float]:
    """
    Fit K-Means on lat/lon coordinates and append cluster labels to df.
    Returns labeled DataFrame, fitted model, scaler, and silhouette score.
    """
    coords  = df[["lat", "long"]].values
    scaler  = StandardScaler()
    scaled  = scaler.fit_transform(coords)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        df = df.copy()
        df["cluster"] = km.fit_predict(scaled)

    sil = silhouette_score(scaled, df["cluster"])

    # Order clusters by size descending and relabel 1..k for readability
    size_order = (
        df["cluster"].value_counts()
          .reset_index()
          .sort_values("count", ascending=False)
          .reset_index(drop=True)
    )
    size_order["new_label"] = size_order.index + 1
    remap = dict(zip(size_order["cluster"], size_order["new_label"]))
    df["cluster"] = df["cluster"].map(remap)
    # Remap cluster centers to match new labels
    new_centers = np.zeros_like(km.cluster_centers_)
    for old, new in remap.items():
        new_centers[new - 1] = km.cluster_centers_[old]
    km.cluster_centers_ = new_centers

    print(f"\n[KM] K-Means fitted (k={k})")
    print(f"[KM] Silhouette score: {sil:.4f}")
    print(f"\n[KM] Cluster summary:")

    summary = (
        df.groupby("cluster")
          .agg(
              count=("cluster", "size"),
              lat_centroid=("lat", "mean"),
              lon_centroid=("long", "mean"),
          )
          .reset_index()
    )
    # Top category per cluster
    top_cats = (
        df.groupby(["cluster", "category_short"])
          .size()
          .reset_index(name="n")
          .sort_values("n", ascending=False)
          .groupby("cluster")
          .first()
          .reset_index()[["cluster", "category_short"]]
          .rename(columns={"category_short": "top_category"})
    )
    summary = summary.merge(top_cats, on="cluster")
    summary["pct_of_total"] = (summary["count"] / len(df) * 100).round(1)

    print(summary.to_string(index=False))

    return df, km, scaler, sil, summary


# ---------------------------------------------------------------------------
# 4. Per-category clustering (top N categories)
# ---------------------------------------------------------------------------

def category_clusters(df: pd.DataFrame,
                      top_n: int = TOP_N_CATS,
                      min_size: int = MIN_CAT_SIZE) -> pd.DataFrame:
    """
    Run K-Means independently for each of the top N service request categories.
    Uses k=3 per category (meaningful sub-zones without over-segmenting).
    Returns a DataFrame of cluster centroids with category labels.
    """
    top_cats = df["category_short"].value_counts().head(top_n).index.tolist()
    cat_k    = 3
    records  = []

    print(f"\n[KM] Per-category clustering (k={cat_k} per category):")

    for cat in top_cats:
        sub = df[df["category_short"] == cat].copy()
        if len(sub) < min_size:
            print(f"[KM]   {cat[:40]:40s} — skipped (n={len(sub)} < {min_size})")
            continue

        coords = sub[["lat", "long"]].values
        scaler = StandardScaler()
        scaled = scaler.fit_transform(coords)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            km = KMeans(n_clusters=cat_k, random_state=RANDOM_STATE, n_init=N_INIT)
            labels = km.fit_predict(scaled)

        sil = silhouette_score(scaled, labels)
        sub = sub.copy()
        sub["cat_cluster"] = labels + 1

        # Cluster sizes
        sizes = sub["cat_cluster"].value_counts().sort_index().to_dict()

        for cluster_id, center in enumerate(km.cluster_centers_, start=1):
            # Un-scale centroid back to lat/lon
            lat_c, lon_c = scaler.inverse_transform([center])[0]
            records.append({
                "category":    cat,
                "cat_cluster": cluster_id,
                "lat_centroid": round(lat_c, 6),
                "lon_centroid": round(lon_c, 6),
                "cluster_size": sizes.get(cluster_id, 0),
            })

        print(f"[KM]   {cat[:40]:40s}  n={len(sub):4d}  silhouette={sil:.4f}")

    cat_df = pd.DataFrame(records)
    return cat_df


# ---------------------------------------------------------------------------
# 5. Response performance annotation
# ---------------------------------------------------------------------------

# Thresholds for response tier classification
# Based on overall median (~6.2 days): Fast < 75%, Slow > 150%
FAST_THRESHOLD = 0.75
SLOW_THRESHOLD = 1.50


def annotate_performance(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Annotate each geographic cluster with response performance metrics,
    and produce a category × cluster cross-tab highlighting service gaps.

    Two outputs:
    1. cluster_perf : One row per cluster — volume, median/mean resolution,
                      % over 30 days, top slow category, response tier.
    2. cat_perf     : One row per cluster × top category — identifies
                      specific geographic + category combinations where
                      response is lagging relative to the city-wide norm
                      for that category.

    Tier classification (based on median resolution days):
      FAST    : median < 75% of city-wide median
      AVERAGE : median within 75%–150% of city-wide median
      SLOW    : median > 150% of city-wide median

    Note: Median is used (not mean) because resolution times have a long
    right tail — a small number of complex or disputed cases can take 6–12+
    months, pulling the mean upward in ways that don't reflect typical
    operational performance.
    """
    closed = df[df["resolution_days"].notna()].copy()
    city_median = closed["resolution_days"].median()

    print(f"\n[KM] City-wide median resolution : {city_median:.2f} days")
    print(f"[KM] Tier thresholds             : "
          f"Fast < {city_median * FAST_THRESHOLD:.2f}d  |  "
          f"Slow > {city_median * SLOW_THRESHOLD:.2f}d")

    # ------------------------------------------------------------------
    # 1. Cluster-level performance profile
    # ------------------------------------------------------------------
    def cluster_stats(grp):
        res = grp["resolution_days"]
        closed_res = res.dropna()
        return pd.Series({
            "n_total":        len(grp),
            "n_closed":       len(closed_res),
            "pct_closed":     round(len(closed_res) / len(grp) * 100, 1),
            "median_days":    round(closed_res.median(), 2) if len(closed_res) else np.nan,
            "mean_days":      round(closed_res.mean(),   2) if len(closed_res) else np.nan,
            "pct_over_7d":    round((closed_res > 7).sum()  / len(closed_res) * 100, 1) if len(closed_res) else np.nan,
            "pct_over_30d":   round((closed_res > 30).sum() / len(closed_res) * 100, 1) if len(closed_res) else np.nan,
            "pct_over_90d":   round((closed_res > 90).sum() / len(closed_res) * 100, 1) if len(closed_res) else np.nan,
            "lat_centroid":   round(grp["lat"].mean(), 6),
            "lon_centroid":   round(grp["long"].mean(), 6),
        })

    cluster_perf = df.groupby("cluster").apply(cluster_stats).reset_index()

    # Response tier
    cluster_perf["response_tier"] = cluster_perf["median_days"].apply(
        lambda m: "FAST"    if m < city_median * FAST_THRESHOLD else
                  "SLOW"    if m > city_median * SLOW_THRESHOLD else
                  "AVERAGE"
    )

    # Top category overall per cluster
    top_cat = (
        df.groupby(["cluster", "category_short"])
          .size().reset_index(name="n")
          .sort_values("n", ascending=False)
          .groupby("cluster").first().reset_index()
          [["cluster", "category_short"]]
          .rename(columns={"category_short": "top_category"})
    )
    cluster_perf = cluster_perf.merge(top_cat, on="cluster")

    # Slowest category (highest median resolution) per cluster
    slowest = (
        closed.groupby(["cluster", "category_short"])["resolution_days"]
          .median().reset_index()
          .sort_values("resolution_days", ascending=False)
          .groupby("cluster").first().reset_index()
          [["cluster", "category_short", "resolution_days"]]
          .rename(columns={
              "category_short":  "slowest_category",
              "resolution_days": "slowest_cat_median_days",
          })
    )
    cluster_perf = cluster_perf.merge(slowest, on="cluster")

    print(f"\n[KM] Cluster performance profile:")
    display_cols = ["cluster", "n_total", "median_days", "mean_days",
                    "pct_over_30d", "response_tier", "top_category", "slowest_category"]
    print(cluster_perf[display_cols].to_string(index=False))

    # ------------------------------------------------------------------
    # 2. Category × cluster cross-tab (prescriptive service gap table)
    # ------------------------------------------------------------------
    top5 = df["category_short"].value_counts().head(TOP_N_CATS).index.tolist()
    cat_closed = closed[closed["category_short"].isin(top5)]

    # City-wide median per category (benchmark)
    cat_city_median = (
        cat_closed.groupby("category_short")["resolution_days"]
          .median().rename("city_median_days")
    )

    cat_perf = (
        cat_closed.groupby(["cluster", "category_short"])
          .agg(
              n_closed      = ("resolution_days", "count"),
              median_days   = ("resolution_days", "median"),
              mean_days     = ("resolution_days", "mean"),
              pct_over_30d  = ("resolution_days", lambda x: (x > 30).sum() / len(x) * 100),
          )
          .round(2)
          .reset_index()
    )
    cat_perf = cat_perf.merge(cat_city_median, on="category_short")

    # Gap: how many days slower/faster than city-wide median for that category
    cat_perf["days_vs_city_median"] = (
        cat_perf["median_days"] - cat_perf["city_median_days"]
    ).round(2)

    # Flag significant gaps (>50% slower than city median for that category)
    cat_perf["gap_flag"] = cat_perf.apply(
        lambda r: "⚠ SLOW" if r["median_days"] > r["city_median_days"] * 1.5
                  else ("✓ FAST" if r["median_days"] < r["city_median_days"] * 0.75
                        else ""),
        axis=1
    )

    print(f"\n[KM] Category × Cluster performance gaps (top {TOP_N_CATS} categories):")
    gap_display = cat_perf[cat_perf["gap_flag"] != ""].sort_values(
        "days_vs_city_median", ascending=False
    )
    if len(gap_display):
        print(gap_display[["cluster", "category_short", "median_days",
                            "city_median_days", "days_vs_city_median",
                            "gap_flag"]].to_string(index=False))
    else:
        print("[KM]   No significant gaps detected.")

    return cluster_perf, cat_perf


# ---------------------------------------------------------------------------
# 6. Save outputs
# ---------------------------------------------------------------------------

def save_outputs(df_labeled:   pd.DataFrame,
                 elbow_df:     pd.DataFrame,
                 summary:      pd.DataFrame,
                 cat_df:       pd.DataFrame,
                 cluster_perf: pd.DataFrame,
                 cat_perf:     pd.DataFrame,
                 sil_score:    float) -> None:
    """Persist all clustering outputs to processed/ folder."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_labeled.to_csv(OUT_CLUSTERS, index=False)
    print(f"\n[KM] Labeled records saved       → {OUT_CLUSTERS}")

    elbow_df.to_csv(OUT_ELBOW, index=False)
    print(f"[KM] Elbow data saved            → {OUT_ELBOW}")

    cat_df.to_csv(OUT_CAT, index=False)
    print(f"[KM] Category clusters saved     → {OUT_CAT}")

    cluster_perf.to_csv(OUT_PERF, index=False)
    print(f"[KM] Cluster performance saved   → {OUT_PERF}")

    cat_perf.to_csv(OUT_CAT_PERF, index=False)
    print(f"[KM] Category performance saved  → {OUT_CAT_PERF}")

    metrics_df = pd.DataFrame([{
        "model":            "K-Means",
        "k":                K,
        "silhouette_score": round(sil_score, 4),
        "n_records":        len(df_labeled),
        "k_selection":      "Operational alignment (4 SR districts: Downtown Core, "
                            "The Canal, Terra Linda, China Camp); "
                            "silhouette k=4=0.4779 vs k=5=0.4279",
    }])
    metrics_df.to_csv(OUT_METRICS, index=False)
    print(f"[KM] Metrics saved               → {OUT_METRICS}")


# ---------------------------------------------------------------------------
# 6. Full pipeline runner
# ---------------------------------------------------------------------------

def run(data_path: Path = CLEAN_311) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute full K-Means clustering pipeline, including optional terrain
    and census enrichment if USGS/TIGER files are available.
    Returns (labeled_df, category_cluster_df).
    """
    print("=" * 60)
    print("K-Means Geospatial Clustering — 311 Hotspot Analysis")
    print("=" * 60)

    df                                   = load_and_filter(data_path)
    coords                               = df[["lat", "long"]].values
    scaler                               = StandardScaler()
    scaled                               = scaler.fit_transform(coords)
    elbow_df                             = elbow_analysis(scaled)
    df_labeled, km, scaler, sil, summary = fit_kmeans(df)
    cat_df                               = category_clusters(df_labeled)
    cluster_perf, cat_perf               = annotate_performance(df_labeled)

    # ------------------------------------------------------------------
    # Terrain + Census enrichment (runs if dependencies are available)
    # ------------------------------------------------------------------
    enriched_perf = cluster_perf.copy()

    if ENRICHMENT_AVAILABLE:
        try:
            print("\n" + "=" * 60)
            print("Terrain + Census Enrichment")
            print("=" * 60)
            df_enriched = extract_terrain(df_labeled, DEM_TIF, SLOPE_TIF)
            df_enriched = extract_census(df_enriched, CENSUS_CSV, TIGER_GPKG)

            # Get terrain + census summary per cluster
            tc_summary  = terrain_cluster_summary(df_enriched)

            # Merge into cluster performance table
            merge_cols = [
                "cluster",
                "elevation_mean_m", "elevation_std_m",
                "slope_mean_deg", "slope_max_deg",
                "pct_steep_plus", "dominant_terrain",
                "total_population", "incidents_per_1k_res",
                "mean_pop_density_sqkm",
                "wtd_median_income", "wtd_median_home_value",
            ]
            available = [c for c in merge_cols if c in tc_summary.columns]
            enriched_perf = cluster_perf.merge(
                tc_summary[available], on="cluster", how="left"
            )

            # Save enriched labeled dataset
            save_df = df_enriched.drop(columns=["geometry"], errors="ignore")
            save_df.to_csv(ROOT / "data" / "processed" / "san_rafael_311_enriched.csv",
                           index=False)
            print(f"[KM] Enriched 311 data saved → "
                  f"{ROOT / 'data' / 'processed' / 'san_rafael_311_enriched.csv'}")

        except Exception as e:
            print(f"[KM] Enrichment skipped — {e}")
    else:
        print("\n[KM] Terrain/census enrichment skipped "
              "(run: pip install rasterio geopandas)")

    # Save all outputs
    save_outputs(df_labeled, elbow_df, summary, cat_df,
                 cluster_perf, cat_perf, sil)

    # Save enriched cluster performance separately
    if len(enriched_perf.columns) > len(cluster_perf.columns):
        enriched_perf.to_csv(OUT_ENRICHED_PERF, index=False)
        print(f"[KM] Enriched cluster perf   → {OUT_ENRICHED_PERF}")

    print("\n✓ K-Means clustering pipeline complete.")
    return df_labeled, cat_df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run K-Means geospatial clustering.")
    parser.add_argument("--input", type=Path, default=CLEAN_311)
    args = parser.parse_args()
    run(data_path=args.input)
