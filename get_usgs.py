from src.ingestion.usgs_elevation import USGSElevationIngestor

print("Downloading USGS elevation data...")
print("Note: Large DEM download, will take ~5 minutes\n")

ingestor = USGSElevationIngestor()

# This will download DEM and calculate slope
results = ingestor.ingest(calculate_slope=True)

if results:
    print("\n✓ USGS elevation data complete!")
    for data_type, filepath in results.items():
        print(f"  {data_type}: {filepath}")
else:
    print("✗ USGS download failed")
    print("\nIf download fails, you can manually download from:")
    print("https://apps.nationalmap.gov/downloader/")
    print("Search: San Rafael, CA")
    print("Product: 1/3 arc-second DEM")