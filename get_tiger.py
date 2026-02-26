from src.ingestion.tiger_shapefiles import TIGERIngestor

print("Downloading TIGER shapefiles...")
print("Note: Large files, will take 5-10 minutes\n")

ingestor = TIGERIngestor()

# Get county, tracts, and city boundaries
datasets = ingestor.ingest(layers=['county', 'tracts', 'places'])

if datasets:
    for name, gdf in datasets.items():
        print(f"✓ {name}: {len(gdf)} features")
        print(f"  CRS: {gdf.crs}")
        print(f"  Columns: {gdf.columns.tolist()[:5]}...")  # First 5 columns
    print("\n✓ TIGER shapefiles complete!")
else:
    print("✗ TIGER download failed")