from src.ingestion.census_data import CensusIngestor

print("Fetching Census data...")
ingestor = CensusIngestor()

# Add include_tracts=True to get granular data
datasets = ingestor.ingest(include_tracts=True)

if datasets:
    for name, df in datasets.items():
        print(f"✓ {name}: {len(df)} records")
        if 'tract' in name.lower():
            print(f"  Columns: {df.columns.tolist()}")
    print("\n✓ Census data complete!")
else:
    print("✗ Census ingestion failed")