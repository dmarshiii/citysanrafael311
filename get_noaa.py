from src.ingestion.noaa_climate import NOAAIngestor

ingestor = NOAAIngestor()
df = ingestor.ingest()

if not df.empty:
    print(f"✓ Got {len(df)} days of climate data")
else:
    print("✗ Failed - check API token")