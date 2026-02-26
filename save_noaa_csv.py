import pandas as pd

# Load the downloaded CSV (adjust path to your actual file)
csv_path = r"C:\Users\dmars\Downloads\4231089.csv"  # Your filename will differ

df = pd.read_csv(csv_path)

print(f"✓ Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# Save to project
df.to_parquet('data/raw/noaa/noaa_climate_latest.parquet')
df.to_csv('data/raw/noaa/noaa_climate_latest.csv', index=False)

print(f"✓ Saved NOAA data to data/raw/noaa/")