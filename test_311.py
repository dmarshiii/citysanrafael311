import pandas as pd

# Replace with your actual download path
csv_path = r"C:\Users\dmars\Downloads\San Rafael 311.csv"

# Load it
df = pd.read_csv(csv_path)

print(f"✓ Loaded {len(df)} records!")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Save it to your data folder
output_path = "data/raw/311_requests/san_rafael_311.csv"
df.to_csv(output_path, index=False)
print(f"\n✓ Saved to {output_path}")