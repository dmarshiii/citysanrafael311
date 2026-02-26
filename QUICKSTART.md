# Quick Start Guide - 311 AI Ingestion Pipeline

## Get Started in 5 Minutes

### Step 1: Setup (2 minutes)

```bash
cd ai_311_project
pip install -r requirements.txt
cp .env.example .env
```

### Step 2: Configure API Keys (if available)

Edit `.env` file:
```bash
# If you have Census API key:
CENSUS_API_KEY=your_census_key_here

# If you have NOAA token:
NOAA_API_TOKEN=your_noaa_token_here
```

**Don't have API keys yet?** No problem - see Step 3.

### Step 3: Choose Your Mode

#### Option A: API Mode (if you have keys)
```bash
python src/run_ingestion.py --skip-usgs
```

#### Option B: CSV Fallback Mode (no keys needed)
```bash
# Set in .env:
USE_CSV_FALLBACK=true

# Download data manually from URLs provided below
# Then run:
python src/run_ingestion.py \
  --csv-311 /path/to/downloaded/311_data.csv \
  --csv-noaa /path/to/downloaded/noaa_data.csv \
  --skip-usgs
```

#### Option C: Mixed Mode
```bash
# Use API for some sources, CSV for others
# Just provide CSV paths for sources without API keys
python src/run_ingestion.py \
  --csv-noaa /path/to/noaa_data.csv \
  --skip-usgs
# Will use API for 311, Census, TIGER
```

### Step 4: Check Results

```bash
# View ingested data
ls -lh data/raw/*/

# Quick peek at 311 data
python -c "import pandas as pd; df = pd.read_parquet('data/raw/311_requests/san_rafael_311_latest.parquet'); print(df.head())"
```

## Quick Download Links for CSV Mode

### 1. San Rafael 311 Data
1. Go to: https://open-data-portal-san-rafael.hub.arcgis.com/datasets/8bbb4a9b7034470784f35dfe91e6be8a
2. Click "Download" → "Spreadsheet" (CSV)
3. Save as `311_requests.csv`

### 2. NOAA Climate Data
1. Go to: https://www.ncei.noaa.gov/cdo-web/search
2. Search: "San Rafael, CA"
3. Select station (e.g., "SAN RAFAEL")
4. Choose: Daily Summaries
5. Date range: 2022-06-01 to 2026-02-15
6. Add to cart → Download CSV
7. Save as `noaa_climate.csv`

### 3. Census Data (Optional for CSV mode)
1. Go to: https://data.census.gov/
2. Search: "Marin County, California"
3. Select tables: B01003 (Population), B19013 (Income)
4. Download CSV
5. Save as `census_data.csv`

## API Key Registration (for future runs)

### Census API (instant)
```bash
# Visit and fill form:
https://api.census.gov/data/key_signup.html

# You'll get key immediately via email
# Add to .env:
CENSUS_API_KEY=your_key_here
```

### NOAA API (1-2 days)
```bash
# Visit and fill form:
https://www.ncdc.noaa.gov/cdo-web/token

# Token arrives via email in 1-2 business days
# Add to .env:
NOAA_API_TOKEN=your_token_here
```

## Testing Individual Sources

Test one source at a time:

```bash
# Test 311 ingestion
python src/ingestion/san_rafael_311.py

# Test Census ingestion
python src/ingestion/census_data.py

# Test TIGER shapefiles
python src/ingestion/tiger_shapefiles.py

# Test NOAA (requires API token)
python src/ingestion/noaa_climate.py

# Test USGS (large download, ~5 min)
python src/ingestion/usgs_elevation.py
```

## Common First-Run Issues

### 1. Missing Dependencies
```bash
# Error: "No module named 'geopandas'"
pip install geopandas

# Error: "No module named 'rasterio'"
pip install rasterio
```

### 2. API Key Not Found
```bash
# You'll see: "⚠ API key not configured"
# Solution: Either add key to .env OR use CSV mode
```

### 3. Network Timeout
```bash
# TIGER/USGS downloads are large
# Solution: Use --skip-usgs flag for testing
python src/run_ingestion.py --skip-usgs
```

## Success Indicators

You'll know it worked when you see:
```
✓ Successfully fetched 11000+ records from API
✓ Saved to parquet: data/raw/311_requests/san_rafael_311_latest.parquet
✓ Saved to CSV: data/raw/311_requests/san_rafael_311_latest.csv
```

## Next Steps After Ingestion

1. **Explore the data:**
   ```bash
   jupyter notebook
   # Open notebooks/01_data_exploration.ipynb
   ```

2. **Start data cleaning:**
   ```bash
   # TBD - cleaning modules coming next
   ```

3. **Begin analysis:**
   ```bash
   # Analyze patterns, correlations, trends
   ```

## Need Help?

1. Check main README.md for detailed documentation
2. Review source code comments in `src/ingestion/`
3. Run with CSV mode if API issues persist
4. Test one source at a time to isolate problems

---

**Time to first data**: ~5 minutes with CSV mode, ~10-15 minutes with API mode
