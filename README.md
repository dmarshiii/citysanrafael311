# 311 AI Analytics Project - Data Ingestion Pipeline

This project implements a comprehensive data ingestion pipeline for the "411 on 311" AI analytics system, designed to analyze municipal service request patterns in San Rafael, California.

## Project Overview

**Objective**: Build an AI-powered analytics system to forecast 311 service demand, identify geographic hotspots, and enable natural language querying of historical data.

**Data Sources** (5 total):
1. San Rafael 311 Service Requests (ArcGIS Open Data Portal)
2. NOAA Climate Data (precipitation, temperature, wind)
3. US Census Bureau Demographics
4. TIGER/Line Shapefiles (geographic boundaries)
5. USGS National Map (elevation/terrain data)

## Project Structure

```
ai_311_project/
├── config/                    # Configuration files
├── data/
│   ├── raw/                  # Raw ingested data
│   │   ├── 311_requests/
│   │   ├── census/
│   │   ├── tiger/
│   │   ├── usgs/
│   │   └── noaa/
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final cleaned data
├── src/
│   ├── ingestion/           # Data ingestion modules
│   │   ├── san_rafael_311.py
│   │   ├── noaa_climate.py
│   │   ├── census_data.py
│   │   ├── tiger_shapefiles.py
│   │   └── usgs_elevation.py
│   ├── cleaning/            # Data cleaning modules (TBD)
│   └── utils/               # Utility functions
│       └── config.py
├── notebooks/               # Jupyter notebooks for exploration
├── tests/                   # Unit tests
├── requirements.txt         # Python dependencies
├── .env.example            # Example environment configuration
└── README.md
```

## Setup Instructions

### 1. Install Dependencies

```bash
# Navigate to project directory
cd ai_311_project

# Install required Python packages
pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas`, `numpy` - Data manipulation
- `geopandas`, `shapely` - Geospatial analysis
- `rasterio` - Raster data processing
- `requests` - API calls
- `python-dotenv` - Configuration management

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor
```

### 3. Register for API Keys

You'll need API keys for Census and NOAA (the other sources don't require keys):

**US Census API** (usually instant):
- URL: https://api.census.gov/data/key_signup.html
- Add key to `.env` as: `CENSUS_API_KEY=your_key_here`

**NOAA Climate Data Online** (1-2 business days):
- URL: https://www.ncdc.noaa.gov/cdo-web/token
- Token delivered via email
- Add to `.env` as: `NOAA_API_TOKEN=your_token_here`

**No keys needed for:**
- San Rafael 311 (public ArcGIS portal)
- TIGER/Line Shapefiles (public FTP)
- USGS National Map (public API)

## Usage

### Option 1: Run Full Pipeline

```bash
# Run complete ingestion pipeline for all 5 data sources
python src/run_ingestion.py
```

This will:
1. Fetch San Rafael 311 requests from ArcGIS API
2. Retrieve NOAA climate data for the date range
3. Pull Census demographics for Marin County
4. Download TIGER shapefiles (county, tracts, places)
5. Fetch and process USGS elevation data

### Option 2: CSV Fallback Mode

If you don't have API keys yet or want to use pre-downloaded data:

```bash
# Enable CSV fallback in .env
USE_CSV_FALLBACK=true

# Run with CSV files
python src/run_ingestion.py \
  --csv-311 /path/to/311_data.csv \
  --csv-noaa /path/to/noaa_data.csv \
  --csv-census /path/to/census_data.csv
```

### Option 3: Run Individual Ingestors

Each data source can be run independently:

```bash
# San Rafael 311 data
python src/ingestion/san_rafael_311.py

# NOAA climate data
python src/ingestion/noaa_climate.py

# Census demographics
python src/ingestion/census_data.py

# TIGER shapefiles
python src/ingestion/tiger_shapefiles.py

# USGS elevation
python src/ingestion/usgs_elevation.py
```

### Command-Line Options

```bash
# Skip USGS (can be slow to download)
python src/run_ingestion.py --skip-usgs

# Use local DEM file
python src/run_ingestion.py --dem /path/to/elevation.tif

# Mix API and CSV modes
python src/run_ingestion.py \
  --csv-311 /path/to/311.csv \
  --csv-noaa /path/to/noaa.csv
  # Census and TIGER will use API
```

## Data Download Locations (Manual Fallback)

If you need to manually download data:

1. **San Rafael 311 Requests**
   - Portal: https://open-data-portal-san-rafael.hub.arcgis.com/
   - Dataset: Service Requests June 2022-Present
   - Format: CSV or GeoJSON

2. **NOAA Climate Data**
   - Portal: https://www.ncei.noaa.gov/cdo-web/search
   - Search: "San Rafael, CA"
   - Dataset: Daily Summaries
   - Format: CSV

3. **US Census Data**
   - Portal: https://data.census.gov/
   - Geography: Marin County, California
   - Tables: B01003 (Population), B19013 (Income), etc.
   - Format: CSV

4. **TIGER/Line Shapefiles**
   - Portal: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
   - Year: 2023
   - Geography: California counties and tracts
   - Format: Shapefile (ZIP)

5. **USGS Elevation**
   - Portal: https://apps.nationalmap.gov/downloader/
   - Search: "San Rafael, CA"
   - Product: 1/3 arc-second DEM
   - Format: GeoTIFF

## Output Files

After ingestion, data is saved in multiple formats:

### San Rafael 311 Requests
- `data/raw/311_requests/san_rafael_311_latest.parquet` (efficient storage)
- `data/raw/311_requests/san_rafael_311_latest.csv` (human-readable)
- `data/raw/311_requests/san_rafael_311_latest_metadata.json` (summary stats)

### NOAA Climate Data
- `data/raw/noaa/noaa_climate_latest.parquet`
- `data/raw/noaa/noaa_climate_latest.csv`
- `data/raw/noaa/noaa_climate_latest_metadata.json`

### Census Demographics
- `data/raw/census/census_acs_2022.parquet` (ACS 5-year estimates)
- `data/raw/census/census_population_2023.parquet` (population estimates)
- `data/raw/census/census_tracts_2022.parquet` (tract-level data)

### TIGER Shapefiles
- `data/raw/tiger/tiger_county_2023.gpkg` (GeoPackage format)
- `data/raw/tiger/tiger_county_2023.geojson` (web-friendly)
- `data/raw/tiger/tiger_tracts_2023.gpkg`
- `data/raw/tiger/tiger_places_2023.gpkg`

### USGS Elevation
- `data/raw/usgs/elevation_dem_TIMESTAMP.tif` (digital elevation model)
- `data/raw/usgs/slope_elevation_dem_TIMESTAMP.tif` (calculated slope)
- `data/raw/usgs/elevation_metadata.json`

## Troubleshooting

### API Key Issues

```bash
# Check API key status
python -c "from src.utils.config import config; print(config.validate_api_keys())"
```

If keys are missing, either:
1. Register for keys (see Setup section)
2. Enable CSV fallback mode: `USE_CSV_FALLBACK=true` in `.env`

### Network/Timeout Errors

- TIGER files are large (100+ MB), downloads may take 5-10 minutes
- USGS elevation downloads can take 2-5 minutes
- Consider using `--skip-usgs` flag for faster testing

### Dependency Issues

```bash
# If you encounter import errors, ensure all dependencies are installed
pip install --upgrade -r requirements.txt

# For geospatial libraries on some systems, you may need system packages:
# Ubuntu/Debian:
sudo apt-get install gdal-bin libgdal-dev

# macOS:
brew install gdal
```

## Next Steps

After successful ingestion, the next phase involves:

1. **Data Cleaning** - Handle missing values, outliers, standardize formats
2. **Feature Engineering** - Create derived features for ML models
3. **Exploratory Analysis** - Understand patterns and correlations
4. **Model Development** - Time series forecasting, clustering, NLP

## Configuration Reference

Key configuration variables in `.env`:

```bash
# API Credentials
CENSUS_API_KEY=your_key
NOAA_API_TOKEN=your_token

# Geographic Scope
TARGET_CITY=San Rafael
TARGET_STATE=CA
TARGET_FIPS_STATE=06
TARGET_FIPS_COUNTY=041
TARGET_LAT=37.9735
TARGET_LON=-122.5311

# Date Range
START_DATE=2022-06-01
END_DATE=2026-02-15

# Fallback Mode
USE_CSV_FALLBACK=false
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review individual module documentation in source files
3. Verify API keys and network connectivity
4. Consider CSV fallback mode for testing

## License

This project is for educational purposes as part of MIS 554: Artificial Intelligence for Business.

---

**Author**: Donald Marsh - Group 8  
**Course**: MIS 554: Artificial Intelligence for Business  
**Institution**: Graduate School Project
