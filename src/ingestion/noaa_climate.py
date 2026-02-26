"""
NOAA Climate Data Online (CDO) API Ingestion
Retrieves weather data for San Rafael area
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import json
import time
from typing import Optional, List
from tqdm import tqdm

from ..utils.config import config


class NOAAIngestor:
    """Ingest climate data from NOAA CDO API"""
    
    def __init__(self):
        self.base_url = config.noaa_base_url
        self.api_token = config.noaa_api_token
        self.output_dir = config.data_raw_dir / 'noaa'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # NOAA station ID for San Rafael area (you may need to adjust)
        # This is typically found via the stations endpoint
        self.station_id = None  # Will be determined dynamically
        
        # Dataset types
        self.dataset_id = 'GHCND'  # Global Historical Climatology Network Daily
    
    def _get_headers(self) -> dict:
        """Get request headers with API token"""
        return {'token': self.api_token}
    
    def _make_request(self, endpoint: str, params: dict) -> Optional[dict]:
        """
        Make API request with error handling and rate limiting
        
        Args:
            endpoint: API endpoint (e.g., 'stations', 'data')
            params: Query parameters
            
        Returns:
            JSON response or None if error
        """
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = requests.get(
                url, 
                headers=self._get_headers(),
                params=params,
                timeout=30
            )
            
            # NOAA API rate limit: 5 requests per second, 10,000 per day
            time.sleep(0.21)  # ~4.7 requests per second to be safe
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"  API request error: {e}")
            return None
    
    def find_nearest_station(self, lat: float, lon: float, 
                            start_date: str, end_date: str) -> Optional[str]:
        """
        Find nearest weather station with data coverage
        
        Args:
            lat: Latitude
            lon: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Station ID or None
        """
        print("Finding nearest NOAA weather station...")
        
        params = {
            'datasetid': self.dataset_id,
            'extent': f'{lat},{lon},{lat},{lon}',  # Point location
            'startdate': start_date,
            'enddate': end_date,
            'limit': 10
        }
        
        data = self._make_request('stations', params)
        
        if not data or 'results' not in data:
            print("  No stations found")
            return None
        
        stations = data['results']
        if stations:
            # Return first station (usually closest)
            station = stations[0]
            print(f"  Found station: {station['name']} ({station['id']})")
            return station['id']
        
        return None
    
    def fetch_climate_data(self, 
                          start_date: str, 
                          end_date: str,
                          datatypes: List[str] = None) -> pd.DataFrame:
        """
        Fetch climate data from NOAA API
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            datatypes: List of data types to fetch
                      (PRCP=precipitation, TMAX=max temp, TMIN=min temp, etc.)
        
        Returns:
            DataFrame with climate data
        """
        if not self.api_token or self.api_token == 'your_noaa_token_here':
            print("⚠ NOAA API token not configured")
            return pd.DataFrame()
        
        print("Fetching NOAA climate data...")
        
        # Default data types
        if datatypes is None:
            datatypes = ['PRCP', 'TMAX', 'TMIN', 'AWND', 'SNOW']  
            # PRCP=Precipitation, TMAX/TMIN=Temp, AWND=Wind, SNOW=Snowfall
        
        # Find nearest station if not set
        if not self.station_id:
            self.station_id = self.find_nearest_station(
                config.target_lat,
                config.target_lon,
                start_date,
                end_date
            )
            
            if not self.station_id:
                print("  Could not find suitable weather station")
                return pd.DataFrame()
        
        all_data = []
        
        # Fetch data for each data type
        for datatype in tqdm(datatypes, desc="Fetching data types"):
            offset = 0
            limit = 1000  # NOAA max per request
            
            while True:
                params = {
                    'datasetid': self.dataset_id,
                    'stationid': self.station_id,
                    'startdate': start_date,
                    'enddate': end_date,
                    'datatypeid': datatype,
                    'units': 'metric',
                    'limit': limit,
                    'offset': offset
                }
                
                data = self._make_request('data', params)
                
                if not data or 'results' not in data:
                    break
                
                results = data['results']
                all_data.extend(results)
                
                if len(results) < limit:
                    break
                
                offset += limit
        
        if not all_data:
            print("  No climate data retrieved")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Pivot to get one row per date with columns for each datatype
        if not df.empty and 'datatype' in df.columns:
            df_pivot = df.pivot_table(
                index='date',
                columns='datatype',
                values='value',
                aggfunc='first'
            ).reset_index()
            
            print(f"✓ Retrieved {len(df_pivot)} days of climate data")
            return df_pivot
        
        return df
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load NOAA data from CSV file (fallback method)
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with climate data
        """
        print(f"Loading NOAA data from CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Parse date column
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            print(f"✓ Loaded {len(df)} records from CSV")
            return df
            
        except Exception as e:
            print(f"  Error loading CSV: {e}")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> Path:
        """
        Save climate data to file
        
        Args:
            df: DataFrame to save
            filename: Base filename (without extension)
            
        Returns:
            Path to saved parquet file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'noaa_climate_{timestamp}'
        
        # Save as parquet
        parquet_path = self.output_dir / f'{filename}.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"✓ Saved to parquet: {parquet_path}")
        
        # Save as CSV
        csv_path = self.output_dir / f'{filename}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved to CSV: {csv_path}")
        
        # Save metadata
        metadata = {
            'records': len(df),
            'columns': list(df.columns),
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            },
            'station_id': self.station_id,
            'ingestion_timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / f'{filename}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return parquet_path
    
    def ingest(self, 
               start_date: str = None,
               end_date: str = None,
               use_csv: bool = None,
               csv_path: str = None) -> pd.DataFrame:
        """
        Main ingestion method
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            use_csv: Force CSV mode
            csv_path: Path to CSV file for fallback
            
        Returns:
            DataFrame with climate data
        """
        # Use config dates if not provided
        start_date = start_date or config.start_date
        end_date = end_date or config.end_date
        
        # Determine ingestion method
        use_csv_mode = use_csv if use_csv is not None else config.use_csv_fallback
        
        df = pd.DataFrame()
        
        if not use_csv_mode:
            # Try API
            try:
                df = self.fetch_climate_data(start_date, end_date)
            except Exception as e:
                print(f"API ingestion failed: {e}")
                print("Falling back to CSV mode...")
                use_csv_mode = True
        
        # Use CSV fallback
        if use_csv_mode and csv_path:
            df = self.load_from_csv(csv_path)
        elif use_csv_mode and not csv_path:
            print("\n⚠ CSV fallback enabled but no CSV path provided")
            print("Please download data from:")
            print("https://www.ncei.noaa.gov/cdo-web/search")
            print("Search for San Rafael, CA and download daily summaries")
        
        # Save ingested data
        if not df.empty:
            self.save_data(df, filename='noaa_climate_latest')
            
            print("\n" + "="*60)
            print("NOAA Climate Data Ingestion Summary")
            print("="*60)
            print(f"Total Records: {len(df)}")
            print(f"Columns: {list(df.columns)}")
            if 'date' in df.columns:
                print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
            print("="*60)
        
        return df


def main():
    """Main execution for standalone testing"""
    ingestor = NOAAIngestor()
    
    # Check configuration
    if config.use_csv_fallback:
        print("\n⚠ CSV Fallback mode enabled")
    
    if not config.noaa_api_token or config.noaa_api_token == 'your_noaa_token_here':
        print("\n⚠ NOAA API token not configured")
        print("Register at: https://www.ncdc.noaa.gov/cdo-web/token")
        print("Or use CSV fallback mode\n")
    
    df = ingestor.ingest()
    
    if not df.empty:
        print("\nFirst few records:")
        print(df.head())


if __name__ == '__main__':
    main()
