"""
San Rafael 311 Service Requests Data Ingestion
Supports both ArcGIS REST API and CSV fallback
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Optional
from tqdm import tqdm

from ..utils.config import config


class SanRafael311Ingestor:
    """Ingest 311 service request data from San Rafael Open Data Portal"""
    
    def __init__(self):
        self.base_url = config.san_rafael_311_url
        self.output_dir = config.data_raw_dir / '311_requests'
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def fetch_from_api(self, max_records: Optional[int] = None) -> pd.DataFrame:
        """
        Fetch data from ArcGIS REST API
        
        Args:
            max_records: Maximum number of records to fetch (None for all)
            
        Returns:
            DataFrame with 311 request data
        """
        print("Fetching 311 data from ArcGIS REST API...")
        
        all_records = []
        offset = 0
        batch_size = 1000  # ArcGIS typical max
        
        while True:
            # Build query parameters
            params = {
                'where': '1=1',  # Get all records
                'outFields': '*',  # All fields
                'returnGeometry': 'true',
                'f': 'json',
                'resultOffset': offset,
                'resultRecordCount': batch_size,
                'orderByFields': 'objectid ASC'
            }
            
            try:
                response = requests.get(self.base_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                if 'features' not in data or len(data['features']) == 0:
                    break
                
                # Extract features
                features = data['features']
                all_records.extend(features)
                
                print(f"  Fetched {len(all_records)} records so far...")
                
                # Check if we've reached max_records or end of data
                if max_records and len(all_records) >= max_records:
                    all_records = all_records[:max_records]
                    break
                
                if len(features) < batch_size:
                    break
                
                offset += batch_size
                
            except requests.exceptions.RequestException as e:
                print(f"  Error fetching data: {e}")
                break
        
        if not all_records:
            print("  No records fetched from API")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = self._features_to_dataframe(all_records)
        print(f"✓ Successfully fetched {len(df)} records from API")
        
        return df
    
    def _features_to_dataframe(self, features: list) -> pd.DataFrame:
        """
        Convert ArcGIS features to pandas DataFrame
        
        Args:
            features: List of feature dictionaries from ArcGIS response
            
        Returns:
            DataFrame with normalized columns
        """
        records = []
        
        for feature in features:
            # Extract attributes
            record = feature.get('attributes', {})
            
            # Extract geometry if present
            geometry = feature.get('geometry', {})
            if geometry:
                record['longitude'] = geometry.get('x')
                record['latitude'] = geometry.get('y')
            
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Convert timestamp fields (typically in milliseconds since epoch)
        timestamp_fields = ['CreationDate', 'EditDate', 'DateCompleted', 'LastEdited']
        for field in timestamp_fields:
            if field in df.columns:
                df[field] = pd.to_datetime(df[field], unit='ms', errors='coerce')
        
        return df
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load 311 data from CSV file (fallback method)
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with 311 request data
        """
        print(f"Loading 311 data from CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            
            # Try to parse date columns
            date_columns = ['CreationDate', 'EditDate', 'DateCompleted', 'LastEdited']
            for col in date_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            print(f"✓ Successfully loaded {len(df)} records from CSV")
            return df
            
        except Exception as e:
            print(f"  Error loading CSV: {e}")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> Path:
        """
        Save DataFrame to multiple formats
        
        Args:
            df: DataFrame to save
            filename: Base filename (without extension)
            
        Returns:
            Path to saved parquet file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'san_rafael_311_{timestamp}'
        
        # Save as parquet (efficient for large datasets)
        parquet_path = self.output_dir / f'{filename}.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"✓ Saved to parquet: {parquet_path}")
        
        # Also save as CSV for easy inspection
        csv_path = self.output_dir / f'{filename}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved to CSV: {csv_path}")
        
        # Save metadata
        metadata = {
            'records': len(df),
            'columns': list(df.columns),
            'date_range': {
                'start': str(df['CreationDate'].min()) if 'CreationDate' in df.columns else None,
                'end': str(df['CreationDate'].max()) if 'CreationDate' in df.columns else None
            },
            'ingestion_timestamp': datetime.now().isoformat()
        }
        
        metadata_path = self.output_dir / f'{filename}_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return parquet_path
    
    def ingest(self, use_csv: bool = None, csv_path: str = None) -> pd.DataFrame:
        """
        Main ingestion method - tries API first, falls back to CSV if needed
        
        Args:
            use_csv: Force CSV mode (overrides config)
            csv_path: Path to CSV file for fallback
            
        Returns:
            DataFrame with ingested data
        """
        # Determine ingestion method
        use_csv_mode = use_csv if use_csv is not None else config.use_csv_fallback
        
        df = pd.DataFrame()
        
        if not use_csv_mode:
            # Try API first
            try:
                df = self.fetch_from_api()
            except Exception as e:
                print(f"API ingestion failed: {e}")
                print("Falling back to CSV mode...")
                use_csv_mode = True
        
        # Use CSV if API failed or CSV mode is enabled
        if use_csv_mode and (df.empty or csv_path):
            if csv_path is None:
                print("\n⚠ CSV fallback enabled but no CSV path provided")
                print("Please download data from:")
                print("https://open-data-portal-san-rafael.hub.arcgis.com/datasets/8bbb4a9b7034470784f35dfe91e6be8a")
                print("And provide the path to the CSV file.")
                return df
            
            df = self.load_from_csv(csv_path)
        
        # Save ingested data
        if not df.empty:
            self.save_data(df, filename='san_rafael_311_latest')
            
            # Print summary
            print("\n" + "="*60)
            print("311 Data Ingestion Summary")
            print("="*60)
            print(f"Total Records: {len(df)}")
            print(f"Columns: {len(df.columns)}")
            if 'CreationDate' in df.columns:
                print(f"Date Range: {df['CreationDate'].min()} to {df['CreationDate'].max()}")
            print("="*60)
        
        return df


def main():
    """Main execution for standalone testing"""
    ingestor = SanRafael311Ingestor()
    
    # Check if CSV fallback is enabled
    if config.use_csv_fallback:
        print("\n⚠ CSV Fallback mode enabled in configuration")
        print("To use API mode, set USE_CSV_FALLBACK=false in .env file\n")
    
    df = ingestor.ingest()
    
    if not df.empty:
        print("\nFirst few records:")
        print(df.head())
        print("\nColumn names:")
        print(df.columns.tolist())


if __name__ == '__main__':
    main()
