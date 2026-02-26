"""
US Census Bureau Data Ingestion
Retrieves demographic data for San Rafael area
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import json
from typing import Optional, List, Dict

from ..utils.config import config


class CensusIngestor:
    """Ingest demographic data from US Census Bureau API"""
    
    def __init__(self):
        self.api_key = config.census_api_key
        self.base_url = "https://api.census.gov/data"
        self.output_dir = config.data_raw_dir / 'census'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Geographic identifiers for San Rafael (Marin County, CA)
        self.state_fips = config.target_fips_state  # California = 06
        self.county_fips = config.target_fips_county  # Marin = 041
    
    def _make_request(self, url: str, params: dict) -> Optional[dict]:
        """
        Make Census API request
        
        Args:
            url: Full API URL
            params: Query parameters
            
        Returns:
            JSON response or None
        """
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"  API request error: {e}")
            return None
    
    def fetch_acs_data(self, 
                       year: int = 2022,
                       variables: List[str] = None) -> pd.DataFrame:
        """
        Fetch American Community Survey (ACS) 5-year estimates
        
        Args:
            year: Year of ACS data (most recent complete dataset)
            variables: List of variable codes to fetch
            
        Returns:
            DataFrame with demographic data
        """
        if not self.api_key or self.api_key == 'your_census_api_key_here':
            print("⚠ Census API key not configured")
            return pd.DataFrame()
        
        print(f"Fetching Census ACS {year} data...")
        
        # Default variables for population and demographics
        if variables is None:
            variables = [
                'B01003_001E',  # Total Population
                'B01002_001E',  # Median Age
                'B19013_001E',  # Median Household Income
                'B25077_001E',  # Median Home Value
                'B11001_001E',  # Total Households
                'B02001_002E',  # White alone
                'B03003_003E',  # Hispanic or Latino
                'B01001_002E',  # Male
                'B01001_026E',  # Female
                'B15003_022E',  # Bachelor's degree
                'B15003_023E',  # Master's degree
            ]
        
        # Build API URL
        url = f"{self.base_url}/{year}/acs/acs5"
        
        # Query parameters
        params = {
            'get': ','.join(['NAME'] + variables),
            'for': f'county:{self.county_fips}',
            'in': f'state:{self.state_fips}',
            'key': self.api_key
        }
        
        data = self._make_request(url, params)
        
        if not data or len(data) < 2:
            print("  No data retrieved from Census API")
            return pd.DataFrame()
        
        # Convert to DataFrame (first row is headers)
        headers = data[0]
        values = data[1:]
        df = pd.DataFrame(values, columns=headers)
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['NAME', 'state', 'county']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Retrieved Census data for {df['NAME'].iloc[0]}")
        
        return df
    
    def fetch_population_estimates(self, year: int = 2023) -> pd.DataFrame:
        """
        Fetch Population Estimates Program (PEP) data
        
        Args:
            year: Year of estimates
            
        Returns:
            DataFrame with population estimates
        """
        if not self.api_key or self.api_key == 'your_census_api_key_here':
            print("⚠ Census API key not configured")
            return pd.DataFrame()
        
        print(f"Fetching Population Estimates {year}...")
        
        url = f"{self.base_url}/{year}/pep/population"
        
        params = {
            'get': 'NAME,POP',
            'for': f'county:{self.county_fips}',
            'in': f'state:{self.state_fips}',
            'key': self.api_key
        }
        
        data = self._make_request(url, params)
        
        if not data or len(data) < 2:
            print("  No population estimates available for this year")
            return pd.DataFrame()
        
        headers = data[0]
        values = data[1:]
        df = pd.DataFrame(values, columns=headers)
        
        # Convert population to numeric
        df['POP'] = pd.to_numeric(df['POP'], errors='coerce')
        
        print(f"✓ Retrieved population estimate: {df['POP'].iloc[0]:,}")
        
        return df
    
    def fetch_census_tracts(self) -> pd.DataFrame:
        """
        Fetch census tract level data for more granular analysis
        
        Returns:
            DataFrame with tract-level data
        """
        if not self.api_key or self.api_key == 'your_census_api_key_here':
            print("⚠ Census API key not configured")
            return pd.DataFrame()
        
        print("Fetching Census Tract data...")
        
        # Use 2022 ACS 5-year estimates
        url = f"{self.base_url}/2022/acs/acs5"
        
        # Key variables for tract level
        variables = [
            'B01003_001E',  # Total Population
            'B19013_001E',  # Median Household Income
            'B25077_001E',  # Median Home Value
        ]
        
        params = {
            'get': ','.join(['NAME'] + variables),
            'for': 'tract:*',
            'in': f'state:{self.state_fips}+county:{self.county_fips}',
            'key': self.api_key
        }
        
        data = self._make_request(url, params)
        
        if not data or len(data) < 2:
            print("  No tract data retrieved")
            return pd.DataFrame()
        
        headers = data[0]
        values = data[1:]
        df = pd.DataFrame(values, columns=headers)
        
        # Convert numeric columns
        for col in df.columns:
            if col not in ['NAME', 'state', 'county', 'tract']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        print(f"✓ Retrieved data for {len(df)} census tracts")
        
        return df
    
    def load_from_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load Census data from CSV file (fallback method)
        
        Args:
            csv_path: Path to CSV file
            
        Returns:
            DataFrame with Census data
        """
        print(f"Loading Census data from CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"✓ Loaded {len(df)} records from CSV")
            return df
        except Exception as e:
            print(f"  Error loading CSV: {e}")
            return pd.DataFrame()
    
    def save_data(self, df: pd.DataFrame, filename: str = None) -> Path:
        """
        Save Census data to file
        
        Args:
            df: DataFrame to save
            filename: Base filename
            
        Returns:
            Path to saved parquet file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'census_data_{timestamp}'
        
        # Save as parquet
        parquet_path = self.output_dir / f'{filename}.parquet'
        df.to_parquet(parquet_path, index=False)
        print(f"✓ Saved to parquet: {parquet_path}")
        
        # Save as CSV
        csv_path = self.output_dir / f'{filename}.csv'
        df.to_csv(csv_path, index=False)
        print(f"✓ Saved to CSV: {csv_path}")
        
        return parquet_path
    
    def ingest(self, 
               use_csv: bool = None,
               csv_path: str = None,
               include_tracts: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Main ingestion method - fetches multiple Census datasets
        
        Args:
            use_csv: Force CSV mode
            csv_path: Path to CSV file for fallback
            include_tracts: Whether to fetch tract-level data
            
        Returns:
            Dictionary of DataFrames with different Census datasets
        """
        use_csv_mode = use_csv if use_csv is not None else config.use_csv_fallback
        
        datasets = {}
        
        if not use_csv_mode:
            # Try API
            try:
                # Fetch ACS demographic data
                acs_df = self.fetch_acs_data(year=2022)
                if not acs_df.empty:
                    datasets['acs_demographics'] = acs_df
                    self.save_data(acs_df, 'census_acs_2022')
                
                # Fetch population estimates
                pop_df = self.fetch_population_estimates(year=2023)
                if not pop_df.empty:
                    datasets['population_estimates'] = pop_df
                    self.save_data(pop_df, 'census_population_2023')
                
                # Optionally fetch tract-level data
                if include_tracts:
                    tract_df = self.fetch_census_tracts()
                    if not tract_df.empty:
                        datasets['census_tracts'] = tract_df
                        self.save_data(tract_df, 'census_tracts_2022')
                
            except Exception as e:
                print(f"API ingestion failed: {e}")
                use_csv_mode = True
        
        # CSV fallback
        if use_csv_mode and csv_path:
            df = self.load_from_csv(csv_path)
            if not df.empty:
                datasets['census_data'] = df
                self.save_data(df, 'census_data_latest')
        elif use_csv_mode and not csv_path:
            print("\n⚠ CSV fallback enabled but no CSV path provided")
            print("Download data from: https://data.census.gov/")
            print("Search for Marin County, California")
        
        # Print summary
        if datasets:
            print("\n" + "="*60)
            print("Census Data Ingestion Summary")
            print("="*60)
            for name, df in datasets.items():
                print(f"{name}: {len(df)} records, {len(df.columns)} columns")
            print("="*60)
        
        return datasets


def main():
    """Main execution for standalone testing"""
    ingestor = CensusIngestor()
    
    # Check configuration
    if not config.census_api_key or config.census_api_key == 'your_census_api_key_here':
        print("\n⚠ Census API key not configured")
        print("Register at: https://api.census.gov/data/key_signup.html")
        print("Or use CSV fallback mode\n")
    
    datasets = ingestor.ingest(include_tracts=True)
    
    # Display sample data
    for name, df in datasets.items():
        print(f"\n{name}:")
        print(df.head())


if __name__ == '__main__':
    main()
