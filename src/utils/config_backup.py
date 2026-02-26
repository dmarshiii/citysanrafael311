"""
Configuration Manager for 311 AI Project
Handles API keys, environment variables, and project settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import yaml


class Config:
    """Centralized configuration management"""
    
    def __init__(self, env_file='.env'):
        """
        Initialize configuration from .env file
        
        Args:
            env_file: Path to .env file (default: '.env')
        """
        # Load environment variables
        project_root = Path(__file__).parent.parent
        env_path = project_root / env_file
        
        if env_path.exists():
            load_dotenv(env_path)
        else:
            print(f"Warning: {env_file} not found. Using default/example configuration.")
            load_dotenv(project_root / '.env.example')
        
        # API Keys
        self.census_api_key = os.getenv('CENSUS_API_KEY', '')
        self.noaa_api_token = os.getenv('NOAA_API_TOKEN', '')
        
        # Data Source URLs
        self.san_rafael_311_url = os.getenv('SAN_RAFAEL_311_URL')
        self.tiger_base_url = os.getenv('TIGER_SHAPEFILES_BASE_URL')
        self.noaa_base_url = os.getenv('NOAA_CDO_BASE_URL')
        
        # Project Paths
        self.project_root = project_root
        self.data_raw_dir = project_root / os.getenv('DATA_RAW_DIR', 'data/raw')
        self.data_interim_dir = project_root / os.getenv('DATA_INTERIM_DIR', 'data/interim')
        self.data_processed_dir = project_root / os.getenv('DATA_PROCESSED_DIR', 'data/processed')
        
        # Geographic Configuration
        self.target_city = os.getenv('TARGET_CITY', 'San Rafael')
        self.target_state = os.getenv('TARGET_STATE', 'CA')
        self.target_county = os.getenv('TARGET_COUNTY', 'Marin')
        self.target_fips_state = os.getenv('TARGET_FIPS_STATE', '06')
        self.target_fips_county = os.getenv('TARGET_FIPS_COUNTY', '041')
        self.target_lat = float(os.getenv('TARGET_LAT', 37.9735))
        self.target_lon = float(os.getenv('TARGET_LON', -122.5311))
        
        # Date Range
        self.start_date = os.getenv('START_DATE', '2022-06-01')
        self.end_date = os.getenv('END_DATE', '2026-02-15')
        
        # CSV Fallback Mode
        self.use_csv_fallback = os.getenv('USE_CSV_FALLBACK', 'false').lower() == 'true'
        
        # Create directories if they don't exist
        self._create_directories()
    
    def _create_directories(self):
        """Create data directories if they don't exist"""
        for directory in [self.data_raw_dir, self.data_interim_dir, self.data_processed_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            # Create subdirectories for each data source
            for subdir in ['311_requests', 'census', 'tiger', 'usgs', 'noaa']:
                (directory / subdir).mkdir(exist_ok=True)
    
    def validate_api_keys(self):
        """
        Validate that required API keys are present
        
        Returns:
            dict: Status of each API key
        """
        status = {
            'census': bool(self.census_api_key and self.census_api_key != 'your_census_api_key_here'),
            'noaa': bool(self.noaa_api_token and self.noaa_api_token != 'your_noaa_token_here'),
            'san_rafael_311': True,  # No key required
            'tiger': True,  # No key required
            'usgs': True  # No key required
        }
        return status
    
    def get_missing_keys(self):
        """
        Get list of missing API keys
        
        Returns:
            list: Names of missing API keys
        """
        status = self.validate_api_keys()
        return [key for key, valid in status.items() if not valid]
    
    def __repr__(self):
        """String representation of configuration"""
        return f"Config(city={self.target_city}, state={self.target_state}, csv_fallback={self.use_csv_fallback})"


# Global config instance
config = Config()


if __name__ == '__main__':
    # Test configuration
    print("Configuration Status:")
    print(f"  Project Root: {config.project_root}")
    print(f"\nAPI Key Status:")
    for source, valid in config.validate_api_keys().items():
        status = "✓" if valid else "✗"
        print(f"  {status} {source}")
    
    missing = config.get_missing_keys()
    if missing:
        print(f"\n⚠ Missing API keys: {', '.join(missing)}")
        print(f"  Consider setting USE_CSV_FALLBACK=true in .env file")
    else:
        print("\n✓ All API keys configured!")
