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
        # Load environment variables - try multiple locations
        project_root = Path(__file__).parent.parent.parent  # Go up to project root from src/utils/config.py
        env_path = project_root / env_file
        
        # Also try current working directory
        cwd_env_path = Path.cwd() / env_file
        
        if env_path.exists():
            load_dotenv(env_path)
            print(f"[Config] Loaded .env from: {env_path}")
        elif cwd_env_path.exists():
            load_dotenv(cwd_env_path)
            print(f"[Config] Loaded .env from: {cwd_env_path}")
        elif (project_root / '.env.example').exists():
            print(f"Warning: {env_file} not found. Using default/example configuration.")
            load_dotenv(project_root / '.env.example')
        else:
            print(f"Warning: No .env file found. Please create one from .env.example")
        
        # API Keys
        self.census_api_key = os.getenv('CENSUS_API_KEY', '')
        self.noaa_api_token = os.getenv('NOAA_API_TOKEN', '')
        
        # Data Source URLs
        self.san_rafael_311_url = os.getenv('SAN_RAFAEL_311_URL')
        self.tiger_base_url = os.getenv('TIGER_SHAPEFILES_BASE_URL')
        self.noaa_base_url = os.getenv('NOAA_CDO_BASE_URL')
        
        # Project Paths - use the detected project root
        self.project_root = project_root if env_path.exists() else Path.cwd()
        self.data_raw_dir = self.project_root / os.getenv('DATA_RAW_DIR', 'data/raw')
        self.data_interim_dir = self.project_root / os.getenv('DATA_INTERIM_DIR', 'data/interim')
        self.data_processed_dir = self.project_root / os.getenv('DATA_PROCESSED_DIR', 'data/processed')
        
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
        # More lenient validation - just check if key exists and has reasonable length
        census_valid = bool(
            self.census_api_key and 
            self.census_api_key != 'your_census_api_key_here' and
            len(self.census_api_key) > 10  # Census keys are typically 40 chars
        )
        
        noaa_valid = bool(
            self.noaa_api_token and 
            self.noaa_api_token != 'your_noaa_token_here' and
            len(self.noaa_api_token) > 10  # NOAA tokens are typically 32 chars
        )
        
        status = {
            'census': census_valid,
            'noaa': noaa_valid,
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
    print(f"\nAPI Keys Loaded:")
    print(f"  Census API Key: '{config.census_api_key[:10]}...' (length: {len(config.census_api_key)})" if config.census_api_key else "  Census API Key: NOT SET")
    print(f"  NOAA API Token: '{config.noaa_api_token[:10]}...' (length: {len(config.noaa_api_token)})" if config.noaa_api_token else "  NOAA API Token: NOT SET")
    
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
