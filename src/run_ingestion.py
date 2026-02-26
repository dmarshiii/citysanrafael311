"""
Master Data Ingestion Pipeline
Orchestrates ingestion from all five data sources
"""

import argparse
from pathlib import Path
from datetime import datetime
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from ingestion import (
    SanRafael311Ingestor,
    NOAAIngestor,
    CensusIngestor,
    TIGERIngestor,
    USGSElevationIngestor
)
from utils.config import config


class MasterPipeline:
    """Orchestrate all data ingestion pipelines"""
    
    def __init__(self):
        self.results = {}
        self.errors = []
    
    def run_311_ingestion(self, csv_path=None):
        """Run San Rafael 311 data ingestion"""
        print("\n" + "="*70)
        print("STEP 1/5: San Rafael 311 Service Requests")
        print("="*70)
        
        try:
            ingestor = SanRafael311Ingestor()
            df = ingestor.ingest(csv_path=csv_path)
            
            if not df.empty:
                self.results['311_requests'] = {
                    'status': 'success',
                    'records': len(df),
                    'columns': len(df.columns)
                }
            else:
                self.results['311_requests'] = {'status': 'failed'}
                self.errors.append("311 ingestion returned no data")
        except Exception as e:
            self.results['311_requests'] = {'status': 'error', 'message': str(e)}
            self.errors.append(f"311 ingestion error: {e}")
    
    def run_noaa_ingestion(self, csv_path=None):
        """Run NOAA climate data ingestion"""
        print("\n" + "="*70)
        print("STEP 2/5: NOAA Climate Data")
        print("="*70)
        
        try:
            ingestor = NOAAIngestor()
            df = ingestor.ingest(csv_path=csv_path)
            
            if not df.empty:
                self.results['noaa_climate'] = {
                    'status': 'success',
                    'records': len(df),
                    'columns': len(df.columns)
                }
            else:
                self.results['noaa_climate'] = {'status': 'failed'}
                self.errors.append("NOAA ingestion returned no data")
        except Exception as e:
            self.results['noaa_climate'] = {'status': 'error', 'message': str(e)}
            self.errors.append(f"NOAA ingestion error: {e}")
    
    def run_census_ingestion(self, csv_path=None):
        """Run Census data ingestion"""
        print("\n" + "="*70)
        print("STEP 3/5: US Census Demographics")
        print("="*70)
        
        try:
            ingestor = CensusIngestor()
            datasets = ingestor.ingest(csv_path=csv_path)
            
            if datasets:
                self.results['census'] = {
                    'status': 'success',
                    'datasets': len(datasets)
                }
                for name, df in datasets.items():
                    self.results['census'][name] = len(df)
            else:
                self.results['census'] = {'status': 'failed'}
                self.errors.append("Census ingestion returned no data")
        except Exception as e:
            self.results['census'] = {'status': 'error', 'message': str(e)}
            self.errors.append(f"Census ingestion error: {e}")
    
    def run_tiger_ingestion(self, local_paths=None):
        """Run TIGER shapefiles ingestion"""
        print("\n" + "="*70)
        print("STEP 4/5: TIGER/Line Shapefiles")
        print("="*70)
        
        try:
            ingestor = TIGERIngestor()
            
            if local_paths:
                datasets = ingestor.ingest(use_local=True, local_paths=local_paths)
            else:
                datasets = ingestor.ingest(layers=['county', 'tracts', 'places'])
            
            if datasets:
                self.results['tiger'] = {
                    'status': 'success',
                    'layers': len(datasets)
                }
                for name, gdf in datasets.items():
                    self.results['tiger'][name] = len(gdf)
            else:
                self.results['tiger'] = {'status': 'failed'}
                self.errors.append("TIGER ingestion returned no data")
        except Exception as e:
            self.results['tiger'] = {'status': 'error', 'message': str(e)}
            self.errors.append(f"TIGER ingestion error: {e}")
    
    def run_usgs_ingestion(self, local_dem_path=None):
        """Run USGS elevation data ingestion"""
        print("\n" + "="*70)
        print("STEP 5/5: USGS Elevation Data")
        print("="*70)
        
        try:
            ingestor = USGSElevationIngestor()
            
            if local_dem_path:
                results = ingestor.ingest(use_local=True, local_dem_path=local_dem_path)
            else:
                results = ingestor.ingest(calculate_slope=True)
            
            if results:
                self.results['usgs'] = {
                    'status': 'success',
                    'files': list(results.keys())
                }
            else:
                self.results['usgs'] = {'status': 'failed'}
                self.errors.append("USGS ingestion failed")
        except Exception as e:
            self.results['usgs'] = {'status': 'error', 'message': str(e)}
            self.errors.append(f"USGS ingestion error: {e}")
    
    def run_all(self, csv_paths=None, tiger_paths=None, dem_path=None):
        """
        Run complete ingestion pipeline
        
        Args:
            csv_paths: Dict with keys '311', 'noaa', 'census' pointing to CSV files
            tiger_paths: Dict with shapefile paths for TIGER data
            dem_path: Path to local DEM file
        """
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("MASTER DATA INGESTION PIPELINE")
        print("="*70)
        print(f"Start time: {start_time}")
        print(f"Target: {config.target_city}, {config.target_state}")
        print(f"CSV Fallback Mode: {config.use_csv_fallback}")
        
        # Check API keys
        print("\nAPI Key Status:")
        key_status = config.validate_api_keys()
        for source, valid in key_status.items():
            status_icon = "✓" if valid else "✗"
            print(f"  {status_icon} {source}")
        
        csv_paths = csv_paths or {}
        
        # Run each ingestion step
        self.run_311_ingestion(csv_path=csv_paths.get('311'))
        self.run_noaa_ingestion(csv_path=csv_paths.get('noaa'))
        self.run_census_ingestion(csv_path=csv_paths.get('census'))
        self.run_tiger_ingestion(local_paths=tiger_paths)
        self.run_usgs_ingestion(local_dem_path=dem_path)
        
        # Print final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*70)
        print("INGESTION PIPELINE SUMMARY")
        print("="*70)
        print(f"Duration: {duration}")
        print(f"\nResults:")
        
        for source, result in self.results.items():
            status = result.get('status', 'unknown')
            status_icon = "✓" if status == 'success' else "✗"
            print(f"  {status_icon} {source}: {status}")
            
            if status == 'success':
                for key, value in result.items():
                    if key != 'status':
                        print(f"      {key}: {value}")
        
        if self.errors:
            print("\nErrors encountered:")
            for error in self.errors:
                print(f"  - {error}")
        
        print("\nData files saved to:")
        print(f"  {config.data_raw_dir}")
        
        return self.results


def main():
    """Command-line interface for master pipeline"""
    parser = argparse.ArgumentParser(
        description='Run 311 AI Project Data Ingestion Pipeline'
    )
    
    parser.add_argument(
        '--csv-311',
        help='Path to San Rafael 311 CSV file (for fallback mode)'
    )
    
    parser.add_argument(
        '--csv-noaa',
        help='Path to NOAA climate CSV file (for fallback mode)'
    )
    
    parser.add_argument(
        '--csv-census',
        help='Path to Census CSV file (for fallback mode)'
    )
    
    parser.add_argument(
        '--dem',
        help='Path to local DEM GeoTIFF file'
    )
    
    parser.add_argument(
        '--skip-usgs',
        action='store_true',
        help='Skip USGS elevation ingestion (can be slow)'
    )
    
    args = parser.parse_args()
    
    # Build CSV paths dict
    csv_paths = {}
    if args.csv_311:
        csv_paths['311'] = args.csv_311
    if args.csv_noaa:
        csv_paths['noaa'] = args.csv_noaa
    if args.csv_census:
        csv_paths['census'] = args.csv_census
    
    # Run pipeline
    pipeline = MasterPipeline()
    
    if args.skip_usgs:
        print("Skipping USGS elevation ingestion")
        pipeline.run_311_ingestion(csv_path=csv_paths.get('311'))
        pipeline.run_noaa_ingestion(csv_path=csv_paths.get('noaa'))
        pipeline.run_census_ingestion(csv_path=csv_paths.get('census'))
        pipeline.run_tiger_ingestion()
    else:
        pipeline.run_all(csv_paths=csv_paths, dem_path=args.dem)


if __name__ == '__main__':
    main()
