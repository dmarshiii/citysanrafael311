"""
TIGER/Line Shapefiles Ingestion
Downloads and processes Census geographic boundary files
"""

import requests
import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import datetime
import zipfile
import io
from typing import Optional, List
from tqdm import tqdm

from ..utils.config import config


class TIGERIngestor:
    """Ingest TIGER/Line shapefiles from US Census Bureau"""
    
    def __init__(self):
        self.base_url = config.tiger_base_url
        self.output_dir = config.data_raw_dir / 'tiger'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target geographic area
        self.state_fips = config.target_fips_state
        self.county_fips = config.target_fips_county
        
        # Year for TIGER files (typically use most recent)
        self.year = 2023
    
    def _download_shapefile(self, url: str, output_name: str) -> Optional[Path]:
        """
        Download and extract shapefile ZIP
        
        Args:
            url: URL to shapefile ZIP
            output_name: Name for output directory
            
        Returns:
            Path to extracted directory or None
        """
        try:
            print(f"  Downloading {output_name}...")
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            
            # Create output directory
            extract_dir = self.output_dir / output_name
            extract_dir.mkdir(exist_ok=True)
            
            # Extract ZIP
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
                zip_ref.extractall(extract_dir)
            
            print(f"  ✓ Extracted to {extract_dir}")
            return extract_dir
            
        except Exception as e:
            print(f"  Error downloading {url}: {e}")
            return None
    
    def fetch_county_boundaries(self) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch county boundary shapefile
        
        Returns:
            GeoDataFrame with county boundaries
        """
        print("Fetching county boundaries...")
        
        # TIGER county file URL pattern
        url = f"{self.base_url}COUNTY/tl_{self.year}_{self.state_fips}_county.zip"
        
        extract_dir = self._download_shapefile(url, f'county_{self.year}')
        
        if not extract_dir:
            return None
        
        # Find .shp file
        shp_files = list(extract_dir.glob('*.shp'))
        if not shp_files:
            print("  No .shp file found in archive")
            return None
        
        # Read shapefile
        gdf = gpd.read_file(shp_files[0])
        
        # Filter to target county
        gdf = gdf[gdf['COUNTYFP'] == self.county_fips]
        
        print(f"  ✓ Loaded {len(gdf)} county boundary(ies)")
        return gdf
    
    def fetch_census_tracts(self) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch census tract boundaries for target county
        
        Returns:
            GeoDataFrame with census tract boundaries
        """
        print("Fetching census tract boundaries...")
        
        # TIGER tract file URL pattern
        url = f"{self.base_url}TRACT/tl_{self.year}_{self.state_fips}_tract.zip"
        
        extract_dir = self._download_shapefile(url, f'tract_{self.year}')
        
        if not extract_dir:
            return None
        
        # Find .shp file
        shp_files = list(extract_dir.glob('*.shp'))
        if not shp_files:
            print("  No .shp file found in archive")
            return None
        
        # Read shapefile
        gdf = gpd.read_file(shp_files[0])
        
        # Filter to target county
        gdf = gdf[gdf['COUNTYFP'] == self.county_fips]
        
        print(f"  ✓ Loaded {len(gdf)} census tracts")
        return gdf
    
    def fetch_roads(self) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch road network for target county
        
        Returns:
            GeoDataFrame with road network
        """
        print("Fetching road network...")
        
        # TIGER roads file - county specific
        county_code = f"{self.state_fips}{self.county_fips}"
        url = f"{self.base_url}ROADS/tl_{self.year}_{county_code}_roads.zip"
        
        extract_dir = self._download_shapefile(url, f'roads_{self.year}')
        
        if not extract_dir:
            return None
        
        # Find .shp file
        shp_files = list(extract_dir.glob('*.shp'))
        if not shp_files:
            print("  No .shp file found in archive")
            return None
        
        # Read shapefile
        gdf = gpd.read_file(shp_files[0])
        
        print(f"  ✓ Loaded {len(gdf)} road segments")
        return gdf
    
    def fetch_places(self) -> Optional[gpd.GeoDataFrame]:
        """
        Fetch incorporated places (cities/towns) boundaries
        
        Returns:
            GeoDataFrame with place boundaries
        """
        print("Fetching place boundaries...")
        
        # TIGER place file URL pattern
        url = f"{self.base_url}PLACE/tl_{self.year}_{self.state_fips}_place.zip"
        
        extract_dir = self._download_shapefile(url, f'place_{self.year}')
        
        if not extract_dir:
            return None
        
        # Find .shp file
        shp_files = list(extract_dir.glob('*.shp'))
        if not shp_files:
            print("  No .shp file found in archive")
            return None
        
        # Read shapefile
        gdf = gpd.read_file(shp_files[0])
        
        # Filter to San Rafael
        gdf = gdf[gdf['NAME'] == config.target_city]
        
        print(f"  ✓ Loaded {len(gdf)} place(s)")
        return gdf
    
    def load_from_shapefile(self, shapefile_path: str) -> gpd.GeoDataFrame:
        """
        Load shapefile from local file (fallback method)
        
        Args:
            shapefile_path: Path to .shp file
            
        Returns:
            GeoDataFrame with shapefile data
        """
        print(f"Loading shapefile from: {shapefile_path}")
        
        try:
            gdf = gpd.read_file(shapefile_path)
            print(f"✓ Loaded {len(gdf)} features")
            return gdf
        except Exception as e:
            print(f"  Error loading shapefile: {e}")
            return gpd.GeoDataFrame()
    
    def save_data(self, gdf: gpd.GeoDataFrame, filename: str) -> Path:
        """
        Save GeoDataFrame to multiple formats
        
        Args:
            gdf: GeoDataFrame to save
            filename: Base filename (without extension)
            
        Returns:
            Path to saved GeoPackage file
        """
        # Save as GeoPackage (modern, efficient format)
        gpkg_path = self.output_dir / f'{filename}.gpkg'
        gdf.to_file(gpkg_path, driver='GPKG')
        print(f"✓ Saved to GeoPackage: {gpkg_path}")
        
        # Save as GeoJSON (web-friendly)
        geojson_path = self.output_dir / f'{filename}.geojson'
        gdf.to_file(geojson_path, driver='GeoJSON')
        print(f"✓ Saved to GeoJSON: {geojson_path}")
        
        # Save attribute table as CSV
        csv_path = self.output_dir / f'{filename}_attributes.csv'
        df_attrs = pd.DataFrame(gdf.drop(columns='geometry'))
        df_attrs.to_csv(csv_path, index=False)
        print(f"✓ Saved attributes to CSV: {csv_path}")
        
        return gpkg_path
    
    def ingest(self, 
               layers: List[str] = None,
               use_local: bool = False,
               local_paths: dict = None) -> dict:
        """
        Main ingestion method - downloads multiple TIGER layers
        
        Args:
            layers: List of layers to download 
                   ('county', 'tracts', 'roads', 'places')
            use_local: Use local shapefiles instead of downloading
            local_paths: Dictionary of layer names to local file paths
            
        Returns:
            Dictionary of GeoDataFrames
        """
        if layers is None:
            layers = ['county', 'tracts', 'places']  # Default layers
        
        datasets = {}
        
        if use_local and local_paths:
            # Load from local files
            for layer, path in local_paths.items():
                gdf = self.load_from_shapefile(path)
                if not gdf.empty:
                    datasets[layer] = gdf
                    self.save_data(gdf, f'tiger_{layer}_latest')
        else:
            # Download from TIGER
            layer_methods = {
                'county': self.fetch_county_boundaries,
                'tracts': self.fetch_census_tracts,
                'roads': self.fetch_roads,
                'places': self.fetch_places
            }
            
            for layer in layers:
                if layer in layer_methods:
                    print(f"\nProcessing {layer}...")
                    gdf = layer_methods[layer]()
                    
                    if gdf is not None and not gdf.empty:
                        datasets[layer] = gdf
                        self.save_data(gdf, f'tiger_{layer}_{self.year}')
                else:
                    print(f"  Unknown layer: {layer}")
        
        # Print summary
        if datasets:
            print("\n" + "="*60)
            print("TIGER/Line Shapefiles Ingestion Summary")
            print("="*60)
            for name, gdf in datasets.items():
                bounds = gdf.total_bounds
                print(f"{name}:")
                print(f"  Features: {len(gdf)}")
                print(f"  CRS: {gdf.crs}")
                print(f"  Bounds: ({bounds[0]:.4f}, {bounds[1]:.4f}) to ({bounds[2]:.4f}, {bounds[3]:.4f})")
            print("="*60)
        
        return datasets


def main():
    """Main execution for standalone testing"""
    ingestor = TIGERIngestor()
    
    print("Starting TIGER/Line shapefile ingestion...")
    print("Note: Large downloads may take several minutes\n")
    
    # Download standard layers
    datasets = ingestor.ingest(layers=['county', 'tracts', 'places'])
    
    # Display summary
    for name, gdf in datasets.items():
        print(f"\n{name} sample:")
        print(gdf.head())


if __name__ == '__main__':
    main()
