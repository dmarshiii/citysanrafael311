"""
USGS National Map - Elevation Data Ingestion
Downloads and processes Digital Elevation Model (DEM) data for slope/terrain analysis
"""

import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from datetime import datetime
import rasterio
from rasterio.plot import show
from rasterio.warp import calculate_default_transform, reproject, Resampling
import json
from typing import Optional, Tuple
from shapely.geometry import box

from ..utils.config import config


class USGSElevationIngestor:
    """Ingest elevation data from USGS National Map"""
    
    def __init__(self):
        self.output_dir = config.data_raw_dir / 'usgs'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Target area (San Rafael, CA)
        self.target_lat = config.target_lat
        self.target_lon = config.target_lon
        
        # USGS 3DEP Elevation API endpoint
        self.api_base = "https://elevation.nationalmap.gov/arcgis/rest/services"
        self.dem_service = f"{self.api_base}/3DEPElevation/ImageServer"
    
    def get_bbox_from_point(self, lat: float, lon: float, 
                           buffer_km: float = 10) -> Tuple[float, float, float, float]:
        """
        Create bounding box around a point
        
        Args:
            lat: Latitude
            lon: Longitude
            buffer_km: Buffer distance in kilometers
            
        Returns:
            Tuple of (min_lon, min_lat, max_lon, max_lat)
        """
        # Approximate degrees per km (rough estimate)
        km_to_deg_lat = 1 / 111.0
        km_to_deg_lon = 1 / (111.0 * np.cos(np.radians(lat)))
        
        buffer_lat = buffer_km * km_to_deg_lat
        buffer_lon = buffer_km * km_to_deg_lon
        
        bbox = (
            lon - buffer_lon,  # min_lon
            lat - buffer_lat,  # min_lat
            lon + buffer_lon,  # max_lon
            lat + buffer_lat   # max_lat
        )
        
        return bbox
    
    def fetch_elevation_data(self, 
                            bbox: Tuple[float, float, float, float] = None,
                            resolution: int = 10) -> Optional[str]:
        """
        Fetch elevation data from USGS 3DEP
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            resolution: Resolution in meters (10 or 30 typically)
            
        Returns:
            Path to downloaded GeoTIFF file
        """
        if bbox is None:
            bbox = self.get_bbox_from_point(self.target_lat, self.target_lon, buffer_km=15)
        
        print(f"Fetching elevation data for bbox: {bbox}")
        print(f"Resolution: {resolution}m")
        
        # Build export image request
        min_lon, min_lat, max_lon, max_lat = bbox
        
        params = {
            'bbox': f'{min_lon},{min_lat},{max_lon},{max_lat}',
            'bboxSR': '4326',  # WGS84
            'size': '2048,2048',  # Image size
            'imageSR': '4326',
            'format': 'tiff',
            'pixelType': 'F32',  # Float32
            'noDataInterpretation': 'esriNoDataMatchAny',
            'interpolation': '+RSP_BilinearInterpolation',
            'f': 'image'
        }
        
        url = f"{self.dem_service}/exportImage"
        
        try:
            print("  Downloading DEM (this may take a few minutes)...")
            response = requests.get(url, params=params, timeout=180)
            response.raise_for_status()
            
            # Save to file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = self.output_dir / f'elevation_dem_{timestamp}.tif'
            
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            print(f"  ✓ Downloaded DEM to {output_path}")
            
            # Verify it's a valid raster
            try:
                with rasterio.open(output_path) as src:
                    print(f"  Dimensions: {src.width} x {src.height}")
                    print(f"  CRS: {src.crs}")
                    print(f"  Bounds: {src.bounds}")
            except Exception as e:
                print(f"  Warning: Could not validate raster: {e}")
            
            return str(output_path)
            
        except requests.exceptions.RequestException as e:
            print(f"  Error downloading elevation data: {e}")
            return None
    
    def calculate_slope(self, dem_path: str) -> Optional[str]:
        """
        Calculate slope from DEM
        
        Args:
            dem_path: Path to DEM GeoTIFF
            
        Returns:
            Path to slope raster
        """
        print("Calculating slope from DEM...")
        
        try:
            with rasterio.open(dem_path) as src:
                elevation = src.read(1)
                
                # Get pixel size
                transform = src.transform
                pixel_size_x = transform[0]
                pixel_size_y = -transform[4]
                
                # Calculate gradients
                dy, dx = np.gradient(elevation, pixel_size_y, pixel_size_x)
                
                # Calculate slope in degrees
                slope = np.arctan(np.sqrt(dx**2 + dy**2)) * (180 / np.pi)
                
                # Replace nodata values
                slope[elevation == src.nodata] = src.nodata
                
                # Save slope raster
                output_path = self.output_dir / f'slope_{Path(dem_path).stem}.tif'
                
                with rasterio.open(
                    output_path, 'w',
                    driver='GTiff',
                    height=slope.shape[0],
                    width=slope.shape[1],
                    count=1,
                    dtype=slope.dtype,
                    crs=src.crs,
                    transform=src.transform,
                    nodata=src.nodata
                ) as dst:
                    dst.write(slope, 1)
                
                print(f"  ✓ Saved slope raster to {output_path}")
                print(f"  Slope range: {np.nanmin(slope):.2f}° to {np.nanmax(slope):.2f}°")
                
                return str(output_path)
                
        except Exception as e:
            print(f"  Error calculating slope: {e}")
            return None
    
    def extract_stats_by_location(self, 
                                  raster_path: str,
                                  locations_gdf: gpd.GeoDataFrame,
                                  buffer_m: float = 100) -> pd.DataFrame:
        """
        Extract raster statistics for point locations
        
        Args:
            raster_path: Path to raster file (elevation or slope)
            locations_gdf: GeoDataFrame with point locations
            buffer_m: Buffer radius in meters
            
        Returns:
            DataFrame with raster statistics per location
        """
        print(f"Extracting raster statistics for {len(locations_gdf)} locations...")
        
        results = []
        
        try:
            with rasterio.open(raster_path) as src:
                for idx, row in locations_gdf.iterrows():
                    point = row.geometry
                    
                    # Create buffer around point
                    buffered = point.buffer(buffer_m / 111320)  # Rough conversion to degrees
                    
                    # Sample raster within buffer
                    try:
                        # Get window
                        window = rasterio.windows.from_bounds(
                            *buffered.bounds, 
                            transform=src.transform
                        )
                        
                        # Read data
                        data = src.read(1, window=window)
                        
                        # Calculate statistics
                        valid_data = data[data != src.nodata]
                        
                        if len(valid_data) > 0:
                            stats = {
                                'location_id': idx,
                                'mean': np.mean(valid_data),
                                'median': np.median(valid_data),
                                'min': np.min(valid_data),
                                'max': np.max(valid_data),
                                'std': np.std(valid_data)
                            }
                            results.append(stats)
                    except Exception:
                        continue
            
            df = pd.DataFrame(results)
            print(f"  ✓ Extracted statistics for {len(df)} locations")
            return df
            
        except Exception as e:
            print(f"  Error extracting statistics: {e}")
            return pd.DataFrame()
    
    def load_from_file(self, file_path: str) -> Optional[str]:
        """
        Load elevation data from local GeoTIFF file (fallback method)
        
        Args:
            file_path: Path to local GeoTIFF
            
        Returns:
            Path to file (for consistency with API method)
        """
        print(f"Using local elevation file: {file_path}")
        
        try:
            with rasterio.open(file_path) as src:
                print(f"  Dimensions: {src.width} x {src.height}")
                print(f"  CRS: {src.crs}")
                print(f"  Bounds: {src.bounds}")
            
            return file_path
        except Exception as e:
            print(f"  Error loading file: {e}")
            return None
    
    def save_metadata(self, dem_path: str, slope_path: str = None):
        """
        Save metadata about ingested elevation data
        
        Args:
            dem_path: Path to DEM file
            slope_path: Path to slope file (optional)
        """
        metadata = {
            'dem_file': dem_path,
            'slope_file': slope_path,
            'target_location': {
                'lat': self.target_lat,
                'lon': self.target_lon
            },
            'ingestion_timestamp': datetime.now().isoformat()
        }
        
        # Extract raster metadata
        try:
            with rasterio.open(dem_path) as src:
                metadata['raster_info'] = {
                    'width': src.width,
                    'height': src.height,
                    'crs': str(src.crs),
                    'bounds': list(src.bounds),
                    'resolution': list(src.res)
                }
        except Exception:
            pass
        
        metadata_path = self.output_dir / 'elevation_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Saved metadata to {metadata_path}")
    
    def ingest(self, 
               use_local: bool = False,
               local_dem_path: str = None,
               calculate_slope: bool = True) -> dict:
        """
        Main ingestion method
        
        Args:
            use_local: Use local DEM file instead of downloading
            local_dem_path: Path to local DEM file
            calculate_slope: Whether to calculate slope from DEM
            
        Returns:
            Dictionary with paths to DEM and slope files
        """
        results = {}
        
        if use_local and local_dem_path:
            dem_path = self.load_from_file(local_dem_path)
        else:
            dem_path = self.fetch_elevation_data()
        
        if dem_path:
            results['dem'] = dem_path
            
            if calculate_slope:
                slope_path = self.calculate_slope(dem_path)
                if slope_path:
                    results['slope'] = slope_path
            
            # Save metadata
            self.save_metadata(dem_path, results.get('slope'))
            
            print("\n" + "="*60)
            print("USGS Elevation Data Ingestion Summary")
            print("="*60)
            print(f"DEM: {dem_path}")
            if 'slope' in results:
                print(f"Slope: {results['slope']}")
            print("="*60)
        else:
            print("\n⚠ Failed to ingest elevation data")
            print("You can download DEM manually from:")
            print("https://apps.nationalmap.gov/downloader/")
            print("Search for San Rafael, CA and download 1/3 arc-second DEM")
        
        return results


def main():
    """Main execution for standalone testing"""
    ingestor = USGSElevationIngestor()
    
    print("Starting USGS elevation data ingestion...")
    print("This will download a DEM for the San Rafael area\n")
    
    results = ingestor.ingest(calculate_slope=True)
    
    if results:
        print("\nIngestion complete!")
        print("Files created:")
        for key, path in results.items():
            print(f"  {key}: {path}")


if __name__ == '__main__':
    main()
