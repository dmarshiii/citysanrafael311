"""
Data Ingestion Modules for 311 AI Project

This package provides data ingestion classes for all five data sources:
- San Rafael 311 Service Requests (ArcGIS)
- NOAA Climate Data
- US Census Demographics
- TIGER/Line Shapefiles
- USGS Elevation Data
"""

from .san_rafael_311 import SanRafael311Ingestor
from .noaa_climate import NOAAIngestor
from .census_data import CensusIngestor
from .tiger_shapefiles import TIGERIngestor
from .usgs_elevation import USGSElevationIngestor

__all__ = [
    'SanRafael311Ingestor',
    'NOAAIngestor',
    'CensusIngestor',
    'TIGERIngestor',
    'USGSElevationIngestor'
]
