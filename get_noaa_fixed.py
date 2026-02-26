import requests
import pandas as pd
from datetime import datetime

# Your NOAA token
import os
from dotenv import load_dotenv
load_dotenv('.env')
token = os.getenv('NOAA_API_TOKEN')

# Known stations near San Rafael, CA
# Let's try a few major Bay Area stations
stations_to_try = [
    'GHCND:USW00023234',  # San Francisco Airport
    'GHCND:USC00047414',  # San Rafael
    'GHCND:USC00048273',  # San Francisco Downtown
]

base_url = "https://www.ncei.noaa.gov/cdo-web/api/v2/data"

for station_id in stations_to_try:
    print(f"\nTrying station: {station_id}")
    
    params = {
        'datasetid': 'GHCND',
        'stationid': station_id,
        'startdate': '2022-06-01',
        'enddate': '2024-12-31',
        'datatypeid': 'PRCP',  # Just precipitation for test
        'units': 'metric',
        'limit': 10
    }
    
    headers = {'token': token}
    
    try:
        response = requests.get(base_url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            if 'results' in data and len(data['results']) > 0:
                print(f"✓ Found data! Station {station_id} works")
                print(f"  Sample records: {len(data['results'])}")
                print(f"  First record: {data['results'][0]}")
                print(f"\n✓ Use this station: {station_id}")
                break
        else:
            print(f"  Status: {response.status_code}")
    except Exception as e:
        print(f"  Error: {e}")
else:
    print("\n✗ None of the stations worked")
    print("Recommendation: Use CSV download instead")