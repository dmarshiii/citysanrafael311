import os
from pathlib import Path
from dotenv import load_dotenv

# Check current directory
print(f"Current directory: {os.getcwd()}")

# Check if .env exists
env_path = Path('.env')
print(f".env exists: {env_path.exists()}")

# Load .env
load_dotenv('.env')

# Check what was loaded
census_key = os.getenv('CENSUS_API_KEY')
noaa_token = os.getenv('NOAA_API_TOKEN')

print(f"\nCensus key: '{census_key}'")
print(f"Census key length: {len(census_key) if census_key else 0}")

print(f"\nNOAA token: '{noaa_token}'")
print(f"NOAA token length: {len(noaa_token) if noaa_token else 0}")

# Check for common issues
if census_key == 'your_census_api_key_here':
    print("\n❌ ERROR: Census key still has placeholder text!")
    
if noaa_token == 'your_noaa_token_here':
    print("❌ ERROR: NOAA token still has placeholder text!")

# Show raw .env content
print("\n--- Raw .env file content ---")
with open('.env', 'r') as f:
    for line in f:
        if 'API_KEY' in line or 'TOKEN' in line:
            print(line.strip())