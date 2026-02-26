"""
Debug script to see what config.py is loading
"""

from pathlib import Path
import os
from dotenv import load_dotenv

print("="*60)
print("CONFIG.PY DEBUG")
print("="*60)

# Simulate what config.py does
print("\n1. Finding project root from config.py location:")
config_file = Path("src/utils/config.py")
project_root = config_file.parent.parent
print(f"   Config file: {config_file}")
print(f"   Project root (2 levels up): {project_root}")
print(f"   Absolute project root: {project_root.absolute()}")

# Check for .env
env_path = project_root / '.env'
print(f"\n2. Looking for .env at: {env_path}")
print(f"   .env exists: {env_path.exists()}")
print(f"   Absolute path: {env_path.absolute()}")

# Try to load it
if env_path.exists():
    load_dotenv(env_path)
    print(f"\n3. After load_dotenv():")
    census = os.getenv('CENSUS_API_KEY', '')
    noaa = os.getenv('NOAA_API_TOKEN', '')
    print(f"   CENSUS_API_KEY: '{census[:10]}...' (length: {len(census)})")
    print(f"   NOAA_API_TOKEN: '{noaa[:10]}...' (length: {len(noaa)})")
else:
    print("\n3. .env file NOT FOUND by config.py path logic!")

# Now check current directory
print(f"\n4. Current working directory: {os.getcwd()}")
print(f"   .env in current dir: {Path('.env').exists()}")

# Try loading from current directory
print(f"\n5. Loading from current directory:")
load_dotenv('.env')
census = os.getenv('CENSUS_API_KEY', '')
noaa = os.getenv('NOAA_API_TOKEN', '')
print(f"   CENSUS_API_KEY: '{census[:10]}...' (length: {len(census)})")
print(f"   NOAA_API_TOKEN: '{noaa[:10]}...' (length: {len(noaa)})")

print("\n" + "="*60)
print("COMPARISON:")
print("="*60)
print("If test_keys.py works but verify_setup.py doesn't, the issue is:")
print("- config.py is looking in the wrong place for .env")
print("- OR config.py is being imported before os.chdir() happens")
print("="*60)
