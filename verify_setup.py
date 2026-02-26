#!/usr/bin/env python3
"""
Setup Verification Script
Run this to verify your environment is properly configured
"""

import sys
from pathlib import Path

def check_python_version():
    """Check Python version is 3.8+"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor} (need 3.8+)")
        return False

def check_dependencies():
    """Check if key dependencies can be imported"""
    required = {
        'pandas': 'pandas',
        'numpy': 'numpy', 
        'geopandas': 'geopandas',
        'requests': 'requests',
        'rasterio': 'rasterio',
        'dotenv': 'python-dotenv'
    }
    
    all_ok = True
    for module, package in required.items():
        try:
            __import__(module)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (run: pip install {package})")
            all_ok = False
    
    return all_ok

def check_directory_structure():
    """Verify directory structure is correct"""
    required_dirs = [
        'data/raw/311_requests',
        'data/raw/census',
        'data/raw/tiger',
        'data/raw/usgs',
        'data/raw/noaa',
        'data/interim',
        'data/processed',
        'src/ingestion',
        'src/utils',
        'notebooks',
        'tests'
    ]
    
    project_root = Path(__file__).parent
    all_ok = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ (missing)")
            all_ok = False
    
    return all_ok

def check_config_file():
    """Check if .env file exists"""
    project_root = Path(__file__).parent
    env_file = project_root / '.env'
    example_file = project_root / '.env.example'
    
    if env_file.exists():
        print("✓ .env file exists")
        return True
    elif example_file.exists():
        print("⚠ .env file missing (copy from .env.example)")
        print(f"  Run: cp .env.example .env")
        return False
    else:
        print("✗ Both .env and .env.example missing")
        return False

def check_api_keys():
    """Check if API keys are configured"""
    try:
        from src.utils.config import config
        
        status = config.validate_api_keys()
        
        print("\nAPI Key Status:")
        for source, valid in status.items():
            icon = "✓" if valid else "⚠"
            print(f"  {icon} {source}")
        
        missing = config.get_missing_keys()
        if missing:
            print(f"\nMissing keys: {', '.join(missing)}")
            print("Options:")
            print("  1. Register for API keys (see README.md)")
            print("  2. Use CSV fallback mode (set USE_CSV_FALLBACK=true in .env)")
        
        return len(missing) == 0
    except Exception as e:
        print(f"✗ Error checking API keys: {e}")
        return False

def main():
    """Run all verification checks"""
    print("="*60)
    print("311 AI Project - Setup Verification")
    print("="*60)
    
    print("\n1. Python Version:")
    py_ok = check_python_version()
    
    print("\n2. Dependencies:")
    deps_ok = check_dependencies()
    
    print("\n3. Directory Structure:")
    dirs_ok = check_directory_structure()
    
    print("\n4. Configuration:")
    config_ok = check_config_file()
    
    print("\n5. API Keys:")
    api_ok = check_api_keys()
    
    print("\n" + "="*60)
    
    if py_ok and deps_ok and dirs_ok and config_ok:
        print("✓ Setup Complete!")
        if not api_ok:
            print("\n⚠ Note: Some API keys are missing")
            print("  You can still use CSV fallback mode for testing")
        print("\nNext steps:")
        print("  1. Review QUICKSTART.md for usage instructions")
        print("  2. Test with: python src/run_ingestion.py --skip-usgs")
    else:
        print("✗ Setup Incomplete - fix the issues above")
        print("\nRefer to README.md for detailed setup instructions")
    
    print("="*60)

if __name__ == '__main__':
    main()
