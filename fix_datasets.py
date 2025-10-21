#!/usr/bin/env python3
"""
Quick runner script to fix the datasets table column issues.
Run this from the backend directory to fix the datasets.file_path error.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the datasets migration
from migrate_datasets_table import main

if __name__ == "__main__":
    print("🚀 FIXING DATASETS TABLE COLUMN ISSUES")
    print("This will fix the 'no such column: datasets.file_path' error")
    print()
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nThe datasets.py workaround should still work even if migration fails.")
        sys.exit(1)
