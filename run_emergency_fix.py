#!/usr/bin/env python3
"""
Quick runner script for the emergency database migration.
Run this from the backend directory to fix the full_name column issue.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the emergency migration
from emergency_migration import main

if __name__ == "__main__":
    print("🚨 RUNNING EMERGENCY DATABASE FIX")
    print("This will fix the 'no such column: users.full_name' error")
    print()
    
    try:
        main()
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\nThe auth.py workaround should still work even if migration fails.")
        sys.exit(1)
