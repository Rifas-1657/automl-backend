#!/usr/bin/env python3
"""
Simple script to run the database migration from the backend directory.
"""

import sys
import os

# Add the app directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

# Import and run the migration
from migrate_add_full_name import add_full_name_column, verify_users_table_structure

if __name__ == "__main__":
    print("🚀 Running database migration...")
    success = add_full_name_column()
    
    if success:
        verify_users_table_structure()
        print("\n✅ Migration completed successfully!")
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)
