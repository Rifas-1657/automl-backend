#!/usr/bin/env python3
"""
Database migration script to add full_name column to users table.
This script can be run without stopping the server and works with both SQLite and MySQL.
"""

import sys
import os
from sqlalchemy import text, inspect

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import engine
from models import User

def check_column_exists(table_name, column_name):
    """Check if a column exists in the specified table"""
    inspector = inspect(engine)
    columns = inspector.get_columns(table_name)
    return any(col['name'] == column_name for col in columns)

def add_full_name_column():
    """Add full_name column to users table if it doesn't exist"""
    try:
        print("🔍 Checking if full_name column exists in users table...")
        
        # Check if column already exists
        if check_column_exists('users', 'full_name'):
            print("✅ full_name column already exists. No migration needed.")
            return True
            
        print("📝 Adding full_name column to users table...")
        
        with engine.connect() as conn:
            # Start a transaction
            trans = conn.begin()
            try:
                # Add the column with a default value of NULL
                # This works for both SQLite and MySQL
                alter_query = text("ALTER TABLE users ADD COLUMN full_name VARCHAR(255)")
                conn.execute(alter_query)
                
                # Commit the transaction
                trans.commit()
                print("✅ Successfully added full_name column to users table!")
                
                # Verify the column was added
                if check_column_exists('users', 'full_name'):
                    print("✅ Verification: full_name column confirmed in users table")
                    return True
                else:
                    print("❌ Verification failed: full_name column not found after migration")
                    return False
                    
            except Exception as e:
                # Rollback the transaction
                trans.rollback()
                print(f"❌ Error during migration: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        return False

def verify_users_table_structure():
    """Verify the users table structure matches the model"""
    try:
        print("\n🔍 Verifying users table structure...")
        inspector = inspect(engine)
        columns = inspector.get_columns('users')
        
        expected_columns = ['id', 'email', 'username', 'hashed_password', 'full_name', 'created_at']
        existing_columns = [col['name'] for col in columns]
        
        print(f"Expected columns: {expected_columns}")
        print(f"Existing columns: {existing_columns}")
        
        missing_columns = set(expected_columns) - set(existing_columns)
        if missing_columns:
            print(f"⚠️  Missing columns: {missing_columns}")
            return False
        else:
            print("✅ All expected columns are present!")
            return True
            
    except Exception as e:
        print(f"❌ Error verifying table structure: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting database migration for full_name column...")
    print("=" * 50)
    
    # Run the migration
    success = add_full_name_column()
    
    if success:
        print("\n" + "=" * 50)
        print("✅ Migration completed successfully!")
        
        # Verify the table structure
        verify_users_table_structure()
        
        print("\n🎉 You can now restart your application or the migration will be applied automatically.")
    else:
        print("\n" + "=" * 50)
        print("❌ Migration failed. Please check the error messages above.")
        sys.exit(1)
