#!/usr/bin/env python3
"""
Emergency database migration script to fix "no such column: users.full_name" error.
This script can be run while the server is running and provides multiple solutions.
"""

import sys
import os
from sqlalchemy import text, inspect

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import engine

def check_column_exists(table_name, column_name):
    """Check if a column exists in the specified table"""
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        return any(col['name'] == column_name for col in columns)
    except Exception as e:
        print(f"Error checking columns: {e}")
        return False

def get_database_type():
    """Determine if we're using SQLite or MySQL"""
    url = str(engine.url)
    if 'sqlite' in url:
        return 'sqlite'
    elif 'mysql' in url:
        return 'mysql'
    else:
        return 'unknown'

def add_full_name_column():
    """Add full_name column to users table"""
    db_type = get_database_type()
    print(f"Database type detected: {db_type}")
    
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                if db_type == 'sqlite':
                    alter_query = text("ALTER TABLE users ADD COLUMN full_name VARCHAR(255)")
                elif db_type == 'mysql':
                    alter_query = text("ALTER TABLE users ADD COLUMN full_name VARCHAR(255)")
                else:
                    # Generic SQL that should work for most databases
                    alter_query = text("ALTER TABLE users ADD COLUMN full_name VARCHAR(255)")
                
                conn.execute(alter_query)
                trans.commit()
                print("✅ Successfully added full_name column!")
                return True
                
            except Exception as e:
                trans.rollback()
                print(f"❌ Error adding column: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def show_table_structure():
    """Show the current structure of the users table"""
    try:
        with engine.connect() as conn:
            db_type = get_database_type()
            
            if db_type == 'sqlite':
                query = text("PRAGMA table_info(users)")
            elif db_type == 'mysql':
                query = text("DESCRIBE users")
            else:
                # Generic approach
                query = text("SELECT * FROM users LIMIT 0")
            
            result = conn.execute(query)
            columns = result.fetchall()
            
            print(f"\n📋 Current users table structure ({db_type}):")
            print("-" * 50)
            for col in columns:
                print(f"  {col}")
                
    except Exception as e:
        print(f"❌ Error showing table structure: {e}")

def verify_migration():
    """Verify that the migration was successful"""
    if check_column_exists('users', 'full_name'):
        print("✅ Verification: full_name column exists!")
        return True
    else:
        print("❌ Verification failed: full_name column not found")
        return False

def main():
    print("🚨 EMERGENCY DATABASE MIGRATION")
    print("=" * 50)
    print("Fixing: sqlite3.OperationalError: no such column: users.full_name")
    print("=" * 50)
    
    # Show current table structure
    show_table_structure()
    
    # Check if column already exists
    if check_column_exists('users', 'full_name'):
        print("\n✅ full_name column already exists!")
        print("The migration has already been applied.")
        return
    
    print(f"\n🔧 Adding full_name column to users table...")
    success = add_full_name_column()
    
    if success:
        print("\n" + "=" * 50)
        verify_migration()
        show_table_structure()
        print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("Your FastAPI application should now work without the column error.")
    else:
        print("\n❌ MIGRATION FAILED!")
        print("Please check the error messages above.")
        print("\nAlternative solutions:")
        print("1. Restart your FastAPI server after the auth.py changes")
        print("2. The raw SQL workaround in auth.py should still work")
        print("3. Contact support if issues persist")

if __name__ == "__main__":
    main()
