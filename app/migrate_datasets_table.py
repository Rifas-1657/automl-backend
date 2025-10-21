#!/usr/bin/env python3
"""
Database migration script to add missing columns to datasets table.
This script can be run without stopping the server and works with both SQLite and MySQL.
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

def add_missing_columns():
    """Add missing columns to datasets table"""
    db_type = get_database_type()
    print(f"Database type detected: {db_type}")
    
    columns_to_add = [
        ('file_path', 'VARCHAR(500)'),
        ('file_size', 'INTEGER'),
        ('uploaded_by', 'INTEGER'),
        ('created_at', 'DATETIME')
    ]
    
    try:
        with engine.connect() as conn:
            trans = conn.begin()
            try:
                for column_name, column_type in columns_to_add:
                    if not check_column_exists('datasets', column_name):
                        print(f"Adding column: {column_name}")
                        
                        if db_type == 'sqlite':
                            alter_query = text(f"ALTER TABLE datasets ADD COLUMN {column_name} {column_type}")
                        elif db_type == 'mysql':
                            alter_query = text(f"ALTER TABLE datasets ADD COLUMN {column_name} {column_type}")
                        else:
                            # Generic SQL
                            alter_query = text(f"ALTER TABLE datasets ADD COLUMN {column_name} {column_type}")
                        
                        conn.execute(alter_query)
                        print(f"✅ Added column: {column_name}")
                    else:
                        print(f"✅ Column already exists: {column_name}")
                
                trans.commit()
                print("✅ All columns added successfully!")
                return True
                
            except Exception as e:
                trans.rollback()
                print(f"❌ Error adding columns: {e}")
                return False
                
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return False

def show_table_structure():
    """Show the current structure of the datasets table"""
    try:
        with engine.connect() as conn:
            db_type = get_database_type()
            
            if db_type == 'sqlite':
                query = text("PRAGMA table_info(datasets)")
            elif db_type == 'mysql':
                query = text("DESCRIBE datasets")
            else:
                query = text("SELECT * FROM datasets LIMIT 0")
            
            result = conn.execute(query)
            columns = result.fetchall()
            
            print(f"\n📋 Current datasets table structure ({db_type}):")
            print("-" * 50)
            for col in columns:
                print(f"  {col}")
                
    except Exception as e:
        print(f"❌ Error showing table structure: {e}")

def verify_migration():
    """Verify that the migration was successful"""
    expected_columns = ['id', 'filename', 'file_path', 'file_size', 'uploaded_by', 'created_at']
    all_exist = True
    
    for column in expected_columns:
        if check_column_exists('datasets', column):
            print(f"✅ Column exists: {column}")
        else:
            print(f"❌ Column missing: {column}")
            all_exist = False
    
    return all_exist

def main():
    print("🚀 DATASETS TABLE MIGRATION")
    print("=" * 50)
    print("Fixing: no such column: datasets.file_path")
    print("=" * 50)
    
    # Show current table structure
    show_table_structure()
    
    # Check if all columns already exist
    expected_columns = ['file_path', 'file_size', 'uploaded_by', 'created_at']
    missing_columns = [col for col in expected_columns if not check_column_exists('datasets', col)]
    
    if not missing_columns:
        print("\n✅ All required columns already exist!")
        print("The datasets table is up to date.")
        return
    
    print(f"\n🔧 Adding missing columns: {missing_columns}")
    success = add_missing_columns()
    
    if success:
        print("\n" + "=" * 50)
        verify_migration()
        show_table_structure()
        print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        print("Your datasets upload and listing should now work without errors.")
    else:
        print("\n❌ MIGRATION FAILED!")
        print("Please check the error messages above.")
        print("\nThe raw SQL workaround in datasets.py should still work.")

if __name__ == "__main__":
    main()
