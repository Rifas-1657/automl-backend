# check_database_status.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from db import engine, SessionLocal
from sqlalchemy import text, inspect

def check_status():
    print("🔍 CHECKING DATABASE STATUS...")
    
    # Check which database we're using
    db_url = str(engine.url)
    print(f"📊 Database URL: {db_url}")
    
    if 'sqlite' in db_url:
        print("💾 Using: SQLite Database")
        db_file = db_url.split('///')[-1]
        if os.path.exists(db_file):
            size = os.path.getsize(db_file)
            print(f"💾 SQLite file: {db_file} ({size} bytes)")
        else:
            print(f"❌ SQLite file not found: {db_file}")
    else:
        print("🐬 Using: MySQL Database")
    
    try:
        # Test connection and get tables
        db = SessionLocal()
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"📋 Tables: {tables}")
        
        # Check row counts
        for table in tables:
            count = db.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            print(f"   📊 {table}: {count} rows")
        
        db.close()
        print("🎉 DATABASE STATUS: HEALTHY ✅")
        
    except Exception as e:
        print(f"❌ DATABASE STATUS: ERROR - {e}")

if __name__ == "__main__":
    check_status()