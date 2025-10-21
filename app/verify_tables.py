import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db import engine

def verify_tables():
    try:
        with engine.connect() as conn:
            result = conn.execute("SHOW TABLES")
            tables = [row[0] for row in result]
            print("✅ Tables in database:")
            for table in tables:
                print(f"   - {table}")
                
            if tables:
                print("✅ Tables created successfully!")
            else:
                print("❌ No tables found!")
                
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    verify_tables()