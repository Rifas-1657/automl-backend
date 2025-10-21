# test_sqlite_connection.py
import sqlite3
import os

def test_sqlite():
    try:
        # Check if SQLite file exists
        if os.path.exists("dev.db"):
            file_size = os.path.getsize("dev.db")
            print(f"✅ SQLite file exists: dev.db ({file_size} bytes)")
        else:
            print("❌ SQLite file not found")
            return False
        
        # Test connection
        conn = sqlite3.connect("dev.db")
        cursor = conn.cursor()
        
        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"✅ Tables in SQLite: {tables}")
        print("🎉 SQLite database is working perfectly!")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ SQLite connection failed: {e}")
        return False

if __name__ == "__main__":
    test_sqlite()