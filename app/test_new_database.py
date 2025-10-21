import sys
import os
import pymysql

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import settings
except ImportError:
    print("❌ Cannot import config. Make sure config.py exists in the backend directory")
    exit(1)

def test_new_database():
    try:
        # Check if settings has the required attributes
        required_attrs = ['MYSQL_HOST', 'MYSQL_USER', 'MYSQL_PASSWORD', 'MYSQL_DATABASE', 'MYSQL_PORT']
        for attr in required_attrs:
            if not hasattr(settings, attr):
                print(f"❌ Missing attribute in settings: {attr}")
                return False
        
        print(f"✅ MySQL Connection Details:")
        print(f"   Host: {settings.MYSQL_HOST}")
        print(f"   User: {settings.MYSQL_USER}")
        print(f"   Database: {settings.MYSQL_DATABASE}")
        print(f"   Port: {settings.MYSQL_PORT}")
        
        connection = pymysql.connect(
            host=settings.MYSQL_HOST,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            port=int(settings.MYSQL_PORT)
        )
        print("✅ New MySQL database connection successful!")
        
        with connection.cursor() as cursor:
            # Check database name
            cursor.execute("SELECT DATABASE()")
            db_name = cursor.fetchone()
            print(f"✅ Connected to database: {db_name[0]}")
            
            # Create a test table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS connection_test (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    test_message VARCHAR(255),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            print("✅ Test table created successfully!")
            
            # Insert test data
            cursor.execute("INSERT INTO connection_test (test_message) VALUES (%s)", 
                         ("Hello from the new database!",))
            connection.commit()
            print("✅ Test data inserted successfully!")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ New database connection failed: {e}")
        return False

if __name__ == "__main__":
    test_new_database()