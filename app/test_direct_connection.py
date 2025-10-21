import pymysql

def test_direct():
    try:
        connection = pymysql.connect(
            host='localhost',
            user='automl_web_user',
            password='rif@123',  # Use your actual password from the output
            database='automl_web_app',
            port=3306
        )
        print("✅ Direct MySQL connection successful!")
        
        with connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE()")
            db_name = cursor.fetchone()
            print(f"✅ Connected to: {db_name[0]}")
            
            # Show tables
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            print(f"✅ Tables in database: {[table[0] for table in tables]}")
        
        connection.close()
        return True
        
    except Exception as e:
        print(f"❌ Direct connection failed: {e}")
        return False

if __name__ == "__main__":
    test_direct()