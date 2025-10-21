import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import Base, engine

def create_tables():
    print("Creating application tables...")
    try:
        Base.metadata.create_all(bind=engine)
        print("All tables created successfully!")
        
        # Verify tables
        with engine.connect() as conn:
            # Use SQLite syntax instead of MySQL
            from sqlalchemy import text
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
            tables = [row[0] for row in result]
            print(f"Tables in database: {tables}")
            
    except Exception as e:
        print(f"Error creating tables: {e}")

if __name__ == "__main__":
    create_tables()