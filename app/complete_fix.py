# complete_fix.py
import sys
import os
sys.path.append(os.path.dirname(__file__))

from db import engine, Base
from models import User, Dataset, TrainedModel
from sqlalchemy import inspect

print("🚀 STARTING COMPLETE DATABASE FIX...")

try:
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("✅ All tables created successfully!")
    
    # Verify tables
    inspector = inspect(engine)
    tables = inspector.get_table_names()
    print(f"✅ Current tables: {tables}")
    
    # Check if our main tables exist
    required_tables = ['users', 'datasets', 'trained_models']
    missing_tables = [table for table in required_tables if table not in tables]
    
    if missing_tables:
        print(f"❌ Missing tables: {missing_tables}")
    else:
        print("🎉 ALL REQUIRED TABLES ARE READY!")
        
except Exception as e:
    print(f"❌ Error: {e}")