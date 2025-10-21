#!/usr/bin/env python3
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

print("Testing main.py import...")

try:
    from app.main import app
    print("SUCCESS: Main app imported")
    print(f"App title: {app.title}")
    print(f"App version: {app.version}")
except Exception as e:
    print(f"FAILED: Main app import error: {e}")
    import traceback
    traceback.print_exc()

print("Main test complete.")
