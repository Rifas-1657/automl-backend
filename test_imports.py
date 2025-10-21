#!/usr/bin/env python3
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

print("Testing router imports...")

try:
    from routers.auth import router as auth_router
    print("SUCCESS: Auth router imported")
except Exception as e:
    print(f"FAILED: Auth router import error: {e}")

try:
    from routers.datasets import router as datasets_router
    print("SUCCESS: Datasets router imported")
except Exception as e:
    print(f"FAILED: Datasets router import error: {e}")

try:
    from routers.ml import router as ml_router
    print("SUCCESS: ML router imported")
except Exception as e:
    print(f"FAILED: ML router import error: {e}")

try:
    from routers.account import router as account_router
    print("SUCCESS: Account router imported")
except Exception as e:
    print(f"FAILED: Account router import error: {e}")

try:
    from routers.history import router as history_router
    print("SUCCESS: History router imported")
except Exception as e:
    print(f"FAILED: History router import error: {e}")

print("Import test complete.")
