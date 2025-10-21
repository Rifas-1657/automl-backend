#!/usr/bin/env python3
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

print("Testing auth router import step by step...")

try:
    print("1. Testing basic imports...")
    from fastapi import APIRouter, Depends, HTTPException, status
    print("   SUCCESS: FastAPI imports")
    
    from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
    print("   SUCCESS: FastAPI security imports")
    
    from sqlalchemy.orm import Session
    from sqlalchemy import text
    print("   SUCCESS: SQLAlchemy imports")
    
    from typing import Optional
    print("   SUCCESS: Typing imports")
    
    print("2. Testing app-specific imports...")
    from db import get_db, Base, engine
    print("   SUCCESS: DB imports")
    
    from models import User
    print("   SUCCESS: Models imports")
    
    from schemas import UserCreate, UserOut, Token
    print("   SUCCESS: Schemas imports")
    
    from security import hash_password, verify_password, create_access_token, decode_access_token
    print("   SUCCESS: Security imports")
    
    from pydantic import EmailStr
    print("   SUCCESS: Pydantic imports")
    
    print("3. Testing router creation...")
    oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")
    router = APIRouter()
    print("   SUCCESS: Router created")
    
    print("4. Testing router import...")
    from routers.auth import router as auth_router
    print("   SUCCESS: Auth router imported")
    
except Exception as e:
    print(f"FAILED at step: {e}")
    import traceback
    traceback.print_exc()

print("Auth import test complete.")
