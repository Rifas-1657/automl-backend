from fastapi import APIRouter, Depends, HTTPException, status  # pyright: ignore[reportMissingImports]
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm  # pyright: ignore[reportMissingImports]
from sqlalchemy.orm import Session  # pyright: ignore[reportMissingImports]
from sqlalchemy import text  # pyright: ignore[reportMissingImports]
from typing import Optional

from db import get_db, Base, engine
from models import User
from schemas import UserCreate, UserOut, Token
from security import hash_password, verify_password, create_access_token, decode_access_token

from pydantic import EmailStr  # pyright: ignore[reportMissingImports]


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")
router = APIRouter()


@router.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)


@router.post("/signup", response_model=UserOut)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    # Normalize inputs to avoid duplicates due to case/whitespace
    email_norm = str(payload.email).strip().lower()
    username_norm = payload.username.strip()
    # Check for existing users using raw SQL to avoid full_name column issues
    existing_query = text("""
        SELECT id FROM users 
        WHERE email = :email OR username = :username
        LIMIT 1
    """)
    existing = db.execute(existing_query, {"email": email_norm, "username": username_norm}).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Email or username already registered")
    
    # Insert user using raw SQL to avoid full_name column issues
    hashed_pwd = hash_password(payload.password)
    
    # Try to insert with full_name column first, fallback if it doesn't exist
    try:
    insert_query = text("""
            INSERT INTO users (email, username, hashed_password, full_name, created_at)
            VALUES (:email, :username, :hashed_password, NULL, datetime('now'))
        """)
        result = db.execute(insert_query, {
            "email": email_norm,
            "username": username_norm,
            "hashed_password": hashed_pwd
        })
    except Exception:
        # Fallback: insert without full_name column
        insert_query = text("""
            INSERT INTO users (email, username, hashed_password, created_at)
            VALUES (:email, :username, :hashed_password, datetime('now'))
        """)
        result = db.execute(insert_query, {
            "email": email_norm,
            "username": username_norm,
            "hashed_password": hashed_pwd
        })
    
    user_id = result.lastrowid
    db.commit()
    
    # Fetch the created user using raw SQL to avoid column issues
    user_query = text("""
        SELECT id, email, username, created_at FROM users WHERE id = :user_id
    """)
    user_data = db.execute(user_query, {"user_id": user_id}).first()
    
    if not user_data:
        raise HTTPException(status_code=500, detail="Failed to create user")
    
    # Create UserOut object manually
    return UserOut(
        id=user_data.id,
        email=user_data.email,
        username=user_data.username,
        full_name=None,  # Set to None since column might not exist
        created_at=user_data.created_at
    )


@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Use raw SQL to avoid full_name column issues
    username_norm = form_data.username.strip()
    user_query = text("""
        SELECT id, username, hashed_password FROM users WHERE username = :username
    """)
    user_data = db.execute(user_query, {"username": username_norm}).first()
    
    if not user_data or not verify_password(form_data.password, user_data.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    token = create_access_token(user_id=user_data.id)
    return Token(access_token=token)


@router.post("/token", response_model=Token)
def get_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """OAuth2 compatible token endpoint"""
    return login(form_data, db)


def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    print(f"Validating token: {token[:20]}...")
    
    user_id = decode_access_token(token)
    print(f"Decoded user_id: {user_id}")
    
    if not user_id:
        print("Token validation failed - no user_id")
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Use raw SQL to avoid full_name column issues
    user_query = text("""
        SELECT id, email, username, hashed_password, created_at FROM users WHERE id = :user_id
    """)
    user_data = db.execute(user_query, {"user_id": user_id}).first()
    
    if not user_data:
        print(f"User not found for user_id: {user_id}")
        raise HTTPException(status_code=401, detail="User not found")
    
    print(f"User found: {user_data.username} (ID: {user_data.id})")
    
    # Create User object manually to avoid column issues
    user = User()
    user.id = user_data.id
    user.email = user_data.email
    user.username = user_data.username
    user.hashed_password = user_data.hashed_password
    user.created_at = user_data.created_at
    # full_name will be None by default since we're not setting it
    
    return user


@router.get("/users/me", response_model=UserOut)
def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return UserOut(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=getattr(current_user, 'full_name', None),
        created_at=current_user.created_at
    )


@router.post("/logout")
def logout():
    return {"detail": "Logged out"}


@router.get("/test-auth")
def test_auth(current_user: User = Depends(get_current_user)):
    """Test endpoint to verify authentication is working"""
    return {
        "message": "Authentication successful!",
        "user_id": current_user.id,
        "username": current_user.username,
        "email": current_user.email
    }


    