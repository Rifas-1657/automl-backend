from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db
from models import User
from schemas import UserOut, UserUpdate
from security import decode_access_token

router = APIRouter()

# Let CORSMiddleware handle preflight; no custom OPTIONS route

@router.get("/account", response_model=UserOut)
def get_account(request: Request, db: Session = Depends(get_db)):
    """Lightweight auth: read Bearer token, decode user_id, and fetch user via raw SQL."""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header.split(" ", 1)[1].strip()
    try:
        user_id = decode_access_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    row = db.execute(text(
        """
        SELECT id, email, username, created_at
        FROM users
        WHERE id = :uid
        LIMIT 1
        """
    ), {"uid": user_id}).first()

    if not row:
        raise HTTPException(status_code=401, detail="User not found")

    return UserOut(
        id=row.id,
        email=row.email,
        username=row.username,
        full_name=None,
        created_at=row.created_at
    )

@router.put("/account", response_model=UserOut)
def update_account(payload: UserUpdate, request: Request, db: Session = Depends(get_db)):
    # Authenticate like GET /account
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = auth_header.split(" ", 1)[1].strip()
    try:
        user_id = decode_access_token(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

    current_row = db.execute(text("SELECT id, username FROM users WHERE id = :uid"), {"uid": user_id}).first()
    if not current_row:
        raise HTTPException(status_code=401, detail="User not found")
    if payload.username:
        new_username = payload.username.strip()
        if new_username and new_username != current_row.username:
            # Ensure username uniqueness
            existing = db.execute(text("""
                SELECT id FROM users WHERE username = :username AND id != :id LIMIT 1
            """), {"username": new_username, "id": user_id}).first()
            if existing:
                raise HTTPException(status_code=400, detail="Username already taken")
            db.execute(text("UPDATE users SET username = :u WHERE id = :id"), {"u": new_username, "id": user_id})
            db.commit()
    # Return updated user
    row = db.execute(text("SELECT id, email, username, created_at FROM users WHERE id = :id"), {"id": user_id}).first()
    return UserOut(id=row.id, email=row.email, username=row.username, full_name=None, created_at=row.created_at)