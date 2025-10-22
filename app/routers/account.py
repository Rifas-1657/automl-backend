from fastapi import APIRouter, Depends, HTTPException, Response, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db
from models import User
from schemas import UserOut, UserUpdate
from routers.auth import get_current_user

router = APIRouter()

# ADD THIS: Explicit CORS for account routes
@router.options("/account")
async def account_options():
    return {"message": "OK"}

@router.get("/account", response_model=UserOut)
def get_account(request: Request, response: Response, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    # Reflect exact Origin for credentialed requests (cannot be "*")
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    return current_user

@router.put("/account", response_model=UserOut)
def update_account(payload: UserUpdate, request: Request, response: Response, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    origin = request.headers.get("origin")
    if origin:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Credentials"] = "true"
    if payload.username:
        new_username = payload.username.strip()
        if new_username and new_username != current_user.username:
            # Ensure username uniqueness
            existing = db.execute(text("""
                SELECT id FROM users WHERE username = :username AND id != :id LIMIT 1
            """), {"username": new_username, "id": current_user.id}).first()
            if existing:
                raise HTTPException(status_code=400, detail="Username already taken")
            current_user.username = new_username
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user