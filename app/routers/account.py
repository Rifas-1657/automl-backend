from fastapi import APIRouter, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

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
def get_account(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return current_user

@router.put("/account", response_model=UserOut)
def update_account(payload: UserUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    if payload.username:
        current_user.username = payload.username
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user