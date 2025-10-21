from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class UserCreate(BaseModel):
    email: str
    username: str
    password: str

class UserResponse(BaseModel):
    email: str
    username: str
    message: str

@router.post("/signup", response_model=UserResponse)
async def signup(user: UserCreate):
    # Simple version without database
    return {
        "email": user.email,
        "username": user.username,
        "message": "User created successfully (simple version)"
    }

@router.post("/login")
async def login(email: str, password: str):
    return {"message": "Login successful (simple version)"}

@router.get("/test")
async def test_auth():
    return {"message": "Auth router is working!"}