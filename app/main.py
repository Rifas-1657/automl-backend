import os
import sys
import traceback
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure app package on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import SessionLocal, engine
from create_tables import create_tables
from sqlalchemy import text, inspect

app = FastAPI(title="AutoML Web App API", version="0.1.0")
_auth_router_loaded = False

# 1) Explicit startup log + robust DB check
@app.on_event("startup")
async def startup_event():
    print("Starting FastAPI backend...")
    try:
        # Ensure DB tables exist
        try:
            create_tables()
        except Exception as e:
            print(f"Auto table creation failed: {e}")

        db = SessionLocal()
        db.execute(text("SELECT 1"))

        inspector = inspect(engine)
        tables = inspector.get_table_names()

        print("DATABASE CONNECTION SUCCESSFUL!")
        print(f"Available tables: {tables}")

        required_tables = ["users", "datasets", "trained_models"]
        missing = [t for t in required_tables if t not in tables]
        if missing:
            print(f"Missing tables: {missing}")
        else:
            print("All required tables present!")
    except Exception as e:
        print(f"Database connection failed: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass

# 2) CORS (Vite + localhost + all)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://automl-frontend-production.up.railway.app",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# 3) CORS preflight handled by CORSMiddleware

# 4) Health endpoints
@app.get("/api/health")
def api_health():
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        ok = True
    except Exception:
        ok = False
    finally:
        try:
            db.close()
        except Exception:
            pass
    return {"status": "ok", "db_connected": ok}

@app.get("/health")  # keep old path too
def health_legacy():
    return {"status": "ok"}

# 4) Routers (with logging)
try:
    from routers.auth import router as auth_router
    app.include_router(auth_router, prefix="/api", tags=["auth"])
    print("Auth router loaded successfully!")
    _auth_router_loaded = True
except Exception as e:
    print(f"Auth router failed: {e}\n{traceback.format_exc()}")

try:
    from routers.datasets import router as datasets_router
    app.include_router(datasets_router, prefix="/api", tags=["datasets"])
    print("Datasets router loaded successfully!")
except Exception as e:
    print(f"Datasets router failed: {e}\n{traceback.format_exc()}")

try:
    from routers.ml import router as ml_router
    app.include_router(ml_router, prefix="/api", tags=["ml"])
    print("ML router loaded successfully!")
except Exception as e:
    print(f"ML router failed: {e}\n{traceback.format_exc()}")

try:
    from routers.account import router as account_router
    app.include_router(account_router, prefix="/api", tags=["account"])
    print("Account router loaded successfully!")
except Exception as e:
    print(f"Account router failed: {e}\n{traceback.format_exc()}")

try:
    from routers.history import router as history_router
    app.include_router(history_router, prefix="/api", tags=["history"])
    print("History router loaded successfully!")
except Exception as e:
    print(f"History router failed: {e}\n{traceback.format_exc()}")

# Visualization feature removed

# Diagnostics endpoint to see what loaded in prod
@app.get("/api/debug/routers")
def debug_routers():
    endpoints = []
    for r in app.routes:
        try:
            # Starlette Mount/Route objects vary; safely access attributes
            path = getattr(r, 'path', getattr(r, 'path_format', str(r)))
            methods = list(getattr(r, 'methods', []) or [])
            name = getattr(r, 'name', type(r).__name__)
            endpoints.append({
                "name": name,
                "path": path,
                "methods": methods,
                "type": type(r).__name__,
            })
        except Exception as e:
            endpoints.append({"error": f"{type(r).__name__}: {e}"})
    return {"auth_router_loaded": _auth_router_loaded, "endpoints": endpoints}

# Attempt dynamic import checks to report exact failures
@app.get("/api/debug/imports")
def debug_imports():
    import importlib
    modules = [
        'routers.auth',
        'routers.datasets',
        'routers.ml',
        'routers.account',
        'routers.history',
        'routers.visualization',
    ]
    results = {}
    for m in modules:
        try:
            importlib.invalidate_caches()
            importlib.import_module(m)
            results[m] = {"ok": True}
        except Exception as e:
            results[m] = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
    return results

# 5) Root and favicon endpoints
@app.get("/")
def root():
    return {
        "message": "AutoML Web App API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/api/health"
    }

@app.get("/favicon.ico")
def favicon():
    return {"message": "No favicon"}

# 6) Static files for plots/uploads
import os
from config import settings

# Mount both storage and uploads directories
if os.path.exists("storage"):
    app.mount("/api/files", StaticFiles(directory="storage"), name="files")
if os.path.exists("storage/uploads"):
    app.mount("/api/uploads", StaticFiles(directory="storage/uploads"), name="uploads")

# Fallback auth endpoints if the auth router failed to import
if not _auth_router_loaded:
    try:
        from pydantic import BaseModel
        from fastapi import Depends, HTTPException, status
        from fastapi.security import OAuth2PasswordRequestForm
        from db import SessionLocal
        from sqlalchemy import text

        class FallbackSignup(BaseModel):
            email: str
            username: str
            password: str

        @app.post("/api/signup")
        def fallback_signup(payload: FallbackSignup):
            db = SessionLocal()
            try:
                email_norm = str(payload.email).strip().lower()
                username_norm = payload.username.strip()

                existing = db.execute(text(
                    """
                    SELECT id FROM users 
                    WHERE email = :email OR username = :username
                    LIMIT 1
                    """
                ), {"email": email_norm, "username": username_norm}).first()
                if existing:
                    raise HTTPException(status_code=400, detail="Email or username already registered")

                from security import hash_password
                hashed_pwd = hash_password(payload.password)

                try:
                    insert_query = text(
                        """
                        INSERT INTO users (email, username, hashed_password, full_name, created_at)
                        VALUES (:email, :username, :hashed_password, NULL, datetime('now'))
                        """
                    )
                    result = db.execute(insert_query, {
                        "email": email_norm,
                        "username": username_norm,
                        "hashed_password": hashed_pwd
                    })
                except Exception:
                    insert_query = text(
                        """
                        INSERT INTO users (email, username, hashed_password, created_at)
                        VALUES (:email, :username, :hashed_password, datetime('now'))
                        """
                    )
                    result = db.execute(insert_query, {
                        "email": email_norm,
                        "username": username_norm,
                        "hashed_password": hashed_pwd
                    })

                user_id = result.lastrowid
                db.commit()

                user_query = text("""
                    SELECT id, email, username, created_at FROM users WHERE id = :user_id
                """)
                user_data = db.execute(user_query, {"user_id": user_id}).first()
                if not user_data:
                    raise HTTPException(status_code=500, detail="Failed to create user")

                return {
                    "id": user_data.id,
                    "email": user_data.email,
                    "username": user_data.username,
                    "full_name": None,
                    "created_at": user_data.created_at,
                }
            finally:
                try:
                    db.close()
                except Exception:
                    pass

        @app.post("/api/token")
        def fallback_token(form_data: OAuth2PasswordRequestForm = Depends()):
            db = SessionLocal()
            try:
                username_norm = form_data.username.strip()
                user_query = text("""
                    SELECT id, username, hashed_password FROM users WHERE username = :username
                """)
                user_data = db.execute(user_query, {"username": username_norm}).first()
                if not user_data:
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

                from security import verify_password, create_access_token
                if not verify_password(form_data.password, user_data.hashed_password):
                    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

                token = create_access_token(user_id=user_data.id)
                return {"access_token": token, "token_type": "bearer"}
            finally:
                try:
                    db.close()
                except Exception:
                    pass
        print("Auth fallback endpoints registered.")
    except Exception as e:
        print(f"Failed to register auth fallbacks: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)