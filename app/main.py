import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Ensure app package on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from db import SessionLocal, engine
from create_tables import create_tables
from sqlalchemy import text, inspect

app = FastAPI(title="AutoML Web App API", version="0.1.0")

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
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3) CORS preflight handler
@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return {"message": "OK"}

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
except Exception as e:
    print(f"Auth router failed: {e}")

try:
    from routers.datasets import router as datasets_router
    app.include_router(datasets_router, prefix="/api", tags=["datasets"])
    print("Datasets router loaded successfully!")
except Exception as e:
    print(f"Datasets router failed: {e}")

try:
    from routers.ml import router as ml_router
    app.include_router(ml_router, prefix="/api", tags=["ml"])
    print("ML router loaded successfully!")
except Exception as e:
    print(f"ML router failed: {e}")

try:
    from routers.account import router as account_router
    app.include_router(account_router, prefix="/api", tags=["account"])
    print("Account router loaded successfully!")
except Exception as e:
    print(f"Account router failed: {e}")

try:
    from routers.history import router as history_router
    app.include_router(history_router, prefix="/api", tags=["history"])
    print("History router loaded successfully!")
except Exception as e:
    print(f"History router failed: {e}")

try:
    from routers.visualization import router as visualization_router
    app.include_router(visualization_router, prefix="/api", tags=["visualization"])
    print("Visualization router loaded successfully!")
except Exception as e:
    print(f"Visualization router failed: {e}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)