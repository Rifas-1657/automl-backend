import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Default to persistent storage DB path in production-like envs
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./storage/app.db")

# Ensure SQLite directory exists when using local file DB
if DATABASE_URL.startswith("sqlite"):
    try:
        # Extract path after 'sqlite:///'
        db_path = DATABASE_URL.replace("sqlite:///", "")
        dir_path = os.path.dirname(db_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
    except Exception:
        pass

connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


