# Replace your config.py with this:
import os
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    PROJECT_NAME: str = "AutoML Web App API"
    
    # Database Configuration - Use MySQL by default
    DATABASE_URL: str = os.getenv("DATABASE_URL", "mysql+pymysql://automl_web_user:AutomlWeb123@localhost:3306/automl_web_app")
    
    # MySQL Configuration
    MYSQL_USER: str = os.getenv("MYSQL_USER", "automl_web_user")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "AutomlWeb123")
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: str = os.getenv("MYSQL_PORT", "3306")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "automl_web_app")
    
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-this-in-production")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = int(os.getenv("JWT_EXPIRE_MINUTES", "10080"))  # 7 days
    ACCESS_TOKEN_EXPIRE: timedelta = timedelta(minutes=JWT_EXPIRE_MINUTES)
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./storage/uploads")
    MODELS_DIR: str = os.getenv("MODELS_DIR", "./storage/models")
    MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", "100"))

settings = Settings()