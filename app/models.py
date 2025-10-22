from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from sqlalchemy.sql import func
from db import Base

class User(Base):
    __tablename__ = "users"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Dataset(Base):
    __tablename__ = "datasets"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)  # ✅ Change uploaded_by to user_id
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500))
    file_size = Column(Integer)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
class TrainedModel(Base):
    __tablename__ = "trained_models"
    __table_args__ = {'extend_existing': True}

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    dataset_id = Column(Integer, nullable=False)
    task_type = Column(String(100), nullable=False)
    algorithm = Column(String(100), nullable=False)
    metrics = Column(Text)  # JSON string
    model_path = Column(String(500))
    created_at = Column(DateTime(timezone=True), server_default=func.now())