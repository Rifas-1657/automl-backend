from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    user_id: int


class UserBase(BaseModel):
    email: EmailStr
    username: str


class UserCreate(UserBase):
    password: str = Field(min_length=8)


class UserUpdate(BaseModel):
    username: Optional[str] = None


class UserOut(UserBase):
    id: int
    full_name: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class DatasetOut(BaseModel):
    id: int
    filename: str
    file_size: int
    created_at: datetime
    file_path: Optional[str] = None
    uploaded_by: Optional[int] = None

    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    dataset: DatasetOut


class AnalyzeRequest(BaseModel):
    dataset_id: int
    target: Optional[str] = None


class AnalyzeResponse(BaseModel):
    task_type: str
    suggestions: List[str]
    target: Optional[str] = None


class TrainRequest(BaseModel):
    dataset_id: int
    task_type: str
    algorithm: str
    target: Optional[str] = None


class TrainResponse(BaseModel):
    model_id: int
    metrics: Dict[str, float]
    sample_predictions: List[Any]
    algorithm_used: str
    cross_validation_score: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None

class PredictionRequest(BaseModel):
    dataset_id: int
    input_features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float]
    input_features: Dict[str, Any]
    algorithm_used: str

# VisualizationResponse removed with visualization feature

class AlgorithmRecommendation(BaseModel):
    name: str
    type: str
    description: str
    pros: List[str]
    cons: List[str]
    best_for: List[str]

class AlgorithmRecommendationsResponse(BaseModel):
    target_column: str
    task_type: str
    dataset_characteristics: Dict[str, Any]
    algorithm_recommendations: List[AlgorithmRecommendation]


class HistoryItem(BaseModel):
    model_id: int
    algorithm: str
    task_type: str
    created_at: datetime
    # Metrics blobs can be nested and contain non-float values; allow any
    metrics: Optional[Dict[str, Any]]


class HistoryResponse(BaseModel):
    datasets: List[DatasetOut]
    models: List[HistoryItem]


