import os
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
import json
import math
import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
import threading

from db import get_db, SessionLocal
from models import User, Dataset, TrainedModel
from schemas import UploadResponse, DatasetOut
from config import settings
from ml_utils import load_dataset, MLPipeline
from routers.auth import get_current_user

router = APIRouter()

def validate_data_integrity(data_before, data_after, operation):
    """Ensure data isn't corrupted during processing"""
    if operation == "upload":
        # After upload, data should be identical to original
        if not data_before.equals(data_after):
            print(f" DATA CORRUPTION DETECTED during upload!")
            print(f" Original shape: {data_before.shape}, After: {data_after.shape}")
            print(f" Original columns: {data_before.columns.tolist()}")
            print(f" After columns: {data_after.columns.tolist()}")
            return False
    
    elif operation == "analysis":
        # Analysis should preserve original data structure
        original_columns = set(data_before.columns)
        analysis_columns = set(data_after.keys()) if isinstance(data_after, dict) else set(data_after.columns)
        if not original_columns.intersection(analysis_columns):
            print(f" COLUMN MISMATCH in analysis!")
            print(f" Original columns: {original_columns}")
            print(f" Analysis columns: {analysis_columns}")
            return False
    
    return True

def ensure_storage():
    """Ensure storage directories exist"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.MODELS_DIR, exist_ok=True)


def _auto_train_new_dataset_async(dataset_id: int, user_id: int):
    """Background job: analyze and train models for the new dataset using ONLY the uploaded data."""
    db = SessionLocal()
    try:
        # Fetch dataset info
        dataset_row = db.execute(text(
            """
            SELECT id, filename, filepath AS file_path, filesize AS file_size, uploaded_at AS created_at
            FROM datasets
            WHERE id = :dataset_id AND user_id = :user_id
            """
        ), {"dataset_id": dataset_id, "user_id": user_id}).first()
        if not dataset_row:
            return

        # Load dataset
        df = load_dataset(dataset_row.file_path)

        # Analyze to determine target and task type
        pipeline = MLPipeline()
        analysis = pipeline.analyze_dataset(df)
        potential_targets = analysis.get("data_quality", {}).get("potential_targets", [])
        if not potential_targets:
            return

        # Choose best target (prefer binary when available)
        preferred = sorted(potential_targets, key=lambda t: (t.get("type") == "binary_classification", "class" in str(t.get("type",""))), reverse=True)
        target_info = preferred[0]
        target_col = target_info.get("column")
        task_type = target_info.get("type")

        # Select inputs
        exclude = {target_col, 'Id', 'id', 'ID', 'index', 'Index'}
        input_cols = [c for c in df.columns if c not in exclude]
        if not input_cols:
            return

        # Preprocess once to determine concrete task
        X, y = pipeline.preprocess_data(df[[*input_cols, target_col]], target_col, task_type or 'auto')

        # Choose algorithms similarly to /train defaults
        if pipeline.task_type in ("classification", "binary_classification"):
            algorithms = ["Logistic Regression", "Random Forest"]
        else:
            algorithms = ["Linear Regression", "Multiple Linear Regression", "Random Forest"]

        # Train each algorithm and persist
        for algo in algorithms:
            try:
                algo_pipeline = MLPipeline()
                algo_pipeline.target_column = target_col
                algo_pipeline.selected_features = input_cols[:]
                algo_pipeline.task_type = pipeline.task_type
                X_a, y_a = algo_pipeline.preprocess_data(df[[*input_cols, target_col]], target_col, algo_pipeline.task_type)
                result = algo_pipeline.train_model(X_a, y_a, algo)

                # Save model to disk
                model_label = result.get("algorithm", algo)
                model_path = os.path.join(settings.MODELS_DIR, f"model_{user_id}_{dataset_id}_{model_label}.pkl")
                os.makedirs(settings.MODELS_DIR, exist_ok=True)
                algo_pipeline.save_model(model_path)

                # Persist minimal metrics with training metadata
                used_features = getattr(algo_pipeline, 'original_feature_columns', input_cols)
                metrics_with_meta = {
                    "metrics": result.get("metrics", {}),
                    "features": used_features,
                    "target": target_col,
                    "cross_validation_score": result.get("cross_validation_score", {}),
                    "samples": result.get("samples", {"first": [], "middle": [], "last": []})
                }

                try:
                    db.execute(text(
                        """
                        INSERT INTO trained_models
                        (user_id, dataset_id, task_type, algorithm, metrics, model_path, created_at, updated_at, status, model_name)
                        VALUES (:user_id, :dataset_id, :task_type, :algorithm, :metrics, :model_path, datetime('now'), datetime('now'), 'completed', :model_name)
                        """
                    ), {
                        "user_id": user_id,
                        "dataset_id": dataset_id,
                        "task_type": algo_pipeline.task_type,
                        "algorithm": model_label,
                        "metrics": json.dumps(metrics_with_meta),
                        "model_path": model_path,
                        "model_name": f"{model_label}_{dataset_id}"
                    })
                except Exception:
                    db.execute(text(
                        """
                        INSERT INTO trained_models
                        (user_id, dataset_id, task_type, algorithm, metrics, model_path, created_at)
                        VALUES (:user_id, :dataset_id, :task_type, :algorithm, :metrics, :model_path, datetime('now'))
                        """
                    ), {
                        "user_id": user_id,
                        "dataset_id": dataset_id,
                        "task_type": algo_pipeline.task_type,
                        "algorithm": model_label,
                        "metrics": json.dumps(metrics_with_meta),
                        "model_path": model_path
                    })
                db.commit()
            except Exception as e:
                try:
                    db.rollback()
                except Exception:
                    pass
                print(f" Auto-train error for {algo}: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass


@router.get("/datasets/test-cors")
async def test_cors():
    """Test CORS endpoint"""
    return {"message": "CORS is working!", "status": "success"}


@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Upload a single file and return dataset info"""
    ensure_storage()
    
    # Validate file type
    allowed_types = [
        "text/csv", 
        "application/vnd.ms-excel", 
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/json"
    ]
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_MB:
        raise HTTPException(status_code=400, detail=f"File too large. Max size: {settings.MAX_UPLOAD_MB}MB")

    # Create unique filename to avoid conflicts
    import uuid
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    unique_filename = f"{file_id}{file_extension}"
    save_path = os.path.join(settings.UPLOAD_DIR, unique_filename)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save file (ensure full write before proceeding)
    with open(save_path, "wb") as f:
        f.write(contents)
        f.flush()
        os.fsync(f.fileno())

    try:
        # Test if file can be loaded - PRESERVE RAW DATA
        df = load_dataset(save_path)
        print(f" Successfully loaded RAW dataset: {df.shape}")
        print(f" RAW DATA SAMPLE: {df.head(3).to_dict('records')}")
        print(f" RAW DATA COLUMNS: {df.columns.tolist()}")
        print(f" RAW DATA TYPES: {df.dtypes.to_dict()}")
    except Exception as e:
        # Clean up file if it can't be loaded
        if os.path.exists(save_path):
            os.remove(save_path)
        raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")

    # Insert dataset into database using actual schema column names
    # Database columns: user_id, filename, filepath, filesize, uploaded_at
    insert_query = text("""
        INSERT INTO datasets (user_id, filename, filepath, filesize, uploaded_at)
        VALUES (:user_id, :filename, :filepath, :filesize, datetime('now'))
    """)
    try:
        result = db.execute(insert_query, {
            "user_id": current_user.id,
            "filename": file.filename,
            "filepath": save_path,
            "filesize": len(contents),
        })
        dataset_id = result.lastrowid
        db.commit()
    except Exception as e:
        # If DB insert fails, remove saved file to avoid orphaned files
        try:
            if os.path.exists(save_path):
                os.remove(save_path)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"Failed to store dataset metadata: {str(e)}")
    
    # Fetch the created dataset, aliasing DB columns to API schema fields
    dataset_query = text("""
        SELECT 
            id,
            filename,
            filepath AS file_path,
            filesize AS file_size,
            uploaded_at AS created_at,
            user_id AS uploaded_by
        FROM datasets 
        WHERE id = :dataset_id
    """)
    dataset_data = db.execute(dataset_query, {"dataset_id": dataset_id}).first()
    
    if not dataset_data:
        raise HTTPException(status_code=500, detail="Failed to create dataset")
    
    # Create DatasetOut object
    dataset = DatasetOut(
        id=dataset_data.id,
        filename=dataset_data.filename,
        file_size=dataset_data.file_size,
        created_at=dataset_data.created_at,
        file_path=dataset_data.file_path,
        uploaded_by=dataset_data.uploaded_by
    )
    
    print(f" Dataset uploaded successfully: ID {dataset_id}, File: {file.filename}")
    # Kick off background auto-training so results are produced from THIS dataset only
    try:
        threading.Thread(target=_auto_train_new_dataset_async, args=(dataset_id, current_user.id), daemon=True).start()
    except Exception as e:
        print(f" Failed to start auto-train thread: {e}")
    return UploadResponse(dataset=dataset)


@router.get("/datasets/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Get RAW dataset preview without any preprocessing"""
    try:
        # Get dataset info
        dataset_query = text("""
            SELECT 
                id,
                filename,
                filepath AS file_path,
                filesize AS file_size,
                uploaded_at AS created_at
            FROM datasets 
            WHERE id = :dataset_id AND user_id = :user_id
        """)
        dataset_data = db.execute(dataset_query, {"dataset_id": dataset_id, "user_id": current_user.id}).first()
        
        if not dataset_data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load RAW dataset without any preprocessing
        df = load_dataset(dataset_data.file_path)
        
        # Clean data for JSON serialization
        df_clean = df.copy()
        
        # Replace NaN values with None for JSON serialization
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        
        # Convert any remaining problematic values
        sample_data = []
        for _, row in df_clean.head(10).iterrows():
            clean_row = {}
            for col, value in row.items():
                if pd.isna(value) or value is None:
                    clean_row[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        clean_row[col] = None
                    else:
                        clean_row[col] = float(value)
                else:
                    clean_row[col] = str(value)
            sample_data.append(clean_row)
        
        # Return RAW data preview with proper JSON serialization
        return {
            "dataset_id": dataset_id,
            "filename": dataset_data.filename,
            "shape": list(df.shape),  # Convert numpy array to list
            "columns": df.columns.tolist(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},  # Convert numpy dtypes to strings
            "sample_data": sample_data,  # Cleaned RAW data
            "preprocessing_applied": False,
            "warning": "This is RAW data - no transformations applied"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset preview: {str(e)}")


@router.post("/datasets/{dataset_id}/recover")
async def recover_corrupted_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Recover original data from corrupted datasets"""
    try:
        # Get dataset info
        dataset_query = text("""
            SELECT 
                id,
                filename,
                filepath AS file_path,
                filesize AS file_size,
                uploaded_at AS created_at
            FROM datasets 
            WHERE id = :dataset_id AND user_id = :user_id
        """)
        dataset_data = db.execute(dataset_query, {"dataset_id": dataset_id, "user_id": current_user.id}).first()
        
        if not dataset_data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Check if original file exists
        if not os.path.exists(dataset_data.file_path):
            raise HTTPException(status_code=404, detail="Original file not found")
        
        # Load RAW data from original file
        df_raw = load_dataset(dataset_data.file_path)
        
        # Validate data integrity
        print(f" Recovering dataset {dataset_id}")
        print(f" RAW DATA: Shape={df_raw.shape}, Columns={df_raw.columns.tolist()}")
        print(f" RAW DATA SAMPLE: {df_raw.head(3).to_dict('records')}")
        
        # Clean data for JSON serialization
        df_clean = df_raw.copy()
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        
        # Convert any remaining problematic values
        sample_data = []
        for _, row in df_clean.head(10).iterrows():
            clean_row = {}
            for col, value in row.items():
                if pd.isna(value) or value is None:
                    clean_row[col] = None
                elif isinstance(value, (np.integer, np.floating)):
                    if np.isnan(value) or np.isinf(value):
                        clean_row[col] = None
                    else:
                        clean_row[col] = float(value)
                else:
                    clean_row[col] = str(value)
            sample_data.append(clean_row)
        
        return {
            "status": "recovered",
            "message": "Original data restored successfully",
            "dataset_id": dataset_id,
            "filename": dataset_data.filename,
            "shape": list(df_raw.shape),  # Convert numpy array to list
            "columns": df_raw.columns.tolist(),
            "sample_data": sample_data,  # Cleaned data
            "preprocessing_applied": False,
            "warning": "This is RAW data - no transformations applied"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Recovery failed: {str(e)}"
        }


@router.get("/datasets", response_model=List[DatasetOut])
def list_datasets(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    # Use actual DB columns and alias to API schema
    datasets_query = text("""
        SELECT 
            id,
            filename,
            filepath AS file_path,
            filesize AS file_size,
            uploaded_at AS created_at,
            user_id AS uploaded_by
        FROM datasets 
        WHERE user_id = :user_id 
        ORDER BY uploaded_at DESC
    """)
    result = db.execute(datasets_query, {"user_id": current_user.id}).fetchall()
    
    datasets = []
    for row in result:
        dataset = DatasetOut(
            id=row.id,
            filename=row.filename,
            file_size=row.file_size,
            created_at=row.created_at,
            file_path=row.file_path,
            uploaded_by=row.uploaded_by
        )
        datasets.append(dataset)
    
    return datasets


@router.get("/datasets/{dataset_id}")
def get_dataset(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Get a specific dataset by ID with column information"""
    dataset_query = text("""
        SELECT 
            id,
            filename,
            filepath AS file_path,
            filesize AS file_size,
            uploaded_at AS created_at,
            user_id AS uploaded_by
        FROM datasets 
        WHERE id = :dataset_id AND user_id = :user_id
    """)
    dataset_data = db.execute(dataset_query, {
        "dataset_id": dataset_id, 
        "user_id": current_user.id
    }).first()
    
    if not dataset_data:
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    def safe_json_serializer(obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return None
        return obj

    try:
        # Load dataset to get column information
        df = load_dataset(dataset_data.file_path)

        # Clean for JSON safety
        df_clean = df.replace([np.nan, np.inf, -np.inf], None)

        # Stats
        stats = {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_types": {k: str(v) for k, v in df_clean.dtypes.astype(str).to_dict().items()},
            "missing_values": {k: int(v) for k, v in df.isnull().sum().to_dict().items()},
            "basic_stats": {}
        }
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        for col in numeric_cols:
            col_series = pd.Series(df_clean[col]).dropna()
            if len(col_series) > 0:
                try:
                    stats["basic_stats"][col] = {
                        "min": float(col_series.min()) if not col_series.empty else None,
                        "max": float(col_series.max()) if not col_series.empty else None,
                        "mean": float(col_series.mean()) if not col_series.empty else None,
                        "std": float(col_series.std()) if not col_series.empty else None,
                    }
                except Exception:
                    stats["basic_stats"][col] = {"min": None, "max": None, "mean": None, "std": None}

        safe_stats = json.loads(json.dumps(stats, default=safe_json_serializer, allow_nan=False))
        preview_records = json.loads(json.dumps(df_clean.head(10).to_dict(orient='records'), default=safe_json_serializer, allow_nan=False))

        return {
            "id": dataset_data.id,
            "filename": dataset_data.filename,
            "file_size": dataset_data.file_size,
            "created_at": dataset_data.created_at,
            "file_path": dataset_data.file_path,
            "uploaded_by": dataset_data.uploaded_by,
            "columns": list(df.columns),
            "data_preview": preview_records,
            "shape": df.shape,
            "statistics": safe_stats
        }
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Return basic info if dataset can't be loaded
        return {
            "id": dataset_data.id,
            "filename": dataset_data.filename,
            "file_size": dataset_data.file_size,
            "created_at": dataset_data.created_at,
            "file_path": dataset_data.file_path,
            "uploaded_by": dataset_data.uploaded_by,
            "columns": [],
            "data_preview": [],
            "shape": [0, 0],
            "error": str(e)
        }


