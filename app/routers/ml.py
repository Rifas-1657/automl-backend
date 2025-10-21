import os
import json
from typing import List, Dict, Any, Optional, Union
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel

from db import get_db
from models import User, Dataset, TrainedModel
from ml_utils import MLPipeline, load_dataset
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
from routers.auth import get_current_user
from config import settings

def clean_for_json(obj):
    """Clean data for JSON serialization by handling NaN, inf, and -inf values"""
    if isinstance(obj, dict):
        return {key: clean_for_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if pd.isna(obj) or np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        else:
            return obj
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if pd.isna(obj) or np.isnan(obj):
            return None
        elif np.isinf(obj):
            return None
        else:
            return float(obj)
    else:
        return obj

router = APIRouter()

# Define missing schemas
class AnalyzeRequest(BaseModel):
    dataset_id: int
    target: Optional[str] = None
    task_type: Optional[str] = "auto"

class AnalyzeResponse(BaseModel):
    task_type: str
    suggestions: List[str]
    target: str
    analysis_summary: Dict[str, Any]

class TrainRequest(BaseModel):
    dataset_id: int
    algorithm: str
    target: Optional[str] = None
    task_type: Optional[str] = None

class TrainResponse(BaseModel):
    model_id: int
    metrics: Dict[str, Any]
    sample_predictions: List[Dict[str, Any]]
    algorithm_used: str
    cross_validation_score: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]

class PredictionRequest(BaseModel):
    dataset_id: int
    input_features: Dict[str, Any]

class PredictionResponse(BaseModel):
    prediction: Union[float, str]
    confidence: Optional[float]
    input_features: Dict[str, Any]
    algorithm_used: str

class VisualizationResponse(BaseModel):
    plot_files: List[str]
    plot_urls: List[str]

# Store ML pipelines in memory (in production, use Redis or database)
ml_pipelines = {}

# --- Data preprocessing helpers ------------------------------------------------
def _encode_categorical_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Encode all object dtype columns to numeric labels.

    Returns a tuple of (encoded_df, column_to_encoder).
    """
    if df is None or df.empty:
        return df, {}
    encoded_df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    for col in encoded_df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
        encoders[col] = le
    return encoded_df, encoders

def ensure_json_serializable(data):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if data is None:
        return data
    if isinstance(data, (str, int, float, bool)):
        return data
    if isinstance(data, (np.integer, np.floating)):
        try:
            return data.item()
        except Exception:
            return float(data) if hasattr(data, "__float__") else int(data)
    if isinstance(data, np.ndarray):
        return data.tolist()
    if isinstance(data, dict):
        return {k: ensure_json_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [ensure_json_serializable(v) for v in data]
    if hasattr(data, "__dict__"):
        return ensure_json_serializable(vars(data))
    # Fallback to string
    return str(data)

def get_simple_algorithm_recommendations(task_type: str, n_samples: int, n_features: int) -> List[Dict[str, Any]]:
    """Get algorithm recommendations without requiring preprocessing"""
    recommendations = []
    
    # Core algorithms for all datasets
    if task_type == "regression":
        recommendations.extend([
            {
                "name": "Linear Regression",
                "type": "linear",
                "description": "Fast baseline model for continuous targets",
                "pros": ["Very fast", "Highly interpretable", "No hyperparameters"],
                "cons": ["Assumes linear relationships", "Sensitive to outliers"],
                "best_for": ["Linear relationships", "Small datasets", "Baseline models"]
            },
            {
                "name": "Multiple Regression",
                "type": "linear", 
                "description": "Linear regression with multiple features",
                "pros": ["Handles multiple features", "Interpretable coefficients", "Fast training"],
                "cons": ["Assumes linear relationships", "Sensitive to multicollinearity"],
                "best_for": ["Continuous targets", "Multiple features", "Linear relationships"]
            },
            {
                "name": "Random Forest",
                "type": "ensemble",
                "description": "Robust ensemble method for regression",
                "pros": ["Robust to outliers", "Handles mixed data types", "Feature importance"],
                "cons": ["Can be slow on large datasets", "Less interpretable"],
                "best_for": ["Most datasets", "Mixed data types", "Robust predictions"]
            }
        ])
    else:  # classification
        recommendations.extend([
            {
                "name": "Logistic Regression",
                "type": "linear",
                "description": "Fast baseline model for classification",
                "pros": ["Very fast", "Highly interpretable", "Probability outputs"],
                "cons": ["Assumes linear decision boundary", "Sensitive to outliers"],
                "best_for": ["Linear relationships", "Small datasets", "Baseline models"]
            },
            {
                "name": "Random Forest",
                "type": "ensemble",
                "description": "Robust ensemble method for classification",
                "pros": ["Robust to outliers", "Handles mixed data types", "Feature importance"],
                "cons": ["Can be slow on large datasets", "Less interpretable"],
                "best_for": ["Most datasets", "Mixed data types", "Robust predictions"]
            },
            {
                "name": "Support Vector Machine",
                "type": "kernel_method",
                "description": "Powerful for complex non-linear patterns",
                "pros": ["Effective in high dimensions", "Memory efficient", "Good for complex patterns"],
                "cons": ["Slow on large datasets", "Sensitive to feature scaling"],
                "best_for": ["Small to medium datasets", "High-dimensional data", "Complex patterns"]
            }
        ])
    
    # Add advanced algorithms for larger datasets
    if n_samples > 1000:
        if task_type == "regression":
            recommendations.append({
                "name": "XGBoost",
                "type": "gradient_boosting",
                "description": "High-performance gradient boosting for regression",
                "pros": ["Very high accuracy", "Built-in regularization", "Fast training"],
                "cons": ["Can overfit", "Many hyperparameters", "Less interpretable"],
                "best_for": ["Large datasets", "High accuracy requirements", "Tabular data"]
            })
        else:
            recommendations.append({
                "name": "XGBoost",
                "type": "gradient_boosting", 
                "description": "High-performance gradient boosting for classification",
                "pros": ["Very high accuracy", "Built-in regularization", "Fast training"],
                "cons": ["Can overfit", "Many hyperparameters", "Less interpretable"],
                "best_for": ["Large datasets", "High accuracy requirements", "Tabular data"]
            })
    
    return recommendations

@router.post("/analyze/{dataset_id}", response_model=AnalyzeResponse)
async def analyze_dataset(
    dataset_id: int,
    target: Optional[str] = None,  # allow optional target override via query param
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Analyze dataset and provide recommendations - PRESERVE RAW DATA"""
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
        
        print(f"Analyze request: dataset {dataset_id} found for user {current_user.id}; path={dataset_data.file_path}")

        # Load dataset - PRESERVE RAW DATA WITHOUT PREPROCESSING
        df = load_dataset(dataset_data.file_path)
        print(f" RAW DATA LOADED: Shape={df.shape}, Columns={df.columns.tolist()}")
        
        # Clean data for logging (avoid JSON serialization issues)
        df_clean = df.copy()
        df_clean = df_clean.where(pd.notnull(df_clean), None)
        clean_sample = []
        for _, row in df_clean.head(3).iterrows():
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
            clean_sample.append(clean_row)
        print(f" RAW DATA SAMPLE: {clean_sample}")
        
        # CRITICAL: Do NOT encode categoricals during analysis - preserve original data
        # df, encoders = _encode_categorical_columns(df)  # REMOVED - This was corrupting data!
        
        # Initialize ML pipeline
        pipeline = MLPipeline()
        
        # Analyze dataset
        analysis = pipeline.analyze_dataset(df)
        
        # Store pipeline for later use
        ml_pipelines[f"{current_user.id}_{dataset_id}"] = pipeline
        
        # Determine task type and suggestions
        potential_targets = analysis["data_quality"]["potential_targets"]
        
        if not potential_targets:
            raise HTTPException(status_code=400, detail="No suitable target columns found")
        
        # If a target is provided and exists, prefer it
        if target and target in df.columns:
            # Infer task type from the provided target (robust and numeric-friendly)
            try:
                nunique = df[target].nunique()
                dtype = df[target].dtype
                dtype_name = getattr(dtype, 'name', str(dtype))
                if str(dtype_name) in ['int64', 'int32', 'float64', 'float32']:
                    # Numeric targets: regression unless strictly binary
                    task_type = "binary_classification" if nunique == 2 else "regression"
                elif dtype == 'object' or dtype_name == 'category':
                    task_type = "classification" if nunique > 2 else "binary_classification"
                else:
                    # Fallback heuristic
                    task_type = "classification" if nunique <= 10 else "regression"
            except Exception:
                task_type = "classification"
            target_column = target
        else:
            # Prioritize likely classification/binary targets (e.g., 'purchased')
            preferred_names = {
                "purchased", "purchase", "bought", "buy", "churn", "default",
                "label", "target", "class", "outcome", "clicked", "converted"
            }
            
            def target_score(item: Dict[str, Any]) -> int:
                name = str(item.get("column", "")).strip().lower()
                ttype = str(item.get("type", ""))
                score = 0
                # Strongly prefer binary classification
                if ttype == "binary_classification":
                    score += 100
                elif "class" in ttype:
                    score += 50
                else:  # regression
                    score += 10
                # Name-based boosts
                if name in preferred_names:
                    score += 40
                if name.startswith("is_") or name.startswith("has_"):
                    score += 30
                # Small cardinality boost
                try:
                    nunique = int(item.get("nunique", 0))
                    if nunique == 2:
                        score += 30
                    elif nunique <= 10:
                        score += 10
                except Exception:
                    pass
                return score
            
            # Sort candidates by score descending
            ranked = sorted(potential_targets, key=target_score, reverse=True)
            target_info = ranked[0]
            target_column = target_info["column"]
            task_type = target_info["type"]
        
        # CRITICAL: Do NOT preprocess data during analysis - preserve raw data
        # X, y = pipeline.preprocess_data(df, target_column, task_type)  # REMOVED - This corrupts data!
        
        # Get algorithm recommendations using RAW data - create simple recommendations without preprocessing
        X_raw = df.drop(columns=[target_column])
        y_raw = df[target_column]
        
        # Create simple algorithm recommendations without requiring numeric data
        algorithm_recommendations = get_simple_algorithm_recommendations(task_type, len(X_raw), len(X_raw.columns))
        
        return AnalyzeResponse(
            task_type=task_type,
            suggestions=[algo["name"] for algo in algorithm_recommendations],
            target=target_column,
            analysis_summary={
                "dataset_shape": analysis["basic_info"]["shape"],
                "missing_values": analysis["missing_values"],
                "data_quality_score": analysis["data_quality"]["completeness"],
                "recommendations": analysis["recommendations"]
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/train")
async def train_models(
    request: dict,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Train multiple models and compare results"""
    try:
        dataset_id = request.get("dataset_id")
        input_columns = request.get("input_columns", [])
        output_column = request.get("output_column")
        algorithms = request.get("algorithms", [])
        # If none provided, choose sensible defaults based on task type
        # Classification should not default to linear regression models
        if not algorithms or len(algorithms) == 0:
            # Choose sensible defaults aligning with Recommendations
            # Classification: 2 (Logistic, Random Forest)
            # Regression: 4 (Linear, Multiple Linear, Logistic, Random Forest)
            prelim_task = request.get("task_type")
            if prelim_task in ("classification", "binary_classification"):
                algorithms = ["Logistic Regression", "Random Forest"]
            else:
                algorithms = [
                    "Linear Regression",
                    "Multiple Linear Regression",
                    "Logistic Regression",
                    "Random Forest"
                ]
        # Enforce only allowed algorithms
        allowed_algorithms = {"Linear Regression", "Multiple Linear Regression", "Logistic Regression", "Random Forest"}
        invalid = [a for a in algorithms if a not in allowed_algorithms]
        if invalid:
            raise HTTPException(status_code=400, detail=f"Unsupported algorithms requested: {invalid}. Allowed: {sorted(list(allowed_algorithms))}")
        request_task_type = request.get("task_type")
        
        if not dataset_id or not output_column or not input_columns:
            raise HTTPException(status_code=400, detail="Missing required parameters")
        
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
        
        # Load dataset
        df = load_dataset(dataset_data.file_path)
        print(f" Loaded dataset: {df.shape}")
        print(f" Available columns: {list(df.columns)}")

        # Validate requested columns exist
        missing_columns = [c for c in (input_columns + [output_column]) if c not in df.columns]
        if missing_columns:
            raise HTTPException(status_code=400, detail=f"Columns not found in dataset: {missing_columns}")

        # Filter to selected columns
        selected_columns = input_columns + [output_column]
        df_filtered = df[selected_columns]
        print(f" Training request columns: inputs={input_columns}, target={output_column}")
        print(f" Selected columns (in order): {selected_columns}")

        # Replace inf with NaN and drop rows with any NaN across selected columns
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan)
        before_drop = len(df_filtered)
        df_filtered = df_filtered.dropna(how='any')
        print(f" Dropped {before_drop - len(df_filtered)} rows with NaN/Inf among selected columns")

        if df_filtered.empty:
            raise HTTPException(status_code=400, detail="No valid data remaining after cleaning selected columns")

        # IMPORTANT: Do NOT encode the target column here. Feature encoding is handled inside MLPipeline.
        # We intentionally avoid encoding the entire DataFrame so that string target labels remain intact
        # for proper classification and human-readable predictions.
        
        # Initialize ML pipeline
        pipeline = MLPipeline()
        
        # Determine task type robustly: honor client hint, but override if data clearly binary
        if request_task_type and request_task_type != 'auto':
            task_type = request_task_type
        else:
            y_series = df_filtered[output_column]
            try:
                nunique = y_series.nunique() if hasattr(y_series, 'nunique') else len(set(y_series))
                dtype_name = getattr(y_series.dtype, 'name', str(y_series.dtype))
            except Exception:
                nunique = len(set(list(y_series)))
                dtype_name = ''
            # Numeric targets -> regression unless strictly binary
            if dtype_name in ['int64', 'int32', 'float64', 'float32']:
                task_type = "binary_classification" if nunique == 2 else "regression"
            else:
                # Non-numeric: classification, treat two levels as binary
                task_type = "binary_classification" if nunique == 2 else "classification"

        # SAFETY GUARD: If target is clearly binary (0/1, yes/no, true/false), force binary_classification
        try:
            unique_vals = set([str(v).strip().lower() for v in pd.unique(df_filtered[output_column])])
            binary_markers = {"0", "1", "yes", "no", "true", "false", "y", "n"}
            if len(unique_vals - {"nan", "none", ""}) == 2 and task_type != "regression":
                task_type = "binary_classification"
            elif unique_vals.issubset(binary_markers) and task_type != "regression":
                task_type = "binary_classification"
        except Exception:
            pass
        
        # Preprocess data
        X, y = pipeline.preprocess_data(df_filtered, output_column, task_type)
        print(f" Preprocessed data: X={X.shape}, y={y.shape}")
        print(f" Original input features captured: {pipeline.original_feature_columns}")
        print(f" Post-preprocessing feature columns: {pipeline.feature_columns}")

        if X.shape[0] == 0 or y.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Empty data after preprocessing")
        
        # Train all algorithms
        results = {}
        # Map from requested name -> actual trained label
        requested_to_trained: Dict[str, str] = {}
        for algorithm_name in algorithms:
            try:
                print(f" Training {algorithm_name}...")
                
                # Create new pipeline instance for each algorithm
                algo_pipeline = MLPipeline()
                # Persist the exact training configuration on the pipeline
                algo_pipeline.target_column = output_column
                algo_pipeline.selected_features = input_columns[:]  # Store selected features
                algo_pipeline.task_type = task_type
                
                # Preprocess data for this algorithm
                X_algo, y_algo = algo_pipeline.preprocess_data(df_filtered, output_column, task_type)
                
                # Train model
                training_result = algo_pipeline.train_model(X_algo, y_algo, algorithm_name)
                trained_label = training_result.get("algorithm", algorithm_name)
                requested_to_trained[algorithm_name] = trained_label
                
                # Store results
                result_payload = ensure_json_serializable({
                    "metrics": training_result["metrics"],
                    "cross_validation_score": training_result["cross_validation_score"],
                    "feature_importance": training_result["feature_importance"],
                    "predictions": training_result.get("predictions", []),
                    "actual": training_result.get("actual", []),
                    "samples": training_result.get("samples", {"first": [], "middle": [], "last": []}),
                    "algorithm_label": trained_label
                })
                # Index by requested name for backward-compat
                results[algorithm_name] = result_payload
                # Also index by the actual trained label for correctness in consumers
                # If multiple requested names map to the same trained label, keep the better score
                try:
                    existing = results.get(trained_label)
                    if existing:
                        prev = existing.get("cross_validation_score", {}).get("mean", 0)
                        curr = result_payload.get("cross_validation_score", {}).get("mean", 0)
                        if curr is not None and (prev is None or curr > prev):
                            results[trained_label] = result_payload
                    else:
                        results[trained_label] = result_payload
                except Exception:
                    results[trained_label] = result_payload
                
                # Save model
                model_path = os.path.join(settings.MODELS_DIR, f"model_{current_user.id}_{dataset_id}_{trained_label}.pkl")
                os.makedirs(settings.MODELS_DIR, exist_ok=True)
                algo_pipeline.save_model(model_path)
                print(f" Saved model to {model_path}")
                print(f" Model original_feature_columns (to be serialized): {algo_pipeline.original_feature_columns}")
                print(f" Model feature_columns (preprocessed): {algo_pipeline.feature_columns}")

                # Safety assertion: ensure DB-stored features exactly match requested input columns
                if set(input_columns) != set(algo_pipeline.original_feature_columns or input_columns):
                    print(f" Mismatch between requested features and serialized original_feature_columns. Using requested input_columns for DB metadata.")
                
                # Save to database with robust insert that satisfies NOT NULL constraints
                # Persist training metadata into metrics blob so we can reload feature list for predictions
                combined_metrics = training_result["metrics"] if isinstance(training_result.get("metrics"), dict) else {}
                # Store the actual features used by the trained model (post-preprocessing model feature list)
                # but prefer the original selected input features if available to enforce user intent
                used_features = getattr(algo_pipeline, 'feature_columns', input_columns)
                if getattr(algo_pipeline, 'original_feature_columns', None):
                    # original_feature_columns reflect user-selected inputs; keep them as canonical
                    used_features = algo_pipeline.original_feature_columns
                metrics_with_meta = {
                    "metrics": ensure_json_serializable(combined_metrics),
                    "features": used_features,
                    "target": output_column,
                    "cross_validation_score": ensure_json_serializable(training_result.get("cross_validation_score", {})),
                    "samples": ensure_json_serializable(training_result.get("samples", {"first": [], "middle": [], "last": []}))
                }
                print(f" Storing training metadata in DB: features={metrics_with_meta['features']}, target={metrics_with_meta['target']}")

                params = {
                    "user_id": current_user.id,
                    "dataset_id": dataset_id,
                    "task_type": task_type,
                    # Persist the actual algorithm used (after any remapping)
                    "algorithm": trained_label,
                    "metrics": json.dumps(metrics_with_meta),
                    "model_path": model_path,
                    "model_name": f"{trained_label}_{dataset_id}",
                }
                try:
                    db.execute(text(
                        """
                        INSERT INTO trained_models 
                        (user_id, dataset_id, task_type, algorithm, metrics, model_path, created_at, updated_at, status, model_name)
                        VALUES (:user_id, :dataset_id, :task_type, :algorithm, :metrics, :model_path, datetime('now'), datetime('now'), 'completed', :model_name)
                        """
                    ), params)
                except Exception:
                    # Fallback minimal insert for schemas without updated_at/status/model_name
                    db.execute(text(
                        """
                        INSERT INTO trained_models 
                        (user_id, dataset_id, task_type, algorithm, metrics, model_path, created_at)
                        VALUES (:user_id, :dataset_id, :task_type, :algorithm, :metrics, :model_path, datetime('now'))
                        """
                    ), params)

                # Ensure model record is persisted before predictions
                db.flush()
                
                print(f" {algorithm_name} trained successfully")
                
            except Exception as e:
                print(f" Error training {algorithm_name}: {e}")
                results[algorithm_name] = {
                    "error": str(e),
                    "metrics": {},
                    "cross_validation_score": {"mean": 0, "std": 0},
                    "feature_importance": {}
                }
        
        db.commit()

        # Safely compute best model
        best_model_key = None
        best_model_label = None
        best_mean = None
        for k, v in results.items():
            try:
                m = v.get("cross_validation_score", {}).get("mean", None)
                if m is not None and (best_mean is None or m > best_mean):
                    best_mean = m
                    best_model_key = k
                    # Prefer the algorithm_label field when available
                    best_model_label = v.get("algorithm_label", k)
            except Exception:
                continue

        # Get the model ID for the best model, with safe fallbacks
        model_id = None
        try:
            if best_model_key:
                # Prefer the best model's algorithm label when available
                model_query = text("""
                    SELECT id FROM trained_models
                    WHERE user_id = :user_id AND dataset_id = :dataset_id AND algorithm = :algorithm
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                model_result = db.execute(model_query, {
                    "user_id": current_user.id,
                    "dataset_id": dataset_id,
                    "algorithm": results.get(best_model_key, {}).get("algorithm_label", best_model_key)
                }).first()
                if model_result:
                    model_id = model_result.id

            # Fallback: pick the latest model for this user+dataset if best_model_key missing or not found
            if model_id is None:
                latest_query = text("""
                    SELECT id FROM trained_models
                    WHERE user_id = :user_id AND dataset_id = :dataset_id
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                latest = db.execute(latest_query, {"user_id": current_user.id, "dataset_id": dataset_id}).first()
                if latest:
                    model_id = latest.id
        except Exception as e:
            print(f" Could not get model ID: {e}")

        return ensure_json_serializable({
            "success": True,
            "dataset_id": dataset_id,
            "task_type": task_type,
            "input_columns": input_columns,
            "output_column": output_column,
            "algorithms_tested": algorithms,
            "results": results,
            # Return the human-correct label for the best model
            "best_model": best_model_label or best_model_key,
            "model_id": model_id,
            # Verification helpers
            "trained_feature_count": len(input_columns),
            "trained_features": input_columns
        })
        
    except Exception as e:
        print(f" Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@router.post("/predict/{dataset_id}", response_model=PredictionResponse)
async def make_prediction(
    dataset_id: int,
    request: PredictionRequest,
    algorithm: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Make predictions on new data"""
    try:
        # Get the trained model for this dataset; prefer a requested algorithm when provided
        if algorithm:
            model_query = text("""
                SELECT id, algorithm, model_path, task_type, metrics FROM trained_models 
                WHERE dataset_id = :dataset_id AND user_id = :user_id AND algorithm = :algorithm
                ORDER BY created_at DESC LIMIT 1
            """)
            params = {"dataset_id": dataset_id, "user_id": current_user.id, "algorithm": algorithm}
        else:
            model_query = text("""
                SELECT id, algorithm, model_path, task_type, metrics FROM trained_models 
                WHERE dataset_id = :dataset_id AND user_id = :user_id 
                ORDER BY created_at DESC LIMIT 1
            """)
            params = {"dataset_id": dataset_id, "user_id": current_user.id}
        model_data = db.execute(model_query, params).first()
        
        if not model_data:
            # Return a friendly 400 instructing to train first
            raise HTTPException(status_code=400, detail="Please train a model first for this dataset")
        
        # Get pipeline
        pipeline_key = f"{current_user.id}_{dataset_id}"
        if pipeline_key not in ml_pipelines:
            # Load model from file
            pipeline = MLPipeline()
            try:
                pipeline.load_model(model_data.model_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load trained model. Please retrain. ({str(e)})")
            ml_pipelines[pipeline_key] = pipeline
        else:
            pipeline = ml_pipelines[pipeline_key]
            # Ensure model is loaded; if missing, reload from disk
            if not getattr(pipeline, 'model', None):
                try:
                    pipeline.load_model(model_data.model_path)
                    ml_pipelines[pipeline_key] = pipeline
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to load trained model. Please retrain. ({str(e)})")
        
        # Validate input features against expected feature columns when available
        # Determine expected feature list: prefer DB-stored metadata, then original training features from model, then pipeline features
        expected_features = None
        stored = None
        try:
            stored = json.loads(model_data.metrics) if getattr(model_data, 'metrics', None) else None
            if isinstance(stored, dict):
                if 'features' in stored and isinstance(stored['features'], list):
                    expected_features = stored['features']
                elif 'metrics' in stored and 'features' in stored:
                    expected_features = stored.get('features')
        except Exception:
            expected_features = None
        # If DB record missing features, try to use original training input columns from the serialized model
        if expected_features is None:
            original_cols = getattr(pipeline, 'original_feature_columns', None)
            if isinstance(original_cols, list) and len(original_cols) > 0:
                expected_features = original_cols
        # Final fallback to post-preprocessing feature columns (may be reduced by selection)
        if expected_features is None:
            expected_features = getattr(pipeline, 'feature_columns', None)
        # Prefer stored target name when available
        target_name = (stored.get('target') if isinstance(stored, dict) else None) or getattr(pipeline, 'target_column', None)
        # Ensure target variable is not part of expected features (defensive filter)
        if isinstance(expected_features, list) and target_name in expected_features:
            expected_features = [f for f in expected_features if f != target_name]
        print(f" Prediction expected features (filtered): {expected_features}")
        print(f" Prediction target column: {target_name}")
        print(f" Prediction payload feature keys: {list(request.input_features.keys())}")
        # Ensure target variable is not part of expected features
        target_name = getattr(pipeline, 'target_column', None)
        if isinstance(expected_features, list) and target_name in expected_features:
            expected_features = [f for f in expected_features if f != target_name]
        if not isinstance(request.input_features, dict):
            raise HTTPException(status_code=400, detail={
                "type": "invalid_payload",
                "msg": "input_features must be an object mapping feature names to values"
            })

        provided_keys = set(request.input_features.keys())
        if isinstance(expected_features, list) and len(expected_features) > 0:
            # Defensive: ensure we only validate against non-target inputs
            safe_expected = [f for f in expected_features if f != target_name]
            missing = [f for f in safe_expected if f not in provided_keys]
            # Sanitize ordering and values; ignore unexpected extras in payload
            sanitized = {}
            for f in safe_expected:
                v = request.input_features.get(f)
                # Lenient handling: if missing/empty, pass None and let pipeline impute
                if v is None or (isinstance(v, str) and v.strip() == ""):
                    sanitized[f] = None
                else:
                    sanitized[f] = v
            if missing:
                print(f" Prediction warning: missing features {missing}. Proceeding with imputation.")
        else:
            # No metadata available; use provided as-is but ensure non-empty
            if len(request.input_features) == 0:
                raise HTTPException(status_code=400, detail={
                    "type": "invalid_payload",
                    "msg": "No input features provided"
                })
            sanitized = request.input_features

        # Validate categorical inputs against known encoders to prevent unseen labels
        try:
            encoders = getattr(pipeline.data_pipeline, 'encoders', {}) or {}
            invalid: list[dict[str, Any]] = []
            for col, enc in encoders.items():
                if isinstance(sanitized, dict) and col in sanitized:
                    val = sanitized[col]
                    if val is None:
                        continue
                    try:
                        classes = [str(c) for c in list(getattr(enc, 'classes_', []))]
                    except Exception:
                        classes = []
                    # Only enforce if we have classes
                    if classes:
                        val_s = str(val)
                        if val_s not in classes:
                            invalid.append({
                                "column": col,
                                "value": val_s,
                                "allowed": classes
                            })
            if invalid:
                raise HTTPException(status_code=400, detail={
                    "type": "invalid_categorical_value",
                    "msg": "One or more categorical values are not recognized",
                    "invalid": invalid
                })
        except HTTPException:
            raise
        except Exception:
            # Be lenient on validation errors; pipeline will fallback, but we try to notify above
            pass

        # Make prediction
        try:
            prediction_result = pipeline.predict(sanitized)
            print(f" Prediction result: {prediction_result}")
            print(f" Task type: {pipeline.task_type}")
            print(f" Target encoder: {getattr(pipeline, 'target_encoder', None)}")
        except Exception as e:
            print(f" Prediction failed: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
        
        return PredictionResponse(
            prediction=prediction_result["prediction"],
            confidence=prediction_result["confidence"],
            input_features=sanitized,
            algorithm_used=model_data.algorithm
        )
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/predict/schema/{dataset_id}")
async def get_prediction_schema(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Expose expected features, target, task type, and categorical options for prediction UI."""
    try:
        model_query = text(
            """
            SELECT id, algorithm, task_type, dataset_id, metrics, model_path
            FROM trained_models
            WHERE user_id = :user_id AND dataset_id = :dataset_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        )
        model_data = db.execute(model_query, {"user_id": current_user.id, "dataset_id": dataset_id}).first()
        if not model_data:
            raise HTTPException(status_code=404, detail="No trained model found for this dataset")

        pipeline_key = f"{current_user.id}_{dataset_id}"
        if pipeline_key not in ml_pipelines:
            pipeline = MLPipeline()
            try:
                pipeline.load_model(model_data.model_path)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to load trained model. Please retrain. ({str(e)})")
            ml_pipelines[pipeline_key] = pipeline
        else:
            pipeline = ml_pipelines[pipeline_key]
            if not getattr(pipeline, 'model', None):
                try:
                    pipeline.load_model(model_data.model_path)
                    ml_pipelines[pipeline_key] = pipeline
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Failed to load trained model. Please retrain. ({str(e)})")

        expected_features = None
        try:
            stored = json.loads(model_data.metrics) if getattr(model_data, 'metrics', None) else None
            if isinstance(stored, dict) and isinstance(stored.get('features'), list):
                expected_features = stored['features']
        except Exception:
            expected_features = None
        if expected_features is None:
            expected_features = getattr(pipeline, 'original_feature_columns', None) or getattr(pipeline, 'feature_columns', [])
        target_name = getattr(pipeline, 'target_column', None)
        if isinstance(expected_features, list) and target_name in expected_features:
            expected_features = [f for f in expected_features if f != target_name]

        categorical_options: dict[str, list[str]] = {}
        try:
            encoders = getattr(pipeline.data_pipeline, 'encoders', {}) or {}
            for col, enc in encoders.items():
                try:
                    classes = [str(c) for c in list(getattr(enc, 'classes_', []))]
                    if classes:
                        categorical_options[col] = classes
                except Exception:
                    continue
        except Exception:
            pass

        return {
            "dataset_id": dataset_id,
            "model_id": model_data.id,
            "algorithm": model_data.algorithm,
            "task_type": getattr(pipeline, 'task_type', model_data.task_type),
            "target": target_name,
            "expected_features": expected_features,
            "categorical_options": categorical_options
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction schema: {str(e)}")


@router.get("/algorithms/{dataset_id}")
async def get_algorithm_recommendations(
    dataset_id: int,
    target_column: Optional[str] = None,
    task_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detailed algorithm recommendations for a dataset"""
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
        
        # Load dataset
        df = load_dataset(dataset_data.file_path)
        
        # Get pipeline
        pipeline_key = f"{current_user.id}_{dataset_id}"
        if pipeline_key not in ml_pipelines:
            pipeline = MLPipeline()
            pipeline.analyze_dataset(df)
            ml_pipelines[pipeline_key] = pipeline
        else:
            pipeline = ml_pipelines[pipeline_key]
        
        # Determine target column and task type
        if not target_column:
            potential_targets = pipeline._assess_data_quality(df)["potential_targets"]
            if not potential_targets:
                raise HTTPException(status_code=400, detail="No suitable target columns found")
            target_column = potential_targets[0]["column"]
            task_type = potential_targets[0]["type"]
        
        # Preprocess data
        X, y = pipeline.preprocess_data(df, target_column, task_type)
        
        # Get detailed algorithm recommendations and restrict to allowed set
        recommendations = [r for r in pipeline.get_algorithm_recommendations(X, y)
                           if r.get("name") in {"Linear Regression", "Multiple Linear Regression", "Logistic Regression", "Random Forest"}]
        
        return {
            "target_column": target_column,
            "task_type": task_type,
            "dataset_characteristics": {
                "n_samples": len(X),
                "n_features": len(X.columns),
                "feature_types": {
                    "numeric": len(X.select_dtypes(include=['number']).columns),
                    "categorical": len(X.select_dtypes(include=['object']).columns)
                }
            },
            "algorithm_recommendations": recommendations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get algorithm recommendations: {str(e)}")

@router.post("/automl/{dataset_id}")
async def run_automl(
    dataset_id: int,
    target_column: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Run complete AutoML pipeline - train all algorithms and compare results"""
    try:
        # Get dataset info
        dataset_query = text("""
            SELECT id, filename, file_path, file_size, created_at FROM datasets 
            WHERE id = :dataset_id AND uploaded_by = :user_id
        """)
        dataset_data = db.execute(dataset_query, {"dataset_id": dataset_id, "user_id": current_user.id}).first()
        
        if not dataset_data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset
        df = load_dataset(dataset_data.file_path)
        
        # Initialize pipeline
        pipeline = MLPipeline()
        
        # Analyze dataset to get target column if not provided
        if not target_column:
            analysis = pipeline.analyze_dataset(df)
            potential_targets = analysis["data_quality"]["potential_targets"]
            if not potential_targets:
                raise HTTPException(status_code=400, detail="No suitable target columns found")
            target_column = potential_targets[0]["column"]
        
        # Preprocess data
        X, y = pipeline.preprocess_data(df, target_column, "auto")
        
        # Get algorithm recommendations
        algorithms = pipeline.get_algorithm_recommendations(X, y)
        algorithm_names = [algo["name"] for algo in algorithms]
        
        # Train all algorithms and compare results
        results = {}
        best_model = None
        best_score = -float('inf')
        
        for algorithm_name in algorithm_names:
            try:
                print(f"Training {algorithm_name}...")
                
                # Create a new pipeline instance for each algorithm
                algo_pipeline = MLPipeline()
                algo_pipeline.target_column = target_column
                algo_pipeline.task_type = pipeline.task_type
                algo_pipeline.feature_columns = pipeline.feature_columns
                
                # Preprocess data for this algorithm
                X_algo, y_algo = algo_pipeline.preprocess_data(df, target_column, algo_pipeline.task_type)
                
                # Train model
                training_result = algo_pipeline.train_model(X_algo, y_algo, algorithm_name)
                
                # Store results
                results[algorithm_name] = {
                    "metrics": training_result["metrics"],
                    "cross_validation_score": training_result["cross_validation_score"],
                    "feature_importance": training_result["feature_importance"],
                    "algorithm_type": next(algo["type"] for algo in algorithms if algo["name"] == algorithm_name)
                }
                
                # Track best model
                score = training_result["cross_validation_score"]["mean"]
                if score > best_score:
                    best_score = score
                    best_model = algorithm_name
                
                # Store pipeline for later use
                ml_pipelines[f"{current_user.id}_{dataset_id}_{algorithm_name}"] = algo_pipeline
                
            except Exception as e:
                print(f"Error training {algorithm_name}: {e}")
                results[algorithm_name] = {
                    "error": str(e),
                    "metrics": {},
                    "cross_validation_score": {"mean": 0, "std": 0},
                    "feature_importance": {}
                }
        
        # Generate visualizations
        plots_dir = os.path.join(settings.UPLOAD_DIR, "plots", str(current_user.id), str(dataset_id))
        plot_files = pipeline.create_visualizations(df, plots_dir)
        plot_urls = [f"/api/uploads/{os.path.relpath(f, settings.UPLOAD_DIR).replace(os.sep, '/')}" for f in plot_files]
        
        return ensure_json_serializable({
            "dataset_id": dataset_id,
            "target_column": target_column,
            "task_type": pipeline.task_type,
            "dataset_info": {
                "filename": dataset_data.filename,
                "shape": df.shape,
                "features": list(X.columns),
                "target_unique_values": y.nunique() if hasattr(y, 'nunique') else len(set(y))
            },
            "algorithms_tested": algorithm_names,
            "results": results,
            "best_model": best_model,
            "best_score": best_score,
            "visualizations": {
                "plot_files": plot_files,
                "plot_urls": plot_urls
            },
            "recommendations": {
                "best_for_accuracy": best_model,
                "best_for_speed": "Linear Regression" if "Linear Regression" in results else algorithm_names[0],
                "best_for_interpretability": "Linear Regression" if "Linear Regression" in results else "Random Forest"
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AutoML failed: {str(e)}")

@router.get("/feature-importance/{dataset_id}")
async def get_feature_importance(
    dataset_id: int,
    algorithm: str = "Random Forest",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get feature importance for a trained model"""
    try:
        # Get the trained model
        model_query = text("""
            SELECT algorithm, model_path FROM trained_models 
            WHERE dataset_id = :dataset_id AND user_id = :user_id AND algorithm = :algorithm
            ORDER BY created_at DESC LIMIT 1
        """)
        model_data = db.execute(model_query, {
            "dataset_id": dataset_id, 
            "user_id": current_user.id,
            "algorithm": algorithm
        }).first()
        
        if not model_data:
            raise HTTPException(status_code=404, detail=f"No trained {algorithm} model found")
        
        # Load the model
        pipeline = MLPipeline()
        pipeline.load_model(model_data.model_path)
        
        feature_importance = pipeline._get_feature_importance()

        return {
            "algorithm": algorithm,
            "feature_importance": feature_importance,
            "top_features": list(feature_importance.keys())[:10] if feature_importance else []
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


@router.get("/models/recent")
async def get_recent_models(
    limit: int = 5,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get recent trained models for the user"""
    try:
        models_query = text("""
            SELECT id, algorithm, task_type, metrics, created_at
            FROM trained_models
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            LIMIT :limit
        """)
        model_rows = db.execute(models_query, {
            "user_id": current_user.id,
            "limit": limit
        }).fetchall()

        models = []
        for row in model_rows:
            metrics_obj = {}
            if row.metrics:
                try:
                    metrics_obj = json.loads(row.metrics)
                except Exception:
                    metrics_obj = {"_raw": row.metrics}
            
            models.append({
                "id": row.id,
                "algorithm": row.algorithm,
                "task_type": row.task_type,
                "created_at": row.created_at,
                "metrics": metrics_obj
            })

        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent models: {str(e)}")

@router.delete("/models/corrupted/{dataset_id}")
async def delete_corrupted_model(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Delete model files and DB records for a dataset to force retraining."""
    try:
        # Fetch trained models for this user/dataset
        rows = db.execute(text(
            """
            SELECT id, model_path FROM trained_models
            WHERE dataset_id = :dataset_id AND user_id = :user_id
            """
        ), {"dataset_id": dataset_id, "user_id": current_user.id}).fetchall()

        # Delete model files on disk
        deleted_files = 0
        for r in rows:
            try:
                if getattr(r, 'model_path', None) and os.path.exists(r.model_path):
                    os.remove(r.model_path)
                    deleted_files += 1
            except Exception:
                # Continue deleting others even if one fails
                pass

        # Delete DB records
        db.execute(text(
            """
            DELETE FROM trained_models
            WHERE dataset_id = :dataset_id AND user_id = :user_id
            """
        ), {"dataset_id": dataset_id, "user_id": current_user.id})
        db.commit()

        # Clear in-memory pipeline cache for this dataset
        key_prefix = f"{current_user.id}_{dataset_id}"
        to_delete = [k for k in list(ml_pipelines.keys()) if k.startswith(key_prefix)]
        for k in to_delete:
            try:
                del ml_pipelines[k]
            except KeyError:
                pass

        return {"status": "success", "deleted_files": deleted_files, "cleared_cache_keys": to_delete}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.get("/train/results/{model_id}")
async def get_training_results(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get training results for a specific model"""
    try:
        # Get model from database with metrics
        model_query = text("""
            SELECT id, algorithm, task_type, dataset_id, created_at, metrics
            FROM trained_models
            WHERE id = :model_id AND user_id = :user_id
        """)
        model_data = db.execute(model_query, {
            "model_id": model_id,
            "user_id": current_user.id
        }).first()
        
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get dataset information
        dataset_query = text("""
            SELECT id, filepath AS file_path, filename
            FROM datasets
            WHERE id = :dataset_id AND user_id = :user_id
        """)
        dataset_data = db.execute(dataset_query, {
            "dataset_id": model_data.dataset_id,
            "user_id": current_user.id
        }).first()
        
        if not dataset_data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Parse stored metrics
        stored_metrics = {}
        if model_data.metrics:
            try:
                stored_metrics = json.loads(model_data.metrics)
            except Exception as e:
                print(f"Failed to parse stored metrics: {e}")
        
        # Load dataset for preview
        try:
            if not dataset_data.file_path or not os.path.exists(dataset_data.file_path):
                raise HTTPException(status_code=404, detail="Dataset file not found")
            
            df = load_dataset(dataset_data.file_path)
            
            # Get focused model results - only input features, output, and predictions
            # Extract the features used in training from stored metrics
            input_features = []
            output_feature = "target"
            if isinstance(stored_metrics, dict):
                if "features" in stored_metrics:
                    input_features = stored_metrics["features"]
                if "target" in stored_metrics:
                    output_feature = stored_metrics["target"]
            
            # Create focused preview with only relevant columns
            if input_features and output_feature in df.columns:
                # Show only input features + output + predictions
                relevant_cols = input_features + [output_feature]
                available_cols = [col for col in relevant_cols if col in df.columns]
                focused_df = df[available_cols].head(10)
                
                dataset_info = {
                    "total_rows": len(df),
                    "columns": available_cols,
                    "data_types": {col: str(focused_df[col].dtype) for col in available_cols},
                    "preview": clean_for_json(focused_df.to_dict('records')),
                    "has_more": len(df) > 10,
                    "input_features": input_features,
                    "output_feature": output_feature
                }
            else:
                # Fallback to original behavior if no feature info available
                dataset_info = {
                    "total_rows": len(df),
                    "columns": df.columns.tolist(),
                    "data_types": {col: str(df[col].dtype) for col in df.columns},
                    "preview": clean_for_json(df.head(10).to_dict('records')),
                    "has_more": len(df) > 10
                }
            
            # Extract metrics from stored data - clean for JSON serialization
            metrics = clean_for_json(stored_metrics.get("metrics", {}) if isinstance(stored_metrics, dict) else {})
            cross_validation_score = clean_for_json(stored_metrics.get("cross_validation_score", {"mean": 0, "std": 0, "scores": []}) if isinstance(stored_metrics, dict) else {"mean": 0, "std": 0, "scores": []})
            feature_importance = clean_for_json(stored_metrics.get("feature_importance", {}) if isinstance(stored_metrics, dict) else {})
            samples = clean_for_json(stored_metrics.get("samples", {"first": [], "middle": [], "last": []}) if isinstance(stored_metrics, dict) else {"first": [], "middle": [], "last": []})
            
            return clean_for_json({
                "model_id": model_data.id,
                "algorithm": model_data.algorithm,
                "task_type": model_data.task_type,
                "created_at": model_data.created_at,
                "dataset_id": model_data.dataset_id,
                "metrics": metrics,
                "cross_validation_score": cross_validation_score,
                "dataset_info": dataset_info,
                "feature_importance": feature_importance,
                "samples": samples
            })
            
        except Exception as dataset_error:
            print(f"Dataset loading error: {dataset_error}")
            # Return basic model info even if dataset loading fails - clean for JSON serialization
            return clean_for_json({
                "model_id": model_data.id,
                "algorithm": model_data.algorithm,
                "task_type": model_data.task_type,
                "created_at": model_data.created_at,
                "dataset_id": getattr(model_data, 'dataset_id', None),
                "metrics": stored_metrics.get("metrics", {}) if isinstance(stored_metrics, dict) else {},
                "cross_validation_score": stored_metrics.get("cross_validation_score", {"mean": 0, "std": 0, "scores": []}) if isinstance(stored_metrics, dict) else {"mean": 0, "std": 0, "scores": []},
                "dataset_info": {
                    "total_rows": 0,
                    "columns": [],
                    "data_types": {},
                    "preview": [],
                    "has_more": False
                },
                "feature_importance": stored_metrics.get("feature_importance", {}) if isinstance(stored_metrics, dict) else {}
            })
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Training results error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch training results: {str(e)}")

@router.post("/train/retry/{model_id}")
async def retry_training(
    model_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retry training for a specific model with live data"""
    try:
        # Get model from database
        model_query = text("""
            SELECT id, algorithm, task_type, dataset_id, created_at
            FROM trained_models
            WHERE id = :model_id AND user_id = :user_id
        """)
        model_data = db.execute(model_query, {
            "model_id": model_id,
            "user_id": current_user.id
        }).first()
        
        if not model_data:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Get dataset information
        dataset_query = text("""
            SELECT id, file_path, filename
            FROM datasets
            WHERE id = :dataset_id AND user_id = :user_id
        """)
        dataset_data = db.execute(dataset_query, {
            "dataset_id": model_data.dataset_id,
            "user_id": current_user.id
        }).first()
        
        if not dataset_data:
            raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset and run live training
        try:
            df = load_dataset(dataset_data.file_path)
            
            # Initialize ML pipeline
            pipeline = MLPipeline()
            
            # Auto-detect target column
            target_column = None
            if 'Price' in df.columns:
                target_column = 'Price'
            elif 'price' in df.columns:
                target_column = 'price'
            else:
                target_column = df.columns[-1]
            
            # Preprocess data
            X, y = pipeline.preprocess_data(df, target_column, 'auto')
            
            # Train model with live data
            result = pipeline.train_model(X, y, model_data.algorithm)
            
            # Update model in database with new results
            update_query = text("""
                UPDATE trained_models 
                SET metrics = :metrics, updated_at = CURRENT_TIMESTAMP
                WHERE id = :model_id
            """)
            db.execute(update_query, {
                "model_id": model_id,
                "metrics": json.dumps(result)
            })
            db.commit()
            
            return {
                "status": "success",
                "message": "Training completed successfully",
                "model_id": model_data.id,
                "algorithm": result["algorithm"],
                "task_type": result["task_type"],
                "metrics": result["metrics"],
                "cross_validation_score": result["cross_validation_score"],
                "dataset_info": result.get("dataset_info", {}),
                "feature_importance": result.get("feature_importance", {})
            }
            
        except Exception as training_error:
            print(f"Retry training error: {training_error}")
            raise HTTPException(status_code=500, detail=f"Retry training failed: {str(training_error)}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retry training: {str(e)}")

@router.get("/datasets/{dataset_id}/data")
async def get_dataset_data(dataset_id: int, offset: int = 0, limit: int = 10, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Get dataset data with pagination"""
    try:
        # Get dataset info using raw SQL query
        dataset_query = text("""
            SELECT id, filename, filepath AS file_path, filesize AS file_size, uploaded_at AS created_at
            FROM datasets 
            WHERE id = :dataset_id AND user_id = :user_id
        """)
        dataset_data = db.execute(dataset_query, {
            "dataset_id": dataset_id, 
            "user_id": current_user.id
        }).first()
        
        if not dataset_data:
            print(f"Dataset {dataset_id} not found for user {current_user.id}")
            # Try to find the correct dataset by checking if this is a model ID instead
            model_query = text("""
                SELECT dataset_id FROM trained_models 
                WHERE id = :model_id AND user_id = :user_id
            """)
            model_data = db.execute(model_query, {
                "model_id": dataset_id,
                "user_id": current_user.id
            }).first()
            
            if model_data:
                print(f"Found model {dataset_id} with dataset_id {model_data.dataset_id}")
                # Use the correct dataset_id
                dataset_query = text("""
                    SELECT id, filename, filepath AS file_path, filesize AS file_size, uploaded_at AS created_at
                    FROM datasets 
                    WHERE id = :dataset_id AND user_id = :user_id
                """)
                dataset_data = db.execute(dataset_query, {
                    "dataset_id": model_data.dataset_id, 
                    "user_id": current_user.id
                }).first()
            
            if not dataset_data:
                raise HTTPException(status_code=404, detail="Dataset not found")
        
        # Load dataset with proper path handling
        file_path = dataset_data.file_path
        if not os.path.exists(file_path):
            # Try alternative paths
            possible_paths = [
                file_path,
                file_path.replace('./storage/uploads\\', './backend/storage/uploads/'),
                file_path.replace('./storage/uploads\\', './backend/storage/uploads\\'),
                f'./backend/storage/uploads/{dataset_data.filename}'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = path
                    break
            else:
                raise HTTPException(status_code=404, detail="Dataset file not found")
        
        # Load dataset
        df = load_dataset(file_path)
        
        # Get paginated data
        start_idx = offset
        end_idx = min(offset + limit, len(df))
        data_slice = df.iloc[start_idx:end_idx]
        
        # Convert to records with proper types and clean for JSON
        records = []
        for idx, row in data_slice.iterrows():
            record = {}
            for col in data_slice.columns:
                value = row[col]
                if pd.api.types.is_numeric_dtype(data_slice[col]):
                    if pd.isna(value):
                        record[col] = None
                    else:
                        record[col] = clean_for_json(float(value))
                else:
                    record[col] = str(value) if not pd.isna(value) else None
            records.append(record)
        
        return clean_for_json({
            "data": records,
            "total_rows": len(df),
            "offset": offset,
            "limit": limit,
            "has_more": end_idx < len(df)
        })
    except Exception as e:
        print(f"Dataset data error: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading dataset: {str(e)}")