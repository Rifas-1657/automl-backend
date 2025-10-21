import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_regression, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Advanced ML Libraries
try:
    import xgboost as xgb
    from lightgbm import LGBMRegressor, LGBMClassifier
    from catboost import CatBoostRegressor, CatBoostClassifier
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    ADVANCED_ML_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

class RobustDataPipeline:
    """Robust data pipeline with feature validation and corruption prevention"""
    
    def __init__(self):
        self.feature_names = None
        self.target_name = None
        self.scaler = None
        self.encoders = {}
        self.target_encoder = None
        self.feature_selector = None
        self.imputer = None
        self.original_dtypes = {}
        self.preprocessing_steps = []
        # Persist columns where zero should be treated as missing (detected at fit)
        self.zero_as_missing_columns = set()
        # Persist imputation statistics learned during fit
        self.numeric_medians: dict[str, float] = {}
        self.categorical_modes: dict[str, str] = {}
        
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str], target_name: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Fit and transform data with comprehensive validation"""
        print(f"Starting robust data pipeline for {len(feature_names)} features")
        
        # VALIDATE INPUTS
        if X.shape[1] != len(feature_names):
            raise ValueError(f"Feature count mismatch: expected {len(feature_names)}, got {X.shape[1]}")
        
        # Store metadata
        self.feature_names = feature_names.copy()
        self.target_name = target_name
        self.original_dtypes = X.dtypes.to_dict()
        
        # Create working copies
        X_work = X.copy()
        y_work = y.copy()
        
        print(f"Original data shape: X={X_work.shape}, y={y_work.shape}")
        
        # Step 1: Handle missing values
        X_work = self._handle_missing_values_robust(X_work)
        
        # Step 2: Encode categorical variables
        X_work = self._encode_categorical_robust(X_work)
        
        # Step 3: Scale features
        X_work = self._scale_features_robust(X_work)
        
        # Step 4: Encode target if needed
        y_work = self._encode_target_robust(y_work)
        
        # Step 5: Remove constant features (EXACTLY like reference code)
        X_work = self._remove_constant_features_robust(X_work)
        
        # Step 6: Feature selection (optional)
        X_work = self._select_features_robust(X_work, y_work)
        
        # VERIFY no corruption (allow for one-hot encoding expansion)
        if X_work.shape[1] < len(self.feature_names):
            raise ValueError(f"Feature corruption detected: expected at least {len(self.feature_names)}, got {X_work.shape[1]}")
        
        print(f"Final data shape: X={X_work.shape}, y={y_work.shape}")
        return X_work, y_work
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted pipeline"""
        if X.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected {len(self.feature_names)} features, got {X.shape[1]}")
        
        X_work = X.copy()
        
        # Apply same transformations
        X_work = self._handle_missing_values_robust(X_work, fit=False)
        X_work = self._encode_categorical_robust(X_work, fit=False)
        X_work = self._scale_features_robust(X_work, fit=False)
        X_work = self._select_features_robust(X_work, fit=False)
        
        return X_work
    
    def _handle_missing_values_robust(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Handle missing values EXACTLY like reference code"""
        print(f"HANDLING MISSING VALUES (Reference Code Method):")
        
        # Detect and convert zero-coded-missing values before standard imputation
        X = X.copy()
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            name_hints = {c for c in numeric_cols if str(c).strip().lower() in {
                'glucose', 'bloodpressure', 'skin_thickness', 'insulin', 'bmi',
                'blood_pressure', 'skinThickness'.lower()
            }}
            if fit:
                detected = set()
                for col in numeric_cols:
                    s = X[col]
                    if len(s) == 0:
                        continue
                    zero_ratio = float((s == 0).mean())
                    median_val = float(s.median()) if s.notnull().any() else 0.0
                    if (col in name_hints) or (zero_ratio >= 0.2 and median_val > 0):
                        detected.add(col)
                if detected:
                    print(f" Treating zeros as missing for columns: {sorted(list(detected))}")
                self.zero_as_missing_columns = detected
            for col in (self.zero_as_missing_columns or set()):
                if col in X.columns:
                    X.loc[X[col] == 0, col] = np.nan
        except Exception as e:
            print(f" Zero-as-missing detection skipped: {e}")
        
        # Check for missing values
        missing_values = X.isnull().sum()
        if missing_values.sum() > 0:
            print("Missing values found:")
            for col, missing_count in missing_values[missing_values > 0].items():
                print(f"  - {col}: {missing_count} missing values")
            
            # Fill numerical columns with median (learned on fit; reused on transform)
            numerical_cols = X.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 0:
                if fit:
                    self.numeric_medians = {}
                    for col in numerical_cols:
                        s = X[col]
                        med = float(s.median()) if s.notnull().any() else 0.0
                        self.numeric_medians[col] = med
                # Use learned medians if available; fallback to 0 for any still-missing
                if getattr(self, 'numeric_medians', None):
                    # pandas fillna accepts dict mapping column->value
                    X[numerical_cols] = X[numerical_cols].fillna(value=self.numeric_medians)
                else:
                    X[numerical_cols] = X[numerical_cols].fillna(0)
                # Final safety: no NaNs allowed for numeric
                if X[numerical_cols].isnull().sum().sum() > 0:
                    X[numerical_cols] = X[numerical_cols].fillna(0)
                print(f"Filled {len(numerical_cols)} numerical columns with median (persisted)")
            
            # Fill categorical columns with mode (persisted across transform)
            categorical_cols = X.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                if fit:
                    mode_value = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
                    self.categorical_modes[col] = mode_value
                mode_value = self.categorical_modes.get(col, 'Unknown')
                X[col] = X[col].fillna(mode_value)
                print(f"Filled {col} with mode: {mode_value}")
            
            print("Missing values handled (Reference Code Method)")
        else:
            print("No missing values found")
        
        return X
    
    def _encode_categorical_robust(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables EXACTLY like reference code"""
        print(f"ENCODING CATEGORICAL VARIABLES (Reference Code Method):")
        
        categorical_cols = X.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"Categorical columns to encode: {list(categorical_cols)}")
            
            # Use label encoding for categorical variables (EXACTLY like reference code)
            if fit:
                if not hasattr(self, 'encoders'):
                    self.encoders = {}
                
                for col in categorical_cols:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[col] = le
                    print(f"  - Encoded: {col}")
                
                print("Categorical variables encoded (Reference Code Method)")
            else:
                # Transform using stored encoders
                for col in categorical_cols:
                    if col in self.encoders:
                        try:
                            enc = self.encoders[col]
                            series = X[col].astype(str)
                            # Handle unseen categories by mapping to the first known class (mode during fit)
                            try:
                                known = set([str(c) for c in list(enc.classes_)])
                                fallback = str(enc.classes_[0])
                                series = series.where(series.isin(known), fallback)
                            except Exception:
                                pass
                            X[col] = enc.transform(series)
                        except Exception as e:
                            # As a last resort, map unseen values to 0 to avoid prediction failure
                            try:
                                X[col] = 0
                            except Exception:
                                pass
        else:
            print("No categorical variables found")
        
        return X
    
    def _scale_features_robust(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale features EXACTLY like reference code"""
        print(f"SCALING FEATURES (Reference Code Method):")
        
        # Use StandardScaler EXACTLY like reference code
        if fit:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            print("Features scaled with StandardScaler (Reference Code Method)")
        else:
            if self.scaler is not None:
                X_scaled = self.scaler.transform(X)
            else:
                X_scaled = X.values
        
        # Convert back to DataFrame with original column names
        X_result = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        return X_result
    
    def _remove_constant_features_robust(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove constant features EXACTLY like reference code"""
        print(f"REMOVING CONSTANT FEATURES (Reference Code Method):")
        
        # Remove constant features (EXACTLY like reference code)
        constant_features = [col for col in X.columns if X[col].nunique() <= 1]
        if constant_features:
            print(f"Removing constant features: {constant_features}")
            X = X.drop(columns=constant_features)
            print("Constant features removed")
        else:
            print("No constant features found")
        
        return X
    
    def _encode_target_robust(self, y: pd.Series) -> pd.Series:
        """Encode target variable if needed"""
        if y.dtype == 'object':
            if self.target_encoder is None:
                self.target_encoder = LabelEncoder()
            y_encoded = self.target_encoder.fit_transform(y.astype(str))
            return pd.Series(y_encoded, name=y.name, index=y.index)
        elif y.dtype == 'bool':
            # Convert boolean to int for binary classification
            return y.astype(int)
        return y
    
    def _select_features_robust(self, X: pd.DataFrame, y: pd.Series = None, fit: bool = True) -> pd.DataFrame:
        """Select features with robust selection"""
        # For now, keep all features to prevent corruption
        # Feature selection can be added later if needed
        return X

class MLPipeline:
    """Comprehensive ML Pipeline for dataset analysis, preprocessing, and predictions"""
    
    def __init__(self):
        self.data_pipeline = RobustDataPipeline()
        self.model = None
        self.task_type = None
        self.target_column = None
        self.feature_columns = None
        self.original_feature_columns = None
        
    def analyze_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive dataset analysis with proper categorical handling"""
        analysis = {
            "basic_info": {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            },
            "missing_values": df.isnull().sum().to_dict(),
            "statistical_summary": self._get_safe_statistical_summary(df),
            "correlation_matrix": self._get_safe_correlation_matrix(df),
            "data_quality": self._assess_data_quality(df),
            "recommendations": self._generate_recommendations(df)
        }
        
        return analysis
    
    def _get_safe_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary that handles categorical data safely"""
        try:
            # Only describe numeric columns to avoid categorical conversion errors
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                return df[numeric_cols].describe().to_dict()
            else:
                return {}
        except Exception as e:
            print(f"Warning: Could not generate statistical summary: {e}")
            return {}
    
    def _get_safe_correlation_matrix(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get correlation matrix that handles categorical data safely"""
        try:
            # Only correlate numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                return df[numeric_cols].corr().to_dict()
            else:
                return {}
        except Exception as e:
            print(f"Warning: Could not generate correlation matrix: {e}")
            return {}
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics with safe categorical handling"""
        quality = {
            "completeness": (1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100,
            "duplicates": df.duplicated().sum(),
            "numeric_columns": len(df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(df.select_dtypes(include=['object']).columns),
            "datetime_columns": len(df.select_dtypes(include=['datetime']).columns)
        }
        
        # Check for potential target columns - handle both numeric and categorical safely
        potential_targets = []
        
        # Check numeric columns for regression/classification
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    nunique = df[col].nunique()
                    if nunique > 2:  # Regression candidate
                        potential_targets.append({"column": col, "type": "regression", "nunique": nunique})
                    elif nunique == 2:  # Binary classification candidate
                        potential_targets.append({"column": col, "type": "binary_classification", "nunique": nunique})
                except Exception as e:
                    print(f"Warning: Could not assess numeric column {col}: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not process numeric columns: {e}")
        
        # Check categorical columns for classification
        try:
            categorical_cols = df.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                try:
                    nunique = df[col].nunique()
                    if nunique <= 10:  # Multi-class classification candidate
                        potential_targets.append({"column": col, "type": "multiclass_classification", "nunique": nunique})
                except Exception as e:
                    print(f"Warning: Could not assess categorical column {col}: {e}")
                    continue
        except Exception as e:
            print(f"Warning: Could not process categorical columns: {e}")
        
        quality["potential_targets"] = potential_targets
        return quality
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate preprocessing and modeling recommendations"""
        recommendations = []
        
        # Missing value recommendations
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            recommendations.append(f"Handle missing values in: {', '.join(missing_cols)}")
        
        # Data type recommendations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        if len(numeric_cols) > 0:
            recommendations.append("Consider scaling numerical features")
        
        if len(categorical_cols) > 0:
            recommendations.append("Encode categorical features")
        
        # Feature engineering recommendations
        if len(numeric_cols) > 5:
            recommendations.append("Consider feature selection for dimensionality reduction")
        
        # Model recommendations
        if len(df) < 1000:
            recommendations.append("Small dataset - use simpler models to avoid overfitting")
        elif len(df) > 10000:
            recommendations.append("Large dataset - consider ensemble methods or deep learning")
        
        return recommendations
    
    def preprocess_data(self, df: pd.DataFrame, target_column: str, 
                       task_type: str = "auto") -> Tuple[pd.DataFrame, pd.Series]:
        """Preprocess data using robust pipeline"""
        self.target_column = target_column
        self.task_type = task_type
        
        # Store original data for display purposes
        self.original_data = df.copy()
        
        # Auto-detect task type if not specified
        if task_type == "auto":
            target_unique = df[target_column].nunique()
            target_dtype = df[target_column].dtype
            target_values = df[target_column].unique()
            
            print(f" Auto-detecting task type: unique={target_unique}, dtype={target_dtype}")
            print(f" Target values: {target_values[:10]}")  # Show first 10 unique values
            
            # Special handling for common target column names
            if target_column.lower() in ['price', 'cost', 'value', 'amount', 'revenue', 'sales', 'income']:
                self.task_type = "regression"
                print(f" Detected as regression (target column name: {target_column})")
            # If target is categorical (object/string), it's classification
            elif target_dtype == 'object' or target_dtype.name == 'category':
                self.task_type = "classification"
                print(f" Detected as classification (categorical target)")
            # Check for binary classification (2 unique values)
            elif target_unique == 2:
                # Check if values are binary-like (Yes/No, 0/1, True/False, etc.)
                unique_vals = set(str(v).lower() for v in target_values)
                binary_indicators = {'yes', 'no', 'true', 'false', '0', '1', 'y', 'n'}
                if any(val in binary_indicators for val in unique_vals):
                    self.task_type = "binary_classification"
                    print(f" Detected as binary classification (binary values: {target_values})")
                else:
                    self.task_type = "classification"
                    print(f" Detected as classification (2 unique values)")
            # Check for regression (many unique numeric values)
            elif target_unique > 10 and target_dtype in ['int64', 'float64']:
                self.task_type = "regression"
                print(f" Detected as regression (numeric target with {target_unique} unique values)")
            # Default to regression for numeric targets
            elif target_dtype in ['int64', 'float64']:
                self.task_type = "regression"
                print(f" Detected as regression (numeric target)")
            # Default to classification for safety
            else:
                self.task_type = "classification"
                print(f" Defaulting to classification (unique={target_unique}, dtype={target_dtype})")
        
        # Separate features and target
        # Exclude common non-predictive columns
        exclude_columns = [target_column, 'Id', 'id', 'ID', 'index', 'Index']
        self.original_feature_columns = [c for c in df.columns if c not in exclude_columns]
        
        # CRITICAL: If we have a specific feature selection from training, use those instead
        if hasattr(self, 'selected_features') and self.selected_features:
            self.original_feature_columns = self.selected_features
            print(f" Using pre-selected features: {self.original_feature_columns}")
        
        # Ensure we have the correct number of features
        print(f" Available columns: {list(df.columns)}")
        print(f" Target column: {target_column}")
        print(f" Excluded columns: {exclude_columns}")
        print(f" Selected features: {self.original_feature_columns}")
        print(f" Number of features: {len(self.original_feature_columns)}")
        
        X = df[self.original_feature_columns]
        y = df[target_column]
        
        print(f" Using robust data pipeline for {len(self.original_feature_columns)} features")
        
        # Warn about small datasets
        if len(df) < 30:
            print(f" WARNING: Small dataset ({len(df)} samples) - results may be unreliable")
            print(f" Consider using Linear Regression or Multiple Linear Regression for better results")
        
        # Pass task type to data pipeline
        self.data_pipeline.task_type = self.task_type
        
        # Use robust data pipeline
        X_processed, y_processed = self.data_pipeline.fit_transform(
            X, y, self.original_feature_columns, target_column
        )
        
        # Store feature information
        self.feature_columns = X_processed.columns.tolist()
        
        # Store target encoder for later use in predictions
        if hasattr(self.data_pipeline, 'target_encoder'):
            self.target_encoder = self.data_pipeline.target_encoder
        
        print(f" Preprocessing complete: X={X_processed.shape}, y={y_processed.shape}")
        return X_processed, y_processed
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with improved strategies"""
        X_clean = X.copy()
        
        # Handle numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Use median for numeric columns (more robust than mean)
            imputer = SimpleImputer(strategy='median')
            X_clean[numeric_cols] = imputer.fit_transform(X_clean[numeric_cols])
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            # Use most frequent for categorical columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            X_clean[categorical_cols] = cat_imputer.fit_transform(X_clean[categorical_cols])
        
        return X_clean
    
    def _encode_categorical(self, X: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            # Store encoders for later use
            if not hasattr(self, 'encoders'):
                self.encoders = {}
            
            for col in categorical_cols:
                try:
                    # Handle missing values first
                    X[col] = X[col].fillna('Unknown')
                    
                    # Use LabelEncoder for categorical variables
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[col] = le
                except Exception as e:
                    print(f"Warning: Could not encode column {col}: {e}")
                    # If encoding fails, drop the column
                    X = X.drop(columns=[col])
        
        return X
    
    def _scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        # Only scale if we have numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return X
            
        self.scaler = StandardScaler()
        X_scaled = X.copy()
        X_scaled[numeric_cols] = self.scaler.fit_transform(X[numeric_cols])
        return X_scaled
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Select best features - disabled for small feature sets to prevent mismatch"""
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        # Skip selection when features are few to preserve training/prediction consistency
        if len(numeric_cols) <= 10:
            return X
        # Only select features if we have enough columns and numeric data
        if len(numeric_cols) > 10 and len(numeric_cols) > 1:
            k = min(10, len(numeric_cols) - 1)
            try:
                if self.task_type == "regression":
                    selector = SelectKBest(score_func=f_regression, k=k)
                else:
                    selector = SelectKBest(score_func=f_classif, k=k)

                X_selected = selector.fit_transform(X[numeric_cols], y)
                selected_features = numeric_cols[selector.get_support()].tolist()
                self.feature_selector = selector

                # Keep only selected numeric features and all categorical features
                categorical_cols = X.select_dtypes(exclude=[np.number]).columns
                all_selected_cols = selected_features + list(categorical_cols)
                return X[all_selected_cols]
            except Exception as e:
                print(f"Warning: Feature selection failed: {e}")
                return X
        return X
    
    def get_algorithm_recommendations(self, X: pd.DataFrame, y: pd.Series) -> List[Dict[str, Any]]:
        """Get comprehensive algorithm recommendations based on dataset characteristics"""
        recommendations = []
        
        # Determine dataset characteristics
        n_samples = len(X)
        n_features = len(X.columns)
        is_small_dataset = n_samples < 1000
        is_large_dataset = n_samples > 10000
        
        # Core algorithms for all datasets
        if self.task_type == "regression":
            core_algorithms = [
                {
                    "name": "Linear Regression",
                    "type": "linear",
                    "description": "Fast baseline model for continuous targets, highly interpretable",
                    "pros": ["Very fast", "Highly interpretable", "No hyperparameters", "Good baseline"],
                    "cons": ["Assumes linear relationships", "Sensitive to outliers"],
                    "best_for": ["Linear relationships", "Small datasets", "Baseline models", "Interpretability"]
                },
                {
                    "name": "Multiple Linear Regression",
                    "type": "linear",
                    "description": "Linear regression with multiple features for continuous targets",
                    "pros": ["Handles multiple features", "Interpretable coefficients", "Fast training", "Good for house prices"],
                    "cons": ["Assumes linear relationships", "Sensitive to multicollinearity"],
                    "best_for": ["House price prediction", "Multiple features", "Linear relationships", "Real estate"]
                },
                {
                    "name": "Random Forest",
                    "type": "ensemble",
                    "description": "Robust ensemble method for regression, handles non-linear relationships",
                    "pros": ["Robust to outliers", "Handles mixed data types", "Feature importance", "No overfitting"],
                    "cons": ["Can be slow on large datasets", "Less interpretable than linear models"],
                    "best_for": ["Most regression tasks", "Mixed data types", "Feature importance", "Robust predictions"]
                }
            ]
        elif self.task_type == "binary_classification":
            core_algorithms = [
                {
                    "name": "Logistic Regression",
                    "type": "linear",
                    "description": "Fast baseline model for binary classification, highly interpretable",
                    "pros": ["Very fast", "Highly interpretable", "Probability outputs", "Good for binary outcomes"],
                    "cons": ["Assumes linear decision boundary", "Sensitive to outliers"],
                    "best_for": ["Binary classification", "Linear relationships", "Probability predictions", "Interpretability"]
                }
            ]
        else:  # multiclass classification
            core_algorithms = [
                {
                    "name": "Logistic Regression",
                    "type": "linear",
                    "description": "Fast baseline model for classification, highly interpretable",
                    "pros": ["Very fast", "Highly interpretable", "Probability outputs", "Good for multiple classes"],
                    "cons": ["Assumes linear decision boundary", "Sensitive to outliers"],
                    "best_for": ["Classification tasks", "Linear relationships", "Probability predictions", "Interpretability"]
                }
            ]
        
        # Add additional algorithms based on task type
        if self.task_type == "regression":
            # Add more regression algorithms
            core_algorithms.extend([
                {
                    "name": "Support Vector Regression",
                    "type": "kernel_method",
                    "description": "Powerful for complex non-linear regression patterns",
                    "pros": ["Effective in high dimensions", "Memory efficient", "Good for complex patterns"],
                    "cons": ["Slow on large datasets", "Sensitive to feature scaling", "Many hyperparameters"],
                    "best_for": ["Small to medium datasets", "High-dimensional data", "Complex patterns"]
                }
            ])
        else:
            # Add Random Forest for classification tasks
            core_algorithms.append({
                "name": "Random Forest",
                "type": "ensemble",
                "description": "Robust ensemble method, handles non-linear relationships well",
                "pros": ["Robust to outliers", "Handles mixed data types", "Feature importance", "No overfitting"],
                "cons": ["Can be slow on large datasets", "Less interpretable than linear models"],
                "best_for": ["Most datasets", "Mixed data types", "Feature importance", "Robust predictions"]
            })
        
        # Add Logistic Regression for all task types
        core_algorithms.append({
            "name": "Logistic Regression",
            "type": "linear",
            "description": "Fast baseline model for classification, highly interpretable",
            "pros": ["Very fast", "Highly interpretable", "Probability outputs", "Good for binary outcomes"],
            "cons": ["Assumes linear decision boundary", "Sensitive to outliers"],
            "best_for": ["Binary classification", "Linear relationships", "Probability predictions", "Interpretability"]
        })
        
        # Add SVM for small to medium datasets
        if not is_large_dataset:
            svm_algorithm = {
                "name": "Support Vector Machine",
                "type": "kernel_method",
                "description": "Powerful for complex non-linear patterns",
                "pros": ["Effective in high dimensions", "Memory efficient", "Versatile kernels", "Good for complex patterns"],
                "cons": ["Slow on large datasets", "Sensitive to feature scaling", "Many hyperparameters"],
                "best_for": ["Small to medium datasets", "High-dimensional data", "Complex patterns"]
            }
            core_algorithms.append(svm_algorithm)
        
        # Add advanced algorithms for larger datasets
        if ADVANCED_ML_AVAILABLE and not is_small_dataset:
            advanced_algorithms = [
                {
                    "name": "XGBoost",
                    "type": "gradient_boosting",
                    "description": "High-performance gradient boosting, excellent for competitions",
                    "pros": ["Very high accuracy", "Built-in regularization", "Handles missing values", "Fast training"],
                    "cons": ["Can overfit", "Many hyperparameters", "Less interpretable", "Memory intensive"],
                    "best_for": ["Large datasets", "High accuracy requirements", "Competitions", "Tabular data"]
                },
                {
                    "name": "LightGBM",
                    "type": "gradient_boosting",
                    "description": "Fast and memory-efficient gradient boosting",
                    "pros": ["Very fast training", "Low memory usage", "Good accuracy", "Handles categorical features"],
                    "cons": ["Can overfit on small datasets", "Less robust to outliers", "Many hyperparameters"],
                    "best_for": ["Large datasets", "Speed requirements", "Memory constraints", "Categorical features"]
                }
            ]
            core_algorithms.extend(advanced_algorithms)
        
        # Add CatBoost if available
        if ADVANCED_ML_AVAILABLE and not is_small_dataset:
            try:
                from catboost import CatBoostRegressor, CatBoostClassifier
                core_algorithms.append({
                    "name": "CatBoost",
                    "type": "gradient_boosting",
                    "description": "Gradient boosting with excellent categorical feature handling",
                    "pros": ["Excellent with categorical features", "No need for encoding", "Good accuracy", "Robust"],
                    "cons": ["Slower than LightGBM", "Memory intensive", "Many hyperparameters"],
                    "best_for": ["Categorical features", "Mixed data types", "High accuracy", "Robust predictions"]
                })
            except ImportError:
                pass
        
        # Deduplicate by algorithm name while preserving order to avoid duplicates (e.g., Logistic Regression)
        seen_names = set()
        deduped: List[Dict[str, Any]] = []
        for algo in core_algorithms:
            name = algo.get("name")
            if name in seen_names:
                continue
            seen_names.add(name)
            deduped.append(algo)
        return deduped
    
    def train_model(self, X: pd.DataFrame, y: pd.Series, algorithm: str) -> Dict[str, Any]:
        """Train the selected algorithm with comprehensive support"""
        # Use stored original data if available, otherwise use current data
        if hasattr(self, 'original_data') and self.original_data is not None:
            # Extract original X and y from the stored original data
            self.original_X = self.original_data.drop(columns=[self.target_column])
            self.original_y = self.original_data[self.target_column]
        else:
            # Fallback to current data
            self.original_X = X.copy()
            self.original_y = y.copy()
        
        # IMPROVED: Better data validation and debugging
        print(f"TRAINING DEBUG INFO:")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  X columns: {list(X.columns)}")
        print(f"  y dtype: {y.dtype}")
        print(f"  y range: {y.min():.2f} to {y.max():.2f}")
        print(f"  y mean: {y.mean():.2f}")
        print(f"  X dtypes: {X.dtypes.to_dict()}")
        
        # Check for any remaining missing values
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum()
        print(f"  Missing values - X: {missing_X}, y: {missing_y}")
        
        if missing_X > 0 or missing_y > 0:
            print(f"⚠️ WARNING: Missing values detected! X: {missing_X}, y: {missing_y}")
            # Fill any remaining missing values
            X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else 0)
            y = y.fillna(y.median() if y.dtype in ['int64', 'float64'] else y.mode()[0] if len(y.mode()) > 0 else 0)
        
        # Smart train/test split based on dataset size
        if len(X) <= 50:
            # Small datasets: Use 80/20 split but ensure we have at least 5 test samples
            test_size = max(0.2, 5 / len(X)) if len(X) > 5 else 0.1
            print(f"Small dataset ({len(X)} rows): Using {test_size:.1%} test split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        else:
            # Large datasets: Use standard 80/20 split
            print(f"Large dataset ({len(X)} rows): Using 80/20 train/test split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize model based on algorithm
        chosen_algo_name = algorithm
        if algorithm == "Random Forest":
            if self.task_type == "regression":
                self.model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        elif algorithm == "Linear Regression":
            if self.task_type == "regression":
                self.model = LinearRegression()
            else:
                # Remap to Logistic Regression model only when task is classification
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                chosen_algo_name = "Logistic Regression"
        
        elif algorithm == "Multiple Regression":
            # Multiple Regression is a regression algorithm; if task is classification, map to Logistic
            if self.task_type in ("classification", "binary_classification"):
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                chosen_algo_name = "Logistic Regression"
            else:
                self.model = LinearRegression()
        
        elif algorithm == "Multiple Linear Regression":
            # Use Ridge regression for small datasets to prevent overfitting
            if len(X) < 50:
                from sklearn.linear_model import Ridge
                self.model = Ridge(alpha=1.0, random_state=42)
                chosen_algo_name = "Multiple Linear Regression (Ridge)"
            else:
                self.model = LinearRegression()
                chosen_algo_name = "Multiple Linear Regression"
        
        elif algorithm == "Logistic Regression":
            if self.task_type == "binary_classification":
                self.model = LogisticRegression(random_state=42, max_iter=1000)
                chosen_algo_name = "Logistic Regression"
            elif self.task_type == "classification":
                self.model = LogisticRegression(random_state=42, max_iter=1000, multi_class='ovr')
                chosen_algo_name = "Logistic Regression"
            else:  # regression - use Linear Regression instead
                self.model = LinearRegression()
                chosen_algo_name = "Linear Regression (Logistic not applicable for regression)"
        
        else:
            # Restrict to allowed algorithms only
            raise ValueError(f"Unsupported algorithm: {algorithm}. Allowed: Linear Regression, Multiple Linear Regression, Logistic Regression, Random Forest")
        
        # Train the model
        print(f" Training model with X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f" X_train dtypes: {X_train.dtypes}")
        print(f" y_train dtype: {y_train.dtype}, unique values: {y_train.unique()}")
        print(f" y_train sample values: {y_train.head()}")
        
        try:
            self.model.fit(X_train, y_train)
            print(f" Model training successful")
        except Exception as e:
            print(f" Model training failed: {e}")
            print(f" X_train sample: {X_train.head()}")
            print(f" y_train sample: {y_train.head()}")
            raise e
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = None
        try:
            if self.task_type in ("binary_classification", "multiclass_classification") and hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_test)
                # For binary classification take positive class probability
                if isinstance(proba, np.ndarray) and proba.ndim == 2 and proba.shape[1] >= 2:
                    y_pred_proba = proba[:, 1]
        except Exception:
            y_pred_proba = None
        
        # IMPROVED: Better prediction debugging
        print(f"PREDICTION DEBUG INFO:")
        print(f"  y_test range: {y_test.min():.2f} to {y_test.max():.2f}")
        print(f"  y_pred range: {y_pred.min():.2f} to {y_pred.max():.2f}")
        print(f"  y_test mean: {y_test.mean():.2f}")
        print(f"  y_pred mean: {y_pred.mean():.2f}")
        print(f"  Sample predictions: {y_pred[:5]}")
        print(f"  Sample actual: {y_test.iloc[:5].values}")
        
        # Calculate metrics (provide optional probabilities and feature count)
        n_features = len(self.feature_columns) if hasattr(self, 'feature_columns') and self.feature_columns is not None else None
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba=y_pred_proba, n_features=n_features)
        
        # IMPROVED: Debug metrics calculation
        print(f"METRICS DEBUG INFO:")
        print(f"  R² Score: {metrics.get('r2_score', 'N/A')}")
        print(f"  MSE: {metrics.get('mse', 'N/A')}")
        print(f"  RMSE: {metrics.get('rmse', 'N/A')}")
        print(f"  MAE: {metrics.get('mae', 'N/A')}")
        
        # Cross-validation score (adaptive to dataset size and task type)
        cv_folds = min(5, len(X_train) // 2) if len(X_train) >= 4 else 2
        if cv_folds < 2:
            cv_folds = 2
        
        # Choose appropriate scoring metric based on task type
        if self.task_type == "regression":
            scoring = 'r2'
        elif self.task_type == "binary_classification":
            scoring = 'accuracy'  # Could also use 'roc_auc' if probabilities are available
        else:  # multiclass classification
            scoring = 'accuracy'
            
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=cv_folds, scoring=scoring)
        
        # Generate samples with predictions
        print(f"SAMPLES GENERATION DEBUG:")
        print(f"  X_test shape: {X_test.shape}")
        print(f"  y_test shape: {y_test.shape}")
        print(f"  y_pred shape: {y_pred.shape}")
        print(f"  y_pred sample: {y_pred[:5]}")
        
        try:
            # Create samples from test data with predictions
            X_test_view = X_test.reset_index(drop=True)
            y_test_view = (y_test.reset_index(drop=True) if hasattr(y_test, 'reset_index') else pd.Series(list(y_test)))
            y_pred_view = pd.Series(y_pred)
            
            n = len(X_test_view)
            block = min(10, n)  # Show first 10 samples
            
            def build_samples(indices: List[int]):
                rows: List[Dict[str, Any]] = []
                for i in indices:
                    if i < len(X_test_view) and i < len(y_test_view) and i < len(y_pred_view):
                        # Create input feature map
                        feature_map = {}
                        for col in X_test_view.columns:
                            value = X_test_view.iloc[i][col]
                            if pd.isna(value):
                                feature_map[col] = None
                            elif isinstance(value, (np.integer, np.floating)):
                                if np.isnan(value) or np.isinf(value):
                                    feature_map[col] = None
                                else:
                                    feature_map[col] = float(value)
                            else:
                                feature_map[col] = str(value)
                        
                        # Get actual and predicted values
                        actual_val = y_test_view.iloc[i]
                        pred_val = y_pred_view.iloc[i]
                        
                        # Convert to appropriate types
                        try:
                            if self.task_type == "regression":
                                actual_out = float(actual_val) if not pd.isna(actual_val) else None
                                pred_out = float(pred_val) if not pd.isna(pred_val) else None
                            else:
                                actual_out = str(actual_val) if not pd.isna(actual_val) else None
                                pred_out = str(pred_val) if not pd.isna(pred_val) else None
                        except Exception as e:
                            print(f"Warning: Could not convert values for sample {i}: {e}")
                            actual_out = actual_val
                            pred_out = pred_val
                        
                        rows.append({
                            "input": feature_map,
                            "actual": actual_out,
                            "prediction": pred_out
                        })
                return rows
            
            # Generate samples
            first_idx = list(range(0, min(block, n)))
            middle_start = max((n // 2) - (block // 2), 0)
            middle_idx = list(range(middle_start, min(middle_start + block, n))) if n > block * 2 else []
            last_idx = list(range(max(n - block, 0), n)) if n > block else []
            
            samples = {
                "first": build_samples(first_idx),
                "middle": build_samples(middle_idx),
                "last": build_samples(last_idx)
            }
            
            print(f"  Generated samples: first={len(samples['first'])}, middle={len(samples['middle'])}, last={len(samples['last'])}")
            if samples['first']:
                print(f"  First sample: input={list(samples['first'][0]['input'].keys())}, actual={samples['first'][0]['actual']}, prediction={samples['first'][0]['prediction']}")
            else:
                print(f"  WARNING: No samples generated! X_test shape: {X_test.shape}, y_test shape: {y_test.shape}, y_pred shape: {y_pred.shape}")
            
        except Exception as e:
            print(f"Error generating samples: {e}")
            samples = {"first": [], "middle": [], "last": []}
        
        # Build dataset info
        try:
            # Use the stored original data if available
            if hasattr(self, 'original_data') and self.original_data is not None:
                # Get first 10 rows of original dataset
                first_10_data = self.original_data.head(10)
                
                # Build dataset preview with original values
                dataset_preview = []
                for idx, row in first_10_data.iterrows():
                    row_data = {}
                    for col in first_10_data.columns:
                        value = row[col]
                        if pd.api.types.is_numeric_dtype(first_10_data[col]):
                            row_data[col] = float(value) if not pd.isna(value) else None
                        else:
                            row_data[col] = str(value) if not pd.isna(value) else None
                    dataset_preview.append(row_data)
                
                # Create dataset info
                dataset_info = {
                    "total_rows": len(self.original_data),
                    "columns": list(self.original_data.columns),
                    "data_types": {col: str(dtype) for col, dtype in self.original_data.dtypes.items()},
                    "preview": dataset_preview,
                    "has_more": len(self.original_data) > 10,
                    "input_features": self.original_feature_columns or [],
                    "output_feature": self.target_column
                }
            else:
                # Fallback if original data not available
                dataset_info = {
                    "total_rows": len(X),
                    "columns": list(X.columns),
                    "data_types": {col: str(dtype) for col, dtype in X.dtypes.items()},
                    "preview": [],
                    "has_more": False,
                    "input_features": self.original_feature_columns or [],
                    "output_feature": self.target_column
                }
        except Exception as e:
            print(f"Warning: Could not build dataset info: {e}")
            dataset_info = {
                "total_rows": len(X),
                "columns": list(X.columns),
                "data_types": {col: str(dtype) for col, dtype in X.dtypes.items()},
                "preview": [],
                "has_more": False,
                "input_features": self.original_feature_columns or [],
                "output_feature": self.target_column
            }
        
        return {
            "algorithm": chosen_algo_name,
            "metrics": metrics,
            "cross_validation_score": {
                "mean": cv_scores.mean(),
                "std": cv_scores.std(),
                "scores": cv_scores.tolist()
            },
            "feature_importance": self._get_feature_importance() if hasattr(self.model, 'feature_importances_') else None,
            "samples": samples,
            "dataset_info": dataset_info,
            "data_transformation_info": self._get_data_transformation_info()
        }
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, *, y_pred_proba: Optional[np.ndarray] = None, n_features: Optional[int] = None) -> Dict[str, float]:
        """Calculate performance metrics based on task type"""
        if self.task_type == "regression":
            # IMPROVED: Better R² calculation with debugging
            r2 = r2_score(y_true, y_pred)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_true - y_pred))
            # Adjusted R²
            try:
                n = int(len(y_true))
                p = int(n_features) if n_features is not None else (len(self.feature_columns) if hasattr(self, 'feature_columns') and self.feature_columns is not None else 0)
                adjusted_r2 = None
                if n > p + 1 and p >= 1:
                    adjusted_r2 = 1.0 - (1.0 - float(r2)) * (float(n - 1) / float(n - p - 1))
            except Exception:
                adjusted_r2 = None
            
            # Debug R² calculation
            print(f"R² CALCULATION DEBUG:")
            print(f"  y_true mean: {y_true.mean():.4f}")
            print(f"  y_pred mean: {y_pred.mean():.4f}")
            print(f"  y_true std: {y_true.std():.4f}")
            print(f"  y_pred std: {y_pred.std():.4f}")
            print(f"  Correlation: {np.corrcoef(y_true, y_pred)[0,1]:.4f}")
            print(f"  R² Score: {r2:.6f}")
            
            # Check for potential issues
            if r2 < -1:
                print(f"WARNING: R² < -1 ({r2:.6f}) - Model is worse than predicting mean!")
            elif r2 < 0:
                print(f"WARNING: R² < 0 ({r2:.6f}) - Model is worse than predicting mean!")
            elif r2 < 0.1:
                print(f"WARNING: R² < 0.1 ({r2:.6f}) - Very poor model performance!")
            
            out: Dict[str, float] = {
                "r2_score": r2,
                "mse": mse,
                "rmse": rmse,
                "mae": mae
            }
            if adjusted_r2 is not None and not np.isnan(adjusted_r2):
                out["adjusted_r2"] = adjusted_r2
            return out
        elif self.task_type == "binary_classification":
            from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
            try:
                metrics = {
                    "accuracy": accuracy_score(y_true, y_pred),
                    "precision": precision_score(y_true, y_pred, average='binary', zero_division=0),
                    "recall": recall_score(y_true, y_pred, average='binary', zero_division=0),
                    "f1_score": f1_score(y_true, y_pred, average='binary', zero_division=0)
                }
                
                # Add AUC if probabilities are available
                if y_pred_proba is not None:
                    try:
                        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
                    except Exception:
                        pass
                        
                return metrics
            except Exception as e:
                print(f"Warning: Could not calculate binary classification metrics: {e}")
                return {"accuracy": accuracy_score(y_true, y_pred)}
        else:  # multiclass classification
            from sklearn.metrics import precision_score, recall_score, f1_score
            return {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
                "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
                "f1_score": f1_score(y_true, y_pred, average='weighted', zero_division=0),
                "classification_report": classification_report(y_true, y_pred, output_dict=True)
            }
    
    def _get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance_dict = dict(zip(self.feature_columns, self.model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        return {}
    
    def _get_data_transformation_info(self) -> Dict[str, Any]:
        """Get information about data transformations applied"""
        info = {
            "data_scaling": "Applied scaling to normalize numerical features",
            "categorical_encoding": "Applied encoding to convert categorical variables to numbers",
            "missing_value_handling": "Applied imputation for missing values",
            "note": "All values shown are from your real uploaded dataset - no fake or pre-built values used"
        }
        
        if hasattr(self, 'original_data') and self.original_data is not None:
            # Add specific transformation details
            numeric_cols = self.original_data.select_dtypes(include=[np.number]).columns
            categorical_cols = self.original_data.select_dtypes(include=['object']).columns
            
            info["original_numeric_columns"] = list(numeric_cols)
            info["original_categorical_columns"] = list(categorical_cols)
            info["transformation_applied"] = True
            info["dataset_source"] = "User uploaded dataset"
            info["fake_values"] = False
        else:
            info["transformation_applied"] = False
            info["dataset_source"] = "Unknown"
            info["fake_values"] = True
            
        return info
    
    def predict(self, new_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions using robust pipeline"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        print(f" Making prediction with robust pipeline")
        print(f" Input data: {new_data}")
        
        # Convert to DataFrame
        df = pd.DataFrame([new_data])
        
        # Ensure all required features are present
        missing_features = []
        for col in self.original_feature_columns:
            if col not in df.columns:
                missing_features.append(col)
                df[col] = 0  # Default value for missing features
        
        if missing_features:
            print(f" Missing features filled with defaults: {missing_features}")
        
        # Use robust pipeline for preprocessing
        try:
            X_processed = self.data_pipeline.transform(df)
            print(f" Processed data shape: {X_processed.shape}")
        except Exception as e:
            print(f" Pipeline transformation failed: {e}")
            raise ValueError(f"Data preprocessing failed: {e}")
        
        # Make prediction
        try:
            prediction = self.model.predict(X_processed)[0]
            print(f" Raw prediction: {prediction}")
        except Exception as e:
            print(f" Prediction failed: {e}")
            raise ValueError(f"Model prediction failed: {e}")
        
        # Get prediction confidence/probability if available
        confidence = None
        if hasattr(self.model, 'predict_proba'):
            # IMPORTANT: use the same preprocessed features used for prediction
            try:
                proba = self.model.predict_proba(X_processed)[0]
            except Exception:
                # Fallback: attempt to preprocess again if needed or skip confidence
                try:
                    proba = self.model.predict_proba(self.data_pipeline.transform(df))[0]
                except Exception:
                    proba = None
            if proba is not None:
                confidence = float(max(proba))
        
        # For classification, return the predicted label as-is (strings supported by sklearn)
        formatted_prediction: Any
        if self.task_type == "classification":
            try:
                # If we encoded the target variable, decode it back to original labels
                if hasattr(self, 'target_encoder') and self.target_encoder:
                    try:
                        # Get the original class names
                        original_labels = self.target_encoder.classes_
                        # Get the prediction index
                        pred_idx = int(prediction)
                        if 0 <= pred_idx < len(original_labels):
                            formatted_prediction = original_labels[pred_idx]
                        else:
                            formatted_prediction = prediction
                    except Exception as e:
                        print(f" Could not decode target prediction: {e}")
                        formatted_prediction = prediction
                else:
                    formatted_prediction = prediction
            except Exception:
                formatted_prediction = prediction
        else:  # regression
            try:
                formatted_prediction = float(prediction)
            except Exception:
                formatted_prediction = prediction
        return {
            "prediction": formatted_prediction,
            "confidence": confidence,
            "input_features": new_data
        }
    
    def create_visualizations(self, df: pd.DataFrame, save_dir: str = "./storage/plots") -> List[str]:
        """Create comprehensive visualizations with robust error handling"""
        os.makedirs(save_dir, exist_ok=True)
        plot_files: List[str] = []
        errors: List[str] = []
        
        # Helper to save PNG alongside HTML when possible
        def save_figure_html_and_png(fig, html_path: str):
            try:
                fig.write_html(html_path)
            except Exception as e:
                errors.append(f"write_html_failed:{html_path}:{str(e)}")
            # Try to also write a PNG (requires kaleido)
            try:
                png_path = os.path.splitext(html_path)[0] + ".png"
                fig.write_image(png_path, format="png", scale=2)  # high-res
            except Exception as e:
                # PNG is optional; log but don't fail the flow
                errors.append(f"write_png_failed:{html_path}:{str(e)}")

        # 1. Data distribution plots
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                fig = make_subplots(
                    rows=(len(numeric_cols) + 1) // 2, cols=2,
                    subplot_titles=numeric_cols.tolist()
                )
                for i, col in enumerate(numeric_cols):
                    row = i // 2 + 1
                    col_idx = i % 2 + 1
                    try:
                        fig.add_trace(
                            go.Histogram(x=df[col], name=col, showlegend=False),
                            row=row, col=col_idx
                        )
                    except Exception as e:
                        errors.append(f"histogram_failed:{col}:{str(e)}")
                fig.update_layout(height=300 * ((len(numeric_cols) + 1) // 2), title_text="Data Distribution")
                plot_file = os.path.join(save_dir, "data_distribution.html")
                save_figure_html_and_png(fig, plot_file)
                plot_files.append(plot_file)
        except Exception as e:
            errors.append(f"distribution_failed:{str(e)}")
        
        # 2. Correlation heatmap
        try:
            if len(df.select_dtypes(include=[np.number]).columns) > 1:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                corr_matrix = df[numeric_cols].corr()
                if corr_matrix.shape[0] > 1:
                    fig = go.Figure(data=go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.columns,
                        colorscale='RdBu',
                        zmid=0
                    ))
                    fig.update_layout(title="Correlation Matrix")
                    plot_file = os.path.join(save_dir, "correlation_matrix.html")
                    save_figure_html_and_png(fig, plot_file)
                    plot_files.append(plot_file)
        except Exception as e:
            errors.append(f"correlation_failed:{str(e)}")
        
        # 3. Missing values visualization
        try:
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                fig = go.Figure(data=go.Bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    name="Missing Values"
                ))
                fig.update_layout(title="Missing Values by Column")
                plot_file = os.path.join(save_dir, "missing_values.html")
                save_figure_html_and_png(fig, plot_file)
                plot_files.append(plot_file)
        except Exception as e:
            errors.append(f"missing_values_failed:{str(e)}")
        
        # 4. Feature importance plot
        try:
            if hasattr(self, 'model') and hasattr(self.model, 'feature_importances_'):
                importance = self._get_feature_importance()
                if importance:
                    fig = go.Figure(data=go.Bar(
                        x=list(importance.values()),
                        y=list(importance.keys()),
                        orientation='h'
                    ))
                    fig.update_layout(title="Feature Importance")
                    plot_file = os.path.join(save_dir, "feature_importance.html")
                    save_figure_html_and_png(fig, plot_file)
                    plot_files.append(plot_file)
        except Exception as e:
            errors.append(f"feature_importance_failed:{str(e)}")
        
        # 5. Model performance visualization
        try:
            if hasattr(self, 'model') and hasattr(self, 'metrics'):
                # Create a simple performance metrics visualization
                metrics_data = []
                if hasattr(self, 'cross_validation_scores') and self.cross_validation_scores:
                    metrics_data.append(('Cross-Validation Score', np.mean(self.cross_validation_scores)))
                if hasattr(self, 'metrics') and 'accuracy' in self.metrics:
                    metrics_data.append(('Accuracy', self.metrics['accuracy']))
                if hasattr(self, 'metrics') and 'r2_score' in self.metrics:
                    metrics_data.append(('R Score', self.metrics['r2_score']))
                
                if metrics_data:
                    fig = go.Figure(data=go.Bar(
                        x=[metric[0] for metric in metrics_data],
                        y=[metric[1] for metric in metrics_data]
                    ))
                    fig.update_layout(title="Model Performance Metrics")
                    plot_file = os.path.join(save_dir, "model_performance.html")
                    save_figure_html_and_png(fig, plot_file)
                    plot_files.append(plot_file)
            else:
                # Do not generate any fake/sample performance metrics. Only real metrics are allowed.
                pass
        except Exception as e:
            errors.append(f"model_performance_failed:{str(e)}")
        
        # 6. Feature analysis visualization
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                # Create box plots for feature analysis
                fig = make_subplots(
                    rows=(len(numeric_cols) + 1) // 2, cols=2,
                    subplot_titles=numeric_cols.tolist()
                )
                for i, col in enumerate(numeric_cols):
                    row = i // 2 + 1
                    col_idx = i % 2 + 1
                    try:
                        fig.add_trace(
                            go.Box(y=df[col], name=col, showlegend=False),
                            row=row, col=col_idx
                        )
                    except Exception as e:
                        errors.append(f"boxplot_failed:{col}:{str(e)}")
                fig.update_layout(height=300 * ((len(numeric_cols) + 1) // 2), title_text="Feature Analysis - Box Plots")
                plot_file = os.path.join(save_dir, "feature_analysis.html")
                fig.write_html(plot_file)
                plot_files.append(plot_file)
        except Exception as e:
            errors.append(f"feature_analysis_failed:{str(e)}")
        
        if errors:
            try:
                print(f" Visualization warnings: {errors}")
            except Exception:
                pass
        return plot_files
    
    def save_model(self, filepath: str):
        """Save the trained model with complete pipeline information"""
        try:
            print(f" Saving model: task_type={self.task_type}, target={self.target_column}")
            print(f" Saving model original_feature_columns={self.original_feature_columns}")
            print(f" Saving model feature_columns={self.feature_columns}")
        except Exception:
            pass
        
        # Save complete pipeline information
        model_data = {
            "model": self.model,
            "data_pipeline": self.data_pipeline,  # Save the entire robust pipeline
            "task_type": self.task_type,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
            "original_feature_columns": self.original_feature_columns,
            # Legacy support
            "scaler": getattr(self.data_pipeline, 'scaler', None),
            "encoder": getattr(self.data_pipeline, 'encoders', {}),
            "feature_selector": getattr(self.data_pipeline, 'feature_selector', None),
            "target_encoder": getattr(self.data_pipeline, 'target_encoder', None)
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load a trained model with complete pipeline restoration"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Load the robust data pipeline if available
        if "data_pipeline" in model_data:
            self.data_pipeline = model_data["data_pipeline"]
            print(" Loaded robust data pipeline")
        else:
            # Fallback to legacy loading
            self.data_pipeline = RobustDataPipeline()
            self.data_pipeline.scaler = model_data.get("scaler")
            self.data_pipeline.encoders = model_data.get("encoder", {})
            self.data_pipeline.feature_selector = model_data.get("feature_selector")
            self.data_pipeline.target_encoder = model_data.get("target_encoder")
            print(" Loaded legacy pipeline components")
        
        # Load model and metadata
        self.model = model_data["model"]
        self.task_type = model_data["task_type"]
        self.target_column = model_data["target_column"]
        self.feature_columns = model_data["feature_columns"]
        self.original_feature_columns = model_data.get("original_feature_columns", None)
        
        # Legacy support
        self.scaler = model_data.get("scaler")
        self.encoder = model_data.get("encoder", {})
        self.feature_selector = model_data.get("feature_selector")
        self.target_encoder = model_data.get("target_encoder")
        
        try:
            print(f" Loaded model: task_type={self.task_type}, target={self.target_column}")
            print(f" Loaded model original_feature_columns={self.original_feature_columns}")
            print(f" Loaded model feature_columns={self.feature_columns}")
        except Exception:
            pass


def load_dataset(file_path: str) -> pd.DataFrame:
    """Load dataset from various file formats"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.csv':
        df = pd.read_csv(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    elif file_extension == '.json':
        df = pd.read_json(file_path)
    elif file_extension == '.parquet':
        df = pd.read_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    # Post-load normalization: convert numeric-looking object columns to numeric
    try:
        for col in df.columns:
            s = df[col]
            if s.dtype == 'object':
                coerced = pd.to_numeric(s.astype(str).str.replace(',', '').str.strip(), errors='coerce')
                non_null_ratio = 0.0
                try:
                    non_null_ratio = float(coerced.notnull().mean())
                except Exception:
                    pass
                # If majority of values are numeric-like, adopt coerced numeric series
                if non_null_ratio >= 0.8:
                    df[col] = coerced
    except Exception:
        # Best-effort; ignore conversion issues
        pass

    return df
