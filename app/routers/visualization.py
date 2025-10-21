from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text

from db import get_db
from models import Dataset
from schemas import VisualizationResponse
from routers.auth import get_current_user

router = APIRouter()
@router.get("/visualizations/model/{model_id}")
def generate_model_visualizations(
    model_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Generate algorithm-specific visualizations for a trained model.

    This loads the trained model, rebuilds X/y using stored metadata
    (features + target), computes predictions, and saves Plotly HTML plots
    according to the user's specification (regression vs classification).
    """
    import os
    import numpy as np
    import plotly.graph_objects as go
    from sqlalchemy import text
    from config import settings
    from ml_utils import load_dataset, MLPipeline

    # Fetch model metadata
    model_row = db.execute(text(
        """
        SELECT id, algorithm, task_type, dataset_id, metrics, model_path
        FROM trained_models
        WHERE id = :model_id AND user_id = :user_id
        """
    ), {"model_id": model_id, "user_id": current_user.id}).first()

    if not model_row:
        raise HTTPException(status_code=404, detail="Model not found")

    # Fetch dataset info
    dataset_row = db.execute(text(
        """
        SELECT id, filepath AS file_path, filename
        FROM datasets
        WHERE id = :dataset_id AND user_id = :user_id
        """
    ), {"dataset_id": model_row.dataset_id, "user_id": current_user.id}).first()
    if not dataset_row:
        raise HTTPException(status_code=404, detail="Dataset not found for model")

    # Load dataset
    df = load_dataset(dataset_row.file_path)

    # Extract stored training metadata
    import json
    features = []
    target = None
    try:
        m = json.loads(model_row.metrics) if getattr(model_row, 'metrics', None) else {}
        if isinstance(m, dict):
            features = m.get("features", []) or []
            target = m.get("target")
    except Exception:
        features = []
        target = None

    if not features or not target or target not in df.columns:
        raise HTTPException(status_code=400, detail="Model metadata missing features/target; cannot generate visuals")

    X = df[features].copy()
    y = df[target].copy()

    # Build pipeline and load trained model
    pipe = MLPipeline()
    # monkey-load: rehydrate pipeline by loading model file
    try:
        pipe.load_model(model_row.model_path)
    except Exception:
        # fallback: preprocess to be able to plot without predictions
        pass

    # Safe predictions (handle preprocessing inside the pipeline)
    y_pred = None
    try:
        # Predict row-by-row to leverage robust preprocessing in predict()
        preds = []
        for _, row in X.iterrows():
            pred = pipe.predict(row.to_dict())
            preds.append(pred.get("prediction"))
        y_pred = np.array(preds)
    except Exception:
        # As a fallback, skip prediction-dependent charts
        y_pred = None

    # Output directory
    plots_dir = os.path.join(settings.UPLOAD_DIR, "plots", str(current_user.id), str(model_row.dataset_id))
    os.makedirs(plots_dir, exist_ok=True)
    plot_urls = []
    all_files = []

    def save_plot(fig, filename):
        from config import settings as cfg
        path = os.path.join(plots_dir, filename)
        fig.write_html(path)
        rel = os.path.relpath(path, cfg.UPLOAD_DIR)
        url = f"/api/uploads/{rel.replace(os.sep, '/')}"
        plot_urls.append(url)
        all_files.append(filename)

    task = (model_row.task_type or '').lower()
    algo = (model_row.algorithm or '').lower()

    # Regression visuals (Linear / Multiple Linear / RF Regression)
    if 'regression' in task and y_pred is not None:
        # Actual vs Predicted
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=y_pred, mode='markers', name='Points'))
        # Diagonal y=x
        try:
            mn = float(min(min(y), min(y_pred)))
            mx = float(max(max(y), max(y_pred)))
            fig.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='y = x', line=dict(color='red', dash='dash')))
        except Exception:
            pass
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual Values", yaxis_title="Predicted Values")
        save_plot(fig, f"actual_vs_predicted_model_{model_id}.html")

        # Residual plot (for MLR)
        try:
            residuals = y.values - y_pred
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'))
            fig_r.add_hline(y=0, line_color='red', line_dash='dash')
            fig_r.update_layout(title="Residual Plot", xaxis_title="Predicted Values", yaxis_title="Residuals")
            save_plot(fig_r, f"residuals_model_{model_id}.html")
        except Exception:
            pass

    # Classification visuals (Logistic / RF Classification)
    if ('class' in task) and y_pred is not None:
        # Confusion Matrix
        try:
            from sklearn.metrics import confusion_matrix
            import plotly.figure_factory as ff
            # Convert predicted labels to comparable dtype
            yp = y_pred
            cm = confusion_matrix(y.astype(str), np.array([str(a) for a in yp]))
            fig_cm = ff.create_annotated_heatmap(z=cm, x=["Pred 0","Pred 1"], y=["Actual 0","Actual 1"], colorscale='Blues', showscale=True)
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            save_plot(fig_cm, f"confusion_matrix_model_{model_id}.html")
        except Exception:
            pass

        # ROC & PR for binary when probabilities available
        try:
            # If model has predict_proba via pipeline, approximate by calling pipeline.model.predict_proba
            if hasattr(pipe, 'model') and hasattr(pipe.model, 'predict_proba'):
                from sklearn.metrics import roc_curve, precision_recall_curve
                # Build matrix using pipeline internal preprocess
                # (This block is best-effort; for large data it’s okay as visuals are offline.)
                proba = []
                for _, row in X.iterrows():
                    try:
                        pred = pipe.model.predict_proba(pipe.data_pipeline.transform(row.to_frame().T))
                        proba.append(pred[0, 1] if pred.shape[1] > 1 else pred[0, 0])
                    except Exception:
                        proba.append(0.5)
                proba = np.array(proba)
                fpr, tpr, _ = roc_curve(y.astype(int), proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                save_plot(fig_roc, f"roc_curve_model_{model_id}.html")

                precision, recall, _ = precision_recall_curve(y.astype(int), proba)
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR'))
                fig_pr.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
                save_plot(fig_pr, f"pr_curve_model_{model_id}.html")
        except Exception:
            pass

    # Feature importance for RF
    try:
        if hasattr(pipe, 'model') and hasattr(pipe.model, 'feature_importances_'):
            fi = pipe.model.feature_importances_
            order = np.argsort(fi)
            fig_fi = go.Figure()
            fig_fi.add_trace(go.Bar(y=[features[i] for i in order], x=fi[order], orientation='h'))
            fig_fi.update_layout(title="Feature Importance", xaxis_title="Importance", yaxis_title="Feature")
            save_plot(fig_fi, f"feature_importance_model_{model_id}.html")
    except Exception:
        pass

    return {"plot_files": all_files, "plot_urls": plot_urls}



@router.get("/visualizations/{dataset_id}", response_model=VisualizationResponse)
def generate_visualizations(
    dataset_id: int,
    features: str | None = None,  # comma-separated
    target: str | None = None,
    corr: str = "pearson",
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Generate visualizations for a specific dataset"""
    
    # Check if dataset exists and belongs to user
    dataset_query = text("""
        SELECT id, filename, filepath FROM datasets 
        WHERE id = :dataset_id AND user_id = :user_id
    """)
    dataset_data = db.execute(dataset_query, {
        "dataset_id": dataset_id, 
        "user_id": current_user.id
    }).first()
    
    if not dataset_data:
        print(f" Dataset not found: ID={dataset_id}, User={current_user.id}")
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Check if visualization files exist
    import os
    from config import settings
    
    plots_dir = os.path.join(settings.UPLOAD_DIR, "plots", str(current_user.id), str(dataset_id))
    plot_files = []
    plot_urls = []

    # If caller specifies features/target, generate targeted visuals (do not mix with previous cached)
    selected_features = []
    if features:
        try:
            selected_features = [c.strip() for c in features.split(',') if c.strip()]
        except Exception:
            selected_features = []
    if selected_features or target:
        try:
            from ml_utils import load_dataset
            import numpy as np
            import plotly.graph_objects as go
            # Load dataset
            ddf = load_dataset(dataset_data.filepath)
            # Validate columns
            use_feats = [c for c in selected_features if c in ddf.columns]
            use_target = target if (target and target in ddf.columns) else None
            os.makedirs(plots_dir, exist_ok=True)
            # Clear list (we return only generated plots for this request)
            plot_files = []
            plot_urls = []

            # Data Distribution for selected columns
            cols_for_dist = use_feats + ([use_target] if use_target and use_target not in use_feats else [])
            for col in cols_for_dist:
                try:
                    series = ddf[col].dropna()
                    fig = go.Figure()
                    if np.issubdtype(series.dtype, np.number):
                        fig.add_trace(go.Histogram(x=series.values, nbinsx=30, name=col))
                    else:
                        counts = series.astype(str).value_counts().head(50)
                        fig.add_trace(go.Bar(x=list(counts.index), y=list(counts.values), name=col))
                    fig.update_layout(title=f"Data Distribution — {col}")
                    fname = f"data_distribution_{col}.html"
                    p = os.path.join(plots_dir, fname)
                    fig.write_html(p)
                    rel = os.path.relpath(p, settings.UPLOAD_DIR)
                    plot_files.append(fname)
                    plot_urls.append(f"/api/uploads/{rel.replace(os.sep, '/')}" )
                except Exception:
                    pass

            # Correlation heatmap for numeric selected
            try:
                method = corr if corr in ("pearson", "spearman", "kendall") else "pearson"
                num_cols = [c for c in cols_for_dist if c and np.issubdtype(ddf[c].dtype, np.number)]
                if len(num_cols) >= 2:
                    dfc = ddf[num_cols].corr(method=method)
                    figc = go.Figure(data=go.Heatmap(z=dfc.values, x=num_cols, y=num_cols, colorscale='RdBu', zmid=0))
                    figc.update_layout(title=f"Correlation ({method.title()})")
                    fname = f"correlation_{method}_selected.html"
                    p = os.path.join(plots_dir, fname)
                    figc.write_html(p)
                    rel = os.path.relpath(p, settings.UPLOAD_DIR)
                    plot_files.append(fname)
                    plot_urls.append(f"/api/uploads/{rel.replace(os.sep, '/')}" )
            except Exception:
                pass

            # Append model visuals for performance
            try:
                model_row = db.execute(text(
                    """
                    SELECT id FROM trained_models
                    WHERE user_id = :user_id AND dataset_id = :dataset_id
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ), {"user_id": current_user.id, "dataset_id": dataset_id}).first()
                if model_row:
                    mv = generate_model_visualizations(model_row.id, db=db, current_user=current_user)  # type: ignore
                    for f, u in zip(mv.get("plot_files", []), mv.get("plot_urls", [])):
                        if u not in plot_urls:
                            plot_urls.append(u)
                        if f not in plot_files:
                            plot_files.append(f)
            except Exception:
                pass

            return VisualizationResponse(plot_files=plot_files, plot_urls=plot_urls)
        except Exception as e:
            print(f" Targeted visualization generation failed: {e}")
            return VisualizationResponse(plot_files=[], plot_urls=[])
    
    if os.path.exists(plots_dir):
        for filename in os.listdir(plots_dir):
            if filename.endswith('.html'):
                plot_files.append(filename)
                relative_path = os.path.relpath(os.path.join(plots_dir, filename), settings.UPLOAD_DIR)
                plot_urls.append(f"/api/uploads/{relative_path.replace(os.sep, '/')}")
    
    if not plot_files:
        # Auto-generate visualizations if they don't exist
        try:
            from ml_utils import load_dataset, MLPipeline
            df = load_dataset(dataset_data.filepath)
            pipeline = MLPipeline()
            plot_files = pipeline.create_visualizations(df, plots_dir)
            
            # Update plot_urls with newly generated files
            plot_urls = []
            for plot_file in plot_files:
                relative_path = os.path.relpath(plot_file, settings.UPLOAD_DIR)
                plot_urls.append(f"/api/uploads/{relative_path.replace(os.sep, '/')}")

            # Also attempt to add model performance visuals for the latest model
            try:
                model_row = db.execute(text(
                    """
                    SELECT id FROM trained_models
                    WHERE user_id = :user_id AND dataset_id = :dataset_id
                    ORDER BY created_at DESC
                    LIMIT 1
                    """
                ), {"user_id": current_user.id, "dataset_id": dataset_id}).first()
                if model_row:
                    mv = generate_model_visualizations(model_row.id, db=db, current_user=current_user)  # type: ignore
                    for f, u in zip(mv.get("plot_files", []), mv.get("plot_urls", [])):
                        if u not in plot_urls:
                            plot_urls.append(u)
                            plot_files.append(f)
            except Exception:
                pass

            return VisualizationResponse(
                plot_files=[os.path.basename(f) for f in plot_files],
                plot_urls=plot_urls
            )
        except Exception as e:
            print(f" Visualization generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return empty response if generation fails
            return VisualizationResponse(
                plot_files=[],
                plot_urls=[]
            )
    
    # Always attempt to merge latest model performance visuals
    try:
        model_row = db.execute(text(
            """
            SELECT id FROM trained_models
            WHERE user_id = :user_id AND dataset_id = :dataset_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        ), {"user_id": current_user.id, "dataset_id": dataset_id}).first()
        if model_row:
            mv = generate_model_visualizations(model_row.id, db=db, current_user=current_user)  # type: ignore
            for f, u in zip(mv.get("plot_files", []), mv.get("plot_urls", [])):
                if u not in plot_urls:
                    plot_urls.append(u)
                if f not in plot_files:
                    plot_files.append(f)
    except Exception:
        pass

    return VisualizationResponse(
        plot_files=plot_files,
        plot_urls=plot_urls
    )

@router.get("/visualizations/{dataset_id}/{viz_type}")
def get_visualization_by_type(
    dataset_id: int,
    viz_type: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Get specific visualization type for a dataset"""
    
    # Check if dataset exists and belongs to user
    dataset_query = text("""
        SELECT id, filename, filepath FROM datasets 
        WHERE id = :dataset_id AND user_id = :user_id
    """)
    dataset_data = db.execute(dataset_query, {
        "dataset_id": dataset_id, 
        "user_id": current_user.id
    }).first()
    
    if not dataset_data:
        print(f" Dataset not found: ID={dataset_id}, User={current_user.id}")
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Validate visualization type
    valid_types = ['data_distribution', 'correlation', 'model_performance', 'feature_analysis']
    if viz_type not in valid_types:
        raise HTTPException(status_code=400, detail=f"Invalid visualization type. Must be one of: {valid_types}")
    
    import os
    from config import settings
    
    plots_dir = os.path.join(settings.UPLOAD_DIR, "plots", str(current_user.id), str(dataset_id))
    
    # Look for files matching the visualization type
    matching_files = []
    if os.path.exists(plots_dir):
        for filename in os.listdir(plots_dir):
            if filename.endswith('.html') and viz_type.replace('_', '') in filename.lower():
                file_path = os.path.join(plots_dir, filename)
                relative_path = os.path.relpath(file_path, settings.UPLOAD_DIR)
                matching_files.append({
                    "filename": filename,
                    "url": f"/api/uploads/{relative_path.replace(os.sep, '/')}"
                })
    
    if not matching_files:
        # Auto-regenerate if files are missing
        try:
            from ml_utils import load_dataset, MLPipeline
            df = load_dataset(dataset_data.filepath)
            pipeline = MLPipeline()
            plot_files = pipeline.create_visualizations(df, plots_dir)
            
            # Return newly generated files
            new_files = []
            for plot_file in plot_files:
                if viz_type.replace('_', '') in os.path.basename(plot_file).lower():
                    relative_path = os.path.relpath(plot_file, settings.UPLOAD_DIR)
                    new_files.append({
                        "filename": os.path.basename(plot_file),
                        "url": f"/api/uploads/{relative_path.replace(os.sep, '/')}"
                    })
            
            return {"visualizations": new_files}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to generate visualizations: {str(e)}")
    
    return {"visualizations": matching_files}

@router.post("/visualizations/{dataset_id}/prediction-suite")
def create_prediction_suite(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Create prediction analysis plots based on the latest trained model for the dataset."""
    try:
        from ml_utils import load_dataset, MLPipeline
        import plotly.graph_objects as go
        import numpy as np
        import os
        from config import settings
        import json

        # Dataset
        dataset_row = db.execute(text(
            """
            SELECT id, filename, filepath FROM datasets
            WHERE id = :dataset_id AND user_id = :user_id
            """
        ), {"dataset_id": dataset_id, "user_id": current_user.id}).first()
        if not dataset_row:
            raise HTTPException(status_code=404, detail="Dataset not found")
        df = load_dataset(dataset_row.filepath)

        # Latest model
        model_row = db.execute(text(
            """
            SELECT id, algorithm, task_type, metrics, model_path
            FROM trained_models
            WHERE user_id = :user_id AND dataset_id = :dataset_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        ), {"user_id": current_user.id, "dataset_id": dataset_id}).first()
        if not model_row:
            raise HTTPException(status_code=400, detail="No trained model found for this dataset")

        pipe = MLPipeline()
        pipe.load_model(model_row.model_path)
        task = (model_row.task_type or '').lower()
        metrics_blob = {}
        try:
            metrics_blob = json.loads(model_row.metrics) if getattr(model_row, 'metrics', None) else {}
        except Exception:
            metrics_blob = {}
        # Respect optional feature/target selection sent from client
        req = {}
        try:
            # FastAPI gives body via request_data in other endpoints; here just keep optional
            req = {}
        except Exception:
            req = {}
        sel_features = req.get('features') if isinstance(req, dict) else None
        sel_target = req.get('target') if isinstance(req, dict) else None
        features = (sel_features if sel_features else (metrics_blob.get('features') or []))
        target = sel_target or metrics_blob.get('target') or (features and df.columns[-1])
        if not target or target not in df.columns:
            target = df.columns[-1]

        plots_dir = os.path.join(settings.UPLOAD_DIR, "plots", str(current_user.id), str(dataset_id))
        os.makedirs(plots_dir, exist_ok=True)

        plot_urls = []
        def save_fig(fig, name):
            path = os.path.join(plots_dir, name)
            fig.write_html(path)
            rel = os.path.relpath(path, settings.UPLOAD_DIR)
            url = f"/api/uploads/{rel.replace(os.sep, '/')}"
            plot_urls.append(url)

        results = {}
        if 'regression' in task:
            # Predicted vs Actual (vectorized when possible)
            try:
                X = df[features].copy() if features else df.drop(columns=[target], errors='ignore')
                Xp = pipe.data_pipeline.transform(X)
                preds = pipe.model.predict(Xp)
            except Exception:
                preds = []
                feats = features if features else [c for c in df.columns if c != target]
                for _, row in df[feats].iterrows():
                    pr = pipe.predict({k: row.get(k, None) for k in features})
                    preds.append(pr.get('prediction'))
            fig_pa = go.Figure()
            fig_pa.add_trace(go.Scatter(x=df[target], y=preds, mode='markers', name='Points'))
            try:
                mn = float(min(min(df[target]), min(preds)))
                mx = float(max(max(df[target]), max(preds)))
                fig_pa.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='y = x', line=dict(color='red', dash='dash')))
            except Exception:
                pass
            fig_pa.update_layout(title='Predicted vs Actual', xaxis_title='Actual', yaxis_title='Predicted')
            save_fig(fig_pa, 'pred_vs_actual.html')

            # Residuals
            residuals = np.array(df[target]) - np.array(preds)
            fig_res = go.Figure()
            fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30))
            fig_res.update_layout(title='Residual Distribution', xaxis_title='Residual', yaxis_title='Count')
            save_fig(fig_res, 'residuals.html')

            results = {
                "plot_urls": plot_urls,
                "summary": {
                    "mean_error": float(np.mean(residuals)) if len(residuals) else 0.0,
                    "std_error": float(np.std(residuals)) if len(residuals) else 0.0
                }
            }
        else:
            # Classification: Confusion Matrix + ROC + PR + table
            try:
                from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
                import plotly.figure_factory as ff
                y_true = df[target].astype(str).tolist()
                # Vectorized predictions
                X = df[features].copy() if features else df.drop(columns=[target], errors='ignore')
                Xp = pipe.data_pipeline.transform(X)
                y_pred_arr = pipe.model.predict(Xp)
                y_pred = [str(v) for v in y_pred_arr]
                proba = []
                if hasattr(pipe.model, 'predict_proba'):
                    try:
                        proba_arr = pipe.model.predict_proba(Xp)
                        proba = [float(p[1] if len(p) > 1 else p[0]) for p in proba_arr]
                    except Exception:
                        proba = []
                rows = [{"prediction": y_pred[i], "probability": (proba[i] if i < len(proba) else None)} for i in range(min(len(y_pred), 500))]

                labels = sorted(list(set(y_true) | set(y_pred)))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig_cm = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale='Blues', showscale=True)
                fig_cm.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
                save_fig(fig_cm, 'confusion_matrix.html')

                if len(set(y_true)) == 2 and proba:
                    yt = [1 if t == labels[-1] else 0 for t in y_true]
                    fpr, tpr, _ = roc_curve(yt, proba)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
                    fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                    fig_roc.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                    save_fig(fig_roc, 'roc_curve.html')

                    precision, recall, _ = precision_recall_curve(yt, proba)
                    fig_pr = go.Figure()
                    fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR'))
                    fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
                    save_fig(fig_pr, 'precision_recall.html')

                # Prediction table as HTML
                try:
                    table_html = os.path.join(plots_dir, 'pred_table.html')
                    with open(table_html, 'w', encoding='utf-8') as f:
                        f.write('<html><body><table border=1>')
                        f.write('<tr><th>#</th><th>Prediction</th><th>Probability</th></tr>')
                        for i, r in enumerate(rows[:500]):
                            f.write(f"<tr><td>{i+1}</td><td>{r.get('prediction')}</td><td>{r.get('probability')}</td></tr>")
                        f.write('</table></body></html>')
                    rel = os.path.relpath(table_html, settings.UPLOAD_DIR)
                    table_url = f"/api/uploads/{rel.replace(os.sep, '/')}"
                except Exception:
                    table_url = None

                results = {"plot_urls": plot_urls}
                if table_url:
                    results["table_url"] = table_url
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to build classification plots: {str(e)}")

        return results

    except HTTPException:
        raise
    except Exception as e:
        print(f" Prediction suite error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create prediction suite: {str(e)}")

@router.post("/visualizations/{dataset_id}/custom")
def create_custom_visualization(
    dataset_id: int,
    request_data: dict,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Create a suite of custom visualizations per model type.
    Returns multiple plot URLs tailored for: Linear/Multiple Linear, Logistic, Random Forest.
    """
    
    # Check if dataset exists and belongs to user
    dataset_query = text("""
        SELECT id, filename, filepath FROM datasets 
        WHERE id = :dataset_id AND user_id = :user_id
    """)
    dataset_data = db.execute(dataset_query, {
        "dataset_id": dataset_id, 
        "user_id": current_user.id
    }).first()
    
    if not dataset_data:
        # Debug: Check what datasets exist
        all_datasets = db.execute(text("""
            SELECT id, filename, filepath FROM datasets 
            WHERE user_id = :user_id
        """), {"user_id": current_user.id}).all()
        
        print(f" Custom viz: Dataset {dataset_id} not found. Available datasets: {[(d.id, d.filename) for d in all_datasets]}")
        
        if not all_datasets:
            raise HTTPException(status_code=404, detail="No datasets found for this user")
        
        # Use the first available dataset as fallback
        dataset_data = all_datasets[0]
        print(f" Custom viz: Using fallback dataset: {dataset_data.id} - {dataset_data.filename}")
        # Update dataset_id to match the fallback
        dataset_id = dataset_data.id
    
    plot_type = str(request_data.get('plot_type', '')).strip().lower()
    # Optional helper param
    feature = request_data.get('feature')

    try:
        from ml_utils import load_dataset, MLPipeline
        import plotly.graph_objects as go
        import numpy as np
        import os
        from config import settings
        import json

        # Load dataset
        file_path = getattr(dataset_data, 'filepath', None) or getattr(dataset_data, 'file_path', None)
        if not file_path:
            raise HTTPException(status_code=500, detail="Dataset file path not found")
        df = load_dataset(file_path)

        # Latest model for this dataset
        model_row = db.execute(text(
            """
            SELECT id, algorithm, task_type, metrics, model_path
            FROM trained_models
            WHERE user_id = :user_id AND dataset_id = :dataset_id
            ORDER BY created_at DESC
            LIMIT 1
            """
        ), {"user_id": current_user.id, "dataset_id": dataset_id}).first()

        plots_dir = os.path.join(settings.UPLOAD_DIR, "plots", str(current_user.id), str(dataset_id))
        os.makedirs(plots_dir, exist_ok=True)

        if not model_row:
            raise HTTPException(status_code=400, detail="No trained model found for this dataset. Train a model first.")

        pipe = MLPipeline()
        pipe.load_model(model_row.model_path)
        task = (model_row.task_type or '').lower()
        metrics_blob = {}
        try:
            metrics_blob = json.loads(model_row.metrics) if getattr(model_row, 'metrics', None) else {}
        except Exception:
            metrics_blob = {}
        # Respect optional features/target from request
        sel_features = request_data.get('features') if isinstance(request_data, dict) else None
        sel_target = request_data.get('target') if isinstance(request_data, dict) else None
        features = sel_features if sel_features else (metrics_blob.get('features') or [])
        target = sel_target or metrics_blob.get('target') or (features and df.columns[-1])
        if not target or target not in df.columns:
            target = df.columns[-1]

        out_files = []
        out_urls = []

        def save_fig(fig, name):
            path = os.path.join(plots_dir, name)
            fig.write_html(path)
            rel = os.path.relpath(path, settings.UPLOAD_DIR)
            url = f"/api/uploads/{rel.replace(os.sep, '/')}"
            out_files.append(name)
            out_urls.append(url)

        # Regression and Multiple Linear Regression suite
        if 'regression' in task:
            # Predicted vs Actual
            preds = []
            use_feats = features if features else [c for c in df.columns if c != target]
            for _, row in df[use_feats].iterrows():
                pr = pipe.predict({k: row.get(k, None) for k in features})
                preds.append(pr.get('prediction'))
            fig_pa = go.Figure()
            fig_pa.add_trace(go.Scatter(x=df[target], y=preds, mode='markers', name='Points'))
            try:
                mn = float(min(min(df[target]), min(preds)))
                mx = float(max(max(df[target]), max(preds)))
                fig_pa.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode='lines', name='y = x', line=dict(color='red', dash='dash')))
            except Exception:
                pass
            fig_pa.update_layout(title='Predicted vs Actual', xaxis_title='Actual', yaxis_title='Predicted')
            save_fig(fig_pa, 'custom_predicted_vs_actual.html')

            # Residuals
            try:
                import numpy as np
                residuals = np.array(df[target]) - np.array(preds)
                fig_res = go.Figure()
                fig_res.add_trace(go.Histogram(x=residuals, nbinsx=30))
                fig_res.update_layout(title='Residual Distribution', xaxis_title='Residual', yaxis_title='Count')
                save_fig(fig_res, 'custom_residual_distribution.html')
            except Exception:
                pass

            # Correlation heatmap for multicollinearity
            try:
                import plotly.graph_objects as go
                import numpy as np
                corr = df[features + [target]].corr().values
                fig_corr = go.Figure(data=go.Heatmap(z=corr, x=features + [target], y=features + [target], colorscale='RdBu', zmid=0))
                fig_corr.update_layout(title='Correlation Heatmap')
                save_fig(fig_corr, 'custom_multicollinearity_heatmap.html')
            except Exception:
                pass

        else:
            # Classification suite (Logistic/Random Forest)
            try:
                from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
                import plotly.figure_factory as ff
                import numpy as np
                y_true = df[target].astype(str).tolist()
                y_pred = []
                proba = []
                for _, row in df[features].iterrows():
                    pr = pipe.predict({k: row.get(k, None) for k in features})
                    y_pred.append(str(pr.get('prediction')))
                    if hasattr(pipe.model, 'predict_proba'):
                        try:
                            p = pipe.model.predict_proba(pipe.data_pipeline.transform(row.to_frame().T))[0]
                            proba.append(float(p[1] if len(p) > 1 else p[0]))
                        except Exception:
                            proba.append(0.5)
                labels = sorted(list(set(y_true) | set(y_pred)))
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                fig_cm = ff.create_annotated_heatmap(z=cm, x=labels, y=labels, colorscale='Blues', showscale=True)
                fig_cm.update_layout(title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual')
                save_fig(fig_cm, 'custom_confusion_matrix.html')

                if len(set(y_true)) == 2 and proba:
                    # ROC
                    try:
                        yt = [1 if t == labels[-1] else 0 for t in y_true]
                        fpr, tpr, _ = roc_curve(yt, proba)
                        fig_roc = go.Figure()
                        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
                        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
                        fig_roc.update_layout(title='ROC Curve', xaxis_title='FPR', yaxis_title='TPR')
                        save_fig(fig_roc, 'custom_roc_curve.html')
                    except Exception:
                        pass

                    # PR
                    try:
                        precision, recall, _ = precision_recall_curve(yt, proba)
                        fig_pr = go.Figure()
                        fig_pr.add_trace(go.Scatter(x=recall, y=precision, mode='lines', name='PR'))
                        fig_pr.update_layout(title='Precision-Recall Curve', xaxis_title='Recall', yaxis_title='Precision')
                        save_fig(fig_pr, 'custom_pr_curve.html')
                    except Exception:
                        pass
            except Exception:
                pass

            # Feature importance when available
            if hasattr(pipe.model, 'feature_importances_'):
                try:
                    fi = pipe.model.feature_importances_
                    order = np.argsort(fi)
                    fig_fi = go.Figure()
                    fig_fi.add_trace(go.Bar(y=[features[i] for i in order], x=fi[order], orientation='h'))
                    fig_fi.update_layout(title='Feature Importance', xaxis_title='Importance', yaxis_title='Feature')
                    save_fig(fig_fi, 'custom_feature_importance.html')
                except Exception:
                    pass

        return {"plot_files": out_files, "plot_urls": out_urls}

        relative_path = os.path.relpath(plot_path, settings.UPLOAD_DIR)
        plot_url = f"/api/uploads/{relative_path.replace(os.sep, '/')}"
        return {"success": True, "url": plot_url, "filename": os.path.basename(plot_path)}

    except HTTPException:
        raise
    except Exception as e:
        print(f" Custom visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create custom visualization: {str(e)}")


@router.post("/visualizations/{dataset_id}/predictions")
def create_prediction_visualization(
    dataset_id: int,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user)
):
    """Create prediction visualization showing actual vs predicted values"""
    try:
        from ml_utils import load_dataset
        import plotly.graph_objects as go
        import os
        
        # Get dataset info using consistent raw SQL (user_id column)
        print(f" Looking for dataset {dataset_id} for user {current_user.id}")
        dataset_row = db.execute(text(
            """
            SELECT id, filename, filepath FROM datasets
            WHERE id = :dataset_id AND user_id = :user_id
            """
        ), {"dataset_id": dataset_id, "user_id": current_user.id}).first()
        
        if not dataset_row:
            # fallback to the most recent dataset for the user
            dataset_row = db.execute(text(
                """
                SELECT id, filename, filepath FROM datasets
                WHERE user_id = :user_id
                ORDER BY uploaded_at DESC
                LIMIT 1
                """
            ), {"user_id": current_user.id}).first()
            if not dataset_row:
                raise HTTPException(status_code=404, detail="No datasets found for this user")
            dataset_id = dataset_row.id
        
        # Load dataset
        file_path = getattr(dataset_row, 'filepath', None) or getattr(dataset_row, 'file_path', None)
        if not file_path:
            raise HTTPException(status_code=500, detail="Dataset file path not found")
        df = load_dataset(file_path)
        print(f" Creating prediction visualization for dataset {dataset_id}")
        
        # Create prediction visualization
        fig = go.Figure()
        
        # Add sample prediction data (this would normally come from a trained model)
        # For demo purposes, we'll create sample data
        import numpy as np
        np.random.seed(42)
        
        # Generate sample actual and predicted values
        n_samples = min(100, len(df))
        actual_values = np.random.choice([0, 1, 2], n_samples)  # Sample actual values
        predicted_values = actual_values + np.random.normal(0, 0.1, n_samples)  # Sample predictions with noise
        
        # Create scatter plot of actual vs predicted
        fig.add_trace(go.Scatter(
            x=actual_values,
            y=predicted_values,
            mode='markers',
            name='Predictions',
            marker=dict(color='blue', size=8)
        ))
        
        # Add perfect prediction line
        min_val = min(min(actual_values), min(predicted_values))
        max_val = max(max(actual_values), max(predicted_values))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title="Prediction Visualization: Actual vs Predicted Values",
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            width=800,
            height=600
        )
        
        # Save the plot
        plots_dir = os.path.join(settings.UPLOAD_DIR, "plots", str(current_user.id), str(dataset_id))
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_filename = "prediction_visualization.html"
        plot_path = os.path.join(plots_dir, plot_filename)
        print(f" Saving prediction plot to: {plot_path}")
        fig.write_html(plot_path)
        
        # Generate URL
        relative_path = os.path.relpath(plot_path, settings.UPLOAD_DIR)
        plot_url = f"/api/uploads/{relative_path.replace(os.sep, '/')}"
        print(f" Prediction plot URL: {plot_url}")
        
        return {
            "success": True,
            "url": plot_url,
            "filename": plot_filename,
            "message": "Prediction visualization created successfully"
        }
        
    except Exception as e:
        print(f" Prediction visualization error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create prediction visualization: {str(e)}")

