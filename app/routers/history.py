from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
import json

from db import get_db
from schemas import HistoryResponse, HistoryItem, DatasetOut
from routers.auth import get_current_user


router = APIRouter()


@router.get("/history", response_model=HistoryResponse)
def history(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    try:
        # Fetch datasets for user (use actual DB columns and alias to API schema)
        datasets_query = text(
            """
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
            """
        )
        dataset_rows = db.execute(datasets_query, {"user_id": current_user.id}).fetchall()
        datasets: List[DatasetOut] = [
            DatasetOut(
                id=row.id,
                filename=row.filename,
                file_size=row.file_size,
                created_at=row.created_at,
                file_path=row.file_path,
                uploaded_by=row.uploaded_by,
            )
            for row in dataset_rows
        ]

        # Fetch trained models for user
        models_query = text(
            """
            SELECT id, dataset_id, task_type, algorithm, metrics, created_at
            FROM trained_models
            WHERE user_id = :user_id
            ORDER BY created_at DESC
            """
        )
        model_rows = db.execute(models_query, {"user_id": current_user.id}).fetchall()

        model_items: List[HistoryItem] = []
        for row in model_rows:
            metrics_obj = {}
            if row.metrics:
                try:
                    metrics_obj = json.loads(row.metrics)
                except Exception:
                    metrics_obj = {"_raw": row.metrics}
            model_items.append(
                HistoryItem(
                    model_id=row.id,
                    algorithm=row.algorithm,
                    task_type=row.task_type,
                    created_at=row.created_at,
                    metrics=metrics_obj,
                )
            )

        return HistoryResponse(datasets=datasets, models=model_items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"History fetch failed: {str(e)}")


