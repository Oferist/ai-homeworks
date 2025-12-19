from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Dict, Optional

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from .core import (
    DatasetSummary,
    compute_quality_flags,
    missing_table,
    summarize_dataset,
)

# Настройка простого логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="EDA Quality Service",
    description="HTTP-сервис для оценки качества данных (HW04)",
    version="0.1.0",
)


# --- Pydantic Models ---

class QualityRequest(BaseModel):
    n_rows: int
    n_cols: int
    missing_share: float


class QualityResponse(BaseModel):
    quality_score: float
    ok_for_model: bool
    latency_ms: float
    flags: Dict[str, Any] = {}


# --- Helper Functions ---

def _read_csv_upload(file: UploadFile) -> pd.DataFrame:
    """Читает загруженный файл в DataFrame."""
    try:
        # Считываем CSV
        df = pd.read_csv(file.file)
        return df
    except Exception as e:
        logger.error(f"Error reading CSV: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {e}")


# --- Endpoints ---

@app.get("/health")
def health() -> Dict[str, str]:
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "service": "eda-cli-api"}


@app.post("/quality", response_model=QualityResponse)
def predict_quality(req: QualityRequest):
    """
    Расчет качества на основе переданных метаданных (без самого файла).
    """
    start_ts = time.time()
    
    # Простая логика скоринга на основе переданных чисел
    score = 1.0
    score -= req.missing_share
    
    if req.n_rows < 100:
        score -= 0.1
    
    score = max(0.0, min(1.0, score))
    
    latency = (time.time() - start_ts) * 1000
    
    return QualityResponse(
        quality_score=round(score, 3),
        ok_for_model=(score > 0.7),
        latency_ms=round(latency, 2),
        flags={
            "too_few_rows": req.n_rows < 100,
            "too_many_missing": req.missing_share > 0.5
        }
    )


@app.post("/quality-from-csv")
def quality_from_csv(file: UploadFile = File(...)):
    """
    Принимает CSV, прогоняет через eda-core и возвращает общий скор.
    """
    request_id = uuid.uuid4()
    start_ts = time.time()
    
    df = _read_csv_upload(file)
    
    # Используем функции ядра
    summary: DatasetSummary = summarize_dataset(df)
    missing_df = missing_table(df)
    
    # Вычисляем флаги (включая новые эвристики)
    flags = compute_quality_flags(summary, missing_df, df=df)
    
    latency = (time.time() - start_ts) * 1000
    
    logger.info(f"REQ_ID={request_id} endpoint=quality-from-csv rows={len(df)} score={flags.get('quality_score')}")

    return {
        "filename": file.filename,
        "n_rows": summary.n_rows,
        "n_cols": summary.n_cols,
        "quality_score": round(flags["quality_score"], 3),
        "ok_for_model": flags["quality_score"] > 0.7,
        "latency_ms": round(latency, 2)
    }


# --- CUSTOM ENDPOINT FOR HW04 ---

@app.post("/quality-flags-from-csv")
def quality_flags_from_csv(file: UploadFile = File(...)):
    """
    [HW04 Requirement]
    Возвращает полный набор флагов качества, включая новые эвристики из HW03:
    - has_constant_columns
    - has_many_zero_values
    - has_suspicious_id_duplicates
    """
    start_ts = time.time()
    df = _read_csv_upload(file)

    # 1. Анализ датасета
    summary = summarize_dataset(df)
    missing_df = missing_table(df)

    # 2. Вычисление флагов (core.py)
    all_flags = compute_quality_flags(summary, missing_df, df=df)

    latency = (time.time() - start_ts) * 1000

    # 3. Формируем ответ, акцентируя внимание на деталях проблем
    return {
        "quality_score": round(all_flags.get("quality_score", 0.0), 3),
        "dataset_stats": {
            "rows": summary.n_rows,
            "cols": summary.n_cols,
        },
        "critical_flags": {
            "too_few_rows": all_flags.get("too_few_rows"),
            "too_many_missing": all_flags.get("too_many_missing"),
        },
        "heuristics_hw04": {
            "has_constant_columns": all_flags.get("has_constant_columns"),
            "constant_columns": all_flags.get("constant_columns_list", []),
            "has_many_zero_values": all_flags.get("has_many_zero_values"),
            "cols_with_many_zeros": all_flags.get("cols_with_many_zeros", []),
            "has_suspicious_id_duplicates": all_flags.get("has_suspicious_id_duplicates"),
        },
        "latency_ms": round(latency, 2)
    }