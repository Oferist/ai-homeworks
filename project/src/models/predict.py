from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import load_config, resolve_path
from src.features.build_features import payload_to_frame


def model_path_from_config() -> Path:
    return resolve_path(load_config()["artifacts"]["model_path"])


@lru_cache(maxsize=1)
def load_model(path: str | Path | None = None) -> Any:
    model_path = resolve_path(path) if path else model_path_from_config()
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}. Run `python -m src.train` first."
        )
    return joblib.load(model_path)


def predict_one(payload: dict[str, Any]) -> float:
    model = load_model()
    frame = payload_to_frame(payload)
    prediction = float(model.predict(frame)[0])
    return max(0.0, prediction)


def predict_batch(records: list[dict[str, Any]]) -> list[float]:
    model = load_model()
    frames = [payload_to_frame(record) for record in records]
    frame = pd.concat(frames, ignore_index=True)
    predictions = model.predict(frame)
    return [max(0.0, float(value)) for value in predictions]
