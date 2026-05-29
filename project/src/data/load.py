from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config, resolve_path
from src.data.download import download_dataset


def ensure_dataset() -> Path:
    config = load_config()
    hour_csv = resolve_path(config["data"]["hour_csv"])
    if not hour_csv.exists():
        download_dataset()
    return hour_csv


def load_hour_data() -> pd.DataFrame:
    hour_csv = ensure_dataset()
    data = pd.read_csv(hour_csv)
    data["dteday"] = pd.to_datetime(data["dteday"])
    return data


def write_sample(data: pd.DataFrame | None = None) -> Path:
    config = load_config()
    sample_path = resolve_path(config["data"]["sample_csv"])
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    source = load_hour_data() if data is None else data
    sample = source.head(int(config["data"]["sample_size"]))
    sample.to_csv(sample_path, index=False)
    return sample_path
