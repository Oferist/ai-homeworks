from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import load_config


@dataclass(frozen=True)
class FeatureSpec:
    categorical: list[str]
    numeric: list[str]
    target: str

    @property
    def all_features(self) -> list[str]:
        return self.categorical + self.numeric


def get_feature_spec() -> FeatureSpec:
    feature_cfg = load_config()["features"]
    return FeatureSpec(
        categorical=list(feature_cfg["categorical"]),
        numeric=list(feature_cfg["numeric"]),
        target=str(feature_cfg["target"]),
    )


def prepare_supervised_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    spec = get_feature_spec()
    missing = [column for column in spec.all_features + [spec.target] if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = data[spec.all_features + [spec.target]].copy()
    frame = frame.dropna()

    X = frame[spec.all_features]
    y = frame[spec.target]
    return X, y


def payload_to_frame(payload: dict) -> pd.DataFrame:
    spec = get_feature_spec()
    missing = [column for column in spec.all_features if column not in payload]
    if missing:
        raise ValueError(f"Missing required prediction fields: {missing}")
    return pd.DataFrame([{column: payload[column] for column in spec.all_features}])
