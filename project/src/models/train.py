from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import load_config, resolve_path
from src.data.load import load_hour_data, write_sample
from src.features.build_features import get_feature_spec, prepare_supervised_frame


matplotlib.use("Agg")
logger = logging.getLogger(__name__)


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    validation_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )
    relative_validation_size = validation_size / (train_size + validation_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_validation_size,
        random_state=random_state,
        shuffle=True,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor() -> ColumnTransformer:
    spec = get_feature_spec()
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), spec.categorical),
            ("numeric", StandardScaler(), spec.numeric),
        ]
    )


def build_models(config: dict) -> dict[str, Pipeline]:
    rf_cfg = config["models"]["random_forest"]
    ridge_cfg = config["models"]["ridge"]

    return {
        "dummy_mean": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "ridge": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", Ridge(alpha=float(ridge_cfg["alpha"]))),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=int(rf_cfg["n_estimators"]),
                        max_depth=int(rf_cfg["max_depth"]),
                        min_samples_leaf=int(rf_cfg["min_samples_leaf"]),
                        n_jobs=int(rf_cfg["n_jobs"]),
                        random_state=int(config["project"]["random_state"]),
                    ),
                ),
            ]
        ),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mse)), 4),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def get_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocess"]
    return list(preprocessor.get_feature_names_out())


def save_feature_importance(model: Pipeline, path: Path) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame()

    importance = pd.DataFrame(
        {
            "feature": get_feature_names(model),
            "importance": estimator.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(path, index=False)
    return importance


def make_plots(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    error_by_hour: pd.DataFrame,
    artifact_cfg: dict,
) -> None:
    sns.set_theme(style="whitegrid")

    hourly = data.groupby("hr", as_index=False)["cnt"].mean()
    plt.figure(figsize=(9, 5))
    sns.lineplot(data=hourly, x="hr", y="cnt", marker="o")
    plt.title("Average bike demand by hour")
    plt.xlabel("Hour")
    plt.ylabel("Average rentals")
    plt.tight_layout()
    plt.savefig(resolve_path(artifact_cfg["hourly_demand_plot"]), dpi=150)
    plt.close()

    sample = predictions.head(400)
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=sample, x="actual", y="prediction", alpha=0.65)
    limit = max(sample["actual"].max(), sample["prediction"].max())
    plt.plot([0, limit], [0, limit], color="black", linestyle="--", linewidth=1)
    plt.title("Actual vs predicted demand")
    plt.xlabel("Actual rentals")
    plt.ylabel("Predicted rentals")
    plt.tight_layout()
    plt.savefig(resolve_path(artifact_cfg["prediction_plot"]), dpi=150)
    plt.close()

    if not feature_importance.empty:
        top = feature_importance.head(15).copy()
        plt.figure(figsize=(9, 6))
        sns.barplot(data=top, x="importance", y="feature", color="#2f7ebc")
        plt.title("Top feature importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(resolve_path(artifact_cfg["feature_importance_plot"]), dpi=150)
        plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(data=error_by_hour, x="hr", y="mae", color="#cc6b49")
    plt.title("Mean absolute error by hour")
    plt.xlabel("Hour")
    plt.ylabel("MAE")
    plt.tight_layout()
    plt.savefig(resolve_path(artifact_cfg["error_by_hour_plot"]), dpi=150)
    plt.close()


def train_and_evaluate() -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config()
    artifact_cfg = config["artifacts"]
    artifact_dir = resolve_path(artifact_cfg["dir"])
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data = load_hour_data()
    write_sample(data)

    X, y = prepare_supervised_frame(data)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
        X,
        y,
        train_size=float(config["split"]["train_size"]),
        validation_size=float(config["split"]["validation_size"]),
        test_size=float(config["split"]["test_size"]),
        random_state=int(config["project"]["random_state"]),
    )

    models = build_models(config)
    results: list[dict] = []
    fitted_models: dict[str, Pipeline] = {}

    for name, model in models.items():
        logger.info("Training %s", name)
        model.fit(X_train, y_train)
        fitted_models[name] = model
        val_pred = model.predict(X_val)
        metrics = regression_metrics(y_val, val_pred)
        results.append({"model": name, "split": "validation", **metrics})

    best_name = min(results, key=lambda row: row["rmse"])["model"]
    best_model = fitted_models[best_name]
    test_pred = best_model.predict(X_test)
    test_metrics = regression_metrics(y_test, test_pred)
    results.append({"model": best_name, "split": "test", **test_metrics})

    model_path = resolve_path(artifact_cfg["model_path"])
    joblib.dump(best_model, model_path)

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(resolve_path(artifact_cfg["metrics_csv"]), index=False)

    predictions = X_test.copy()
    predictions["actual"] = y_test.to_numpy()
    predictions["prediction"] = np.maximum(0, test_pred)
    predictions["absolute_error"] = (predictions["actual"] - predictions["prediction"]).abs()
    predictions.to_csv(resolve_path(artifact_cfg["predictions_csv"]), index=False)

    error_by_hour = (
        predictions.groupby("hr", as_index=False)["absolute_error"]
        .mean()
        .rename(columns={"absolute_error": "mae"})
    )
    error_by_hour.to_csv(resolve_path(artifact_cfg["error_by_hour_csv"]), index=False)

    importance = save_feature_importance(best_model, resolve_path(artifact_cfg["feature_importance_csv"]))
    make_plots(data, predictions, importance, error_by_hour, artifact_cfg)

    metadata = {
        "project": config["project"]["name"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "UCI Bike Sharing Dataset, hourly data",
        "target": config["features"]["target"],
        "features": config["features"],
        "final_model": best_name,
        "validation_results": [row for row in results if row["split"] == "validation"],
        "test_metrics": test_metrics,
        "model_path": artifact_cfg["model_path"],
    }

    with resolve_path(artifact_cfg["metadata_path"]).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    with resolve_path(artifact_cfg["metrics_json"]).open("w", encoding="utf-8") as file:
        json.dump({"results": results, "metadata": metadata}, file, ensure_ascii=False, indent=2)

    logger.info("Final model: %s, test metrics: %s", best_name, test_metrics)
    return metadata


if __name__ == "__main__":
    train_and_evaluate()
