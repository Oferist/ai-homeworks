from pathlib import Path

from src.config import resolve_path


def pytest_sessionstart(session):
    model_path = resolve_path("artifacts/model.joblib")
    metadata_path = resolve_path("artifacts/model_metadata.json")
    if model_path.exists() and metadata_path.exists():
        return

    from src.models.train import train_and_evaluate

    train_and_evaluate()
