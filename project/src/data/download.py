from __future__ import annotations

import logging
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from src.config import load_config, resolve_path


logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> Path:
    config = load_config()
    data_cfg = config["data"]
    raw_dir = resolve_path(data_cfg["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    hour_csv = resolve_path(data_cfg["hour_csv"])
    if hour_csv.exists() and not force:
        logger.info("Dataset already exists: %s", hour_csv)
        return raw_dir

    zip_path = raw_dir / "Bike-Sharing-Dataset.zip"
    logger.info("Downloading UCI Bike Sharing dataset from %s", data_cfg["dataset_url"])
    urlretrieve(data_cfg["dataset_url"], zip_path)

    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(raw_dir)

    nested_dir = raw_dir / "Bike-Sharing-Dataset"
    if nested_dir.exists():
        for item in nested_dir.iterdir():
            target = raw_dir / item.name
            if target.exists():
                target.unlink()
            item.replace(target)
        nested_dir.rmdir()

    if not hour_csv.exists():
        raise FileNotFoundError(f"Expected dataset file was not created: {hour_csv}")

    logger.info("Dataset is ready in %s", raw_dir)
    return raw_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_dataset()
