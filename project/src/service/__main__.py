from __future__ import annotations

import os

import uvicorn

from src.config import load_config


def main() -> None:
    config = load_config()["service"]
    host = os.getenv("APP_HOST", config["host"])
    port = int(os.getenv("APP_PORT", config["port"]))
    log_level = os.getenv("LOG_LEVEL", config["log_level"]).lower()
    uvicorn.run("src.service.app:app", host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
