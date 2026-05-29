from __future__ import annotations

import json

import requests


PAYLOAD = {
    "season": 3,
    "mnth": 7,
    "hr": 17,
    "holiday": 0,
    "weekday": 3,
    "workingday": 1,
    "weathersit": 1,
    "yr": 1,
    "temp": 0.76,
    "atemp": 0.72,
    "hum": 0.45,
    "windspeed": 0.18,
}


def main() -> None:
    response = requests.post("http://127.0.0.1:8000/predict", json=PAYLOAD, timeout=10)
    response.raise_for_status()
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
