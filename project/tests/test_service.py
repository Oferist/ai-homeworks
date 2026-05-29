from fastapi.testclient import TestClient

from src.service.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "model_missing"}
    assert "version" in body


def test_ui_page_is_available():
    response = client.get("/")
    assert response.status_code == 200
    assert "Прогноз спроса на велопрокат" in response.text
    assert "predictForm" in response.text


def test_predict_endpoint_uses_model():
    payload = {
        "season": 2,
        "mnth": 5,
        "hr": 8,
        "holiday": 0,
        "weekday": 1,
        "workingday": 1,
        "weathersit": 2,
        "yr": 1,
        "temp": 0.58,
        "atemp": 0.55,
        "hum": 0.6,
        "windspeed": 0.2,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] >= 0
    assert body["rounded_prediction"] >= 0
