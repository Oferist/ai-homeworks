from src.models.predict import predict_one


def test_predict_one_returns_non_negative_number():
    payload = {
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

    prediction = predict_one(payload)

    assert isinstance(prediction, float)
    assert prediction >= 0
