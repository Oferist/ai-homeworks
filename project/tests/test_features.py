import pandas as pd

from src.features.build_features import get_feature_spec, prepare_supervised_frame


def test_prepare_supervised_frame_removes_leakage_columns():
    data = pd.DataFrame(
        {
            "season": [1],
            "mnth": [1],
            "hr": [8],
            "holiday": [0],
            "weekday": [2],
            "workingday": [1],
            "weathersit": [1],
            "yr": [1],
            "temp": [0.24],
            "atemp": [0.25],
            "hum": [0.6],
            "windspeed": [0.1],
            "casual": [10],
            "registered": [120],
            "cnt": [130],
        }
    )

    X, y = prepare_supervised_frame(data)
    spec = get_feature_spec()

    assert list(X.columns) == spec.all_features
    assert "casual" not in X.columns
    assert "registered" not in X.columns
    assert y.iloc[0] == 130
