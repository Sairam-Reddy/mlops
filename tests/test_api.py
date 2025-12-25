# tests/test_api.py
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from src.api.app import app

@pytest.fixture
def mock_model():
    mock = MagicMock()
    # predict_proba returns [[0.2, 0.8]] for test input
    mock.predict_proba.return_value = [[0.2, 0.8]]
    return mock

def test_predict(mock_model):
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145,
        "chol": 233, "fbs": 1, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
    }

    # patch the global `model` in src.api.app with the mock
    with patch("src.api.app.model", mock_model):
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
            assert response.status_code == 200
            data = response.json()
            assert data["prediction"] == 1
            assert data["confidence"] == 0.8
