from fastapi.testclient import TestClient
from src.api.app import app
import pytest

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "stock_code": "AAPL",
        "live_data": {
            "Close": 150.0,
            "volatility_30d": 0.02,
            # ... other features
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data['confidence'] <= 1
    assert data['prediction'] in [0, 1]
