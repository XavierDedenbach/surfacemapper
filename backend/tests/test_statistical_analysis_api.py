import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

API_URL = "/api/analysis/statistics"

class TestStatisticalAnalysisAPI:
    def test_valid_normal_data(self):
        data = {"values": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        response = client.post(API_URL, json=data)
        assert response.status_code == 200
        result = response.json()
        assert "mean_value" in result
        assert result["mean_value"] == pytest.approx(5.5)
        assert result["sample_count"] == 10

    def test_single_value(self):
        data = {"values": [42.0]}
        response = client.post(API_URL, json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["mean_value"] == 42.0
        assert result["standard_deviation"] == 0.0
        assert result["sample_count"] == 1

    def test_all_same_values(self):
        data = {"values": [5.0] * 100}
        response = client.post(API_URL, json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["mean_value"] == 5.0
        assert result["standard_deviation"] == 0.0
        assert result["sample_count"] == 100

    def test_negative_and_large_values(self):
        data = {"values": [-1000, 0, 1000, 1e6, -1e6]}
        response = client.post(API_URL, json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["sample_count"] == 5
        assert "mean_value" in result

    def test_empty_data(self):
        data = {"values": []}
        response = client.post(API_URL, json=data)
        assert response.status_code == 422 or response.status_code == 400

    def test_nan_values(self):
        data = {"values": [1, 2, float('nan'), 4, 5]}
        response = client.post(API_URL, json=data)
        assert response.status_code == 422 or response.status_code == 400

    def test_inf_values(self):
        data = {"values": [1, 2, float('inf'), 4, 5]}
        response = client.post(API_URL, json=data)
        assert response.status_code == 422 or response.status_code == 400

    def test_non_numeric_values(self):
        data = {"values": [1, 2, "a", 4, 5]}
        response = client.post(API_URL, json=data)
        assert response.status_code == 422 or response.status_code == 400

    def test_missing_values_key(self):
        data = {}
        response = client.post(API_URL, json=data)
        assert response.status_code == 422 or response.status_code == 400

    def test_large_dataset(self):
        data = {"values": list(range(100000))}
        response = client.post(API_URL, json=data)
        assert response.status_code == 200
        result = response.json()
        assert result["sample_count"] == 100000
        assert result["mean_value"] == pytest.approx(49999.5) 