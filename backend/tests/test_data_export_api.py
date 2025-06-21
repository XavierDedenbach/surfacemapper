import pytest
import tempfile
import os
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

API_URL = "/api/analysis/export"

class TestDataExportAPI:
    @pytest.fixture
    def sample_analysis_results(self):
        """Sample analysis results for testing"""
        return {
            "data_type": "analysis_results",
            "data": {
                "volume_results": [
                    {
                        "layer_designation": "Surface 0 to Surface 1",
                        "volume_cubic_yards": 1000.5,
                        "confidence_interval": [950.0, 1050.0],
                        "uncertainty": 50.0
                    }
                ],
                "thickness_results": [
                    {
                        "layer_designation": "Surface 0 to Surface 1",
                        "average_thickness_feet": 2.5,
                        "min_thickness_feet": 1.0,
                        "max_thickness_feet": 4.0,
                        "confidence_interval": [2.0, 3.0]
                    }
                ],
                "compaction_results": [
                    {
                        "layer_designation": "Surface 0 to Surface 1",
                        "compaction_rate_lbs_per_cubic_yard": 120.0,
                        "tonnage_used": 50.0
                    }
                ],
                "processing_metadata": {
                    "processing_time": 120.5,
                    "points_processed": 10000,
                    "algorithm_version": "1.0"
                }
            }
        }

    @pytest.fixture
    def sample_statistical_data(self):
        """Sample statistical data for testing"""
        return {
            "data_type": "statistical_data",
            "data": {
                "mean_value": 10.5,
                "median_value": 10.0,
                "standard_deviation": 2.0,
                "variance": 4.0,
                "skewness": 0.1,
                "kurtosis": -0.2,
                "sample_count": 1000,
                "confidence_interval_95": [9.5, 11.5],
                "percentiles": {"25": 8.0, "50": 10.0, "75": 12.0, "90": 14.0, "95": 15.0, "99": 18.0}
            }
        }

    @pytest.fixture
    def sample_surface_data(self):
        """Sample surface data for testing"""
        return {
            "data_type": "surface_data",
            "data": {
                "vertices": [[0, 0, 0], [1, 0, 1], [0, 1, 2], [1, 1, 3]],
                "faces": [[0, 1, 2], [1, 3, 2]],
                "metadata": {
                    "surface_name": "Test Surface",
                    "point_count": 4,
                    "face_count": 2
                }
            }
        }

    def test_export_analysis_results_csv(self, sample_analysis_results):
        """Test exporting analysis results to CSV"""
        payload = {
            **sample_analysis_results,
            "format": "csv"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert "format" in result
        assert result["format"] == "csv"

    def test_export_analysis_results_json(self, sample_analysis_results):
        """Test exporting analysis results to JSON"""
        payload = {
            **sample_analysis_results,
            "format": "json"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "json"

    def test_export_analysis_results_excel(self, sample_analysis_results):
        """Test exporting analysis results to Excel"""
        payload = {
            **sample_analysis_results,
            "format": "excel"
        }
        response = client.post(API_URL, json=payload)
        print(f"DEBUG: Status code: {response.status_code}")
        print(f"DEBUG: Response: {response.text}")
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "excel"

    def test_export_statistical_data_csv(self, sample_statistical_data):
        """Test exporting statistical data to CSV"""
        payload = {
            **sample_statistical_data,
            "format": "csv"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "csv"

    def test_export_statistical_data_json(self, sample_statistical_data):
        """Test exporting statistical data to JSON"""
        payload = {
            **sample_statistical_data,
            "format": "json"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "json"

    def test_export_surface_data_ply(self, sample_surface_data):
        """Test exporting surface data to PLY format"""
        payload = {
            **sample_surface_data,
            "format": "ply"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "ply"

    def test_export_surface_data_obj(self, sample_surface_data):
        """Test exporting surface data to OBJ format"""
        payload = {
            **sample_surface_data,
            "format": "obj"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "obj"

    def test_export_surface_data_stl(self, sample_surface_data):
        """Test exporting surface data to STL format"""
        payload = {
            **sample_surface_data,
            "format": "stl"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "stl"

    def test_export_surface_data_xyz(self, sample_surface_data):
        """Test exporting surface data to XYZ format"""
        payload = {
            **sample_surface_data,
            "format": "xyz"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result
        assert result["format"] == "xyz"

    def test_export_invalid_format(self, sample_analysis_results):
        """Test exporting with invalid format"""
        payload = {
            **sample_analysis_results,
            "format": "invalid_format"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 400 or response.status_code == 422

    def test_export_missing_data_type(self, sample_analysis_results):
        """Test exporting with missing data type"""
        payload = {
            "data": sample_analysis_results["data"],
            "format": "csv"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 400 or response.status_code == 422

    def test_export_missing_data(self, sample_analysis_results):
        """Test exporting with missing data"""
        payload = {
            "data_type": "analysis_results",
            "format": "csv"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 400 or response.status_code == 422

    def test_export_missing_format(self, sample_analysis_results):
        """Test exporting with missing format"""
        payload = {
            "data_type": "analysis_results",
            "data": sample_analysis_results["data"]
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 400 or response.status_code == 422

    def test_export_unsupported_format_for_data_type(self, sample_analysis_results):
        """Test exporting with unsupported format for data type"""
        payload = {
            **sample_analysis_results,
            "format": "ply"  # PLY not supported for analysis results
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 400

    def test_export_large_dataset(self):
        """Test exporting large dataset"""
        large_data = {
            "data_type": "analysis_results",
            "data": {
                "volume_results": [
                    {
                        "layer_designation": f"Layer {i}",
                        "volume_cubic_yards": float(i * 100),
                        "confidence_interval": [float(i * 95), float(i * 105)],
                        "uncertainty": float(i * 5)
                    } for i in range(1000)
                ],
                "thickness_results": [
                    {
                        "layer_designation": f"Layer {i}",
                        "average_thickness_feet": float(i * 0.1),
                        "min_thickness_feet": float(i * 0.05),
                        "max_thickness_feet": float(i * 0.15),
                        "confidence_interval": [float(i * 0.08), float(i * 0.12)]
                    } for i in range(1000)
                ],
                "compaction_results": [
                    {
                        "layer_designation": f"Layer {i}",
                        "compaction_rate_lbs_per_cubic_yard": float(i * 10),
                        "tonnage_used": float(i * 5)
                    } for i in range(1000)
                ],
                "processing_metadata": {"large_dataset": True}
            },
            "format": "csv"
        }
        response = client.post(API_URL, json=large_data)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result

    def test_export_with_custom_filename(self, sample_analysis_results):
        """Test exporting with custom filename"""
        payload = {
            **sample_analysis_results,
            "format": "csv",
            "filename": "custom_analysis_results.csv"
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result

    def test_export_with_metadata(self, sample_analysis_results):
        """Test exporting with additional metadata"""
        payload = {
            **sample_analysis_results,
            "format": "json",
            "metadata": {
                "export_timestamp": "2024-01-01T12:00:00Z",
                "export_format": "json",
                "data_version": "1.0"
            }
        }
        response = client.post(API_URL, json=payload)
        assert response.status_code == 200
        result = response.json()
        assert "file_path" in result or "download_url" in result 