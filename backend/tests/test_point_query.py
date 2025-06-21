"""
Tests for point query functionality
"""
import pytest
import numpy as np
from unittest.mock import patch, Mock
from fastapi.testclient import TestClient
from app.main import app

class TestPointQuery:
    """Test point query functionality"""
    
    def setup_method(self):
        """Setup test data"""
        self.mock_thickness_response = {
            "thickness_layers": [
                {
                    "layer_name": "Surface 1 to Surface 2",
                    "thickness_feet": 15.5
                },
                {
                    "layer_name": "Surface 2 to Surface 3", 
                    "thickness_feet": 8.2
                }
            ],
            "query_point": {"x": 100.0, "y": 200.0, "coordinate_system": "utm"},
            "interpolation_method": "linear"
        }
    
    def test_point_query_success(self):
        """Test successful point query"""
        # Mock the executor to return analysis results
        mock_results = {
            "status": "completed",
            "surface_tins": [Mock(), Mock()],
            "surface_names": ["Surface 1", "Surface 2"],
            "georef": {"lat": 37.7749, "lon": -122.4194, "orientation": 0.0, "scale": 1.0}
        }
        
        with patch('app.services.analysis_executor.AnalysisExecutor.get_results', return_value=mock_results):
            with patch('app.services.thickness_calculator._interpolate_z_at_point', return_value=10.0):
                client = TestClient(app)
                payload = {"x": 100.0, "y": 200.0, "coordinate_system": "utm"}
                response = client.post("/api/analysis/test-id/point_query", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                assert "thickness_layers" in data
                assert "query_point" in data
                assert "interpolation_method" in data
                assert len(data["thickness_layers"]) == 1  # Only one layer between 2 surfaces
                assert data["thickness_layers"][0]["thickness_feet"] == 0.0  # 10.0 - 10.0
    
    def test_point_query_invalid_coordinates(self):
        """Test point query with invalid coordinates"""
        client = TestClient(app)
        payload = {"x": "invalid", "y": 200.0}
        response = client.post("/api/analysis/test-id/point_query", json=payload)
        
        assert response.status_code == 400
        assert "Invalid coordinates" in response.json()["error"]
    
    def test_batch_point_query_success(self):
        """Test successful batch point query"""
        # Mock the executor to return analysis results
        mock_results = {
            "status": "completed",
            "surface_tins": [Mock(), Mock()],
            "surface_names": ["Surface 1", "Surface 2"],
            "georef": {"lat": 37.7749, "lon": -122.4194, "orientation": 0.0, "scale": 1.0}
        }
        
        with patch('app.services.analysis_executor.AnalysisExecutor.get_results', return_value=mock_results):
            with patch('app.services.thickness_calculator._interpolate_z_at_point', return_value=10.0):
                client = TestClient(app)
                payload = {
                    "points": [
                        {"x": 100.0, "y": 200.0},
                        {"x": 150.0, "y": 250.0}
                    ]
                }
                response = client.post("/api/analysis/test-id/batch_point_query", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                assert "results" in data
                assert len(data["results"]) == 2
                assert len(data["results"][0]["thickness_layers"]) == 1
    
    def test_batch_point_query_too_many_points(self):
        """Test batch point query with too many points"""
        client = TestClient(app)
        # Create 1001 points
        points = [{"x": float(i), "y": float(i)} for i in range(1001)]
        payload = {"points": points}
        response = client.post("/api/analysis/test-id/batch_point_query", json=payload)
        
        assert response.status_code == 400
        assert "Too many points" in response.json()["error"]
    
    def test_point_query_analysis_not_found(self):
        """Test point query with non-existent analysis"""
        # Mock the executor to return None (analysis not found)
        with patch('app.services.analysis_executor.AnalysisExecutor.get_results', return_value=None):
            client = TestClient(app)
            payload = {"x": 100.0, "y": 200.0}
            response = client.post("/api/analysis/nonexistent/point_query", json=payload)
            
            assert response.status_code == 404
            assert "not found" in response.json()["error"].lower()
    
    def test_point_query_outside_bounds(self):
        """Test point query for point outside surface bounds"""
        # Mock the executor to return analysis results
        mock_results = {
            "status": "completed",
            "surface_tins": [Mock(), Mock()],
            "surface_names": ["Surface 1", "Surface 2"],
            "georef": {"lat": 37.7749, "lon": -122.4194, "orientation": 0.0, "scale": 1.0}
        }
        
        with patch('app.services.analysis_executor.AnalysisExecutor.get_results', return_value=mock_results):
            # Mock interpolation to return NaN (outside bounds)
            with patch('app.services.thickness_calculator._interpolate_z_at_point', return_value=np.nan):
                client = TestClient(app)
                payload = {"x": 999999.0, "y": 999999.0}
                response = client.post("/api/analysis/test-id/point_query", json=payload)
                
                assert response.status_code == 200
                data = response.json()
                assert data["thickness_layers"][0]["thickness_feet"] is None
    
    def test_point_query_coordinate_transformation(self):
        """Test point query with coordinate transformation"""
        # Mock the executor to return analysis results
        mock_results = {
            "status": "completed",
            "surface_tins": [Mock(), Mock()],
            "surface_names": ["Surface 1", "Surface 2"],
            "georef": {"lat": 37.7749, "lon": -122.4194, "orientation": 0.0, "scale": 1.0}
        }
        
        with patch('app.services.analysis_executor.AnalysisExecutor.get_results', return_value=mock_results):
            with patch('app.services.thickness_calculator._interpolate_z_at_point', return_value=10.0):
                # Patch CoordinateTransformer to return a mock pipeline with transform_to_utm
                mock_pipeline = Mock()
                mock_pipeline.transform_to_utm.return_value = np.array([[100.0, 200.0]])
                mock_transformer = Mock()
                mock_transformer.transform_to_utm = mock_pipeline.transform_to_utm
                with patch('app.routes.analysis.CoordinateTransformer', return_value=mock_transformer):
                    client = TestClient(app)
                    payload = {"x": -122.4194, "y": 37.7749, "coordinate_system": "wgs84"}
                    response = client.post("/api/analysis/test-id/point_query", json=payload)
                    
                    assert response.status_code == 200
                    data = response.json()
                    assert data["query_point"]["coordinate_system"] == "utm" 