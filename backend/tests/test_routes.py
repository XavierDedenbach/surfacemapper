"""
Integration tests for API routes
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
from app.main import app
from app.models.data_models import ProcessingRequest, GeoreferenceParams, AnalysisBoundary

@pytest.fixture
def client():
    return TestClient(app)

class TestSurfaceRoutes:
    """Test cases for surface-related API routes"""
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "surface-mapper-backend"
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    @patch('app.routes.surfaces.upload_surface')
    def test_upload_surface(self, mock_upload, client):
        """Test surface upload endpoint"""
        # Mock the upload response
        mock_upload.return_value = {
            "message": "Surface uploaded successfully",
            "filename": "test.ply",
            "status": "pending"
        }
        
        # Create a mock file upload
        files = {"file": ("test.ply", b"mock file content", "application/octet-stream")}
        response = client.post("/surfaces/upload", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Surface uploaded successfully"
        assert data["filename"] == "test.ply"
    
    def test_process_surfaces(self, client):
        """Test surface processing endpoint"""
        # Create test request data
        request_data = {
            "surface_files": ["surface1.ply", "surface2.ply"],
            "georeference_params": [
                {
                    "wgs84_lat": 40.0,
                    "wgs84_lon": -120.0,
                    "orientation_degrees": 0.0,
                    "scaling_factor": 1.0
                },
                {
                    "wgs84_lat": 40.0,
                    "wgs84_lon": -120.0,
                    "orientation_degrees": 0.0,
                    "scaling_factor": 1.0
                }
            ],
            "analysis_boundary": {
                "wgs84_coordinates": [
                    (40.0, -120.0),
                    (40.0, -119.0),
                    (41.0, -119.0),
                    (41.0, -120.0)
                ]
            },
            "generate_base_surface": False
        }
        
        response = client.post("/surfaces/process", json=request_data)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "status" in data
        assert "job_id" in data
    
    def test_get_processing_status(self, client):
        """Test processing status endpoint"""
        job_id = "test-job-123"
        response = client.get(f"/surfaces/status/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data

class TestDataValidation:
    """Test cases for data validation"""
    
    def test_invalid_georeference_params(self, client):
        """Test validation of georeferencing parameters"""
        request_data = {
            "surface_files": ["surface1.ply"],
            "georeference_params": [
                {
                    "wgs84_lat": 91.0,  # Invalid latitude
                    "wgs84_lon": -120.0,
                    "orientation_degrees": 0.0,
                    "scaling_factor": 1.0
                }
            ],
            "analysis_boundary": {
                "wgs84_coordinates": [
                    (40.0, -120.0),
                    (40.0, -119.0),
                    (41.0, -119.0),
                    (41.0, -120.0)
                ]
            }
        }
        
        response = client.post("/surfaces/process", json=request_data)
        # Should return validation error
        assert response.status_code in [400, 422]
    
    def test_invalid_boundary_coordinates(self, client):
        """Test validation of boundary coordinates"""
        request_data = {
            "surface_files": ["surface1.ply"],
            "georeference_params": [
                {
                    "wgs84_lat": 40.0,
                    "wgs84_lon": -120.0,
                    "orientation_degrees": 0.0,
                    "scaling_factor": 1.0
                }
            ],
            "analysis_boundary": {
                "wgs84_coordinates": [
                    (40.0, -120.0),
                    (40.0, -119.0)  # Invalid: only 2 coordinates instead of 4
                ]
            }
        }
        
        response = client.post("/surfaces/process", json=request_data)
        # Should return validation error
        assert response.status_code in [400, 422] 