import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import tempfile
import os

from app.main import app
from app.services.surface_processor import SurfaceProcessor


class Test3DVisualizationDataAPI:
    """Test suite for 3D visualization data API endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def mock_analysis_data(self):
        """Mock analysis data for testing"""
        # Create a mock TIN object
        mock_tin = Mock()
        mock_tin.points = np.array([
            [0.0, 0.0, 10.0],
            [1.0, 0.0, 11.0],
            [0.0, 1.0, 12.0],
            [1.0, 1.0, 13.0],
            [0.5, 0.5, 14.0]
        ])
        mock_tin.simplices = np.array([
            [0, 1, 4],
            [1, 3, 4],
            [3, 2, 4],
            [2, 0, 4]
        ])
        
        return {
            "analysis_id": "test-analysis-123",
            "status": "completed",
            "surface_tins": [mock_tin]
        }
    
    @pytest.fixture
    def large_mesh_data(self):
        """Generate large mesh data for performance testing"""
        # Create a 100x100 grid mesh
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = 10 + 0.1 * X + 0.05 * Y + 0.01 * np.random.randn(100, 100)
        
        vertices = []
        faces = []
        
        for i in range(99):
            for j in range(99):
                # Create two triangles for each grid cell
                v1 = i * 100 + j
                v2 = i * 100 + j + 1
                v3 = (i + 1) * 100 + j
                v4 = (i + 1) * 100 + j + 1
                
                vertices.extend([
                    [X[i, j], Y[i, j], Z[i, j]],
                    [X[i, j+1], Y[i, j+1], Z[i, j+1]],
                    [X[i+1, j], Y[i+1, j], Z[i+1, j]],
                    [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]]
                ])
                
                faces.extend([[v1, v2, v3], [v2, v4, v3]])
        
        return {
            "vertices": np.array(vertices),
            "faces": np.array(faces),
            "bounds": {
                "x_min": 0.0, "x_max": 10.0,
                "y_min": 0.0, "y_max": 10.0,
                "z_min": 9.0, "z_max": 12.0
            }
        }

    def test_3d_mesh_data_retrieval(self, client, mock_analysis_data):
        """Test basic 3D mesh data retrieval"""
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis_data):
            response = client.get("/api/analysis/test-analysis-123/surface/0/mesh")
            
            assert response.status_code == 200
            data = response.json()
            
            assert "vertices" in data
            assert "faces" in data
            assert "bounds" in data
            
            # Validate data structure
            vertices = data["vertices"]
            faces = data["faces"]
            assert len(vertices) > 0
            assert len(faces) > 0
            assert all(len(vertex) == 3 for vertex in vertices)  # x, y, z
            assert all(len(face) == 3 for face in faces)  # triangle indices
            
            # Validate bounds
            bounds = data["bounds"]
            assert "x_min" in bounds
            assert "x_max" in bounds
            assert "y_min" in bounds
            assert "y_max" in bounds
            assert "z_min" in bounds
            assert "z_max" in bounds

    def test_mesh_simplification_levels(self, client, large_mesh_data):
        """Test mesh simplification for different levels of detail"""
        mock_analysis = {
            "analysis_id": "large-analysis-123",
            "status": "completed",
            "surface_tins": [Mock(points=large_mesh_data["vertices"], simplices=large_mesh_data["faces"])]
        }
        
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis):
            # Test different levels of detail
            lod_results = {}
            
            for lod in ["high", "medium", "low"]:
                response = client.get(
                    "/api/analysis/large-analysis-123/surface/0/mesh",
                    params={"level_of_detail": lod}
                )
                assert response.status_code == 200
                mesh_data = response.json()
                lod_results[lod] = len(mesh_data["vertices"])
            
            # Lower LOD should have fewer vertices
            assert lod_results["low"] < lod_results["medium"]
            assert lod_results["medium"] < lod_results["high"]

    def test_mesh_data_format_consistency(self, client, mock_analysis_data):
        """Test that mesh data format is consistent across requests"""
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis_data):
            # Make multiple requests
            responses = []
            for _ in range(3):
                response = client.get("/api/analysis/test-analysis-123/surface/0/mesh")
                assert response.status_code == 200
                responses.append(response.json())
            
            # All responses should have the same structure
            for i in range(1, len(responses)):
                assert set(responses[i].keys()) == set(responses[0].keys())
                
                # Vertex and face counts should be consistent
                assert len(responses[i]["vertices"]) == len(responses[0]["vertices"])
                assert len(responses[i]["faces"]) == len(responses[0]["faces"])

    def test_large_mesh_handling(self, client, large_mesh_data):
        """Test handling of large mesh data"""
        mock_analysis = {
            "analysis_id": "large-analysis-123",
            "status": "completed",
            "surface_tins": [Mock(points=large_mesh_data["vertices"], simplices=large_mesh_data["faces"])]
        }
        
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis):
            response = client.get("/api/analysis/large-analysis-123/surface/0/mesh")
            
            assert response.status_code == 200
            data = response.json()
            
            # Should handle large meshes without timeout
            assert len(data["vertices"]) > 1000
            assert len(data["faces"]) > 1000
            
            # Response time should be reasonable
            assert response.elapsed.total_seconds() < 5.0

    def test_mesh_quality_validation(self, client, mock_analysis_data):
        """Test mesh quality validation"""
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis_data):
            response = client.get("/api/analysis/test-analysis-123/surface/0/mesh")
            
            assert response.status_code == 200
            data = response.json()
            
            vertices = data["vertices"]
            faces = data["faces"]
            
            # Validate face indices are within bounds
            max_vertex_index = len(vertices) - 1
            for face in faces:
                assert all(0 <= idx <= max_vertex_index for idx in face)
            
            # Validate no degenerate faces (all vertices different)
            for face in faces:
                assert len(set(face)) == 3
            
            # Validate mesh connectivity (each edge appears in at least one face)
            edges = set()
            for face in faces:
                for i in range(3):
                    edge = tuple(sorted([face[i], face[(i + 1) % 3]]))
                    edges.add(edge)
            
            # For a valid mesh, all edges should be connected to at least one face
            assert len(edges) > 0

    def test_invalid_analysis_id(self, client):
        """Test handling of invalid analysis ID"""
        with patch('app.routes.analysis.executor.get_results', return_value=None):
            response = client.get("/api/analysis/invalid-id/surface/0/mesh")
            assert response.status_code == 404

    def test_invalid_surface_id(self, client, mock_analysis_data):
        """Test handling of invalid surface ID"""
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis_data):
            response = client.get("/api/analysis/test-analysis-123/surface/999/mesh")
            assert response.status_code == 404

    def test_analysis_not_completed(self, client):
        """Test handling of analysis that is not completed"""
        mock_analysis = {
            "analysis_id": "incomplete-analysis-123",
            "status": "processing",
            "surface_tins": []
        }
        
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis):
            response = client.get("/api/analysis/incomplete-analysis-123/surface/0/mesh")
            assert response.status_code == 400

    def test_mesh_simplification_parameters(self, client, large_mesh_data):
        """Test mesh simplification with custom parameters"""
        mock_analysis = {
            "analysis_id": "custom-analysis-123",
            "status": "completed",
            "surface_tins": [Mock(points=large_mesh_data["vertices"], simplices=large_mesh_data["faces"])]
        }
        
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis):
            # Test with custom simplification parameters
            response = client.get(
                "/api/analysis/custom-analysis-123/surface/0/mesh",
                params={
                    "level_of_detail": "medium",
                    "max_vertices": 1000,
                    "preserve_boundaries": "true"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            
            # Should respect max_vertices parameter
            assert len(data["vertices"]) <= 1000

    def test_mesh_data_serialization_performance(self, client, large_mesh_data):
        """Test performance of mesh data serialization"""
        mock_analysis = {
            "analysis_id": "perf-analysis-123",
            "status": "completed",
            "surface_tins": [Mock(points=large_mesh_data["vertices"], simplices=large_mesh_data["faces"])]
        }
        
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis):
            import time
            start_time = time.time()
            
            response = client.get("/api/analysis/perf-analysis-123/surface/0/mesh")
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            assert response.status_code == 200
            assert processing_time < 2.0  # Should complete within 2 seconds

    def test_mesh_bounds_accuracy(self, client, mock_analysis_data):
        """Test that mesh bounds are accurately calculated"""
        with patch('app.routes.analysis.executor.get_results', return_value=mock_analysis_data):
            response = client.get("/api/analysis/test-analysis-123/surface/0/mesh")
            
            assert response.status_code == 200
            data = response.json()
            
            bounds = data["bounds"]
            vertices = data["vertices"]
            
            # Calculate actual bounds from vertices
            x_coords = [v[0] for v in vertices]
            y_coords = [v[1] for v in vertices]
            z_coords = [v[2] for v in vertices]
            
            actual_x_min, actual_x_max = min(x_coords), max(x_coords)
            actual_y_min, actual_y_max = min(y_coords), max(y_coords)
            actual_z_min, actual_z_max = min(z_coords), max(z_coords)
            
            # Bounds should match calculated values
            assert abs(bounds["x_min"] - actual_x_min) < 1e-6
            assert abs(bounds["x_max"] - actual_x_max) < 1e-6
            assert abs(bounds["y_min"] - actual_y_min) < 1e-6
            assert abs(bounds["y_max"] - actual_y_max) < 1e-6
            assert abs(bounds["z_min"] - actual_z_min) < 1e-6
            assert abs(bounds["z_max"] - actual_z_max) < 1e-6 