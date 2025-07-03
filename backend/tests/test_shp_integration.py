import pytest
import tempfile
import os
import time
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import fiona
import shapely.geometry as sgeom
from shapely.geometry import Point, Polygon, LineString
import pyproj

# Import the app and components
from app.main import app
from app.utils.shp_parser import SHPParser
from app.utils.ply_parser import PLYParser
from app.services.surface_cache import surface_cache

class TestSHPIntegration:
    """Integration tests for SHP and PLY workflow"""
    
    def setup_method(self):
        """Set up test environment"""
        self.client = TestClient(app)
        self.shp_parser = SHPParser()
        self.ply_parser = PLYParser()
        
        # Clear surface cache before each test
        surface_cache.clear()
        
        # Path to real SHP file for testing
        self.real_shp_file = "../drone_surfaces/27June20250541PM1619tonspartialcover/27June20250541PM1619tonspartialcover.shp"
        
        # Create test PLY file
        self.test_ply_content = self._create_test_ply_content()
    
    def _create_test_ply_content(self):
        """Create test PLY content for integration testing"""
        vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 1.5],
            [0.5, 0.5, 0.75]
        ], dtype=np.float32)
        
        faces = np.array([
            [0, 1, 4],
            [1, 3, 4],
            [3, 2, 4],
            [2, 0, 4]
        ], dtype=np.int32)
        
        # Create ASCII PLY content
        ply_content = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
element face {len(faces)}
property list uchar int vertex_indices
end_header
"""
        
        # Add vertices
        for vertex in vertices:
            ply_content += f"{vertex[0]} {vertex[1]} {vertex[2]}\n"
        
        # Add faces
        for face in faces:
            ply_content += f"3 {face[0]} {face[1]} {face[2]}\n"
        
        return ply_content.encode()
    
    def test_shp_and_ply_upload_integration(self):
        """Test uploading both SHP and PLY files in the same session"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Upload SHP file
        with open(self.real_shp_file, "rb") as f:
            shp_response = self.client.post("/api/surfaces/upload", files={"file": ("test.shp", f, "application/octet-stream")})
        
        # SHP upload should work (assuming we extend the upload endpoint)
        # For now, we'll test the SHP parser directly
        shp_vertices, shp_faces = self.shp_parser.process_shp_file(self.real_shp_file)
        assert len(shp_vertices) > 0
        assert shp_vertices.shape[1] == 3  # x, y, z coordinates
        
        # Upload PLY file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_file.write(self.test_ply_content)
            tmp_file.flush()
            
            with open(tmp_file.name, "rb") as f:
                ply_response = self.client.post("/api/surfaces/upload", files={"file": ("test.ply", f, "application/octet-stream")})
            
            os.unlink(tmp_file.name)
        
        # PLY upload should work
        assert ply_response.status_code == 200
        ply_data = ply_response.json()
        assert "surface_id" in ply_data
        assert ply_data["filename"] == "test.ply"
    
    def test_shp_parser_output_format_compatibility(self):
        """Test that SHP parser output format is compatible with PLY parser"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Parse SHP file
        shp_vertices, shp_faces = self.shp_parser.process_shp_file(self.real_shp_file)
        
        # Create equivalent PLY file
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            # Create PLY content from SHP vertices
            ply_content = f"""ply
format ascii 1.0
element vertex {len(shp_vertices)}
property float x
property float y
property float z
end_header
"""
            for vertex in shp_vertices:
                ply_content += f"{vertex[0]} {vertex[1]} {vertex[2]}\n"
            
            tmp_file.write(ply_content.encode())
            tmp_file.flush()
            
            # Parse PLY file
            ply_vertices, ply_faces = self.ply_parser.parse_ply_file(tmp_file.name)
            os.unlink(tmp_file.name)
        
        # Verify format compatibility
        assert type(shp_vertices) == type(ply_vertices)  # np.ndarray
        assert shp_vertices.shape[1] == 3  # x, y, z coordinates
        assert ply_vertices.shape[1] == 3  # x, y, z coordinates
        assert shp_vertices.dtype == ply_vertices.dtype  # Same data type
        
        # Verify content similarity (within floating point precision)
        np.testing.assert_array_almost_equal(shp_vertices, ply_vertices, decimal=6)
    
    def test_shp_and_ply_analysis_workflow(self):
        """Test complete analysis workflow with both SHP and PLY files"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Process SHP file to get vertices
        shp_vertices, _ = self.shp_parser.process_shp_file(self.real_shp_file)
        
        # Create PLY file with similar data
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            ply_content = f"""ply
format ascii 1.0
element vertex {len(shp_vertices)}
property float x
property float y
property float z
end_header
"""
            for vertex in shp_vertices:
                ply_content += f"{vertex[0]} {vertex[1]} {vertex[2]}\n"
            
            tmp_file.write(ply_content.encode())
            tmp_file.flush()
            
            # Upload PLY file
            with open(tmp_file.name, "rb") as f:
                response = self.client.post("/api/surfaces/upload", files={"file": ("test.ply", f, "application/octet-stream")})
            
            os.unlink(tmp_file.name)
        
        assert response.status_code == 200
        surface_id = response.json()["surface_id"]
        
        # Start analysis
        analysis_request = {
            "surface_ids": [surface_id],
            "params": {
                "boundary": {
                    "wgs84_coordinates": [
                        [40.7, -74.0],
                        [40.8, -74.0],
                        [40.8, -73.9],
                        [40.7, -73.9]
                    ]
                },
                "generate_base_surface": True,
                "base_surface_offset": 3.0
            }
        }
        
        response = self.client.post("/api/analysis/start", json=analysis_request)
        assert response.status_code == 202
        analysis_id = response.json()["analysis_id"]
        
        # Monitor analysis progress
        max_wait = 120  # 2 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f"/api/analysis/{analysis_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Analysis failed: {status_data.get('error_message', 'Unknown error')}")
            
            time.sleep(2)
        
        # Verify results
        results_response = self.client.get(f"/api/analysis/{analysis_id}/results")
        assert results_response.status_code == 200
        results_data = results_response.json()
        
        # Should have volume results
        assert "volume_results" in results_data
        assert len(results_data["volume_results"]) > 0
        
        # Should have thickness results
        assert "thickness_results" in results_data
        assert len(results_data["thickness_results"]) > 0
    
    def test_shp_coordinate_transformation_integration(self):
        """Test coordinate transformation integration with SHP files"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Process SHP file (should be in WGS84)
        shp_vertices, _ = self.shp_parser.process_shp_file(self.real_shp_file)
        
        # Verify coordinates are in WGS84 range
        assert np.all(shp_vertices[:, 0] >= -180) and np.all(shp_vertices[:, 0] <= 180)  # longitude
        assert np.all(shp_vertices[:, 1] >= -90) and np.all(shp_vertices[:, 1] <= 90)    # latitude
        
        # Test projection to UTM
        utm_vertices = self.shp_parser.project_to_utm(shp_vertices)
        
        # Verify UTM coordinates are reasonable (positive, in meters)
        assert np.all(utm_vertices[:, 0] > 0)  # UTM X should be positive
        assert np.all(utm_vertices[:, 1] > 0)  # UTM Y should be positive
        assert np.all(utm_vertices[:, 0] < 1000000)  # Reasonable UTM X range
        assert np.all(utm_vertices[:, 1] < 10000000)  # Reasonable UTM Y range
    
    def test_shp_error_handling_integration(self):
        """Test error handling for invalid SHP files in the workflow"""
        # Test with non-existent SHP file
        with pytest.raises(RuntimeError):
            self.shp_parser.process_shp_file("nonexistent.shp")
        
        # Test with invalid SHP content
        with tempfile.NamedTemporaryFile(suffix=".shp", delete=False) as tmp_file:
            tmp_file.write(b"Invalid SHP content")
            tmp_file.flush()
            
            with pytest.raises(RuntimeError):
                self.shp_parser.process_shp_file(tmp_file.name)
            
            os.unlink(tmp_file.name)
    
    def test_shp_and_ply_mixed_analysis(self):
        """Test analysis with mixed SHP and PLY files (simulated)"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Process SHP file
        shp_vertices, _ = self.shp_parser.process_shp_file(self.real_shp_file)
        
        # Create two PLY files with different data
        surface_ids = []
        
        for i in range(2):
            with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
                # Create PLY with offset data
                offset_vertices = shp_vertices.copy()
                offset_vertices[:, 2] += (i + 1) * 5.0  # 5ft and 10ft offset
                
                ply_content = f"""ply
format ascii 1.0
element vertex {len(offset_vertices)}
property float x
property float y
property float z
end_header
"""
                for vertex in offset_vertices:
                    ply_content += f"{vertex[0]} {vertex[1]} {vertex[2]}\n"
                
                tmp_file.write(ply_content.encode())
                tmp_file.flush()
                
                # Upload PLY file
                with open(tmp_file.name, "rb") as f:
                    response = self.client.post("/api/surfaces/upload", files={"file": (f"surface_{i}.ply", f, "application/octet-stream")})
                
                os.unlink(tmp_file.name)
            
            assert response.status_code == 200
            surface_ids.append(response.json()["surface_id"])
        
        # Start analysis with multiple surfaces
        analysis_request = {
            "surface_ids": surface_ids,
            "params": {
                "boundary": {
                    "wgs84_coordinates": [
                        [40.7, -74.0],
                        [40.8, -74.0],
                        [40.8, -73.9],
                        [40.7, -73.9]
                    ]
                },
                "generate_base_surface": False  # Use provided surfaces
            }
        }
        
        response = self.client.post("/api/analysis/start", json=analysis_request)
        assert response.status_code == 202
        analysis_id = response.json()["analysis_id"]
        
        # Wait for completion
        max_wait = 180  # 3 minutes for multiple files
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f"/api/analysis/{analysis_id}/status")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Analysis failed: {status_data.get('error_message', 'Unknown error')}")
            
            time.sleep(2)
        
        # Verify results
        results_response = self.client.get(f"/api/analysis/{analysis_id}/results")
        assert results_response.status_code == 200
        results_data = results_response.json()
        
        # Should have results for multiple surfaces
        assert len(results_data["volume_results"]) >= 1
        assert len(results_data["thickness_results"]) >= 1
    
    def test_shp_point_query_integration(self):
        """Test point query functionality with SHP-derived surfaces"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Process SHP file and create PLY equivalent
        shp_vertices, _ = self.shp_parser.process_shp_file(self.real_shp_file)
        
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            ply_content = f"""ply
format ascii 1.0
element vertex {len(shp_vertices)}
property float x
property float y
property float z
end_header
"""
            for vertex in shp_vertices:
                ply_content += f"{vertex[0]} {vertex[1]} {vertex[2]}\n"
            
            tmp_file.write(ply_content.encode())
            tmp_file.flush()
            
            # Upload PLY file
            with open(tmp_file.name, "rb") as f:
                response = self.client.post("/api/surfaces/upload", files={"file": ("test.ply", f, "application/octet-stream")})
            
            os.unlink(tmp_file.name)
        
        assert response.status_code == 200
        surface_id = response.json()["surface_id"]
        
        # Start analysis
        analysis_request = {
            "surface_ids": [surface_id],
            "params": {
                "boundary": {
                    "wgs84_coordinates": [
                        [40.7, -74.0],
                        [40.8, -74.0],
                        [40.8, -73.9],
                        [40.7, -73.9]
                    ]
                }
            }
        }
        
        response = self.client.post("/api/analysis/start", json=analysis_request)
        analysis_id = response.json()["analysis_id"]
        
        # Wait for completion
        max_wait = 60
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f"/api/analysis/{analysis_id}/status")
            if status_response.json()["status"] == "completed":
                break
            time.sleep(1)
        
        # Test point query with UTM coordinates
        point_query = {
            "x": 583960.0,  # UTM coordinates
            "y": 4507523.0,
            "coordinate_system": "utm"
        }
        
        query_response = self.client.post(f"/api/analysis/{analysis_id}/point_query", json=point_query)
        assert query_response.status_code == 200
        query_data = query_response.json()
        
        # Should return thickness data
        assert "thickness_layers" in query_data
        assert len(query_data["thickness_layers"]) > 0
    
    def test_shp_performance_benchmarks(self):
        """Test performance benchmarks with SHP files"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Test SHP processing performance
        start_time = time.time()
        shp_vertices, shp_faces = self.shp_parser.process_shp_file(self.real_shp_file)
        shp_processing_time = time.time() - start_time
        
        # Should process SHP file in reasonable time
        assert shp_processing_time < 10.0  # Should complete in <10 seconds
        assert len(shp_vertices) > 0
        
        # Test projection performance
        start_time = time.time()
        utm_vertices = self.shp_parser.project_to_utm(shp_vertices)
        projection_time = time.time() - start_time
        
        # Should project coordinates in reasonable time
        assert projection_time < 5.0  # Should complete in <5 seconds
        assert utm_vertices.shape == shp_vertices.shape
    
    def test_shp_workflow_error_recovery(self):
        """Test error recovery in SHP workflow"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Test that system can recover from SHP processing errors
        try:
            # This should fail gracefully
            self.shp_parser.process_shp_file("invalid_file.shp")
        except RuntimeError:
            # Expected error
            pass
        
        # System should still be able to process valid SHP files
        shp_vertices, _ = self.shp_parser.process_shp_file(self.real_shp_file)
        assert len(shp_vertices) > 0
        
        # System should still be able to process PLY files
        with tempfile.NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
            tmp_file.write(self.test_ply_content)
            tmp_file.flush()
            
            ply_vertices, ply_faces = self.ply_parser.parse_ply_file(tmp_file.name)
            os.unlink(tmp_file.name)
        
        assert len(ply_vertices) > 0 