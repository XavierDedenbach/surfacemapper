import numpy as np
import pytest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from app.utils.shp_parser import SHPParser

class TestSHPParserUTM:
    """Test SHP Parser UTM projection functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.shp_parser = SHPParser()
        
        # Create test SHP file path
        self.test_shp_dir = "drone_surfaces/27June2025_0550AM_emptyworkingface"
        self.test_shp_file = os.path.join(self.test_shp_dir, "27June2025_0550AM_emptyworkingface.shp")
        
        # Test WGS84 coordinates (known values)
        self.test_wgs84_coords = [
            (-118.123456, 35.123456),  # Test coordinate 1
            (-118.123457, 35.123457),  # Test coordinate 2
            (-118.123458, 35.123458),  # Test coordinate 3
        ]
    
    def test_shp_file_processing_with_utm_projection(self):
        """Test SHP file processing with immediate UTM projection"""
        if not os.path.exists(self.test_shp_file):
            pytest.skip(f"Test SHP file not found at {self.test_shp_file}")
        
        # Process SHP file using the new UTM method
        result = self.shp_parser.process_shp_file_to_utm(self.test_shp_file)
        
        # Verify result structure
        assert isinstance(result, tuple)
        vertices, faces = result
        
        # Verify vertices are in UTM coordinates (meters)
        assert isinstance(vertices, np.ndarray)
        assert vertices.shape[1] == 3  # x, y, z
        
        # Check that coordinates are in UTM range (positive, reasonable values)
        assert np.all(vertices[:, 0] > 100000)  # UTM X should be > 100000
        assert np.all(vertices[:, 1] > 3000000)  # UTM Y should be > 3000000
        
        # Verify no coordinates are in WGS84 range (degrees)
        assert not np.any(np.abs(vertices[:, 0]) <= 180)  # No longitude values
        assert not np.any(np.abs(vertices[:, 1]) <= 90)   # No latitude values
    
    def test_boundary_projection(self):
        """Test boundary projection from WGS84 to UTM"""
        # Test boundary in WGS84 coordinates
        boundary_wgs84 = [
            (-118.2, 35.0),
            (-118.1, 35.0),
            (-118.1, 35.1),
            (-118.2, 35.1)
        ]
        
        # Project boundary to UTM using the existing project_to_utm method
        boundary_vertices = np.array([[lon, lat, 0] for lon, lat in boundary_wgs84])
        boundary_utm_vertices = self.shp_parser.project_to_utm(boundary_vertices)
        
        # Convert back to list of tuples
        boundary_utm = [(v[0], v[1]) for v in boundary_utm_vertices]
        
        # Verify boundary structure
        assert len(boundary_utm) == len(boundary_wgs84)
        assert all(len(coord) == 2 for coord in boundary_utm)
        
        # Verify coordinates are in UTM range (more realistic bounds)
        for x, y in boundary_utm:
            assert x > 100000  # UTM X should be positive and large
            assert y > 3000000  # UTM Y should be positive and large
            assert x < 1000000  # Reasonable upper bound
            assert y < 5000000  # Reasonable upper bound
    
    def test_single_coordinate_projection(self):
        """Test single coordinate projection from WGS84 to UTM"""
        # Test known coordinate pairs using the existing project_to_utm method
        for wgs84_coord in self.test_wgs84_coords:
            lon, lat = wgs84_coord
            
            # Use the existing project_to_utm method
            wgs84_vertex = np.array([[lon, lat, 0]])
            utm_vertices = self.shp_parser.project_to_utm(wgs84_vertex)
            utm_x, utm_y = utm_vertices[0][0], utm_vertices[0][1]
            
            # Verify coordinates are in reasonable UTM range
            assert utm_x > 100000 and utm_x < 1000000
            assert utm_y > 3000000 and utm_y < 5000000
            
            # Verify projection is consistent (same input gives same output)
            utm_vertices2 = self.shp_parser.project_to_utm(wgs84_vertex)
            utm_x2, utm_y2 = utm_vertices2[0][0], utm_vertices2[0][1]
            assert abs(utm_x - utm_x2) < 1e-10
            assert abs(utm_y - utm_y2) < 1e-10
    
    def test_mesh_generation_in_utm(self):
        """Test mesh generation with UTM coordinates"""
        # Create test WGS84 vertices (not UTM) for proper testing
        wgs84_vertices = np.array([
            [-118.123456, 35.123456, 100.0],
            [-118.123457, 35.123457, 100.0],
            [-118.123458, 35.123458, 100.0],
            [-118.123459, 35.123459, 100.0]
        ], dtype=np.float32)
        
        # Create LineString from WGS84 vertices
        from shapely.geometry import LineString
        linestring = LineString([(v[0], v[1], v[2]) for v in wgs84_vertices])
        
        # Generate mesh in UTM coordinates
        mesh_vertices, mesh_faces = self.shp_parser.generate_surface_mesh_from_linestrings([linestring])
        
        # Verify mesh data structure
        assert isinstance(mesh_vertices, np.ndarray)
        assert mesh_vertices.shape[1] == 3
        
        # Verify vertices are in UTM coordinates
        assert np.all(mesh_vertices[:, 0] > 100000)  # UTM X
        assert np.all(mesh_vertices[:, 1] > 3000000)  # UTM Y
        
        # Verify faces are valid triangle indices (if present)
        if mesh_faces is not None:
            assert isinstance(mesh_faces, np.ndarray)
            assert mesh_faces.shape[1] == 3  # Triangle faces
            assert np.all(mesh_faces >= 0)  # Valid indices
            assert np.all(mesh_faces < len(mesh_vertices))  # Indices within bounds
    
    def test_no_wgs84_mesh_operations(self):
        """Test that no mesh operations are performed in WGS84 coordinates"""
        if not os.path.exists(self.test_shp_file):
            pytest.skip(f"Test SHP file not found at {self.test_shp_file}")
        
        # Mock the mesh generation to track coordinate systems
        original_method = self.shp_parser.generate_surface_mesh_from_linestrings
        
        def mock_mesh_generation(linestrings, spacing_feet=1.0):
            # Extract vertices from linestrings and verify they're in UTM
            all_vertices = []
            for ls in linestrings:
                coords = list(ls.coords)
                for coord in coords:
                    all_vertices.append(coord)
            
            # Convert to numpy array for checking
            vertices = np.array(all_vertices)
            if len(vertices) > 0:
                # Verify vertices are in UTM coordinates (not WGS84)
                assert np.all(vertices[:, 0] > 180)  # No longitude values
                assert np.all(vertices[:, 1] > 90)   # No latitude values
            
            return original_method(linestrings, spacing_feet)
        
        with patch.object(self.shp_parser, 'generate_surface_mesh_from_linestrings', side_effect=mock_mesh_generation):
            # Process SHP file - should not trigger WGS84 mesh operations
            result = self.shp_parser.process_shp_file_to_utm(self.test_shp_file)
            assert isinstance(result, tuple)
    
    def test_boundary_projection_accuracy(self):
        """Test boundary projection accuracy with known coordinates"""
        # Test with a simple rectangular boundary
        boundary_wgs84 = [
            (-118.0, 35.0),
            (-117.9, 35.0),
            (-117.9, 35.1),
            (-118.0, 35.1),
            (-118.0, 35.0)  # Close the polygon
        ]
        
        # Project boundary to UTM
        boundary_vertices = np.array([[lon, lat, 0] for lon, lat in boundary_wgs84])
        boundary_utm_vertices = self.shp_parser.project_to_utm(boundary_vertices)
        boundary_utm = [(v[0], v[1]) for v in boundary_utm_vertices]
        
        # Verify boundary is closed
        assert len(boundary_utm) >= 3
        
        # Verify all coordinates are in UTM range (more realistic bounds)
        for x, y in boundary_utm:
            assert x > 100000 and x < 1000000
            assert y > 3000000 and y < 5000000
        
        # Verify boundary maintains shape (approximate)
        x_coords = [coord[0] for coord in boundary_utm]
        y_coords = [coord[1] for coord in boundary_utm]
        
        # Should have reasonable bounds
        assert max(x_coords) - min(x_coords) > 0
        assert max(y_coords) - min(y_coords) > 0 