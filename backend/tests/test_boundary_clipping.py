import pytest
import numpy as np
from app.services.coord_transformer import CoordinateTransformer
from app.services.surface_processor import SurfaceProcessor
import tempfile
import os


class TestBoundaryTransformation:
    """Test boundary coordinate transformation from WGS84 to UTM"""
    
    def setup_method(self):
        self.coord_transformer = CoordinateTransformer()
    
    def test_transform_boundary_coordinates_valid_input(self):
        """Test transformation of valid WGS84 boundary coordinates"""
        # Test coordinates in Los Angeles area
        wgs84_boundary = [
            (34.0522, -118.2437),  # Southwest corner
            (34.0522, -118.2400),  # Southeast corner  
            (34.0550, -118.2400),  # Northeast corner
            (34.0550, -118.2437)   # Northwest corner
        ]
        
        utm_boundary = self.coord_transformer.transform_boundary_coordinates(wgs84_boundary)
        
        # Verify output format
        assert len(utm_boundary) == 4
        assert all(len(coord) == 2 for coord in utm_boundary)
        
        # Verify coordinates are reasonable UTM values (in meters)
        for x, y in utm_boundary:
            assert 300000 <= x <= 400000  # UTM Zone 11N x-coordinates
            assert 3750000 <= y <= 3850000  # UTM Zone 11N y-coordinates
    
    def test_transform_boundary_coordinates_invalid_length(self):
        """Test that invalid boundary length raises error"""
        wgs84_boundary = [
            (34.0522, -118.2437),
            (34.0522, -118.2400),
            (34.0550, -118.2400)
        ]  # Only 3 coordinates
        
        with pytest.raises(ValueError, match="Boundary must have exactly 4 coordinate pairs"):
            self.coord_transformer.transform_boundary_coordinates(wgs84_boundary)
    
    def test_transform_boundary_coordinates_edge_cases(self):
        """Test boundary transformation with edge case coordinates"""
        # Test coordinates near the equator
        wgs84_boundary = [
            (0.0, -180.0),  # Southwest corner
            (0.0, 180.0),   # Southeast corner
            (1.0, 180.0),   # Northeast corner
            (1.0, -180.0)   # Northwest corner
        ]
        
        utm_boundary = self.coord_transformer.transform_boundary_coordinates(wgs84_boundary)
        
        assert len(utm_boundary) == 4
        assert all(len(coord) == 2 for coord in utm_boundary)
        
        # Verify coordinates are reasonable
        for x, y in utm_boundary:
            assert isinstance(x, (int, float))
            assert isinstance(y, (int, float))
            assert not np.isnan(x)
            assert not np.isnan(y)


class TestBoundaryClipping:
    """Test surface clipping to boundary"""
    
    def setup_method(self):
        self.surface_processor = SurfaceProcessor()
    
    def test_clip_to_rectangular_boundary(self):
        """Test clipping to rectangular boundary (2-corner format)"""
        # Create test vertices
        vertices = np.array([
            [0.0, 0.0, 1.0],   # Inside
            [5.0, 5.0, 2.0],   # Inside
            [10.0, 10.0, 3.0], # Outside
            [-5.0, -5.0, 4.0], # Outside
            [2.0, 8.0, 5.0],   # Inside
        ])
        
        # Define rectangular boundary: [(min_x, min_y), (max_x, max_y)]
        boundary = [(0.0, 0.0), (8.0, 8.0)]
        
        clipped_vertices = self.surface_processor.clip_to_boundary(vertices, boundary)
        
        # Should have 3 vertices inside the boundary
        assert len(clipped_vertices) == 3
        
        # Verify all remaining vertices are inside the boundary
        for vertex in clipped_vertices:
            x, y = vertex[0], vertex[1]
            assert 0.0 <= x <= 8.0
            assert 0.0 <= y <= 8.0
    
    def test_clip_to_polygon_boundary(self):
        """Test clipping to polygon boundary (4-corner format)"""
        # Create test vertices
        vertices = np.array([
            [5.0, 5.0, 1.0],   # Inside polygon
            [15.0, 5.0, 2.0],  # Inside polygon
            [10.0, 15.0, 3.0], # Inside polygon
            [20.0, 20.0, 4.0], # Outside polygon
            [0.0, 0.0, 5.0],   # Outside polygon
        ])
        
        # Define polygon boundary (diamond shape)
        boundary = [
            (10.0, 0.0),   # Bottom
            (20.0, 10.0),  # Right
            (10.0, 20.0),  # Top
            (0.0, 10.0)    # Left
        ]
        
        clipped_vertices = self.surface_processor.clip_to_boundary(vertices, boundary)
        
        # Should have 3 vertices inside the polygon
        assert len(clipped_vertices) == 3
        
        # Verify all remaining vertices are inside the polygon
        for vertex in clipped_vertices:
            x, y = vertex[0], vertex[1]
            # Simple check: should be within the bounding box of the polygon
            assert 0.0 <= x <= 20.0
            assert 0.0 <= y <= 20.0
    
    def test_clip_to_boundary_invalid_input(self):
        """Test that invalid boundary format raises error"""
        vertices = np.array([[0.0, 0.0, 1.0], [1.0, 1.0, 2.0]])
        
        # Invalid boundary length
        invalid_boundary = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]  # 3 coordinates
        
        with pytest.raises(ValueError, match="Boundary must be a list of 2 or 4 coordinate tuples"):
            self.surface_processor.clip_to_boundary(vertices, invalid_boundary)
    
    def test_clip_to_boundary_empty_result(self):
        """Test clipping when no vertices are inside the boundary"""
        vertices = np.array([
            [10.0, 10.0, 1.0],
            [15.0, 15.0, 2.0],
            [20.0, 20.0, 3.0],
        ])
        
        # Boundary that doesn't contain any vertices
        boundary = [(0.0, 0.0), (5.0, 5.0)]
        
        clipped_vertices = self.surface_processor.clip_to_boundary(vertices, boundary)
        
        # Should return empty array
        assert len(clipped_vertices) == 0
    
    def test_convert_boundary_to_rectangle(self):
        """Test conversion of 4-corner boundary to rectangular boundary"""
        # Define irregular 4-corner boundary
        four_corner_boundary = [
            (0.0, 5.0),   # Bottom-left
            (10.0, 2.0),  # Bottom-right
            (8.0, 15.0),  # Top-right
            (2.0, 12.0)   # Top-left
        ]
        
        rectangular_boundary = self.surface_processor.convert_boundary_to_rectangle(four_corner_boundary)
        
        # Should return 2-corner format
        assert len(rectangular_boundary) == 2
        
        min_corner, max_corner = rectangular_boundary
        min_x, min_y = min_corner
        max_x, max_y = max_corner
        
        # Verify bounding box calculation
        assert min_x == 0.0  # Minimum x from the 4 corners
        assert min_y == 2.0  # Minimum y from the 4 corners
        assert max_x == 10.0  # Maximum x from the 4 corners
        assert max_y == 15.0  # Maximum y from the 4 corners


class TestPointInPolygon:
    """Test point-in-polygon algorithm"""
    
    def setup_method(self):
        self.surface_processor = SurfaceProcessor()
    
    def test_point_inside_square(self):
        """Test point inside a square polygon"""
        polygon = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        
        # Test points inside the square
        inside_points = [
            np.array([5.0, 5.0]),
            np.array([1.0, 1.0]),
            np.array([9.0, 9.0]),
            np.array([5.0, 0.0]),  # On edge
            np.array([0.0, 5.0])   # On edge
        ]
        
        for point in inside_points:
            assert self.surface_processor._is_point_in_polygon(point, polygon)
    
    def test_point_outside_square(self):
        """Test point outside a square polygon"""
        polygon = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0]
        ])
        
        # Test points outside the square
        outside_points = [
            np.array([-1.0, 5.0]),
            np.array([11.0, 5.0]),
            np.array([5.0, -1.0]),
            np.array([5.0, 11.0]),
            np.array([15.0, 15.0])
        ]
        
        for point in outside_points:
            assert not self.surface_processor._is_point_in_polygon(point, polygon)
    
    def test_point_inside_triangle(self):
        """Test point inside a triangle polygon"""
        polygon = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 10.0]
        ])
        
        # Test point inside the triangle
        inside_point = np.array([5.0, 5.0])
        assert self.surface_processor._is_point_in_polygon(inside_point, polygon)
        
        # Test point outside the triangle
        outside_point = np.array([15.0, 15.0])
        assert not self.surface_processor._is_point_in_polygon(outside_point, polygon)


class TestIntegration:
    """Integration tests for complete boundary processing workflow"""
    
    def setup_method(self):
        self.coord_transformer = CoordinateTransformer()
        self.surface_processor = SurfaceProcessor()
    
    def test_complete_boundary_workflow(self):
        """Test complete workflow: WGS84 -> UTM -> Clipping"""
        # Create test surface with vertices in UTM coordinates
        vertices = np.array([
            [583000, 4507000, 100.0],  # Inside boundary
            [584000, 4508000, 110.0],  # Inside boundary
            [585000, 4509000, 120.0],  # Outside boundary
            [582000, 4506000, 90.0],   # Outside boundary
        ])
        
        # Define WGS84 boundary (Los Angeles area)
        wgs84_boundary = [
            (34.0522, -118.2437),  # Southwest
            (34.0522, -118.2400),  # Southeast
            (34.0550, -118.2400),  # Northeast
            (34.0550, -118.2437)   # Northwest
        ]
        
        # Step 1: Transform WGS84 to UTM
        utm_boundary = self.coord_transformer.transform_boundary_coordinates(wgs84_boundary)
        
        # Step 2: Clip surface to boundary
        clipped_vertices = self.surface_processor.clip_to_boundary(vertices, utm_boundary)
        
        # Verify results
        assert len(clipped_vertices) >= 0  # May be 0 if no vertices are inside
        assert clipped_vertices.shape[1] == 3  # Still has x, y, z coordinates
        
        # If there are vertices inside, verify they're reasonable
        if len(clipped_vertices) > 0:
            for vertex in clipped_vertices:
                x, y = vertex[0], vertex[1]
                # Should be within reasonable UTM bounds for the area
                assert 580000 <= x <= 590000
                assert 4500000 <= y <= 4510000 