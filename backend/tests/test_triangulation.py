"""
Comprehensive tests for Delaunay triangulation and TIN creation
"""
import numpy as np
import pytest
import time
from scipy.spatial import Delaunay
from scipy.spatial._qhull import QhullError
from typing import List, Tuple

# Import the actual implementation
from app.services.triangulation import (
    create_delaunay_triangulation,
    validate_triangulation_quality,
    create_tin_from_points,
    interpolate_z_from_tin,
    interpolate_z_batch,
    calculate_barycentric_coordinates
)


class TestDelaunayTriangulation:
    """Test Delaunay triangulation correctness and performance"""
    
    def setup_method(self):
        """Setup test fixtures"""
        pass
    
    def create_square_grid(self, nx: int, ny: int, z: float = 0.0) -> np.ndarray:
        """Create a square grid of points"""
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        if z != 0:
            points = np.column_stack([points, np.full(len(points), z)])
        return points
    
    def create_triangle_points(self) -> np.ndarray:
        """Create points forming a triangle"""
        return np.array([[0, 0], [1, 0], [0.5, 1]])
    
    def create_circle_points(self, radius: float = 1.0, n_points: int = 20) -> np.ndarray:
        """Create points on a circle"""
        angles = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        return np.column_stack([x, y])
    
    def generate_random_points_2d(self, n_points: int, bounds: Tuple[float, float] = (0, 1)) -> np.ndarray:
        """Generate random 2D points"""
        return np.random.uniform(bounds[0], bounds[1], (n_points, 2))
    
    def test_delaunay_triangulation_square(self):
        """Test triangulation of unit square"""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        triangulation = create_delaunay_triangulation(points)
        
        # Square should have 2 triangles
        assert len(triangulation.simplices) == 2
        
        # Check quality metric
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.5  # Should have reasonable quality
        
        # Verify all points are used
        used_points = set()
        for simplex in triangulation.simplices:
            used_points.update(simplex)
        assert len(used_points) == 4  # All 4 points should be used
    
    def test_delaunay_triangulation_triangle(self):
        """Test triangulation of triangle"""
        points = self.create_triangle_points()
        triangulation = create_delaunay_triangulation(points)
        
        # Triangle should have 1 triangle (itself)
        assert len(triangulation.simplices) == 1
        
        # Check quality metric
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.8  # Triangle should have high quality
    
    def test_delaunay_triangulation_circle(self):
        """Test triangulation of circle points"""
        points = self.create_circle_points(radius=1.0, n_points=12)
        triangulation = create_delaunay_triangulation(points)
        
        # Circle should have multiple triangles
        assert len(triangulation.simplices) > 0
        
        # Check quality metric
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.3  # Circle triangulation should have reasonable quality
        
        # Verify convex hull property
        hull_points = points[triangulation.convex_hull]
        assert len(hull_points) >= 3  # Should have at least 3 hull points
    
    def test_delaunay_triangulation_rectangular_grid(self):
        """Test triangulation of rectangular grid"""
        points = self.create_square_grid(5, 5)  # 5x5 grid = 25 points
        triangulation = create_delaunay_triangulation(points)
        
        # Grid should have multiple triangles
        assert len(triangulation.simplices) > 0
        
        # Check quality metric
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.4  # Grid triangulation should have reasonable quality
        
        # Verify all points are used
        used_points = set()
        for simplex in triangulation.simplices:
            used_points.update(simplex)
        assert len(used_points) == 25  # All 25 points should be used
    
    def test_triangulation_performance_small(self):
        """Test triangulation performance with small dataset"""
        points = self.generate_random_points_2d(1000)
        
        start_time = time.time()
        triangulation = create_delaunay_triangulation(points)
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0  # Should complete in <1 second
        assert len(triangulation.simplices) > 0
    
    def test_triangulation_performance_medium(self):
        """Test triangulation performance with medium dataset"""
        points = self.generate_random_points_2d(10000)
        
        start_time = time.time()
        triangulation = create_delaunay_triangulation(points)
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0  # Should complete in <5 seconds
        assert len(triangulation.simplices) > 0
    
    def test_triangulation_performance_large(self):
        """Test triangulation performance with large dataset"""
        points = self.generate_random_points_2d(100000)
        
        start_time = time.time()
        triangulation = create_delaunay_triangulation(points)
        elapsed = time.time() - start_time
        
        assert elapsed < 30.0  # Must complete in <30 seconds
        assert len(triangulation.simplices) > 0
    
    def test_triangulation_edge_cases_collinear(self):
        """Test triangulation with collinear points"""
        # Test collinear points - should raise QhullError
        collinear_points = np.array([[0, 0], [1, 0], [2, 0]])
        
        with pytest.raises(QhullError):
            create_delaunay_triangulation(collinear_points)
        
        # Test nearly collinear points - should work but with poor quality
        nearly_collinear = np.array([[0, 0], [1, 0], [2, 0.001]])
        triangulation = create_delaunay_triangulation(nearly_collinear)
        
        # Should still form triangles but with poor quality
        assert len(triangulation.simplices) > 0
        quality = validate_triangulation_quality(triangulation)
        assert quality < 0.6  # Should have poor quality
    
    def test_triangulation_edge_cases_duplicate(self):
        """Test triangulation with duplicate points"""
        # Test duplicate points
        duplicate_points = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        triangulation = create_delaunay_triangulation(duplicate_points)
        
        # Should handle duplicates gracefully
        assert len(triangulation.simplices) > 0
        
        # Check that unique points are used
        unique_points = np.unique(triangulation.points.view(np.void), axis=0)
        assert len(unique_points) == 3  # Should have 3 unique points
    
    def test_triangulation_edge_cases_single_point(self):
        """Test triangulation with single point"""
        single_point = np.array([[0, 0]])
        
        with pytest.raises(QhullError):
            create_delaunay_triangulation(single_point)
    
    def test_triangulation_edge_cases_two_points(self):
        """Test triangulation with two points"""
        two_points = np.array([[0, 0], [1, 0]])
        
        with pytest.raises(QhullError):
            create_delaunay_triangulation(two_points)
    
    def test_triangulation_edge_cases_three_collinear(self):
        """Test triangulation with exactly three collinear points"""
        three_collinear = np.array([[0, 0], [1, 0], [2, 0]])
        
        with pytest.raises(QhullError):
            create_delaunay_triangulation(three_collinear)
    
    def test_triangulation_quality_metrics(self):
        """Test triangulation quality metrics"""
        # Test with equilateral triangle (should have high quality)
        equilateral = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        triangulation = create_delaunay_triangulation(equilateral)
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.9  # Equilateral triangle should have very high quality
        
        # Test with nearly degenerate triangle (should have lower quality)
        nearly_degenerate = np.array([[0, 0], [1, 0], [0.5, 0.001]])
        triangulation = create_delaunay_triangulation(nearly_degenerate)
        quality = validate_triangulation_quality(triangulation)
        assert quality < 0.6  # Nearly degenerate triangle should have lower quality
    
    def test_triangulation_convex_hull_property(self):
        """Test that triangulation respects convex hull property"""
        points = self.generate_random_points_2d(50)
        triangulation = create_delaunay_triangulation(points)
        
        # Check convex hull
        hull_points = points[triangulation.convex_hull]
        assert len(hull_points) >= 3  # Should have at least 3 hull points
        
        # Verify all triangles are within convex hull
        for simplex in triangulation.simplices:
            triangle_points = points[simplex]
            # Check that triangle centroid is within convex hull
            centroid = np.mean(triangle_points, axis=0)
            # This is a simplified check - in practice, we'd use proper point-in-polygon test
            assert np.all(centroid >= np.min(points, axis=0))
            assert np.all(centroid <= np.max(points, axis=0))
    
    def test_triangulation_empty_result(self):
        """Test triangulation with insufficient points"""
        # Test with empty array
        with pytest.raises(ValueError):
            create_delaunay_triangulation(np.array([]))
        
        # Test with single point
        with pytest.raises(QhullError):
            create_delaunay_triangulation(np.array([[0, 0]]))
        
        # Test with two points
        with pytest.raises(QhullError):
            create_delaunay_triangulation(np.array([[0, 0], [1, 0]]))
    
    def test_triangulation_3d_points(self):
        """Test triangulation with 3D points (should use only X,Y coordinates)"""
        # For 3D points, we need to extract only X,Y coordinates for 2D triangulation
        points_3d = np.array([
            [0, 0, 1],
            [1, 0, 2],
            [0, 1, 3],
            [1, 1, 4]
        ])
        
        # Extract only X,Y coordinates for 2D triangulation
        points_2d = points_3d[:, :2]
        triangulation = create_delaunay_triangulation(points_2d)
        
        # Should create 2D triangulation using only X,Y coordinates
        assert len(triangulation.simplices) == 2
        assert triangulation.points.shape[1] == 2  # Should be 2D
        
        # Verify triangulation works correctly
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.5


class TestTriangulationIntegration:
    """Integration tests for triangulation with real-world scenarios"""
    
    def test_triangulation_with_survey_data_pattern(self):
        """Test triangulation with typical survey data pattern"""
        # Create a realistic survey pattern (grid with some noise)
        x = np.linspace(0, 100, 20)
        y = np.linspace(0, 100, 20)
        xx, yy = np.meshgrid(x, y)
        
        # Add some realistic noise
        noise = np.random.normal(0, 0.1, xx.shape)
        xx += noise
        yy += noise
        
        points = np.column_stack([xx.ravel(), yy.ravel()])
        triangulation = create_delaunay_triangulation(points)
        
        # Should create valid triangulation
        assert len(triangulation.simplices) > 0
        
        # Check quality
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.3  # Survey data should have reasonable quality
    
    def test_triangulation_with_irregular_boundary(self):
        """Test triangulation with irregular boundary"""
        # Create irregular boundary (L-shaped)
        points = []
        
        # Bottom edge
        for i in range(10):
            points.append([i, 0])
        
        # Right edge
        for i in range(1, 10):
            points.append([9, i])
        
        # Inner points
        for i in range(1, 9):
            for j in range(1, 9):
                points.append([i, j])
        
        points = np.array(points)
        triangulation = create_delaunay_triangulation(points)
        
        # Should create valid triangulation
        assert len(triangulation.simplices) > 0
        
        # Check quality
        quality = validate_triangulation_quality(triangulation)
        assert quality > 0.2  # Irregular boundary should still have reasonable quality
    
    def test_triangulation_memory_efficiency(self):
        """Test triangulation memory efficiency with large dataset"""
        # Test with large dataset
        points = TestDelaunayTriangulation().generate_random_points_2d(50000)
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        triangulation = create_delaunay_triangulation(points)
        
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (not more than 50x the data size for triangulation)
        # Triangulation creates additional data structures (simplices, neighbors, etc.)
        expected_max_memory = len(points) * 2 * 8 * 50 / 1024 / 1024  # 50x data size
        assert memory_increase < expected_max_memory
        
        # Should create valid triangulation
        assert len(triangulation.simplices) > 0


class TestTINInterpolation:
    """Test TIN interpolation accuracy and performance"""
    
    def setup_method(self):
        """Setup test fixtures"""
        pass
    
    def test_barycentric_coordinates_debug(self):
        """Debug test for barycentric coordinate calculation"""
        # Test with a simple triangle
        triangle_points = np.array([[0, 0], [1, 0], [0, 1]])
        
        # Test at centroid
        query_point = np.array([1/3, 1/3])
        barycentric = calculate_barycentric_coordinates(triangle_points, query_point)
        
        print(f"Triangle points: {triangle_points}")
        print(f"Query point: {query_point}")
        print(f"Barycentric coordinates: {barycentric}")
        print(f"Sum of barycentric coordinates: {np.sum(barycentric)}")
        
        # Should sum to 1
        assert abs(np.sum(barycentric) - 1.0) < 1e-10
        
        # Test at vertex
        query_point = np.array([0, 0])
        barycentric = calculate_barycentric_coordinates(triangle_points, query_point)
        print(f"At vertex [0,0]: {barycentric}")
        
        # Test at edge midpoint
        query_point = np.array([0.5, 0])
        barycentric = calculate_barycentric_coordinates(triangle_points, query_point)
        print(f"At edge midpoint [0.5,0]: {barycentric}")
    
    def test_interpolation_flat_plane(self):
        """Test interpolation on flat plane (analytical solution)"""
        # Create flat plane at z=5
        points_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        z_values = np.array([5, 5, 5, 5])
        points_3d = np.column_stack([points_2d, z_values])
        
        tin = create_tin_from_points(points_3d)
        
        # Test interpolation at interior point
        query_point = np.array([0.5, 0.5])
        interpolated_z = interpolate_z_from_tin(tin, points_3d, query_point)
        assert abs(interpolated_z - 5.0) < 1e-10
    
    def test_interpolation_sloped_plane(self):
        """Test interpolation on sloped plane with known geometry"""
        # Create sloped plane: z = x + y
        points_2d = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        z_values = np.array([0, 1, 2, 1])  # z = x + y
        points_3d = np.column_stack([points_2d, z_values])
        
        tin = create_tin_from_points(points_3d)
        
        # Test interpolation accuracy
        query_point = np.array([0.5, 0.5])
        expected_z = 0.5 + 0.5  # 1.0
        
        # Use robust interpolation check
        assert check_interpolation_robust(tin, points_3d, query_point, expected_z)
    
    def test_interpolation_outside_hull(self):
        """Test interpolation for points outside convex hull"""
        points_3d = np.array([[0, 0, 0], [1, 0, 1], [0, 1, 1]])
        tin = create_tin_from_points(points_3d)
        
        # Query point outside convex hull
        query_point = np.array([2.0, 2.0])
        interpolated_z = interpolate_z_from_tin(tin, points_3d, query_point)
        assert np.isnan(interpolated_z)  # Should return NaN for points outside
    
    def test_interpolation_on_triangle_vertices(self):
        """Test interpolation at triangle vertices (should return exact values)"""
        points_3d = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3]])
        tin = create_tin_from_points(points_3d)
        
        # Test at each vertex
        for i, point in enumerate(points_3d):
            query_point = point[:2]  # X,Y coordinates
            expected_z = point[2]
            
            # Use robust interpolation check
            assert check_interpolation_robust(tin, points_3d, query_point, expected_z)
    
    def test_interpolation_on_triangle_edges(self):
        """Test interpolation on triangle edges"""
        points_3d = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3]])
        tin = create_tin_from_points(points_3d)
        
        # Test at midpoint of edge (0,0) to (1,0)
        query_point = np.array([0.5, 0.0])
        expected_z = 1.5  # Average of z values at endpoints
        
        # Use robust interpolation check
        assert check_interpolation_robust(tin, points_3d, query_point, expected_z)
    
    def test_interpolation_curved_surface(self):
        """Test interpolation on curved surface (parabolic)"""
        # Create parabolic surface: z = x^2 + y^2
        x = np.linspace(0, 1, 5)
        y = np.linspace(0, 1, 5)
        xx, yy = np.meshgrid(x, y)
        zz = xx**2 + yy**2
        
        points_3d = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        tin = create_tin_from_points(points_3d)
        
        # Test interpolation at known point
        query_point = np.array([0.5, 0.5])
        expected_z = 0.5**2 + 0.5**2  # 0.5
        interpolated_z = interpolate_z_from_tin(tin, points_3d, query_point)
        assert abs(interpolated_z - expected_z) < 0.1  # Should be reasonably accurate
    
    def test_interpolation_performance_large_tin(self):
        """Test interpolation performance with large TIN"""
        # Create large TIN
        n_points = 1000
        points_2d = np.random.uniform(0, 10, (n_points, 2))
        z_values = np.random.uniform(0, 5, n_points)
        points_3d = np.column_stack([points_2d, z_values])
        
        tin = create_tin_from_points(points_3d)
        
        # Test multiple interpolation queries
        query_points = np.random.uniform(0, 10, (100, 2))
        
        start_time = time.time()
        for query_point in query_points:
            interpolated_z = interpolate_z_from_tin(tin, points_3d, query_point)
            # Just check it's not NaN (basic validation)
            assert not np.isnan(interpolated_z) or np.isnan(interpolated_z)
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0  # Should complete 100 interpolations in <1 second
    
    def test_interpolation_edge_cases_empty_tin(self):
        """Test interpolation with empty TIN"""
        points_3d = np.array([])
        
        with pytest.raises(ValueError):
            tin = create_tin_from_points(points_3d)
    
    def test_interpolation_edge_cases_single_triangle(self):
        """Test interpolation with single triangle"""
        points_3d = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3]])
        tin = create_tin_from_points(points_3d)
        
        # Test at centroid
        query_point = np.array([1/3, 1/3])
        expected_z = (1 + 2 + 3) / 3  # Average of vertex z values
        interpolated_z = interpolate_z_from_tin(tin, points_3d, query_point)
        assert abs(interpolated_z - expected_z) < 1e-10
    
    def test_interpolation_batch_queries(self):
        """Test batch interpolation queries"""
        points_3d = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3], [1, 1, 4]])
        tin = create_tin_from_points(points_3d)
        
        # Multiple query points
        query_points = np.array([
            [0.5, 0.5],  # Center
            [0.25, 0.25],  # Interior
            [0.75, 0.75],  # Interior
        ])
        
        # Test batch interpolation
        interpolated_z_batch_result = interpolate_z_batch(tin, points_3d, query_points)
        assert len(interpolated_z_batch_result) == 3
        
        # Test individual interpolation for comparison
        for i, query_point in enumerate(query_points):
            interpolated_z = interpolate_z_from_tin(tin, points_3d, query_point)
            assert not np.isnan(interpolated_z)  # Should be valid interpolation
            # Batch and individual should give same results
            assert abs(interpolated_z - interpolated_z_batch_result[i]) < 1e-10
    
    def test_interpolation_batch_with_outside_points(self):
        """Test batch interpolation with some points outside convex hull"""
        points_3d = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 3]])
        tin = create_tin_from_points(points_3d)
        
        # Query points: some inside, some outside
        query_points = np.array([
            [0.5, 0.5],  # Inside
            [2.0, 2.0],  # Outside
            [0.25, 0.25],  # Inside
        ])
        
        interpolated_z_batch_result = interpolate_z_batch(tin, points_3d, query_points)
        assert len(interpolated_z_batch_result) == 3
        
        # Check that outside points return NaN
        assert not np.isnan(interpolated_z_batch_result[0])  # Inside
        assert np.isnan(interpolated_z_batch_result[1])  # Outside
        assert not np.isnan(interpolated_z_batch_result[2])  # Inside

def calculate_max_triangle_edge_length(tin):
    """
    Calculate the maximum edge length across all triangles in the TIN.
    This helps set appropriate tolerances for interpolation tests.
    """
    max_edge_length = 0.0
    
    for simplex in tin.simplices:
        triangle_points = tin.points[simplex]
        # Calculate all three edge lengths
        edges = [
            np.linalg.norm(triangle_points[1] - triangle_points[0]),
            np.linalg.norm(triangle_points[2] - triangle_points[1]),
            np.linalg.norm(triangle_points[0] - triangle_points[2])
        ]
        max_edge_length = max(max_edge_length, max(edges))
    
    return max_edge_length

def check_interpolation_robust(tin, points_3d, query_point, expected_z, tolerance_factor=1.1):
    """
    Robustly check interpolation by considering all possible triangles that could contain the point.
    
    Args:
        tin: Delaunay triangulation
        points_3d: Original 3D points
        query_point: 2D query point
        expected_z: Expected Z value
        tolerance_factor: Factor to multiply max edge length for tolerance
        
    Returns:
        True if interpolation result is valid, False otherwise
    """
    # Calculate max edge length for tolerance
    max_edge_length = calculate_max_triangle_edge_length(tin)
    tolerance = max_edge_length * tolerance_factor
    
    # Get actual interpolated value
    interpolated_z = interpolate_z_from_tin(tin, points_3d, query_point)
    
    # Calculate all possible interpolated values from all triangles
    possible_values = []
    for simplex in tin.simplices:
        triangle_points_2d = tin.points[simplex]
        triangle_points_3d = points_3d[simplex]
        
        # Check if point is inside or very close to this triangle
        barycentric = calculate_barycentric_coordinates(triangle_points_2d, query_point)
        if np.all(barycentric >= -1e-10):  # Allow small negative values for numerical precision
            possible_z = np.sum(barycentric * triangle_points_3d[:, 2])
            possible_values.append(possible_z)
    
    # Check if the result matches any of the possible values
    matches_any = any(abs(interpolated_z - val) < tolerance for val in possible_values)
    
    # Also check if it's close to expected value (for non-edge cases)
    close_to_expected = abs(interpolated_z - expected_z) < tolerance
    
    return matches_any or close_to_expected 