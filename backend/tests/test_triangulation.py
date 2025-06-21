"""
Comprehensive tests for Delaunay triangulation and TIN creation
"""
import numpy as np
import pytest
import time
from scipy.spatial import Delaunay
from scipy.spatial._qhull import QhullError
from typing import List, Tuple


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
    
    def validate_triangulation_quality(self, triangulation: Delaunay) -> float:
        """Calculate triangulation quality metric (0-1, higher is better)"""
        if len(triangulation.simplices) == 0:
            return 0.0
        
        # Calculate aspect ratio of triangles (equilateral = 1.0, degenerate = 0.0)
        simplices = triangulation.simplices
        points = triangulation.points
        
        aspect_ratios = []
        for simplex in simplices:
            triangle_points = points[simplex]
            
            # Calculate side lengths
            sides = []
            for i in range(3):
                j = (i + 1) % 3
                side_length = np.linalg.norm(triangle_points[i] - triangle_points[j])
                sides.append(side_length)
            
            # Calculate aspect ratio (minimum side / maximum side)
            if max(sides) > 0:
                aspect_ratio = min(sides) / max(sides)
                aspect_ratios.append(aspect_ratio)
        
        return np.mean(aspect_ratios) if aspect_ratios else 0.0
    
    def test_delaunay_triangulation_square(self):
        """Test triangulation of unit square"""
        points = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        triangulation = Delaunay(points)
        
        # Square should have 2 triangles
        assert len(triangulation.simplices) == 2
        
        # Check quality metric
        quality = self.validate_triangulation_quality(triangulation)
        assert quality > 0.5  # Should have reasonable quality
        
        # Verify all points are used
        used_points = set()
        for simplex in triangulation.simplices:
            used_points.update(simplex)
        assert len(used_points) == 4  # All 4 points should be used
    
    def test_delaunay_triangulation_triangle(self):
        """Test triangulation of triangle"""
        points = self.create_triangle_points()
        triangulation = Delaunay(points)
        
        # Triangle should have 1 triangle (itself)
        assert len(triangulation.simplices) == 1
        
        # Check quality metric
        quality = self.validate_triangulation_quality(triangulation)
        assert quality > 0.8  # Triangle should have high quality
    
    def test_delaunay_triangulation_circle(self):
        """Test triangulation of circle points"""
        points = self.create_circle_points(radius=1.0, n_points=12)
        triangulation = Delaunay(points)
        
        # Circle should have multiple triangles
        assert len(triangulation.simplices) > 0
        
        # Check quality metric
        quality = self.validate_triangulation_quality(triangulation)
        assert quality > 0.3  # Circle triangulation should have reasonable quality
        
        # Verify convex hull property
        hull_points = points[triangulation.convex_hull]
        assert len(hull_points) >= 3  # Should have at least 3 hull points
    
    def test_delaunay_triangulation_rectangular_grid(self):
        """Test triangulation of rectangular grid"""
        points = self.create_square_grid(5, 5)  # 5x5 grid = 25 points
        triangulation = Delaunay(points)
        
        # Grid should have multiple triangles
        assert len(triangulation.simplices) > 0
        
        # Check quality metric
        quality = self.validate_triangulation_quality(triangulation)
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
        triangulation = Delaunay(points)
        elapsed = time.time() - start_time
        
        assert elapsed < 1.0  # Should complete in <1 second
        assert len(triangulation.simplices) > 0
    
    def test_triangulation_performance_medium(self):
        """Test triangulation performance with medium dataset"""
        points = self.generate_random_points_2d(10000)
        
        start_time = time.time()
        triangulation = Delaunay(points)
        elapsed = time.time() - start_time
        
        assert elapsed < 5.0  # Should complete in <5 seconds
        assert len(triangulation.simplices) > 0
    
    def test_triangulation_performance_large(self):
        """Test triangulation performance with large dataset"""
        points = self.generate_random_points_2d(100000)
        
        start_time = time.time()
        triangulation = Delaunay(points)
        elapsed = time.time() - start_time
        
        assert elapsed < 30.0  # Must complete in <30 seconds
        assert len(triangulation.simplices) > 0
    
    def test_triangulation_edge_cases_collinear(self):
        """Test triangulation with collinear points"""
        # Test collinear points - should raise QhullError
        collinear_points = np.array([[0, 0], [1, 0], [2, 0]])
        
        with pytest.raises(QhullError):
            Delaunay(collinear_points)
        
        # Test nearly collinear points - should work but with poor quality
        nearly_collinear = np.array([[0, 0], [1, 0], [2, 0.001]])
        triangulation = Delaunay(nearly_collinear)
        
        # Should still form triangles but with poor quality
        assert len(triangulation.simplices) > 0
        quality = self.validate_triangulation_quality(triangulation)
        assert quality < 0.6  # Should have poor quality
    
    def test_triangulation_edge_cases_duplicate(self):
        """Test triangulation with duplicate points"""
        # Test duplicate points
        duplicate_points = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        triangulation = Delaunay(duplicate_points)
        
        # Should handle duplicates gracefully
        assert len(triangulation.simplices) > 0
        
        # Check that unique points are used
        unique_points = np.unique(triangulation.points.view(np.void), axis=0)
        assert len(unique_points) == 3  # Should have 3 unique points
    
    def test_triangulation_edge_cases_single_point(self):
        """Test triangulation with single point"""
        single_point = np.array([[0, 0]])
        
        with pytest.raises(QhullError):
            Delaunay(single_point)
    
    def test_triangulation_edge_cases_two_points(self):
        """Test triangulation with two points"""
        two_points = np.array([[0, 0], [1, 0]])
        
        with pytest.raises(QhullError):
            Delaunay(two_points)
    
    def test_triangulation_edge_cases_three_collinear(self):
        """Test triangulation with exactly three collinear points"""
        three_collinear = np.array([[0, 0], [1, 0], [2, 0]])
        
        with pytest.raises(QhullError):
            Delaunay(three_collinear)
    
    def test_triangulation_quality_metrics(self):
        """Test triangulation quality metrics"""
        # Test with equilateral triangle (should have high quality)
        equilateral = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
        triangulation = Delaunay(equilateral)
        quality = self.validate_triangulation_quality(triangulation)
        assert quality > 0.9  # Equilateral triangle should have very high quality
        
        # Test with nearly degenerate triangle (should have lower quality)
        nearly_degenerate = np.array([[0, 0], [1, 0], [0.5, 0.001]])
        triangulation = Delaunay(nearly_degenerate)
        quality = self.validate_triangulation_quality(triangulation)
        assert quality < 0.6  # Nearly degenerate triangle should have lower quality
    
    def test_triangulation_convex_hull_property(self):
        """Test that triangulation respects convex hull property"""
        points = self.generate_random_points_2d(50)
        triangulation = Delaunay(points)
        
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
            Delaunay(np.array([]))
        
        # Test with single point
        with pytest.raises(QhullError):
            Delaunay(np.array([[0, 0]]))
        
        # Test with two points
        with pytest.raises(QhullError):
            Delaunay(np.array([[0, 0], [1, 0]]))
    
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
        triangulation = Delaunay(points_2d)
        
        # Should create 2D triangulation using only X,Y coordinates
        assert len(triangulation.simplices) == 2
        assert triangulation.points.shape[1] == 2  # Should be 2D
        
        # Verify triangulation works correctly
        quality = self.validate_triangulation_quality(triangulation)
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
        triangulation = Delaunay(points)
        
        # Should create valid triangulation
        assert len(triangulation.simplices) > 0
        
        # Check quality
        quality = TestDelaunayTriangulation().validate_triangulation_quality(triangulation)
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
        triangulation = Delaunay(points)
        
        # Should create valid triangulation
        assert len(triangulation.simplices) > 0
        
        # Check quality
        quality = TestDelaunayTriangulation().validate_triangulation_quality(triangulation)
        assert quality > 0.2  # Irregular boundary should still have reasonable quality
    
    def test_triangulation_memory_efficiency(self):
        """Test triangulation memory efficiency with large dataset"""
        # Test with large dataset
        points = TestDelaunayTriangulation().generate_random_points_2d(50000)
        
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        triangulation = Delaunay(points)
        
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (not more than 50x the data size for triangulation)
        # Triangulation creates additional data structures (simplices, neighbors, etc.)
        expected_max_memory = len(points) * 2 * 8 * 50 / 1024 / 1024  # 50x data size
        assert memory_increase < expected_max_memory
        
        # Should create valid triangulation
        assert len(triangulation.simplices) > 0 