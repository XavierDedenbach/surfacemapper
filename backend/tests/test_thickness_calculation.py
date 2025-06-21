import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
import time

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.triangulation import create_delaunay_triangulation
from services.thickness_calculator import (
    calculate_point_to_surface_distance,
    calculate_batch_point_to_surface_distances,
    generate_uniform_sample_points,
    generate_adaptive_sample_points,
    generate_boundary_aware_sample_points
)


class TestThicknessCalculation:
    """Test thickness calculation accuracy for various surface types"""
    
    def test_point_to_flat_plane_distance(self):
        """Test distance from point to flat plane (analytical solution)"""
        # Create flat plane at z=10
        x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        z = np.full_like(x, 10.0)
        plane_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from plane points
        plane_tin = create_delaunay_triangulation(plane_points[:, :2])
        plane_tin.z_values = plane_points[:, 2]
        
        # Test point 5 units above plane center
        test_point = np.array([5, 5, 15])
        distance = calculate_point_to_surface_distance(test_point, plane_tin)
        
        assert abs(distance - 5.0) < 1e-10
    
    def test_point_to_sloped_plane_distance(self):
        """Test distance to sloped plane with known geometry"""
        # Create sloped plane: z = 0.5*x + 0.3*y + 10
        x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        z = 0.5*x + 0.3*y + 10
        plane_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from plane points
        plane_tin = create_delaunay_triangulation(plane_points[:, :2])
        plane_tin.z_values = plane_points[:, 2]
        
        # Test point above known location
        test_x, test_y = 5.0, 5.0
        expected_surface_z = 0.5*test_x + 0.3*test_y + 10  # 12.5
        test_point = np.array([test_x, test_y, 15.0])
        distance = calculate_point_to_surface_distance(test_point, plane_tin)
        expected_distance = 15.0 - expected_surface_z  # 2.5
        
        assert abs(distance - expected_distance) < 1e-6
    
    def test_point_to_curved_surface_distance(self):
        """Test distance to curved surface (sine wave)"""
        # Create sine wave surface: z = 2*sin(π*x/5) + 10
        x, y = np.meshgrid(np.linspace(0, 10, 21), np.linspace(0, 10, 21))
        z = 2*np.sin(np.pi*x/5) + 10
        surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Test point above peak of sine wave
        test_point = np.array([2.5, 5.0, 15.0])  # Above peak (z ≈ 12)
        distance = calculate_point_to_surface_distance(test_point, surface_tin)
        
        # Should be approximately 3 units above surface
        assert 2.5 < distance < 3.5
        assert distance > 0
    
    def test_point_outside_surface_boundary(self):
        """Test distance calculation for points outside surface boundary"""
        # Create small surface
        x, y = np.meshgrid(np.linspace(0, 5, 6), np.linspace(0, 5, 6))
        z = np.full_like(x, 10.0)
        surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Test point outside convex hull
        test_point = np.array([10.0, 10.0, 15.0])
        distance = calculate_point_to_surface_distance(test_point, surface_tin)
        
        # Should return NaN for points outside boundary
        assert np.isnan(distance)
    
    def test_point_on_surface_vertex(self):
        """Test distance calculation for point exactly on surface vertex"""
        # Create simple triangular surface
        surface_points = np.array([[0, 0, 10], [1, 0, 10], [0, 1, 10]])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Test point exactly on vertex
        test_point = np.array([0, 0, 10])
        distance = calculate_point_to_surface_distance(test_point, surface_tin)
        
        # Should be very close to zero
        assert abs(distance) < 1e-10
    
    def test_point_on_surface_edge(self):
        """Test distance calculation for point on surface edge"""
        # Create simple triangular surface
        surface_points = np.array([[0, 0, 10], [1, 0, 10], [0, 1, 10]])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Test point on edge (midpoint)
        test_point = np.array([0.5, 0, 10])
        distance = calculate_point_to_surface_distance(test_point, surface_tin)
        
        # Should be very close to zero
        assert abs(distance) < 1e-10
    
    def test_batch_distance_calculation(self):
        """Test batch distance calculation performance"""
        # Create test surface
        x, y = np.meshgrid(np.linspace(0, 10, 21), np.linspace(0, 10, 21))
        z = 0.5*x + 0.3*y + 10
        surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Generate test points above surface
        np.random.seed(42)  # For reproducible results
        test_points = np.column_stack([
            np.random.uniform(1, 9, 1000),  # x coordinates
            np.random.uniform(1, 9, 1000),  # y coordinates
            np.random.uniform(15, 20, 1000)  # z coordinates above surface
        ])
        
        start_time = time.time()
        distances = calculate_batch_point_to_surface_distances(test_points, surface_tin)
        elapsed = time.time() - start_time
        
        assert len(distances) == 1000
        assert elapsed < 1.0  # Must complete 1000 points in <1 second
        assert not np.any(np.isnan(distances))  # No NaN values for points inside boundary
        # Note: distances can be negative if points are below surface
    
    def test_distance_calculation_accuracy(self):
        """Test distance calculation accuracy with known geometry"""
        # Create flat plane at z=5
        x, y = np.meshgrid(np.linspace(0, 5, 6), np.linspace(0, 5, 6))
        z = np.full_like(x, 5.0)
        surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Test multiple points with known distances
        test_cases = [
            (np.array([2.5, 2.5, 8.0]), 3.0),   # 3 units above
            (np.array([2.5, 2.5, 2.0]), 3.0),   # 3 units below
            (np.array([2.5, 2.5, 5.0]), 0.0),   # On surface
            (np.array([1.0, 1.0, 7.0]), 2.0),   # 2 units above
        ]
        
        for test_point, expected_distance in test_cases:
            calculated_distance = calculate_point_to_surface_distance(test_point, surface_tin)
            assert abs(calculated_distance - expected_distance) < 1e-6
    
    def test_distance_with_irregular_surface(self):
        """Test distance calculation with irregular surface"""
        # Create irregular surface with varying elevation
        x, y = np.meshgrid(np.linspace(0, 10, 21), np.linspace(0, 10, 21))
        z = 5 + 2*np.sin(np.pi*x/5) + 1.5*np.cos(np.pi*y/5)
        surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Test point above surface
        test_point = np.array([5.0, 5.0, 12.0])
        distance = calculate_point_to_surface_distance(test_point, surface_tin)
        
        # Should be positive and reasonable
        assert distance > 0
        assert distance < 10  # Shouldn't be too large
        assert not np.isnan(distance)
    
    def test_distance_performance_large_surface(self):
        """Test distance calculation performance with large surface"""
        # Create large surface
        x, y = np.meshgrid(np.linspace(0, 100, 101), np.linspace(0, 100, 101))
        z = 10 + 0.1*x + 0.05*y
        surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        
        # Create TIN from surface points
        surface_tin = create_delaunay_triangulation(surface_points[:, :2])
        surface_tin.z_values = surface_points[:, 2]
        
        # Test single point calculation performance
        test_point = np.array([50.0, 50.0, 20.0])
        
        start_time = time.time()
        distance = calculate_point_to_surface_distance(test_point, surface_tin)
        elapsed = time.time() - start_time
        
        assert elapsed < 0.1  # Must complete in <100ms
        assert distance > 0
        assert not np.isnan(distance)
    
    def test_uniform_grid_sampling(self):
        """Test uniform grid sampling strategy"""
        # Test regular grid generation
        boundary = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        sample_spacing = 1.0
        
        sample_points = generate_uniform_sample_points(boundary, sample_spacing)
        
        # Should generate 11x11 = 121 points
        assert len(sample_points) == 121
        
        # Check point spacing
        x_coords = np.unique(sample_points[:, 0])
        y_coords = np.unique(sample_points[:, 1])
        assert len(x_coords) == 11
        assert len(y_coords) == 11
        
        # Check spacing accuracy
        x_spacing = np.diff(x_coords)
        y_spacing = np.diff(y_coords)
        assert np.allclose(x_spacing, sample_spacing, atol=1e-10)
        assert np.allclose(y_spacing, sample_spacing, atol=1e-10)
        
        # Check boundary coverage
        assert np.min(sample_points[:, 0]) >= 0
        assert np.max(sample_points[:, 0]) <= 10
        assert np.min(sample_points[:, 1]) >= 0
        assert np.max(sample_points[:, 1]) <= 10
    
    def test_adaptive_sampling(self):
        """Test adaptive sampling based on surface complexity"""
        # Create rough surface (high complexity)
        x, y = np.meshgrid(np.linspace(0, 10, 21), np.linspace(0, 10, 21))
        z_rough = 10 + 2*np.sin(np.pi*x/2) + 1.5*np.cos(np.pi*y/3) + 0.5*np.random.randn(21, 21)
        rough_surface = np.column_stack([x.ravel(), y.ravel(), z_rough.ravel()])
        
        # Create smooth surface (low complexity)
        z_smooth = 10 + 0.1*x + 0.05*y
        smooth_surface = np.column_stack([x.ravel(), y.ravel(), z_smooth.ravel()])
        
        boundary = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        max_spacing = 2.0
        
        rough_samples = generate_adaptive_sample_points(rough_surface, boundary, max_spacing)
        smooth_samples = generate_adaptive_sample_points(smooth_surface, boundary, max_spacing)
        
        # Rough surface should generate more sample points
        assert len(rough_samples) > len(smooth_samples)
        
        # Both should respect maximum spacing
        if len(rough_samples) > 1:
            rough_spacing = np.min(np.diff(np.unique(rough_samples[:, 0])))
            assert rough_spacing <= max_spacing
        
        if len(smooth_samples) > 1:
            smooth_spacing = np.min(np.diff(np.unique(smooth_samples[:, 0])))
            assert smooth_spacing <= max_spacing
    
    def test_boundary_aware_sampling(self):
        """Test sampling within irregular boundaries"""
        # Create irregular boundary (L-shaped)
        boundary = np.array([
            [0, 0], [5, 0], [5, 3], [8, 3], [8, 8], [0, 8]
        ])
        sample_spacing = 1.0
        
        sample_points = generate_boundary_aware_sample_points(boundary, sample_spacing)
        
        # All points should be inside boundary
        for point in sample_points:
            assert is_point_in_polygon(point, boundary)
        
        # Should have reasonable number of points
        assert len(sample_points) > 0
        assert len(sample_points) < 100  # Not too many for L-shape
    
    def test_sampling_density_control(self):
        """Test sampling density control parameters"""
        boundary = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        
        # Test different spacing values
        spacings = [0.5, 1.0, 2.0, 5.0]
        expected_counts = [441, 121, 36, 9]  # Approximate expected counts
        
        for spacing, expected_count in zip(spacings, expected_counts):
            sample_points = generate_uniform_sample_points(boundary, spacing)
            
            # Should be close to expected count
            assert abs(len(sample_points) - expected_count) < expected_count * 0.1
            
            # Check spacing is respected
            if len(sample_points) > 1:
                actual_spacing = np.min(np.diff(np.unique(sample_points[:, 0])))
                assert abs(actual_spacing - spacing) < spacing * 0.1
    
    def test_sampling_performance(self):
        """Test sampling performance with large boundaries"""
        # Create large boundary
        boundary = np.array([[0, 0], [100, 0], [100, 100], [0, 100]])
        sample_spacing = 5.0
        
        start_time = time.time()
        sample_points = generate_uniform_sample_points(boundary, sample_spacing)
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 1.0  # Must complete in <1 second
        
        # Should generate reasonable number of points
        expected_count = (21 * 21)  # 100/5 + 1 = 21 points per dimension
        assert abs(len(sample_points) - expected_count) < expected_count * 0.1
    
    def test_sampling_edge_cases(self):
        """Test sampling with edge cases"""
        # Test very small boundary
        small_boundary = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        sample_points = generate_uniform_sample_points(small_boundary, 0.5)
        assert len(sample_points) == 9  # 3x3 grid
        
        # Test very large spacing
        large_boundary = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        sample_points = generate_uniform_sample_points(large_boundary, 20.0)
        assert len(sample_points) == 1  # Only center point
        
        # Test degenerate boundary (single point)
        degenerate_boundary = np.array([[5, 5]])
        sample_points = generate_uniform_sample_points(degenerate_boundary, 1.0)
        assert len(sample_points) == 1
        assert np.allclose(sample_points[0], [5, 5])


def create_test_surface_tin():
    """Helper function to create a test surface TIN"""
    x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    z = 10 + 0.5*x + 0.3*y
    surface_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    surface_tin = create_delaunay_triangulation(surface_points[:, :2])
    surface_tin.z_values = surface_points[:, 2]
    return surface_tin


def generate_test_points_above_surface(num_points):
    """Helper function to generate test points above surface"""
    np.random.seed(42)
    return np.column_stack([
        np.random.uniform(1, 9, num_points),
        np.random.uniform(1, 9, num_points),
        np.random.uniform(15, 20, num_points)
    ])


def is_point_in_polygon(point, polygon):
    """Helper function to check if point is inside polygon"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside 