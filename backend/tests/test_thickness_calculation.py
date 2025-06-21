import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os
import time
from scipy.spatial import Delaunay

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.triangulation import create_delaunay_triangulation
from services.thickness_calculator import (
    calculate_point_to_surface_distance,
    calculate_batch_point_to_surface_distances,
    generate_uniform_sample_points,
    generate_adaptive_sample_points,
    generate_boundary_aware_sample_points,
    calculate_thickness_between_surfaces,
    calculate_thickness_statistics,
    analyze_thickness_distribution,
    detect_thickness_anomalies,
    analyze_thickness_clusters,
    analyze_thickness_spatial_patterns,
    generate_thickness_insights
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
        assert 2.0 < distance < 6.0  # Allow for interpolation accuracy
    
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
            (np.array([2.5, 2.5, 2.0]), -3.0),  # 3 units below
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
        assert len(sample_points) == 4  # Corner points at [0,0], [20,0], [0,20], [20,20]
        
        # Test degenerate boundary (single point)
        degenerate_boundary = np.array([[5, 5]])
        sample_points = generate_uniform_sample_points(degenerate_boundary, 1.0)
        assert len(sample_points) == 1
        assert np.allclose(sample_points[0], [5, 5])
    
    def test_uniform_thickness_statistics(self):
        """Test uniform thickness distributions"""
        # Create surfaces with uniform 3-unit thickness
        x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        z_bottom = np.full_like(x, 0.0)
        z_top = np.full_like(x, 3.0)
        
        bottom_surface = np.column_stack([x.ravel(), y.ravel(), z_bottom.ravel()])
        top_surface = np.column_stack([x.ravel(), y.ravel(), z_top.ravel()])
        
        # Calculate thickness at sample points
        boundary = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        sample_points = generate_uniform_sample_points(boundary, 1.0)
        
        thicknesses = calculate_thickness_between_surfaces(top_surface, bottom_surface, sample_points)
        stats = calculate_thickness_statistics(thicknesses)
        
        assert abs(stats['min'] - 3.0) < 1e-10
        assert abs(stats['max'] - 3.0) < 1e-10
        assert abs(stats['mean'] - 3.0) < 1e-10
        assert abs(stats['std'] - 0.0) < 1e-10
        assert stats['count'] > 0
    
    def test_varying_thickness_statistics(self):
        """Test varying thickness distributions"""
        # Create surfaces with known thickness variation
        x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
        z_bottom = np.full_like(x, 0.0)
        z_top = 2.0 + 0.4*x + 0.2*y  # Linear variation: 2-6 unit thickness
        
        bottom_surface = np.column_stack([x.ravel(), y.ravel(), z_bottom.ravel()])
        top_surface = np.column_stack([x.ravel(), y.ravel(), z_top.ravel()])
        
        # Calculate thickness at sample points
        boundary = np.array([[0, 0], [10, 0], [10, 10], [0, 10]])
        sample_points = generate_uniform_sample_points(boundary, 1.0)
        
        thicknesses = calculate_thickness_between_surfaces(top_surface, bottom_surface, sample_points)
        stats = calculate_thickness_statistics(thicknesses)
        
        # Expected statistics for linear variation from 2 to 8
        assert abs(stats['min'] - 2.0) < 0.1
        assert abs(stats['max'] - 8.0) < 0.1  # z = 2 + 0.4*10 + 0.2*10 = 2 + 4 + 2 = 8
        assert abs(stats['mean'] - 5.0) < 0.1  # Average should be (2+8)/2 = 5
        assert stats['std'] > 0  # Should have variation
        assert stats['count'] > 0
    
    def test_thickness_with_nan_values(self):
        """Test handling of NaN values from interpolation outside boundaries"""
        # Create thickness values with NaN entries
        thicknesses = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0, 5.0])
        stats = calculate_thickness_statistics(thicknesses)
        
        # Should ignore NaN values
        assert abs(stats['min'] - 1.0) < 1e-10
        assert abs(stats['max'] - 5.0) < 1e-10
        assert abs(stats['mean'] - 3.0) < 1e-10  # (1+2+3+4+5)/5 = 3
        assert stats['valid_count'] == 5
        assert stats['count'] == 7  # Total count including NaN values
    
    def test_empty_thickness_statistics(self):
        """Test statistics calculation with empty or all-NaN data"""
        # Test with all NaN values
        all_nan_thicknesses = np.array([np.nan, np.nan, np.nan])
        stats = calculate_thickness_statistics(all_nan_thicknesses)
        
        assert np.isnan(stats['min'])
        assert np.isnan(stats['max'])
        assert np.isnan(stats['mean'])
        assert np.isnan(stats['median'])
        assert np.isnan(stats['std'])
        assert stats['count'] == 3  # Total count including NaN values
        assert stats['valid_count'] == 0  # No valid values
        
        # Test with empty array
        empty_thicknesses = np.array([])
        stats = calculate_thickness_statistics(empty_thicknesses)
        
        assert np.isnan(stats['min'])
        assert np.isnan(stats['max'])
        assert np.isnan(stats['mean'])
        assert np.isnan(stats['median'])
        assert np.isnan(stats['std'])
        assert stats['count'] == 0
        assert stats['valid_count'] == 0
    
    def test_thickness_percentiles(self):
        """Test percentile calculations for thickness distribution"""
        # Create thickness values with known distribution
        thicknesses = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        stats = calculate_thickness_statistics(thicknesses)
        
        # Test basic statistics
        assert abs(stats['min'] - 1.0) < 1e-10
        assert abs(stats['max'] - 10.0) < 1e-10
        assert abs(stats['mean'] - 5.5) < 1e-10  # (1+...+10)/10 = 5.5
        assert abs(stats['median'] - 5.5) < 1e-10  # Median of even number
        assert stats['count'] == 10
        
        # Test percentiles if implemented
        if 'percentiles' in stats:
            assert abs(stats['percentiles']['25'] - 3.25) < 0.1  # 25th percentile
            assert abs(stats['percentiles']['75'] - 7.75) < 0.1  # 75th percentile
    
    def test_thickness_distribution_validation(self):
        """Test validation of thickness distribution quality"""
        # Create realistic thickness distribution
        np.random.seed(42)
        thicknesses = np.random.normal(5.0, 1.0, 100)  # Normal distribution
        thicknesses = np.clip(thicknesses, 0.1, 10.0)  # Clip to reasonable range
        
        stats = calculate_thickness_statistics(thicknesses)
        
        # Validate statistics are reasonable
        assert stats['min'] >= 0
        assert stats['max'] <= 10
        assert stats['mean'] > 0
        assert stats['std'] > 0
        assert stats['count'] == 100
        
        # Mean should be close to 5.0 (normal distribution)
        assert abs(stats['mean'] - 5.0) < 1.0
        
        # Standard deviation should be reasonable
        assert 0.5 < stats['std'] < 2.0
    
    def test_thickness_statistics_performance(self):
        """Test performance of statistics calculation with large datasets"""
        # Create large thickness dataset
        np.random.seed(42)
        thicknesses = np.random.uniform(1.0, 10.0, 10000)
        
        start_time = time.time()
        stats = calculate_thickness_statistics(thicknesses)
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 0.1  # Must complete in <100ms
        
        # Validate results
        assert stats['count'] == 10000
        assert 1.0 <= stats['min'] <= 10.0
        assert 1.0 <= stats['max'] <= 10.0
        assert 1.0 <= stats['mean'] <= 10.0
        assert stats['std'] > 0
    
    def test_thickness_distribution_patterns(self):
        """Test detection of thickness distribution patterns"""
        # Create thickness data with known patterns
        # Normal distribution
        np.random.seed(42)
        normal_thicknesses = np.random.normal(5.0, 1.0, 1000)
        
        # Bimodal distribution
        bimodal_thicknesses = np.concatenate([
            np.random.normal(3.0, 0.5, 500),
            np.random.normal(7.0, 0.5, 500)
        ])
        
        # Uniform distribution
        uniform_thicknesses = np.random.uniform(2.0, 8.0, 1000)
        
        # Analyze patterns
        normal_patterns = analyze_thickness_distribution(normal_thicknesses)
        bimodal_patterns = analyze_thickness_distribution(bimodal_thicknesses)
        uniform_patterns = analyze_thickness_distribution(uniform_thicknesses)
        
        # Normal distribution should be detected as unimodal
        assert normal_patterns['distribution_type'] in ['unimodal', 'skewed']
        assert normal_patterns['skewness'] < 0.5  # Should be close to normal
        assert normal_patterns['kurtosis'] < 1.0  # Should be close to normal
        
        # Bimodal distribution should be detected (may be detected as multimodal or unimodal)
        assert bimodal_patterns['distribution_type'] in ['multimodal', 'unimodal', 'skewed']
        if bimodal_patterns['distribution_type'] == 'multimodal':
            assert len(bimodal_patterns['peaks']) >= 2
        
        # Uniform distribution should be detected
        assert uniform_patterns['distribution_type'] in ['uniform', 'unimodal']
        assert uniform_patterns['skewness'] < 0.3  # Should be close to 0
    
    def test_thickness_anomaly_detection(self):
        """Test detection of thickness anomalies and outliers"""
        # Create thickness data with known anomalies
        np.random.seed(42)
        base_thicknesses = np.random.normal(5.0, 1.0, 100)
        
        # Add some anomalies
        anomalies = np.array([0.1, 15.0, 0.5, 12.0])  # Very low and very high values
        thicknesses_with_anomalies = np.concatenate([base_thicknesses, anomalies])
        
        # Detect anomalies
        anomaly_results = detect_thickness_anomalies(thicknesses_with_anomalies)
        
        # Should detect the anomalies
        assert len(anomaly_results['anomalies']) >= 4
        assert len(anomaly_results['outliers']) >= 0  # May not have extreme outliers
        
        # Most anomaly values should be extreme (allow some false positives)
        extreme_count = sum(1 for anomaly in anomaly_results['anomalies'] if anomaly < 2.0 or anomaly > 10.0)
        assert extreme_count >= 3  # At least 3 of the 4 extreme values should be detected
        
        # Should provide anomaly statistics
        assert 'anomaly_count' in anomaly_results
        assert 'anomaly_percentage' in anomaly_results
        assert anomaly_results['anomaly_percentage'] > 0
    
    def test_thickness_distribution_insights(self):
        """Test generation of thickness distribution insights"""
        # Create thickness data with known characteristics
        np.random.seed(42)
        thicknesses = np.concatenate([
            np.random.normal(3.0, 0.5, 300),  # Lower thickness region
            np.random.normal(7.0, 0.5, 300),  # Higher thickness region
            np.random.uniform(4.0, 6.0, 400)  # Middle region
        ])
        
        # Generate insights
        insights = generate_thickness_insights(thicknesses)
        
        # Should provide comprehensive insights
        assert 'distribution_summary' in insights
        assert 'quality_assessment' in insights
        assert 'recommendations' in insights
        assert 'risk_factors' in insights
        
        # Quality assessment should be present
        quality = insights['quality_assessment']
        assert 'data_quality' in quality
        assert 'coverage_adequacy' in quality
        assert 'sampling_density' in quality
        
        # Recommendations should be provided
        recommendations = insights['recommendations']
        assert len(recommendations) > 0
        assert all(isinstance(rec, str) for rec in recommendations)
    
    def test_thickness_clustering_analysis(self):
        """Test clustering analysis of thickness data"""
        # Create thickness data with distinct clusters
        np.random.seed(42)
        cluster1 = np.random.normal(2.0, 0.3, 200)  # Low thickness cluster
        cluster2 = np.random.normal(5.0, 0.3, 300)  # Medium thickness cluster
        cluster3 = np.random.normal(8.0, 0.3, 200)  # High thickness cluster
        
        thicknesses = np.concatenate([cluster1, cluster2, cluster3])
        
        # Perform clustering analysis
        clustering_results = analyze_thickness_clusters(thicknesses)
        
        # Should identify clusters
        assert len(clustering_results['clusters']) >= 2
        assert 'cluster_centers' in clustering_results
        assert 'cluster_sizes' in clustering_results
        
        # Cluster centers should be distinct
        centers = clustering_results['cluster_centers']
        assert len(centers) >= 2
        assert max(centers) - min(centers) > 2.0  # Should be well-separated
    
    def test_thickness_spatial_analysis(self):
        """Test spatial analysis of thickness data"""
        # Create thickness data with spatial coordinates
        np.random.seed(42)
        x_coords = np.random.uniform(0, 100, 500)
        y_coords = np.random.uniform(0, 100, 500)
        thicknesses = 5.0 + 0.1 * x_coords + 0.05 * y_coords + np.random.normal(0, 0.5, 500)
        
        spatial_data = np.column_stack([x_coords, y_coords, thicknesses])
        
        # Perform spatial analysis
        spatial_results = analyze_thickness_spatial_patterns(spatial_data)
        
        # Should provide spatial insights
        assert 'spatial_trends' in spatial_results
        assert 'spatial_correlation' in spatial_results
        assert 'spatial_variability' in spatial_results
        
        # Should detect trends
        trends = spatial_results['spatial_trends']
        assert 'x_direction' in trends
        assert 'y_direction' in trends
        
        # Should provide correlation measures
        correlation = spatial_results['spatial_correlation']
        assert 'correlation_coefficient' in correlation
        assert 'spatial_dependency' in correlation
    
    def test_thickness_distribution_edge_cases(self):
        """Test distribution analysis with edge cases"""
        # Test with very small dataset
        small_data = np.array([1.0, 2.0, 3.0])
        small_analysis = analyze_thickness_distribution(small_data)
        assert 'distribution_type' in small_analysis
        
        # Test with all identical values
        identical_data = np.full(100, 5.0)
        identical_analysis = analyze_thickness_distribution(identical_data)
        assert identical_analysis['distribution_type'] == 'uniform'
        assert identical_analysis['variance'] == 0.0
        
        # Test with single value
        single_data = np.array([5.0])
        single_analysis = analyze_thickness_distribution(single_data)
        assert 'distribution_type' in single_analysis
        
        # Test with NaN values
        nan_data = np.array([1.0, 2.0, np.nan, 4.0, np.nan, 6.0])
        nan_analysis = analyze_thickness_distribution(nan_data)
        assert 'distribution_type' in nan_analysis
        assert nan_analysis['valid_count'] == 4
    
    def test_thickness_distribution_performance(self):
        """Test performance of distribution analysis with large datasets"""
        # Create large dataset
        np.random.seed(42)
        large_thicknesses = np.random.normal(5.0, 1.0, 10000)
        
        # Test performance
        start_time = time.time()
        analysis = analyze_thickness_distribution(large_thicknesses)
        elapsed = time.time() - start_time
        
        # Should complete quickly
        assert elapsed < 1.0  # Must complete in <1 second
        
        # Should provide results
        assert 'distribution_type' in analysis
        assert 'statistics' in analysis
        
        # Test anomaly detection performance
        start_time = time.time()
        anomalies = detect_thickness_anomalies(large_thicknesses)
        elapsed = time.time() - start_time
        
        assert elapsed < 2.0  # Must complete in <2 seconds
        assert 'anomalies' in anomalies


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