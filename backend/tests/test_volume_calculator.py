import pytest
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the app directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'app'))

from services.volume_calculator import (
    calculate_volume_between_surfaces,
    create_square_grid,
    create_rectangular_grid,
    create_sine_wave_surface,
    calculate_surface_area,
    calculate_real_surface_area
)


class TestVolumeCalculation:
    """Test volume calculation accuracy for simple geometric shapes"""
    
    def test_pyramid_volume_calculation(self):
        """Test pyramid volume: V = (1/3) * base_area * height"""
        # Create pyramid with square base 500'x500', height 15'
        base_points = create_square_grid(500, 500, z=0)
        apex_point = np.array([[250, 250, 15]])  # Apex at center, 15' height
        bottom_surface = base_points
        top_surface = np.full((len(base_points), 3), [250, 250, 15])  # All points at apex
        
        print(f"Pyramid test: {len(base_points)} points in base surface")
        
        calculated_volume = calculate_volume_between_surfaces(bottom_surface, top_surface)
        expected_volume = (500 * 500 * 15) / 3  # 1,250,000 cubic units
        
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.01  # Within 1% accuracy
        assert calculated_volume > 0  # Volume should be positive
    
    def test_rectangular_prism_volume(self):
        """Test rectangular prism volume: V = length * width * height"""
        # Create two parallel rectangular surfaces
        bottom = create_rectangular_grid(10, 20, z=0)  # 10x20 base
        top = create_rectangular_grid(10, 20, z=5)     # 5 units higher
        
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        expected_volume = 10 * 20 * 5  # 1000 cubic units
        
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.005  # Within 0.5% for simple shapes
        assert calculated_volume > 0
    
    def test_irregular_surface_volume(self):
        """Test volume calculation for irregular surfaces with known thickness"""
        # Create surfaces with known volume difference
        bottom = create_sine_wave_surface(amplitude=2, wavelength=10)
        top = bottom + 3.0  # Uniform 3-unit thickness
        
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        
        # Calculate real surface area using triangulation
        real_surface_area = calculate_real_surface_area(bottom)
        expected_volume = real_surface_area * 3.0
        
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.02  # Within 2% for irregular shapes
        assert calculated_volume > 0
    
    def test_zero_thickness_volume(self):
        """Test volume calculation when surfaces are identical (zero thickness)"""
        # Create identical surfaces
        surface = create_square_grid(5, 5, z=10)
        
        calculated_volume = calculate_volume_between_surfaces(surface, surface)
        expected_volume = 0.0
        
        assert abs(calculated_volume - expected_volume) < 1e-10
        assert calculated_volume >= 0  # Should be exactly zero or very small positive
    
    def test_negative_thickness_volume(self):
        """Test volume calculation when top surface is below bottom surface"""
        # Create surfaces with negative thickness
        bottom = create_square_grid(5, 5, z=10)
        top = create_square_grid(5, 5, z=5)  # 5 units below bottom
        
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        expected_volume = 5 * 5 * 5  # 125 cubic units (absolute value)
        
        # Volume should be positive even for negative thickness
        assert calculated_volume > 0
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.01
    
    def test_small_volume_accuracy(self):
        """Test volume calculation accuracy for small volumes"""
        # Create small volume: 1x1x1 cube
        bottom = create_square_grid(1, 1, z=0)
        top = create_square_grid(1, 1, z=1)
        
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        expected_volume = 1.0  # 1 cubic unit
        
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.01  # Within 1% for small volumes
    
    def test_large_volume_accuracy(self):
        """Test volume calculation accuracy for large volumes"""
        # Create large volume: 100x100x50
        bottom = create_rectangular_grid(100, 100, z=0)
        top = create_rectangular_grid(100, 100, z=50)
        
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        expected_volume = 100 * 100 * 50  # 500,000 cubic units
        
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.01  # Within 1% for large volumes
    
    def test_volume_with_single_point_surfaces(self):
        """Test volume calculation with minimal point sets"""
        # Single point surfaces (degenerate case)
        bottom = np.array([[0, 0, 0]])
        top = np.array([[0, 0, 1]])
        
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        # Should handle gracefully, may return 0 or small positive value
        assert calculated_volume >= 0
    
    def test_volume_with_three_point_surfaces(self):
        """Test volume calculation with triangular surfaces"""
        # Triangular surfaces
        bottom = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        top = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]])
        
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        expected_volume = 0.5 * 2  # Triangle area * height = 1 cubic unit
        
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        assert relative_error < 0.05  # Within 5% for triangular surfaces
        assert calculated_volume > 0
    
    def test_volume_calculation_performance(self):
        """Test volume calculation performance with moderate dataset"""
        # Create moderate-sized surfaces
        bottom = create_rectangular_grid(50, 50, z=0)
        top = create_rectangular_grid(50, 50, z=10)
        
        import time
        start_time = time.time()
        calculated_volume = calculate_volume_between_surfaces(bottom, top)
        elapsed_time = time.time() - start_time
        
        expected_volume = 50 * 50 * 10  # 25,000 cubic units
        relative_error = abs(calculated_volume - expected_volume) / expected_volume
        
        assert relative_error < 0.01  # Within 1% accuracy
        assert elapsed_time < 5.0  # Should complete in <5 seconds
        assert calculated_volume > 0


def create_square_grid(size_x, size_y, z=0):
    """Create a square grid of points at specified Z level"""
    x = np.linspace(0, size_x, size_x + 1)
    y = np.linspace(0, size_y, size_y + 1)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    return points


def create_rectangular_grid(length, width, z=0):
    """Create a rectangular grid of points at specified Z level"""
    x = np.linspace(0, length, length + 1)
    y = np.linspace(0, width, width + 1)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    return points


def create_sine_wave_surface(amplitude=2, wavelength=10, size=20):
    """Create a surface with sine wave variation"""
    x = np.linspace(0, size, size + 1)
    y = np.linspace(0, size, size + 1)
    X, Y = np.meshgrid(x, y)
    Z = amplitude * np.sin(2 * np.pi * X / wavelength)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return points


def calculate_surface_area(surface_points):
    """Calculate approximate surface area of a point cloud surface"""
    # Simple approximation: bounding box area
    x_coords = surface_points[:, 0]
    y_coords = surface_points[:, 1]
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    return x_range * y_range 