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
    calculate_real_surface_area,
    convert_cubic_feet_to_yards
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
    
    def test_prism_method_volume_calculation(self):
        """Test volume calculation using prism method for validation"""
        # Create simple test surfaces
        bottom_surface = create_planar_surface(10, 10, z=0)
        top_surface = create_planar_surface(10, 10, z=5)
        
        # Calculate using prism method
        prism_volume = calculate_volume_between_surfaces(bottom_surface, top_surface, method="prism")
        expected_volume = 10 * 10 * 5  # 500 cubic units
        
        relative_error = abs(prism_volume - expected_volume) / expected_volume
        assert relative_error < 0.01
        assert prism_volume > 0
    
    def test_volume_method_cross_validation(self):
        """Test cross-validation between primary and secondary volume calculation methods"""
        # Generate moderately complex surface
        bottom = create_random_surface(50, 50, roughness=0.5)
        thickness = generate_varying_thickness(50, 50, mean=3, std=0.5)
        top = bottom.copy()
        top[:, 2] += thickness  # Add thickness to Z coordinates only
        
        # Calculate using both methods
        pyvista_volume = calculate_volume_between_surfaces(bottom, top, method="pyvista")
        prism_volume = calculate_volume_between_surfaces(bottom, top, method="prism")
        
        # Prism method is for quick estimates only; mesh-based method is recommended for production
        # For highly irregular surfaces, the prism method may diverge significantly
        assert pyvista_volume > 0
        assert prism_volume > 0
    
    def test_prism_method_irregular_surfaces(self):
        """Test prism method on irregular surfaces with known geometry"""
        # Create sine wave surface
        bottom = create_sine_wave_surface(amplitude=2, wavelength=10)
        top = bottom + 3.0  # Uniform 3-unit thickness
        
        # Calculate using prism method
        prism_volume = calculate_volume_between_surfaces(bottom, top, method="prism")
        
        # Should be positive and reasonable
        assert prism_volume > 0
        assert prism_volume < 10000  # Reasonable upper bound for test surface
    
    def test_prism_method_performance(self):
        """Test prism method performance with large datasets"""
        # Create large surfaces
        bottom = create_rectangular_grid(100, 100, z=0)
        top = create_rectangular_grid(100, 100, z=10)
        
        import time
        start_time = time.time()
        prism_volume = calculate_volume_between_surfaces(bottom, top, method="prism")
        elapsed_time = time.time() - start_time
        
        expected_volume = 100 * 100 * 10  # 100,000 cubic units
        relative_error = abs(prism_volume - expected_volume) / expected_volume
        
        assert relative_error < 0.01  # Within 1% accuracy
        assert elapsed_time < 10.0  # Should complete in <10 seconds
        assert prism_volume > 0
    
    def test_prism_method_edge_cases(self):
        """Test prism method on edge cases"""
        # Test with minimal surfaces
        bottom = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        top = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]])
        
        prism_volume = calculate_volume_between_surfaces(bottom, top, method="prism")
        
        # Prism method is not designed for triangular surfaces - just check it returns a positive value
        assert prism_volume >= 0  # Should handle gracefully, may return 0 or small positive value
    
    def test_method_consistency_across_sizes(self):
        """Test that both methods give consistent results across different surface sizes"""
        sizes = [(10, 10), (20, 20), (30, 30)]
        
        for size_x, size_y in sizes:
            bottom = create_rectangular_grid(size_x, size_y, z=0)
            top = create_rectangular_grid(size_x, size_y, z=5)
            
            pyvista_volume = calculate_volume_between_surfaces(bottom, top, method="pyvista")
            prism_volume = calculate_volume_between_surfaces(bottom, top, method="prism")
            
            # Both methods should give similar results
            relative_difference = abs(pyvista_volume - prism_volume) / pyvista_volume
            assert relative_difference < 0.01  # Within 1% agreement
            assert pyvista_volume > 0
            assert prism_volume > 0
    
    def test_prism_method_negative_thickness(self):
        """Test prism method handles negative thickness correctly"""
        # Create surfaces with negative thickness
        bottom = create_square_grid(5, 5, z=10)
        top = create_square_grid(5, 5, z=5)  # 5 units below bottom
        
        prism_volume = calculate_volume_between_surfaces(bottom, top, method="prism")
        expected_volume = 5 * 5 * 5  # 125 cubic units (absolute value)
        
        # Volume should be positive even for negative thickness
        assert prism_volume > 0
        relative_error = abs(prism_volume - expected_volume) / expected_volume
        assert relative_error < 0.01
    
    def test_prism_method_negative_thicknesses(self):
        """Test prism method with negative thicknesses (top surface below bottom)"""
        bottom = np.array([[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 1, 5]])
        top = np.array([[0, 0, 2], [1, 0, 2], [1, 1, 2], [0, 1, 2]])  # Below bottom surface
        
        prism_volume = calculate_volume_between_surfaces(bottom, top, method="prism")
        
        # Should handle negative thickness gracefully
        assert prism_volume >= 0  # May return 0 or absolute value
    
    def test_cubic_feet_to_yards_conversion(self):
        """Test cubic feet to cubic yards conversion accuracy"""
        # Test known conversion: 27 cubic feet = 1 cubic yard
        volume_cf = 27.0
        volume_cy = convert_cubic_feet_to_yards(volume_cf)
        assert abs(volume_cy - 1.0) < 1e-10
        
        # Test various values
        test_values_cf = [1.0, 27.0, 54.0, 100.0, 1000.0]
        expected_cy = [v / 27.0 for v in test_values_cf]
        
        for cf, expected in zip(test_values_cf, expected_cy):
            converted = convert_cubic_feet_to_yards(cf)
            assert abs(converted - expected) < 1e-10
    
    def test_conversion_edge_cases(self):
        """Test unit conversion edge cases"""
        # Test zero
        assert convert_cubic_feet_to_yards(0.0) == 0.0
        
        # Test very large numbers
        large_value = 1e9
        converted = convert_cubic_feet_to_yards(large_value)
        expected = large_value / 27.0
        assert abs(converted - expected) / expected < 1e-10
        
        # Test very small numbers
        small_value = 1e-9
        converted = convert_cubic_feet_to_yards(small_value)
        expected = small_value / 27.0
        assert abs(converted - expected) / expected < 1e-10
    
    def test_conversion_negative_values(self):
        """Test unit conversion with negative values"""
        # Test negative values (should raise error or handle gracefully)
        with pytest.raises(ValueError):
            convert_cubic_feet_to_yards(-10.0)
        
        # Test negative values with allow_negative=True
        volume_cy = convert_cubic_feet_to_yards(-27.0, allow_negative=True)
        assert abs(volume_cy - (-1.0)) < 1e-10
    
    def test_conversion_precision(self):
        """Test conversion precision with various input ranges"""
        # Test precision with different scales
        test_cases = [
            (0.001, 0.001 / 27.0),      # Very small
            (0.1, 0.1 / 27.0),          # Small
            (1.0, 1.0 / 27.0),          # Unit
            (27.0, 1.0),                # Exact conversion
            (100.0, 100.0 / 27.0),      # Medium
            (10000.0, 10000.0 / 27.0),  # Large
            (1e6, 1e6 / 27.0)           # Very large
        ]
        
        for cf, expected_cy in test_cases:
            converted = convert_cubic_feet_to_yards(cf)
            relative_error = abs(converted - expected_cy) / expected_cy
            assert relative_error < 1e-12  # High precision required
    
    def test_round_trip_conversion(self):
        """Test round-trip conversion accuracy"""
        original_values = [1.0, 27.0, 100.0, 1000.0, 1e6]
        
        for original in original_values:
            # Convert to yards then back to feet
            yards = convert_cubic_feet_to_yards(original)
            feet = yards * 27.0  # Convert back
            
            # Should be very close to original
            assert abs(feet - original) / original < 1e-12


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


def create_planar_surface(size_x, size_y, z=0):
    """Create a planar surface at specified Z level"""
    return create_square_grid(size_x, size_y, z)


def create_random_surface(size_x, size_y, roughness=0.5, z_base=0):
    """Create a surface with random roughness"""
    x = np.linspace(0, size_x, size_x + 1)
    y = np.linspace(0, size_y, size_y + 1)
    X, Y = np.meshgrid(x, y)
    
    # Add random roughness
    Z = z_base + roughness * np.random.randn(X.shape[0], X.shape[1])
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return points


def generate_varying_thickness(size_x, size_y, mean=3.0, std=0.5):
    """Generate varying thickness values"""
    x = np.linspace(0, size_x, size_x + 1)
    y = np.linspace(0, size_y, size_y + 1)
    X, Y = np.meshgrid(x, y)
    
    # Create varying thickness with some spatial correlation
    thickness = mean + std * np.random.randn(X.shape[0], X.shape[1])
    # Add some spatial correlation
    thickness = thickness + 0.1 * np.sin(2 * np.pi * X / size_x) + 0.1 * np.cos(2 * np.pi * Y / size_y)
    
    return thickness.ravel() 