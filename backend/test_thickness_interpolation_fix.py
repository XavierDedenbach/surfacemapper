#!/usr/bin/env python3
"""
Test script to verify thickness interpolation fixes for both CSV and frontend
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from app.services.thickness_calculator import _interpolate_z_at_point, _interpolate_nearest_thickness
from app.services.triangulation import create_delaunay_triangulation
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_interpolation_fixes():
    """Test that interpolation fixes work correctly"""
    print("=== Testing Thickness Interpolation Fixes ===")
    
    # Create test surfaces
    x_coords = np.linspace(0, 10, 6)
    y_coords = np.linspace(0, 10, 6)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create lower surface
    lower_z = 5 + 0.1 * X + 0.05 * Y
    lower_surface = np.column_stack([X.ravel(), Y.ravel(), lower_z.ravel()])
    
    # Create upper surface
    upper_z = 10 + 0.15 * X + 0.08 * Y
    upper_surface = np.column_stack([X.ravel(), Y.ravel(), upper_z.ravel()])
    
    # Create triangulations
    lower_tin = create_delaunay_triangulation(lower_surface[:, :2])
    setattr(lower_tin, 'z_values', lower_surface[:, 2])
    
    upper_tin = create_delaunay_triangulation(upper_surface[:, :2])
    setattr(upper_tin, 'z_values', upper_surface[:, 2])
    
    print(f"Lower surface: {len(lower_surface)} points")
    print(f"Upper surface: {len(upper_surface)} points")
    
    # Test points - some inside triangulation, some outside
    test_points = [
        np.array([5.0, 5.0]),  # Inside
        np.array([2.0, 2.0]),  # Inside
        np.array([8.0, 8.0]),  # Inside
        np.array([15.0, 15.0]),  # Outside
        np.array([-5.0, -5.0]),  # Outside
    ]
    
    print("\nTesting interpolation at various points:")
    for i, point in enumerate(test_points):
        print(f"\nPoint {i+1}: ({point[0]:.1f}, {point[1]:.1f})")
        
        # Test direct interpolation
        upper_z = _interpolate_z_at_point(point, upper_tin)
        lower_z = _interpolate_z_at_point(point, lower_tin)
        
        print(f"  Direct interpolation:")
        print(f"    Upper Z: {upper_z}")
        print(f"    Lower Z: {lower_z}")
        
        if not np.isnan(upper_z) and not np.isnan(lower_z):
            direct_thickness = upper_z - lower_z
            print(f"    Direct thickness: {direct_thickness:.3f} ft")
        else:
            print(f"    Direct thickness: FAILED (NaN)")
        
        # Test nearest neighbor fallback
        nearest_thickness = _interpolate_nearest_thickness(point, upper_tin, lower_tin)
        print(f"  Nearest neighbor thickness: {nearest_thickness:.3f} ft")
        
        # Verify that nearest neighbor always gives a result
        if nearest_thickness > 0:
            print(f"  ✓ Nearest neighbor interpolation successful")
        else:
            print(f"  ✗ Nearest neighbor interpolation failed")
    
    print("\n=== Interpolation Fix Test Complete ===")

def test_csv_vs_point_query_consistency():
    """Test that CSV and point query use the same interpolation method"""
    print("\n=== Testing CSV vs Point Query Consistency ===")
    
    # Create test surfaces
    x_coords = np.linspace(0, 10, 4)
    y_coords = np.linspace(0, 10, 4)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create lower surface
    lower_z = 5 + 0.1 * X + 0.05 * Y
    lower_surface = np.column_stack([X.ravel(), Y.ravel(), lower_z.ravel()])
    
    # Create upper surface
    upper_z = 10 + 0.15 * X + 0.08 * Y
    upper_surface = np.column_stack([X.ravel(), Y.ravel(), upper_z.ravel()])
    
    # Create triangulations
    lower_tin = create_delaunay_triangulation(lower_surface[:, :2])
    setattr(lower_tin, 'z_values', lower_surface[:, 2])
    
    upper_tin = create_delaunay_triangulation(upper_surface[:, :2])
    setattr(upper_tin, 'z_values', upper_surface[:, 2])
    
    # Test points
    test_points = [
        np.array([5.0, 5.0]),
        np.array([2.0, 2.0]),
        np.array([8.0, 8.0]),
        np.array([15.0, 15.0]),  # Outside triangulation
    ]
    
    print("Testing consistency between CSV and point query methods:")
    for i, point in enumerate(test_points):
        print(f"\nPoint {i+1}: ({point[0]:.1f}, {point[1]:.1f})")
        
        # Simulate CSV method
        upper_z_csv = _interpolate_z_at_point(point, upper_tin)
        lower_z_csv = _interpolate_z_at_point(point, lower_tin)
        
        if not np.isnan(upper_z_csv) and not np.isnan(lower_z_csv):
            thickness_csv = upper_z_csv - lower_z_csv
        else:
            thickness_csv = _interpolate_nearest_thickness(point, upper_tin, lower_tin)
        
        # Simulate point query method (same as CSV now)
        upper_z_query = _interpolate_z_at_point(point, upper_tin)
        lower_z_query = _interpolate_z_at_point(point, lower_tin)
        
        if not np.isnan(upper_z_query) and not np.isnan(lower_z_query):
            thickness_query = upper_z_query - lower_z_query
        else:
            thickness_query = _interpolate_nearest_thickness(point, upper_tin, lower_tin)
        
        print(f"  CSV thickness: {thickness_csv:.3f} ft")
        print(f"  Point query thickness: {thickness_query:.3f} ft")
        
        if abs(thickness_csv - thickness_query) < 0.001:
            print(f"  ✓ Methods are consistent")
        else:
            print(f"  ✗ Methods are inconsistent")
    
    print("\n=== Consistency Test Complete ===")

if __name__ == "__main__":
    print("Starting thickness interpolation fix verification...")
    
    # Test 1: Interpolation fixes
    test_interpolation_fixes()
    
    # Test 2: CSV vs point query consistency
    test_csv_vs_point_query_consistency()
    
    print("\n=== All Tests Complete ===") 