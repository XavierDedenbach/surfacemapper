#!/usr/bin/env python3
"""
Test script to verify volume calculation and point query functionality
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from app.services.volume_calculator import VolumeCalculator, calculate_volume_between_surfaces
from app.services.thickness_calculator import _interpolate_z_at_point
from scipy.spatial import Delaunay

def test_volume_calculation():
    """Test volume calculation between two surfaces"""
    print("=== Testing Volume Calculation ===")
    
    # Create test surfaces with known volume
    # Surface 1: 10x10 grid at z=0
    x1 = np.linspace(0, 10, 11)
    y1 = np.linspace(0, 10, 11)
    X1, Y1 = np.meshgrid(x1, y1)
    Z1 = np.zeros_like(X1)
    surface1 = np.column_stack([X1.ravel(), Y1.ravel(), Z1.ravel()])
    
    # Surface 2: 10x10 grid at z=5 (5 units higher)
    x2 = np.linspace(0, 10, 11)
    y2 = np.linspace(0, 10, 11)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = np.full_like(X2, 5.0)
    surface2 = np.column_stack([X2.ravel(), Y2.ravel(), Z2.ravel()])
    
    # Expected volume: 10 * 10 * 5 = 500 cubic units
    expected_volume = 500.0
    
    # Test volume calculation
    calculated_volume = calculate_volume_between_surfaces(surface1, surface2)
    
    print(f"Expected volume: {expected_volume}")
    print(f"Calculated volume: {calculated_volume}")
    print(f"Relative error: {abs(calculated_volume - expected_volume) / expected_volume:.2%}")
    
    assert abs(calculated_volume - expected_volume) / expected_volume < 0.1, "Volume calculation error too large"
    print("✅ Volume calculation test passed!")
    
    return calculated_volume

def test_point_query():
    """Test point query functionality"""
    print("\n=== Testing Point Query ===")
    
    # Create test surfaces
    # Surface 1: 5x5 grid at z=0
    x1 = np.linspace(0, 5, 6)
    y1 = np.linspace(0, 5, 6)
    X1, Y1 = np.meshgrid(x1, y1)
    Z1 = np.zeros_like(X1)
    surface1 = np.column_stack([X1.ravel(), Y1.ravel(), Z1.ravel()])
    
    # Surface 2: 5x5 grid at z=3 (3 units higher)
    x2 = np.linspace(0, 5, 6)
    y2 = np.linspace(0, 5, 6)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = np.full_like(X2, 3.0)
    surface2 = np.column_stack([X2.ravel(), Y2.ravel(), Z2.ravel()])
    
    # Create TINs for interpolation
    tin1 = Delaunay(surface1[:, :2])
    setattr(tin1, 'z_values', surface1[:, 2])
    
    tin2 = Delaunay(surface2[:, :2])
    setattr(tin2, 'z_values', surface2[:, 2])
    
    # Test point query at center
    query_point = np.array([2.5, 2.5])  # Center of the grid
    
    # Interpolate Z values
    z1 = _interpolate_z_at_point(query_point, tin1)
    z2 = _interpolate_z_at_point(query_point, tin2)
    
    thickness = z2 - z1
    
    print(f"Query point: {query_point}")
    print(f"Surface 1 Z: {z1}")
    print(f"Surface 2 Z: {z2}")
    print(f"Thickness: {thickness}")
    
    # Expected thickness should be around 3.0
    assert abs(thickness - 3.0) < 0.1, f"Thickness calculation error: expected ~3.0, got {thickness}"
    print("✅ Point query test passed!")
    
    return thickness

def test_multiple_layers():
    """Test multiple layer thickness calculation"""
    print("\n=== Testing Multiple Layers ===")
    
    # Create 3 surfaces with different heights
    surfaces = []
    for i in range(3):
        x = np.linspace(0, 5, 6)
        y = np.linspace(0, 5, 6)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, i * 2.0)  # Heights: 0, 2, 4
        surface = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        surfaces.append(surface)
    
    # Test thickness between each pair
    query_point = np.array([2.5, 2.5])
    thicknesses = []
    
    for i in range(len(surfaces) - 1):
        tin1 = Delaunay(surfaces[i][:, :2])
        setattr(tin1, 'z_values', surfaces[i][:, 2])
        
        tin2 = Delaunay(surfaces[i+1][:, :2])
        setattr(tin2, 'z_values', surfaces[i+1][:, 2])
        
        z1 = _interpolate_z_at_point(query_point, tin1)
        z2 = _interpolate_z_at_point(query_point, tin2)
        
        thickness = z2 - z1
        thicknesses.append(thickness)
        
        print(f"Layer {i} to {i+1}: {thickness:.2f} units")
    
    # Expected thicknesses: 2.0, 2.0
    expected_thicknesses = [2.0, 2.0]
    
    for i, (calculated, expected) in enumerate(zip(thicknesses, expected_thicknesses)):
        assert abs(calculated - expected) < 0.1, f"Layer {i} thickness error: expected {expected}, got {calculated}"
    
    print("✅ Multiple layers test passed!")
    return thicknesses

if __name__ == "__main__":
    print("Testing volume calculation and point query functionality...")
    
    try:
        volume = test_volume_calculation()
        thickness = test_point_query()
        layer_thicknesses = test_multiple_layers()
        
        print("\n=== All Tests Passed! ===")
        print(f"Volume calculation: {volume:.2f} cubic units")
        print(f"Single point thickness: {thickness:.2f} units")
        print(f"Multiple layer thicknesses: {[f'{t:.2f}' for t in layer_thicknesses]}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        sys.exit(1) 