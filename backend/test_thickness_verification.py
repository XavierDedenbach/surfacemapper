#!/usr/bin/env python3
"""
Test script to verify thickness calculation and trace data flow
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from app.services.thickness_calculator import calculate_thickness_between_surfaces, calculate_thickness_statistics
from app.services.surface_processor import SurfaceProcessor
from app.services.analysis_executor import AnalysisExecutor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_thickness_calculation():
    """Test basic thickness calculation"""
    print("=== Testing Basic Thickness Calculation ===")
    
    # Create simple test surfaces
    # Upper surface: z = 10 + 0.1*x + 0.05*y
    # Lower surface: z = 5 + 0.08*x + 0.03*y
    # Expected thickness: 5 + 0.02*x + 0.02*y
    
    x_coords = np.linspace(0, 100, 11)
    y_coords = np.linspace(0, 100, 11)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create upper surface
    upper_z = 10 + 0.1 * X + 0.05 * Y
    upper_surface = np.column_stack([X.ravel(), Y.ravel(), upper_z.ravel()])
    
    # Create lower surface
    lower_z = 5 + 0.08 * X + 0.03 * Y
    lower_surface = np.column_stack([X.ravel(), Y.ravel(), lower_z.ravel()])
    
    print(f"Upper surface shape: {upper_surface.shape}")
    print(f"Lower surface shape: {lower_surface.shape}")
    print(f"Upper surface Z range: {np.min(upper_surface[:, 2]):.2f} to {np.max(upper_surface[:, 2]):.2f}")
    print(f"Lower surface Z range: {np.min(lower_surface[:, 2]):.2f} to {np.max(lower_surface[:, 2]):.2f}")
    
    # Calculate thickness
    thickness_data, invalid_points = calculate_thickness_between_surfaces(
        upper_surface, lower_surface, sample_spacing=10.0
    )
    
    print(f"Thickness calculation results:")
    print(f"  Thickness data shape: {thickness_data.shape}")
    print(f"  Valid thickness values: {np.sum(~np.isnan(thickness_data))}")
    print(f"  Invalid points: {len(invalid_points)}")
    print(f"  Thickness range: {np.nanmin(thickness_data):.3f} to {np.nanmax(thickness_data):.3f}")
    
    # Calculate statistics
    thickness_stats = calculate_thickness_statistics(thickness_data, invalid_points)
    print(f"Thickness statistics:")
    print(f"  Mean: {thickness_stats.get('mean', 'N/A')}")
    print(f"  Min: {thickness_stats.get('min', 'N/A')}")
    print(f"  Max: {thickness_stats.get('max', 'N/A')}")
    print(f"  Std: {thickness_stats.get('std', 'N/A')}")
    print(f"  Count: {thickness_stats.get('count', 'N/A')}")
    print(f"  Valid count: {thickness_stats.get('valid_count', 'N/A')}")
    
    # Verify expected thickness
    expected_thickness = 5.0  # Base difference
    actual_mean = thickness_stats.get('mean')
    if actual_mean is not None:
        print(f"Expected mean thickness: {expected_thickness:.3f}")
        print(f"Actual mean thickness: {actual_mean:.3f}")
        print(f"Difference: {abs(actual_mean - expected_thickness):.3f}")
        
        if abs(actual_mean - expected_thickness) < 0.1:
            print("✓ Thickness calculation is working correctly!")
        else:
            print("✗ Thickness calculation may have issues")
    else:
        print("✗ No valid thickness data calculated")
    
    return thickness_stats

def test_surface_processor_integration():
    """Test thickness calculation through surface processor"""
    print("\n=== Testing Surface Processor Integration ===")
    
    # Create test surfaces
    surface_processor = SurfaceProcessor()
    
    # Create simple test surfaces
    x_coords = np.linspace(0, 50, 6)
    y_coords = np.linspace(0, 50, 6)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Upper surface
    upper_z = 10 + 0.1 * X + 0.05 * Y
    upper_surface = np.column_stack([X.ravel(), Y.ravel(), upper_z.ravel()])
    
    # Lower surface  
    lower_z = 5 + 0.08 * X + 0.03 * Y
    lower_surface = np.column_stack([X.ravel(), Y.ravel(), lower_z.ravel()])
    
    # Create surface data structures
    surfaces_to_process = [
        {
            'vertices': lower_surface,
            'name': 'Lower Surface',
            'file_type': 'PLY'
        },
        {
            'vertices': upper_surface,
            'name': 'Upper Surface', 
            'file_type': 'PLY'
        }
    ]
    
    # Test parameters
    params = {
        'georeference_params': [{
            'wgs84_lat': 40.0,
            'wgs84_lon': -75.0,
            'orientation_degrees': 0.0,
            'scaling_factor': 1.0
        }],
        'analysis_boundary': None,
        'tonnage_data': {}
    }
    
    print("Testing surface processing with thickness calculation...")
    
    try:
        # Process surfaces
        result = surface_processor.process_surfaces(surfaces_to_process, params)
        
        print("Surface processing completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        
        # Check thickness results
        thickness_results = result.get('thickness_results', [])
        print(f"Thickness results count: {len(thickness_results)}")
        
        for i, thickness_result in enumerate(thickness_results):
            print(f"Layer {i}:")
            print(f"  Layer designation: {thickness_result.get('layer_designation')}")
            print(f"  Average thickness: {thickness_result.get('average_thickness_feet')}")
            print(f"  Min thickness: {thickness_result.get('min_thickness_feet')}")
            print(f"  Max thickness: {thickness_result.get('max_thickness_feet')}")
            print(f"  Std dev thickness: {thickness_result.get('std_dev_thickness_feet')}")
        
        return result
        
    except Exception as e:
        print(f"Error in surface processing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_analysis_executor():
    """Test full analysis executor with thickness calculation"""
    print("\n=== Testing Analysis Executor ===")
    
    # Create test surfaces
    x_coords = np.linspace(0, 30, 4)
    y_coords = np.linspace(0, 30, 4)
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Upper surface
    upper_z = 10 + 0.1 * X + 0.05 * Y
    upper_surface = np.column_stack([X.ravel(), Y.ravel(), upper_z.ravel()])
    
    # Lower surface
    lower_z = 5 + 0.08 * X + 0.03 * Y  
    lower_surface = np.column_stack([X.ravel(), Y.ravel(), lower_z.ravel()])
    
    # Create analysis parameters
    params = {
        'surfaces': [
            {
                'vertices': lower_surface.tolist(),
                'name': 'Lower Surface',
                'file_type': 'PLY'
            },
            {
                'vertices': upper_surface.tolist(),
                'name': 'Upper Surface',
                'file_type': 'PLY'
            }
        ],
        'georeference_params': [{
            'wgs84_lat': 40.0,
            'wgs84_lon': -75.0,
            'orientation_degrees': 0.0,
            'scaling_factor': 1.0
        }],
        'analysis_boundary': None,
        'tonnage_data': {}
    }
    
    print("Testing analysis executor...")
    
    try:
        executor = AnalysisExecutor()
        analysis_id = executor.generate_analysis_id()
        print(f"Generated analysis ID: {analysis_id}")
        
        # Run analysis
        result = executor.run_analysis_sync(analysis_id, params)
        
        print("Analysis completed successfully!")
        print(f"Result keys: {list(result.keys())}")
        
        # Check thickness results
        thickness_results = result.get('thickness_results', [])
        print(f"Thickness results count: {len(thickness_results)}")
        
        for i, thickness_result in enumerate(thickness_results):
            print(f"Layer {i}:")
            print(f"  Layer designation: {thickness_result.get('layer_designation')}")
            print(f"  Average thickness: {thickness_result.get('average_thickness_feet')}")
            print(f"  Min thickness: {thickness_result.get('min_thickness_feet')}")
            print(f"  Max thickness: {thickness_result.get('max_thickness_feet')}")
            print(f"  Std dev thickness: {thickness_result.get('std_dev_thickness_feet')}")
        
        return result
        
    except Exception as e:
        print(f"Error in analysis executor: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting thickness verification tests...")
    
    # Test 1: Basic thickness calculation
    test_thickness_calculation()
    
    # Test 2: Surface processor integration
    test_surface_processor_integration()
    
    # Test 3: Full analysis executor
    test_analysis_executor()
    
    print("\n=== Thickness Verification Tests Complete ===") 