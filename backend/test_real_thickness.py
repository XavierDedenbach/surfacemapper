#!/usr/bin/env python3
"""
Test script to verify thickness calculation with real surface data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from app.services.surface_processor import SurfaceProcessor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_real_surfaces():
    """Test thickness calculation with real surface data"""
    print("=== Testing Thickness Calculation with Real Surfaces ===")
    
    # Create surface processor
    surface_processor = SurfaceProcessor()
    
    # Create realistic test surfaces (similar to what we'd get from real data)
    # Create a grid of points
    x_coords = np.linspace(0, 100, 21)  # 21 points
    y_coords = np.linspace(0, 100, 21)  # 21 points
    X, Y = np.meshgrid(x_coords, y_coords)
    
    # Create lower surface with some variation
    lower_z = 5 + 0.05 * X + 0.03 * Y + 0.01 * np.sin(X/10) * np.cos(Y/10)
    lower_surface = np.column_stack([X.ravel(), Y.ravel(), lower_z.ravel()])
    
    # Create upper surface with more variation
    upper_z = 10 + 0.08 * X + 0.05 * Y + 0.02 * np.sin(X/8) * np.cos(Y/8) + 0.5 * np.random.randn(*X.shape)
    upper_surface = np.column_stack([X.ravel(), Y.ravel(), upper_z.ravel()])
    
    print(f"Lower surface shape: {lower_surface.shape}")
    print(f"Upper surface shape: {upper_surface.shape}")
    print(f"Lower surface Z range: {np.min(lower_surface[:, 2]):.2f} to {np.max(lower_surface[:, 2]):.2f}")
    print(f"Upper surface Z range: {np.min(upper_surface[:, 2]):.2f} to {np.max(upper_surface[:, 2]):.2f}")
    
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
    
    print("Processing surfaces with thickness calculation...")
    
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
        
        # Check volume results
        volume_results = result.get('volume_results', [])
        print(f"Volume results count: {len(volume_results)}")
        
        for i, volume_result in enumerate(volume_results):
            print(f"Volume result {i}:")
            print(f"  Layer designation: {volume_result.get('layer_designation')}")
            print(f"  Volume cubic yards: {volume_result.get('volume_cubic_yards')}")
        
        # Check analysis summary
        analysis_summary = result.get('analysis_summary', [])
        print(f"Analysis summary count: {len(analysis_summary)}")
        
        for i, summary in enumerate(analysis_summary):
            print(f"Summary {i}:")
            print(f"  Layer designation: {summary.get('layer_designation')}")
            print(f"  Volume cubic yards: {summary.get('volume_cubic_yards')}")
            print(f"  Avg thickness feet: {summary.get('avg_thickness_feet')}")
            print(f"  Min thickness feet: {summary.get('min_thickness_feet')}")
            print(f"  Max thickness feet: {summary.get('max_thickness_feet')}")
        
        # Verify that thickness values are reasonable
        for thickness_result in thickness_results:
            avg_thickness = thickness_result.get('average_thickness_feet')
            min_thickness = thickness_result.get('min_thickness_feet')
            max_thickness = thickness_result.get('max_thickness_feet')
            
            if avg_thickness is not None and not np.isnan(avg_thickness):
                print(f"✓ Thickness calculation successful: avg={avg_thickness:.3f} ft")
                if min_thickness is not None and max_thickness is not None:
                    print(f"  Range: {min_thickness:.3f} to {max_thickness:.3f} ft")
            else:
                print("✗ Thickness calculation failed - no valid values")
        
        return result
        
    except Exception as e:
        print(f"Error in surface processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Starting real thickness verification test...")
    
    # Test with realistic surface data
    test_real_surfaces()
    
    print("\n=== Real Thickness Verification Test Complete ===") 