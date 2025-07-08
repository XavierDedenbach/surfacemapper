#!/usr/bin/env python3
"""
End-to-end test for SHP file processing workflow
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from shapely.geometry import Polygon
from app.services.analysis_executor import AnalysisExecutor
from app.services.surface_cache import surface_cache
from app.utils.shp_parser import SHPParser
import json
import uuid
import time
import threading

def test_end_to_end_shp_workflow():
    """Test the complete SHP file processing workflow"""
    print("=== End-to-End SHP Workflow Test ===")
    
    # Step 1: Set up test data
    shp_file_path = "../drone_surfaces/27June2025_0550AM_emptyworkingface/27June2025_0550AM_emptyworkingface.shp"
    
    # Create a test boundary (smaller than the full area for testing)
    test_boundary = [
        [40.151, -79.855],  # lat, lon
        [40.151, -79.854],  # lat, lon
        [40.150, -79.854],  # lat, lon
        [40.150, -79.855],  # lat, lon
        [40.151, -79.855]   # lat, lon (close the polygon)
    ]
    
    print(f"Step 1: Loading SHP file from {shp_file_path}")
    
    # Step 2: Test SHP parser directly
    shp_parser = SHPParser()
    try:
        vertices, faces = shp_parser.process_shp_file(shp_file_path)
        print(f"✓ SHP parser loaded {len(vertices)} vertices and {len(faces) if faces is not None else 0} faces")
    except Exception as e:
        print(f"✗ SHP parser failed: {e}")
        return False
    
    # Step 3: Test surface cache integration
    surface_id = str(uuid.uuid4())
    surface_cache.set(surface_id, {
        "file_path": shp_file_path,
        "file_type": "SHP",
        "filename": "test_surface.shp"
    })
    print(f"✓ Surface added to cache with ID: {surface_id}")
    
    # Step 4: Test analysis executor
    analysis_id = str(uuid.uuid4())
    executor = AnalysisExecutor()
    
    # Create analysis parameters
    params = {
        "surface_ids": [surface_id],
        "analysis_boundary": {
            "wgs84_coordinates": test_boundary
        },
        "params": {}  # No additional processing parameters
    }
    
    print(f"Step 4: Running analysis with ID: {analysis_id}")
    print(f"  - Surface IDs: {params['surface_ids']}")
    print(f"  - Boundary: {params['analysis_boundary']}")
    
    try:
        # Start the analysis execution (this creates the job entry)
        start_result = executor.start_analysis_execution(analysis_id, params)
        print(f"✓ Analysis started successfully")
        print(f"  - Start result: {start_result}")
        
        # Run the analysis in a background thread (simulating FastAPI background tasks)
        def run_analysis():
            try:
                executor.run_analysis_sync(analysis_id, params)
            except Exception as e:
                print(f"    Background analysis failed: {e}")
        
        analysis_thread = threading.Thread(target=run_analysis)
        analysis_thread.start()
        
        # Wait for the analysis to complete by polling the status
        print(f"  - Waiting for analysis to complete...")
        max_wait = 300  # 5 minutes
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = executor.get_analysis_status(analysis_id)
            status = status_response.get("status")
            progress = status_response.get("progress_percent", 0)
            current_step = status_response.get("current_step", "unknown")
            
            print(f"    Status: {status}, Progress: {progress:.1f}%, Step: {current_step}")
            
            if status == "completed":
                print(f"    ✓ Analysis completed!")
                break
            elif status in ["failed", "cancelled"]:
                error_msg = status_response.get("error_message", "Unknown error")
                print(f"    ✗ Analysis failed: {error_msg}")
                return False
            
            time.sleep(2)  # Poll every 2 seconds
        else:
            print(f"    ✗ Analysis did not complete within {max_wait} seconds")
            return False
        
        # Wait for the background thread to finish
        analysis_thread.join()
        
        # Now check the results
        results = executor.get_results(analysis_id)
        if results:
            print(f"✓ Results retrieved successfully")
            print(f"  - Result keys: {list(results.keys())}")
            
            # Check for surface data
            if 'surfaces' in results:
                surfaces = results['surfaces']
                print(f"  - Number of surfaces: {len(surfaces)}")
                for i, surface in enumerate(surfaces):
                    vertex_count = len(surface.get('vertices', []))
                    face_count = len(surface.get('faces', [])) if surface.get('faces') else 0
                    print(f"    Surface {i+1}: {vertex_count} vertices, {face_count} faces")
            
            # Check for analysis results
            if 'analysis_results' in results:
                analysis_results = results['analysis_results']
                print(f"  - Analysis results: {list(analysis_results.keys())}")
                
                # Check for volume calculations
                if 'volume_analysis' in analysis_results:
                    volume_data = analysis_results['volume_analysis']
                    print(f"    Volume analysis: {list(volume_data.keys())}")
                    if 'total_volume' in volume_data:
                        print(f"    Total volume: {volume_data['total_volume']:.2f} cubic units")
            
            return True
        else:
            print(f"✗ No results available after completion")
            return False
            
    except Exception as e:
        print(f"✗ Analysis execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_surface_cache_integration():
    """Test surface cache integration with SHP files"""
    print("\n=== Surface Cache Integration Test ===")
    
    # Test adding SHP file to cache
    surface_id = str(uuid.uuid4())
    shp_file_path = "../drone_surfaces/27June2025_0550AM_emptyworkingface/27June2025_0550AM_emptyworkingface.shp"
    
    surface_cache.set(surface_id, {
        "file_path": shp_file_path,
        "file_type": "SHP",
        "filename": "test_surface.shp"
    })
    
    # Test retrieving from cache
    cached_surface = surface_cache.get(surface_id)
    if cached_surface and cached_surface.get('file_type') == 'SHP':
        print(f"✓ Surface cache integration working")
        print(f"  - File type: {cached_surface.get('file_type')}")
        print(f"  - File path: {cached_surface.get('file_path')}")
        return True
    else:
        print(f"✗ Surface cache integration failed")
        return False

if __name__ == "__main__":
    print("Starting end-to-end SHP workflow tests...")
    
    # Test 1: Surface cache integration
    cache_success = test_surface_cache_integration()
    
    # Test 2: Full workflow
    workflow_success = test_end_to_end_shp_workflow()
    
    print(f"\n=== Test Results ===")
    print(f"Surface Cache Integration: {'✓ PASS' if cache_success else '✗ FAIL'}")
    print(f"End-to-End Workflow: {'✓ PASS' if workflow_success else '✗ FAIL'}")
    
    if cache_success and workflow_success:
        print(f"\n🎉 All tests passed! SHP workflow is fully integrated.")
    else:
        print(f"\n❌ Some tests failed. Check the output above for details.") 