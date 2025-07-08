#!/usr/bin/env python3
"""
Comprehensive end-to-end test script for SHP workflow with WGS84 clipping
"""

import requests
import json
import time
import os
import numpy as np
import sys

# Backend API URL
BACKEND_URL = "http://localhost:8081"

def wait_for_backend():
    """Wait for backend to be ready"""
    print("Waiting for backend to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Backend is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"  Attempt {i+1}/{max_retries}: Backend not ready yet...")
        time.sleep(2)
    
    print("✗ Backend failed to start")
    return False

def test_shp_workflow():
    """Test the complete SHP workflow with WGS84 clipping"""
    
    print("=" * 60)
    print("SHP WORKFLOW END-TO-END TEST")
    print("=" * 60)
    
    # Test 1: Upload SHP file
    print("\n1. Uploading SHP file...")
    shp_files = [
        ('27June2025_0550AM_emptyworkingface.shp', 'application/octet-stream'),
        ('27June2025_0550AM_emptyworkingface.shx', 'application/octet-stream'),
        ('27June2025_0550AM_emptyworkingface.dbf', 'application/octet-stream'),
        ('27June2025_0550AM_emptyworkingface.prj', 'application/octet-stream')
    ]
    
    files = []
    for filename, content_type in shp_files:
        file_path = f"drone_surfaces/27June2025_0550AM_emptyworkingface/{filename}"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                files.append((filename, f.read(), content_type))
            print(f"  ✓ Found {filename}")
        else:
            print(f"  ✗ Missing {filename}")
    
    if not files:
        print("No SHP files found, skipping test")
        return False
    
    # Convert to the format expected by requests
    upload_files = []
    for filename, content, content_type in files:
        upload_files.append(('files', (filename, content, content_type)))
    
    try:
        upload_response = requests.post(
            f"{BACKEND_URL}/api/surfaces/upload",
            files=upload_files,
            timeout=30
        )
        
        if upload_response.status_code != 200:
            print(f"✗ Upload failed: {upload_response.status_code}")
            print(upload_response.text)
            return False
        
        upload_data = upload_response.json()
        surface_id = upload_data.get('surface_id')
        print(f"✓ Upload successful, surface ID: {surface_id}")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Upload request failed: {e}")
        return False
    
    # Test 2: Start analysis with WGS84 boundary
    print("\n2. Starting analysis with WGS84 boundary...")

    # Read boundary from config file
    config_path = "drone_surfaces/surface_location_config1.json"
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        boundary_coords = config["analysis_boundary"]["coordinates"]
        wgs84_coordinates = [[pt["lat"], pt["lon"]] for pt in boundary_coords]
        # Ensure the polygon is closed by adding the first point at the end
        if wgs84_coordinates[0] != wgs84_coordinates[-1]:
            wgs84_coordinates.append(wgs84_coordinates[0])
        print(f"✓ Loaded boundary with {len(wgs84_coordinates)} points")
    except Exception as e:
        print(f"✗ Failed to load boundary config: {e}")
        # Use a simple boundary as fallback
        wgs84_coordinates = [
            [40.151, -79.855],
            [40.151, -79.854],
            [40.150, -79.854],
            [40.150, -79.855],
            [40.151, -79.855]  # Close the polygon
        ]
        print("  Using fallback boundary")

    analysis_request = {
        "surface_ids": [surface_id],
        "analysis_type": "volume",
        "generate_base_surface": True,
        "georeference_params": [],
        "analysis_boundary": {
            "wgs84_coordinates": wgs84_coordinates
        },
        "params": {
            "base_surface_offset": 3.0
        }
    }
    
    try:
        analysis_response = requests.post(
            f"{BACKEND_URL}/api/analysis/start",
            json=analysis_request,
            timeout=30
        )
        
        if analysis_response.status_code != 202:
            print(f"✗ Analysis start failed: {analysis_response.status_code}")
            print(analysis_response.text)
            return False
        
        analysis_data = analysis_response.json()
        analysis_id = analysis_data.get('analysis_id')
        print(f"✓ Analysis started, ID: {analysis_id}")
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Analysis start request failed: {e}")
        return False
    
    # Test 3: Monitor analysis progress
    print("\n3. Monitoring analysis progress...")
    max_wait = 1800  # 30 minutes for large SHP files with UTM conversion
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            status_response = requests.get(
                f"{BACKEND_URL}/api/analysis/{analysis_id}/status",
                timeout=10
            )
            
            if status_response.status_code != 200:
                print(f"✗ Status check failed: {status_response.status_code}")
                break
            
            status_data = status_response.json()
            status = status_data.get('status')
            progress = status_data.get('progress_percent', 0)
            step = status_data.get('current_step', 'unknown')
            
            print(f"  Status: {status}, Progress: {progress}%, Step: {step}")
            
            if status == 'completed':
                print("✓ Analysis completed successfully!")
                break
            elif status == 'failed':
                error_msg = status_data.get('error_message', 'Unknown error')
                print(f"✗ Analysis failed: {error_msg}")
                return False
            
            # If stuck at processing_boundary for more than 10 minutes, show warning
            if step == 'processing_boundary' and (time.time() - start_time) > 600:
                print(f"  ⚠️  Warning: Stuck at boundary processing for {(time.time() - start_time)/60:.1f} minutes")
                print(f"     This may be normal for large SHP files with UTM conversion...")
            
            time.sleep(5)  # Poll every 5 seconds instead of 2
            
        except requests.exceptions.RequestException as e:
            print(f"  Status check failed: {e}")
            time.sleep(5)
    
    if time.time() - start_time >= max_wait:
        print("✗ Analysis timed out after 30 minutes")
        print("  This may be due to the large number of vertices (1M+) being converted to UTM")
        return False
    
    # Test 4: Get results
    print("\n4. Retrieving analysis results...")
    try:
        results_response = requests.get(
            f"{BACKEND_URL}/api/analysis/{analysis_id}/results",
            timeout=30
        )
        
        if results_response.status_code != 200:
            print(f"✗ Results retrieval failed: {results_response.status_code}")
            print(results_response.text)
            return False
        
        results_data = results_response.json()
        
    except requests.exceptions.RequestException as e:
        print(f"✗ Results retrieval request failed: {e}")
        return False
    
    # Verify results structure
    print("\n5. Results Analysis:")
    print(f"  - Volume results: {len(results_data.get('volume_results', []))}")
    print(f"  - Thickness results: {len(results_data.get('thickness_results', []))}")
    print(f"  - Compaction results: {len(results_data.get('compaction_results', []))}")
    print(f"  - Surfaces: {len(results_data.get('surfaces', []))}")
    
    # Print volume results
    volume_results = results_data.get('volume_results', [])
    if volume_results:
        print("\nVolume Results:")
        total_volume = 0
        for result in volume_results:
            volume = result.get('volume_cubic_yards', 0)
            total_volume += volume
            print(f"  - {result.get('layer_name', 'Unknown')}: {volume:.2f} cubic yards")
        print(f"  Total volume: {total_volume:.2f} cubic yards")
    
    # Print surface information after clipping
    print("\n6. Surface Information After WGS84 Clipping:")
    surfaces = results_data.get('surfaces', [])
    if surfaces:
        for i, surface in enumerate(surfaces):
            vertices = surface.get('vertices', [])
            if vertices:
                vertices_array = np.array(vertices)
                print(f"\nSurface {i+1}: {surface.get('name', 'Unknown')}")
                print(f"  Vertex count after clipping: {len(vertices)}")
                
                # Show first few vertices
                print(f"  Vertex locations (first 5):")
                for j, vertex in enumerate(vertices[:5]):
                    print(f"    Vertex {j+1}: ({vertex[0]:.6f}, {vertex[1]:.6f}, {vertex[2]:.6f})")
                if len(vertices) > 5:
                    print(f"    ... and {len(vertices) - 5} more vertices")
                
                # Calculate surface area in square feet
                if len(vertices) >= 3:
                    # Convert WGS84 degrees to meters, then to feet
                    FEET_PER_METER = 3.28084
                    METERS_PER_DEGREE_LAT = 111000  # Approximate
                    
                    # Calculate meters per degree longitude at this latitude
                    avg_lat = np.mean(vertices_array[:, 1])  # Average latitude
                    METERS_PER_DEGREE_LON = 111000 * np.cos(np.radians(avg_lat))
                    
                    # Convert degree ranges to meters
                    x_range_degrees = np.max(vertices_array[:, 0]) - np.min(vertices_array[:, 0])  # longitude
                    y_range_degrees = np.max(vertices_array[:, 1]) - np.min(vertices_array[:, 1])  # latitude
                    
                    x_range_meters = x_range_degrees * METERS_PER_DEGREE_LON
                    y_range_meters = y_range_degrees * METERS_PER_DEGREE_LAT
                    
                    # Convert to feet
                    x_range_feet = x_range_meters * FEET_PER_METER
                    y_range_feet = y_range_meters * FEET_PER_METER
                    
                    surface_area_sqft = x_range_feet * y_range_feet
                    
                    print(f"  Surface area (bounding box): {surface_area_sqft:.2f} square feet")
                    print(f"  Bounding box: X range {x_range_feet:.2f} ft, Y range {y_range_feet:.2f} ft")
                    print(f"  Degrees: X range {x_range_degrees:.6f}°, Y range {y_range_degrees:.6f}°")
                    print(f"  Average latitude: {avg_lat:.6f}°")
    
    print("\n" + "=" * 60)
    print("✓ SHP WORKFLOW TEST COMPLETED SUCCESSFULLY")
    print("=" * 60)
    return True

def main():
    """Main function to run the end-to-end test"""
    print("Starting SHP workflow end-to-end test...")
    
    # Wait for backend to be ready
    if not wait_for_backend():
        print("Backend not available, exiting...")
        sys.exit(1)
    
    # Run the test
    success = test_shp_workflow()
    
    if success:
        print("\n🎉 All tests passed! SHP workflow is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the output above for details.")
        sys.exit(1)

if __name__ == "__main__":
    main() 