#!/usr/bin/env python3
"""
Test script to verify API endpoints work after serialization fix
"""
import requests
import json
import time

BASE_URL = "http://localhost:8081"

def test_health():
    """Test health endpoint"""
    print("=== Testing Health Endpoint ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_surface_upload():
    """Test surface upload"""
    print("\n=== Testing Surface Upload ===")
    try:
        # Upload a test surface
        files = {'file': open('data/test_files/test_surface_500ft.ply', 'rb')}
        response = requests.post(f"{BASE_URL}/api/surfaces/upload", files=files)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            return response.json().get('surface_id')
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_analysis_start(surface_id):
    """Test analysis start"""
    print(f"\n=== Testing Analysis Start (Surface: {surface_id}) ===")
    try:
        payload = {
            "surface_ids": [surface_id],
            "analysis_type": "volume",
            "generate_base_surface": True,
            "georeference_params": [
                {
                    "wgs84_lat": 40.7128,
                    "wgs84_lon": -74.0060,
                    "orientation_degrees": 0.0,
                    "scaling_factor": 1.0
                }
            ],
            "analysis_boundary": {
                "min_x": 0.0,
                "max_x": 1000.0,
                "min_y": 0.0,
                "max_y": 1000.0
            },
            "params": {
                "tonnage_per_layer": [
                    {"layer_index": 0, "tonnage": 100.0}
                ]
            }
        }
        
        response = requests.post(f"{BASE_URL}/api/analysis/start", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 202:
            return response.json().get('analysis_id')
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def test_analysis_status(analysis_id):
    """Test analysis status endpoint"""
    print(f"\n=== Testing Analysis Status (ID: {analysis_id}) ===")
    try:
        response = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_analysis_results(analysis_id):
    """Test analysis results endpoint"""
    print(f"\n=== Testing Analysis Results (ID: {analysis_id}) ===")
    try:
        response = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}/results")
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Results keys: {list(data.keys())}")
            print(f"Analysis summary: {len(data.get('analysis_summary', []))} layers")
            return True
        else:
            print(f"Response: {response.json()}")
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    """Run all tests"""
    print("Starting API tests...")
    
    # Test health
    if not test_health():
        print("❌ Health check failed")
        return
    
    # Test surface upload
    surface_id = test_surface_upload()
    if not surface_id:
        print("❌ Surface upload failed")
        return
    
    # Test analysis start
    analysis_id = test_analysis_start(surface_id)
    if not analysis_id:
        print("❌ Analysis start failed")
        return
    
    # Wait a bit for analysis to progress
    print(f"\nWaiting 5 seconds for analysis to progress...")
    time.sleep(5)
    
    # Test status endpoint
    if not test_analysis_status(analysis_id):
        print("❌ Analysis status failed")
        return
    
    # Wait for analysis to complete
    print(f"\nWaiting for analysis to complete...")
    max_wait = 60  # 60 seconds max
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        try:
            response = requests.get(f"{BASE_URL}/api/analysis/{analysis_id}/status")
            if response.status_code == 200:
                status_data = response.json()
                current_status = status_data.get('status')
                progress = status_data.get('progress_percent', 0)
                print(f"Status: {current_status}, Progress: {progress}%")
                
                if current_status == 'completed':
                    break
                elif current_status in ['failed', 'cancelled']:
                    print(f"❌ Analysis {current_status}")
                    return
            else:
                print(f"❌ Status check failed: {response.status_code}")
                return
                
        except Exception as e:
            print(f"Error checking status: {e}")
            return
            
        time.sleep(2)
    
    # Test results endpoint
    if not test_analysis_results(analysis_id):
        print("❌ Analysis results failed")
        return
    
    print("\n🎉 ALL TESTS PASSED - Serialization fix is working!")

if __name__ == "__main__":
    main() 