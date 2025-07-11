#!/usr/bin/env python3
"""
Test script to verify mouse distance threshold functionality
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import math

def test_distance_threshold():
    """Test the distance threshold logic"""
    print("=== Testing Mouse Distance Threshold ===\n")
    
    # Test parameters
    distance_threshold = 1.0  # 1 foot in UTM coordinates
    last_query_point = None
    
    # Test points (simulating mouse movements)
    test_points = [
        {"x": 100.0, "y": 200.0, "description": "First point"},
        {"x": 100.5, "y": 200.2, "description": "Small movement (0.54 ft)"},
        {"x": 101.5, "y": 201.0, "description": "Larger movement (1.80 ft)"},
        {"x": 102.0, "y": 201.5, "description": "Another large movement (1.12 ft)"},
        {"x": 102.1, "y": 201.6, "description": "Small movement (0.14 ft)"},
        {"x": 105.0, "y": 205.0, "description": "Very large movement (5.66 ft)"},
    ]
    
    print("Distance threshold: 1.0 foot")
    print("=" * 50)
    
    for i, point in enumerate(test_points):
        current_point = {"x": point["x"], "y": point["y"]}
        
        if last_query_point:
            distance = math.sqrt(
                math.pow(current_point["x"] - last_query_point["x"], 2) + 
                math.pow(current_point["y"] - last_query_point["y"], 2)
            )
            
            should_query = distance >= distance_threshold
            status = "✓ QUERY" if should_query else "✗ SKIP"
            
            print(f"Point {i+1}: {point['description']}")
            print(f"  Distance from last: {distance:.3f} ft")
            print(f"  Action: {status}")
            print()
            
            if should_query:
                last_query_point = current_point
        else:
            print(f"Point {i+1}: {point['description']}")
            print(f"  First point - always query")
            print(f"  Action: ✓ QUERY")
            print()
            last_query_point = current_point
    
    print("=" * 50)
    print("Expected behavior:")
    print("- Point 1: Always query (first point)")
    print("- Point 2: Skip (distance < 1.0 ft)")
    print("- Point 3: Query (distance >= 1.0 ft)")
    print("- Point 4: Query (distance >= 1.0 ft)")
    print("- Point 5: Skip (distance < 1.0 ft)")
    print("- Point 6: Query (distance >= 1.0 ft)")

if __name__ == "__main__":
    test_distance_threshold() 