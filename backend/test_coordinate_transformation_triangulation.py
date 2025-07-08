#!/usr/bin/env python3
"""
Test script to verify WGS84 triangulation -> UTM conversion -> UTM triangulation logic
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.spatial import Delaunay
from pyproj import Transformer
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_surface_wgs84():
    """Create a test surface in WGS84 coordinates (roughly 300x200 ft area)"""
    # Create a rectangular area in WGS84 (roughly 300x200 ft)
    # Approximate: 1 degree ≈ 364,173 feet
    # 300 ft ≈ 0.00082 degrees, 200 ft ≈ 0.00055 degrees
    
    # Center around a test location (e.g., UTM Zone 17N)
    center_lat = 28.0  # Approximate latitude
    center_lon = -82.0  # Approximate longitude
    
    # Create a grid in WGS84
    lat_range = 0.00082  # ~300 ft
    lon_range = 0.00055  # ~200 ft
    
    # Create 10x10 grid
    lats = np.linspace(center_lat - lat_range/2, center_lat + lat_range/2, 10)
    lons = np.linspace(center_lon - lon_range/2, center_lon + lon_range/2, 10)
    
    # Create meshgrid
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    
    # Add some elevation variation (simple sine wave)
    elevation = 10 + 2 * np.sin(lon_grid * 100) * np.cos(lat_grid * 100)
    
    # Flatten to vertices
    vertices = np.column_stack([lon_grid.ravel(), lat_grid.ravel(), elevation.ravel()])
    
    # Create triangulation in WGS84
    tri = Delaunay(vertices[:, :2])
    faces = tri.simplices
    
    return vertices, faces

def convert_wgs84_to_utm(vertices, utm_zone=32617):
    """Convert WGS84 vertices to UTM coordinates"""
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_zone}", always_xy=True)
    
    utm_vertices = []
    for vertex in vertices:
        lon, lat, z = vertex
        utm_x, utm_y = transformer.transform(lon, lat)
        utm_vertices.append([utm_x, utm_y, z])
    
    return np.array(utm_vertices)

def test_triangulation_transfer():
    """Test the triangulation transfer logic"""
    print("=== Testing WGS84 Triangulation -> UTM Conversion -> UTM Triangulation ===")
    
    # Step 1: Create test surface in WGS84
    print("\n1. Creating test surface in WGS84...")
    wgs84_vertices, wgs84_faces = create_test_surface_wgs84()
    print(f"   WGS84 vertices: {len(wgs84_vertices)}")
    print(f"   WGS84 faces: {len(wgs84_faces)}")
    print(f"   WGS84 bounds: X({wgs84_vertices[:, 0].min():.6f} to {wgs84_vertices[:, 0].max():.6f})")
    print(f"   WGS84 bounds: Y({wgs84_vertices[:, 1].min():.6f} to {wgs84_vertices[:, 1].max():.6f})")
    
    # Step 2: Convert to UTM
    print("\n2. Converting WGS84 to UTM...")
    utm_vertices = convert_wgs84_to_utm(wgs84_vertices)
    print(f"   UTM vertices: {len(utm_vertices)}")
    print(f"   UTM bounds: X({utm_vertices[:, 0].min():.2f} to {utm_vertices[:, 0].max():.2f})")
    print(f"   UTM bounds: Y({utm_vertices[:, 1].min():.2f} to {utm_vertices[:, 1].max():.2f})")
    
    # Step 3: Create new triangulation in UTM
    print("\n3. Creating new triangulation in UTM...")
    utm_tri = Delaunay(utm_vertices[:, :2])
    utm_faces = utm_tri.simplices
    print(f"   UTM faces: {len(utm_faces)}")
    
    # Step 4: Compare triangulations
    print("\n4. Comparing triangulations...")
    print(f"   Original WGS84 faces: {len(wgs84_faces)}")
    print(f"   New UTM faces: {len(utm_faces)}")
    
    # Check if face counts are similar
    face_count_diff = abs(len(wgs84_faces) - len(utm_faces))
    print(f"   Face count difference: {face_count_diff}")
    
    # Calculate areas to check for distortion
    def calculate_surface_area(vertices, faces):
        total_area = 0.0
        for face in faces:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            # Calculate triangle area using cross product
            edge1 = v2[:2] - v1[:2]
            edge2 = v3[:2] - v1[:2]
            area = 0.5 * abs(np.cross(edge1, edge2))
            total_area += area
        return total_area
    
    wgs84_area = calculate_surface_area(wgs84_vertices, wgs84_faces)
    utm_area = calculate_surface_area(utm_vertices, utm_faces)
    
    print(f"\n5. Area comparison:")
    print(f"   WGS84 surface area: {wgs84_area:.6f} square degrees")
    print(f"   UTM surface area: {utm_area:.2f} square meters")
    
    # Convert UTM area to square feet for comparison
    utm_area_sqft = utm_area * 10.764  # 1 sq meter = 10.764 sq feet
    print(f"   UTM surface area: {utm_area_sqft:.2f} square feet")
    
    # Expected area should be roughly 300x200 = 60,000 sq ft
    expected_area_sqft = 300 * 200
    print(f"   Expected area: {expected_area_sqft:.2f} square feet")
    
    area_error = abs(utm_area_sqft - expected_area_sqft) / expected_area_sqft * 100
    print(f"   Area error: {area_error:.1f}%")
    
    # Check for extreme distortion
    if area_error > 50:
        print(f"   ⚠️  WARNING: High area distortion detected!")
        print(f"   This suggests the triangulation transfer is not working correctly.")
    
    return wgs84_vertices, wgs84_faces, utm_vertices, utm_faces

def test_vertex_mapping():
    """Test if vertices are being properly mapped during transformation"""
    print("\n=== Testing Vertex Mapping ===")
    
    # Create simple test case
    wgs84_vertices = np.array([
        [-82.0, 28.0, 10.0],  # Bottom-left
        [-81.999, 28.0, 10.0], # Bottom-right  
        [-82.0, 28.001, 10.0], # Top-left
        [-81.999, 28.001, 10.0] # Top-right
    ])
    
    # Create triangulation
    tri = Delaunay(wgs84_vertices[:, :2])
    wgs84_faces = tri.simplices
    
    print(f"Original WGS84 vertices: {len(wgs84_vertices)}")
    print(f"Original WGS84 faces: {len(wgs84_faces)}")
    print(f"Face indices: {wgs84_faces}")
    
    # Convert to UTM
    utm_vertices = convert_wgs84_to_utm(wgs84_vertices)
    
    print(f"UTM vertices: {len(utm_vertices)}")
    print(f"UTM vertex coordinates:")
    for i, vertex in enumerate(utm_vertices):
        print(f"  Vertex {i}: ({vertex[0]:.2f}, {vertex[1]:.2f}, {vertex[2]:.2f})")
    
    # Create new triangulation
    utm_tri = Delaunay(utm_vertices[:, :2])
    utm_faces = utm_tri.simplices
    
    print(f"New UTM faces: {len(utm_faces)}")
    print(f"New face indices: {utm_faces}")
    
    # Check if face indices are the same
    if np.array_equal(wgs84_faces, utm_faces):
        print("✅ Face indices are preserved!")
    else:
        print("❌ Face indices are different!")
        print("This confirms the triangulation is being recreated, not transferred.")

if __name__ == "__main__":
    print("Testing coordinate transformation and triangulation logic...")
    
    # Run the main test
    wgs84_vertices, wgs84_faces, utm_vertices, utm_faces = test_triangulation_transfer()
    
    # Run vertex mapping test
    test_vertex_mapping()
    
    print("\n=== Summary ===")
    print("The issue is that the original WGS84 triangulation is being discarded")
    print("and a new triangulation is being created in UTM coordinates.")
    print("This causes distortion because:")
    print("1. WGS84 triangulation is based on geographic coordinates (degrees)")
    print("2. UTM triangulation is based on projected coordinates (meters)")
    print("3. The scales are completely different")
    print("4. New triangles may not preserve the original surface topology")
    
    print("\nSolution: Transfer the triangulation indices, don't recreate them!") 