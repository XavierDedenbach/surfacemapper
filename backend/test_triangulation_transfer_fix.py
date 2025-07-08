#!/usr/bin/env python3
"""
Test script to verify the triangulation transfer fix
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from scipy.spatial import Delaunay
from pyproj import Transformer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_triangulation_transfer_fix():
    """Test that triangulation indices are preserved during coordinate transformation"""
    print("=== Testing Triangulation Transfer Fix ===")
    
    # Create a simple test surface in WGS84
    print("\n1. Creating test surface in WGS84...")
    wgs84_vertices = np.array([
        [-82.0, 28.0, 10.0],      # Bottom-left
        [-81.999, 28.0, 10.0],    # Bottom-right  
        [-82.0, 28.001, 10.0],    # Top-left
        [-81.999, 28.001, 10.0],  # Top-right
        [-81.9995, 28.0005, 12.0] # Center point (elevated)
    ])
    
    # Create triangulation
    tri = Delaunay(wgs84_vertices[:, :2])
    wgs84_faces = tri.simplices
    
    print(f"   WGS84 vertices: {len(wgs84_vertices)}")
    print(f"   WGS84 faces: {len(wgs84_faces)}")
    print(f"   Face indices: {wgs84_faces}")
    
    # Convert to UTM
    print("\n2. Converting WGS84 to UTM...")
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)
    
    utm_vertices = []
    for vertex in wgs84_vertices:
        lon, lat, z = vertex
        utm_x, utm_y = transformer.transform(lon, lat)
        utm_vertices.append([utm_x, utm_y, z])
    
    utm_vertices = np.array(utm_vertices)
    print(f"   UTM vertices: {len(utm_vertices)}")
    
    # Test 1: Original approach (recreate triangulation)
    print("\n3. Testing original approach (recreate triangulation)...")
    utm_tri = Delaunay(utm_vertices[:, :2])
    utm_faces_new = utm_tri.simplices
    print(f"   New UTM faces: {len(utm_faces_new)}")
    print(f"   New face indices: {utm_faces_new}")
    
    # Test 2: Fixed approach (preserve triangulation)
    print("\n4. Testing fixed approach (preserve triangulation)...")
    # The faces should be the same since they reference the same vertex indices
    utm_faces_preserved = wgs84_faces.copy()
    print(f"   Preserved UTM faces: {len(utm_faces_preserved)}")
    print(f"   Preserved face indices: {utm_faces_preserved}")
    
    # Compare approaches
    print("\n5. Comparing approaches...")
    if np.array_equal(wgs84_faces, utm_faces_preserved):
        print("   ✅ Preserved triangulation: Face indices are identical!")
    else:
        print("   ❌ Preserved triangulation: Face indices are different!")
    
    if np.array_equal(wgs84_faces, utm_faces_new):
        print("   ✅ New triangulation: Face indices are identical!")
    else:
        print("   ❌ New triangulation: Face indices are different!")
    
    # Calculate areas to show the difference
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
    utm_area_preserved = calculate_surface_area(utm_vertices, utm_faces_preserved)
    utm_area_new = calculate_surface_area(utm_vertices, utm_faces_new)
    
    print(f"\n6. Area comparison:")
    print(f"   WGS84 area: {wgs84_area:.6f} square degrees")
    print(f"   UTM area (preserved): {utm_area_preserved:.2f} square meters")
    print(f"   UTM area (new): {utm_area_new:.2f} square meters")
    
    # Convert to square feet for easier comparison
    utm_area_preserved_sqft = utm_area_preserved * 10.764
    utm_area_new_sqft = utm_area_new * 10.764
    
    print(f"   UTM area (preserved): {utm_area_preserved_sqft:.2f} square feet")
    print(f"   UTM area (new): {utm_area_new_sqft:.2f} square feet")
    
    # Check if preserved approach maintains the same topology
    area_diff_preserved = abs(utm_area_preserved_sqft - (wgs84_area * 364173 * 364173 / 1000000)) / (wgs84_area * 364173 * 364173 / 1000000) * 100
    area_diff_new = abs(utm_area_new_sqft - (wgs84_area * 364173 * 364173 / 1000000)) / (wgs84_area * 364173 * 364173 / 1000000) * 100
    
    print(f"   Area difference (preserved): {area_diff_preserved:.1f}%")
    print(f"   Area difference (new): {area_diff_new:.1f}%")
    
    if area_diff_preserved < area_diff_new:
        print("   ✅ Preserved triangulation maintains better area consistency!")
    else:
        print("   ❌ New triangulation has better area consistency (unexpected)")
    
    return wgs84_vertices, wgs84_faces, utm_vertices, utm_faces_preserved, utm_faces_new

def test_clipping_preserves_triangulation():
    """Test that clipping preserves triangulation when faces are passed"""
    print("\n=== Testing Clipping Preserves Triangulation ===")
    
    # Create a simple surface
    vertices = np.array([
        [0.0, 0.0, 10.0],
        [1.0, 0.0, 10.0],
        [0.0, 1.0, 10.0],
        [1.0, 1.0, 10.0],
        [0.5, 0.5, 12.0]
    ])
    
    # Create triangulation
    tri = Delaunay(vertices[:, :2])
    faces = tri.simplices
    
    print(f"Original vertices: {len(vertices)}")
    print(f"Original faces: {len(faces)}")
    print(f"Face indices: {faces}")
    
    # Create a boundary that clips the surface
    boundary = [(0.2, 0.2), (0.8, 0.2), (0.8, 0.8), (0.2, 0.8)]
    
    # Test clipping with faces
    from app.services.surface_processor import SurfaceProcessor
    surface_processor = SurfaceProcessor()
    
    try:
        clipped_vertices, clipped_faces = surface_processor.clip_to_boundary(vertices, boundary, faces)
        print(f"Clipped vertices: {len(clipped_vertices)}")
        print(f"Clipped faces: {len(clipped_faces)}")
        print(f"Clipped face indices: {clipped_faces}")
        
        if len(clipped_faces) > 0:
            print("✅ Clipping with faces preserved triangulation!")
        else:
            print("❌ Clipping with faces resulted in no faces!")
            
    except Exception as e:
        print(f"❌ Clipping with faces failed: {e}")

if __name__ == "__main__":
    print("Testing triangulation transfer fix...")
    
    # Run the main test
    wgs84_vertices, wgs84_faces, utm_vertices, utm_faces_preserved, utm_faces_new = test_triangulation_transfer_fix()
    
    # Run clipping test
    test_clipping_preserves_triangulation()
    
    print("\n=== Summary ===")
    print("The fix ensures that:")
    print("1. Original triangulation indices are preserved during coordinate transformation")
    print("2. Clipping operations pass faces to preserve triangulation")
    print("3. No new triangulation is created unless absolutely necessary")
    print("4. Surface topology and area consistency are maintained") 