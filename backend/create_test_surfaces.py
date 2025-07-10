#!/usr/bin/env python3
"""
Create simple test PLY files for testing volume calculation and point query
"""
import numpy as np
import os

def create_test_ply_files():
    """Create simple test PLY files"""
    
    # Create directory if it doesn't exist
    os.makedirs("test_surfaces", exist_ok=True)
    
    # Surface 1: 10x10 grid at z=0
    x1 = np.linspace(0, 10, 11)
    y1 = np.linspace(0, 10, 11)
    X1, Y1 = np.meshgrid(x1, y1)
    Z1 = np.zeros_like(X1)
    vertices1 = np.column_stack([X1.ravel(), Y1.ravel(), Z1.ravel()])
    
    # Surface 2: 10x10 grid at z=5 (5 units higher)
    x2 = np.linspace(0, 10, 11)
    y2 = np.linspace(0, 10, 11)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = np.full_like(X2, 5.0)
    vertices2 = np.column_stack([X2.ravel(), Y2.ravel(), Z2.ravel()])
    
    # Create faces for triangulation (simple grid triangulation)
    faces1 = []
    faces2 = []
    
    nx, ny = 11, 11
    for i in range(ny - 1):
        for j in range(nx - 1):
            # Create two triangles for each grid cell
            v1 = i * nx + j
            v2 = i * nx + j + 1
            v3 = (i + 1) * nx + j
            v4 = (i + 1) * nx + j + 1
            
            # Triangle 1
            faces1.append([v1, v2, v3])
            faces2.append([v1, v2, v3])
            
            # Triangle 2
            faces1.append([v2, v4, v3])
            faces2.append([v2, v4, v3])
    
    faces1 = np.array(faces1)
    faces2 = np.array(faces2)
    
    # Write PLY files
    with open("test_surfaces/surface1.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices1)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces1)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices
        for vertex in vertices1:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write faces
        for face in faces1:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    with open("test_surfaces/surface2.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices2)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces2)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        # Write vertices
        for vertex in vertices2:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")
        
        # Write faces
        for face in faces2:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print("Created test PLY files:")
    print("  - test_surfaces/surface1.ply (z=0)")
    print("  - test_surfaces/surface2.ply (z=5)")
    print(f"  Expected volume: 10 * 10 * 5 = 500 cubic units")
    print(f"  Expected thickness: 5 units")

if __name__ == "__main__":
    create_test_ply_files() 