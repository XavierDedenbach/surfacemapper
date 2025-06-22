#!/usr/bin/env python3
"""
Script to create a split-shifted test surface from tv_test.ply.
Half the surface shifted by 3 feet, other half by 1 foot.
Also calculates volume estimate between the surfaces, using UTM projection and feet units.
"""

import numpy as np
import struct
import os
from pyproj import CRS, Transformer

def read_binary_ply_file(filename):
    """Read binary PLY file and return vertices and faces"""
    vertices = []
    faces = []
    
    with open(filename, 'rb') as f:
        # Read header
        header_lines = []
        while True:
            line = f.readline().decode('ascii').strip()
            header_lines.append(line)
            if line == 'end_header':
                break
        
        # Parse header
        vertex_count = 0
        face_count = 0
        vertex_properties = []
        face_properties = []
        current_element = None
        
        for line in header_lines:
            if line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
                current_element = 'vertex'
            elif line.startswith('element face'):
                face_count = int(line.split()[-1])
                current_element = 'face'
            elif line.startswith('property'):
                parts = line.split()
                if current_element == 'vertex':
                    vertex_properties.append(parts[1:])
                elif current_element == 'face':
                    face_properties.append(parts[1:])
        
        print(f"Reading {vertex_count} vertices and {face_count} faces...")
        print(f"Vertex properties: {vertex_properties}")
        print(f"Face properties: {face_properties}")
        
        # Build struct format for vertex
        vertex_fmt = ''
        for prop in vertex_properties:
            if prop[0] == 'float':
                vertex_fmt += 'f'
            elif prop[0] == 'uchar':
                vertex_fmt += 'B'
        vertex_size = struct.calcsize('<' + vertex_fmt)
        
        # Read vertices
        for i in range(vertex_count):
            data = f.read(vertex_size)
            if len(data) != vertex_size:
                print(f"Warning: Incomplete vertex data at vertex {i}")
                break
            values = struct.unpack('<' + vertex_fmt, data)
            # Only keep x, y, z (first three floats)
            vertices.append(values[:3])
        
        # Read faces
        for i in range(face_count):
            count_data = f.read(1)
            if len(count_data) != 1:
                print(f"Warning: Incomplete face count at face {i}")
                break
            count = struct.unpack('<B', count_data)[0]
            if count == 3:
                indices_data = f.read(3 * 4)
                if len(indices_data) != 12:
                    print(f"Warning: Incomplete face data at face {i}")
                    break
                v1, v2, v3 = struct.unpack('<3i', indices_data)
                faces.append([v1, v2, v3])
            else:
                f.read(count * 4)
    vertices = np.array(vertices)
    faces = np.array(faces)
    return vertices, faces

def lonlat_to_utm_feet(vertices):
    """
    Convert (lon, lat) to UTM (meters), then to feet. Returns new array (x_ft, y_ft, z_ft)
    """
    lons = vertices[:, 0]
    lats = vertices[:, 1]
    zs = vertices[:, 2]
    # Use centroid to determine UTM zone
    lon0 = np.mean(lons)
    lat0 = np.mean(lats)
    utm_crs = CRS.from_proj4(f"+proj=utm +zone={((int((lon0 + 180) / 6) % 60) + 1)} +datum=WGS84 +units=m +south" if lat0 < 0 else f"+proj=utm +zone={((int((lon0 + 180) / 6) % 60) + 1)} +datum=WGS84 +units=m")
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
    xs_m, ys_m = transformer.transform(lons, lats)
    # Convert meters to feet
    xs_ft = xs_m * 3.28084
    ys_ft = ys_m * 3.28084
    return np.column_stack([xs_ft, ys_ft, zs]), utm_crs

def write_ascii_ply_file(filename, vertices, faces):
    """Write PLY file in ASCII format with vertices and faces"""
    with open(filename, 'w') as f:
        # Write header
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        # Write vertices
        for vertex in vertices:
            f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
        # Write faces
        for face in faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

def create_split_shifted_surface(vertices, faces):
    """
    Create a surface where half the vertices are shifted by 3 feet,
    and the other half by 1 foot. Split based on X coordinate.
    """
    x_coords = vertices[:, 0]
    x_mid = (x_coords.min() + x_coords.max()) / 2.0
    print(f"X coordinate range: {x_coords.min():.2f} to {x_coords.max():.2f}")
    print(f"Splitting at X = {x_mid:.2f}")
    shifted_vertices = vertices.copy()
    left_half = x_coords < x_mid
    right_half = x_coords >= x_mid
    left_count = np.sum(left_half)
    right_count = np.sum(right_half)
    print(f"Left half (X < {x_mid:.2f}): {left_count} vertices (shift by 3 feet)")
    print(f"Right half (X >= {x_mid:.2f}): {right_count} vertices (shift by 1 foot)")
    shifted_vertices[left_half, 2] -= 3.0
    shifted_vertices[right_half, 2] -= 1.0
    return shifted_vertices, left_half, right_half

def calculate_volume_estimate(vertices1, vertices2, faces):
    total_volume = 0.0
    for face in faces:
        v1_idx, v2_idx, v3_idx = face
        p1_1 = vertices1[v1_idx]
        p2_1 = vertices1[v2_idx]
        p3_1 = vertices1[v3_idx]
        p1_2 = vertices2[v1_idx]
        p2_2 = vertices2[v2_idx]
        p3_2 = vertices2[v3_idx]
        area = 0.5 * abs((p2_1[0] - p1_1[0]) * (p3_1[1] - p1_1[1]) - (p3_1[0] - p1_1[0]) * (p2_1[1] - p1_1[1]))
        h1 = abs(p1_2[2] - p1_1[2])
        h2 = abs(p2_2[2] - p2_1[2])
        h3 = abs(p3_2[2] - p3_1[2])
        avg_height = (h1 + h2 + h3) / 3.0
        prism_volume = area * avg_height
        total_volume += prism_volume
    return total_volume

def calculate_detailed_volume_analysis(vertices1, vertices2, faces, left_half, right_half):
    left_faces = []
    right_faces = []
    for face in faces:
        v1_idx, v2_idx, v3_idx = face
        if left_half[v1_idx] and left_half[v2_idx] and left_half[v3_idx]:
            left_faces.append(face)
        elif right_half[v1_idx] and right_half[v2_idx] and right_half[v3_idx]:
            right_faces.append(face)
    print(f"Left half faces: {len(left_faces)}")
    print(f"Right half faces: {len(right_faces)}")
    print(f"Mixed faces (spanning boundary): {len(faces) - len(left_faces) - len(right_faces)}")
    left_volume = calculate_volume_estimate(vertices1, vertices2, np.array(left_faces)) if left_faces else 0.0
    right_volume = calculate_volume_estimate(vertices1, vertices2, np.array(right_faces)) if right_faces else 0.0
    total_volume = calculate_volume_estimate(vertices1, vertices2, faces)
    return {
        'left_volume': left_volume,
        'right_volume': right_volume,
        'total_volume': total_volume,
        'left_faces': len(left_faces),
        'right_faces': len(right_faces),
        'mixed_faces': len(faces) - len(left_faces) - len(right_faces)
    }

def main():
    original_file = "data/test_files/tv_test.ply"
    shifted_file = "data/test_files/tv_test_split_shifted_feet.ply"
    print("Reading original binary PLY file...")
    vertices_wgs, faces = read_binary_ply_file(original_file)
    print(f"Original surface: {len(vertices_wgs)} vertices, {len(faces)} faces")
    print(f"Lon range: {vertices_wgs[:, 0].min():.6f} to {vertices_wgs[:, 0].max():.6f}")
    print(f"Lat range: {vertices_wgs[:, 1].min():.6f} to {vertices_wgs[:, 1].max():.6f}")
    print(f"Z range: {vertices_wgs[:, 2].min():.3f} to {vertices_wgs[:, 2].max():.3f}")
    print("\nConverting to UTM (feet)...")
    vertices_feet, utm_crs = lonlat_to_utm_feet(vertices_wgs)
    print(f"X (ft) range: {vertices_feet[:, 0].min():.2f} to {vertices_feet[:, 0].max():.2f}")
    print(f"Y (ft) range: {vertices_feet[:, 1].min():.2f} to {vertices_feet[:, 1].max():.2f}")
    print(f"Z (ft) range: {vertices_feet[:, 2].min():.3f} to {vertices_feet[:, 2].max():.3f}")
    print(f"UTM CRS used: {utm_crs}")
    print("\nCreating split-shifted surface...")
    shifted_vertices, left_half, right_half = create_split_shifted_surface(vertices_feet, faces)
    print(f"Shifted surface Z range: {shifted_vertices[:, 2].min():.3f} to {shifted_vertices[:, 2].max():.3f}")
    print(f"\nWriting shifted surface to {shifted_file}...")
    write_ascii_ply_file(shifted_file, shifted_vertices, faces)
    print("\nCalculating detailed volume analysis...")
    volume_analysis = calculate_detailed_volume_analysis(vertices_feet, shifted_vertices, faces, left_half, right_half)
    left_volume_cy = volume_analysis['left_volume'] / 27.0
    right_volume_cy = volume_analysis['right_volume'] / 27.0
    total_volume_cy = volume_analysis['total_volume'] / 27.0
    print(f"\n📊 Volume Analysis Results:")
    print(f"Left half (3ft shift): {volume_analysis['left_volume']:.2f} cubic feet ({left_volume_cy:.2f} cubic yards)")
    print(f"Right half (1ft shift): {volume_analysis['right_volume']:.2f} cubic feet ({right_volume_cy:.2f} cubic yards)")
    print(f"Total volume: {volume_analysis['total_volume']:.2f} cubic feet ({total_volume_cy:.2f} cubic yards)")
    x_range = vertices_feet[:, 0].max() - vertices_feet[:, 0].min()
    y_range = vertices_feet[:, 1].max() - vertices_feet[:, 1].min()
    total_area = x_range * y_range
    left_area = total_area * (volume_analysis['left_faces'] / len(faces))
    right_area = total_area * (volume_analysis['right_faces'] / len(faces))
    print(f"\n📐 Surface Analysis:")
    print(f"Total surface area: {total_area:.2f} square feet")
    print(f"Left half area (estimated): {left_area:.2f} square feet")
    print(f"Right half area (estimated): {right_area:.2f} square feet")
    left_avg_thickness = volume_analysis['left_volume'] / left_area if left_area > 0 else 0
    right_avg_thickness = volume_analysis['right_volume'] / right_area if right_area > 0 else 0
    total_avg_thickness = volume_analysis['total_volume'] / total_area
    print(f"\n📏 Thickness Analysis:")
    print(f"Left half average thickness: {left_avg_thickness:.3f} feet (expected: 3.000)")
    print(f"Right half average thickness: {right_avg_thickness:.3f} feet (expected: 1.000)")
    print(f"Overall average thickness: {total_avg_thickness:.3f} feet (expected: 2.000)")
    z_diff = vertices_feet[:, 2] - shifted_vertices[:, 2]
    left_z_diff = z_diff[left_half]
    right_z_diff = z_diff[right_half]
    print(f"\n✅ Verification:")
    if len(left_z_diff) > 0:
        print(f"Left half Z difference: {left_z_diff.min():.3f} to {left_z_diff.max():.3f} feet (expected: 3.000)")
    else:
        print("Left half Z difference: No vertices in left half")
    if len(right_z_diff) > 0:
        print(f"Right half Z difference: {right_z_diff.min():.3f} to {right_z_diff.max():.3f} feet (expected: 1.000)")
    else:
        print("Right half Z difference: No vertices in right half")
    print(f"\n✅ Successfully created {shifted_file}")
    print(f"📊 Total volume estimate for testing: {total_volume_cy:.2f} cubic yards")

if __name__ == "__main__":
    main() 