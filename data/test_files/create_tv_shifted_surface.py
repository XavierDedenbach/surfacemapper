#!/usr/bin/env python3
"""
Go-to script for shifting a PLY surface (ASCII or binary) by a Z offset.
- Handles variable-length faces (triangles, quads, etc.)
- Shifts all Z coordinates by a specified offset (default 3 ft)
- Writes the shifted surface to a new PLY file, preserving face structure
- Prints summary info and volume estimate (using only triangle faces)

Usage:
    python create_tv_shifted_surface.py [input_file] [output_file] [z_offset]
    (all args optional, defaults: test_ply_tv.ply, test_ply_tv_shifted_3ft.ply, 3.0)
"""
import sys
import os
import numpy as np
import struct
from pathlib import Path

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Shift a PLY surface by a Z offset.")
    parser.add_argument('input_file', nargs='?', default='test_ply_tv.ply', help='Input PLY file')
    parser.add_argument('output_file', nargs='?', default='test_ply_tv_shifted_3ft.ply', help='Output PLY file')
    parser.add_argument('--z_offset', type=float, default=3.0, help='Z offset (feet)')
    return parser.parse_args()

def read_ply_header(f):
    header_lines = []
    format_type = None
    vertex_count = None
    face_count = None
    properties = []
    face_property = None
    while True:
        line = f.readline().decode('ascii')
        header_lines.append(line)
        if line.startswith('format'):
            format_type = line.split()[1]
        elif line.startswith('element vertex'):
            vertex_count = int(line.split()[2])
        elif line.startswith('element face'):
            face_count = int(line.split()[2])
        elif line.startswith('property') and vertex_count is not None and face_count is None:
            properties.append(line.strip())
        elif line.startswith('property list') and face_count is not None:
            face_property = line.strip()
        if line.strip() == 'end_header':
            break
    return {
        'header_lines': header_lines,
        'format_type': format_type,
        'vertex_count': vertex_count,
        'face_count': face_count,
        'properties': properties,
        'face_property': face_property,
        'header_size': f.tell()
    }

def read_ascii_ply_vertices(f, vertex_count, num_props):
    vertices = []
    for _ in range(vertex_count):
        vals = f.readline().decode('ascii').strip().split()
        vertices.append([float(vals[0]), float(vals[1]), float(vals[2])])
    return np.array(vertices, dtype=np.float32)

def read_ascii_ply_faces(f, face_count):
    faces = []
    for _ in range(face_count):
        vals = f.readline().decode('ascii').strip().split()
        n = int(vals[0])
        faces.append([int(x) for x in vals[1:1+n]])
    return faces

def read_binary_ply_vertices(f, vertex_count, properties):
    # Only x, y, z are used
    prop_types = []
    for p in properties:
        if 'float' in p:
            prop_types.append('f')
        elif 'uchar' in p:
            prop_types.append('B')
        else:
            raise ValueError(f"Unsupported property: {p}")
    stride = struct.calcsize('<' + ''.join(prop_types))
    vertices = []
    for _ in range(vertex_count):
        data = f.read(stride)
        vals = struct.unpack('<' + ''.join(prop_types), data)
        vertices.append([vals[0], vals[1], vals[2]])
    return np.array(vertices, dtype=np.float32)

def read_binary_ply_faces(f, face_count, face_property):
    # property list uchar int vertex_indices
    faces = []
    for _ in range(face_count):
        n = struct.unpack('<B', f.read(1))[0]
        idxs = struct.unpack(f'<{n}i', f.read(n*4))
        faces.append(list(idxs))
    return faces

def write_binary_ply(file_path, header_lines, vertices, faces):
    # Write header
    with open(file_path, 'wb') as f:
        for line in header_lines:
            f.write(line.encode('ascii'))
        # Write vertex data (pad with zeros for extra properties)
        for v in vertices:
            f.write(struct.pack('<fff', v[0], v[1], v[2]))
            # Write zeros for any extra properties (assume 6 extra: nx,ny,nz,red,green,blue)
            f.write(struct.pack('<fffBBB', 0.0, 0.0, 0.0, 0, 0, 0))
        # Write face data
        for face in faces:
            f.write(struct.pack('<B', len(face)))
            f.write(struct.pack(f'<{len(face)}i', *face))

def write_ascii_ply(file_path, header_lines, vertices, faces):
    with open(file_path, 'w') as f:
        for line in header_lines:
            f.write(line)
        for v in vertices:
            f.write(f"{v[0]} {v[1]} {v[2]} 0 0 0 0 0 0\n")
        for face in faces:
            f.write(f"{len(face)} {' '.join(str(i) for i in face)}\n")

def calculate_volume_estimate(vertices, faces):
    # Only use triangle faces
    total_volume = 0.0
    tri_count = 0
    for face in faces:
        if len(face) == 3:
            v1, v2, v3 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
            edge1 = np.array([v2[0] - v1[0], v2[1] - v1[1]])
            edge2 = np.array([v3[0] - v1[0], v3[1] - v1[1]])
            area = 0.5 * abs(edge1[0] * edge2[1] - edge1[1] * edge2[0])
            avg_z = (v1[2] + v2[2] + v3[2]) / 3.0
            volume_cubic_feet = area * avg_z
            volume_cubic_yards = volume_cubic_feet / 27.0
            total_volume += volume_cubic_yards
            tri_count += 1
    return total_volume, tri_count

def main():
    args = parse_args()
    input_file = args.input_file
    output_file = args.output_file
    z_offset = args.z_offset

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    with open(input_file, 'rb') as f:
        header_info = read_ply_header(f)
        fmt = header_info['format_type']
        vertex_count = header_info['vertex_count']
        face_count = header_info['face_count']
        properties = header_info['properties']
        face_property = header_info['face_property']
        header_lines = header_info['header_lines']
        # Read vertices and faces
        if fmt == 'ascii':
            vertices = read_ascii_ply_vertices(f, vertex_count, len(properties))
            faces = read_ascii_ply_faces(f, face_count)
        elif fmt == 'binary_little_endian':
            vertices = read_binary_ply_vertices(f, vertex_count, properties)
            faces = read_binary_ply_faces(f, face_count, face_property)
        else:
            print(f"Unsupported PLY format: {fmt}")
            sys.exit(1)

    print(f"Read {len(vertices)} vertices, {len(faces)} faces from {input_file}")
    orig_vol, tri_count = calculate_volume_estimate(vertices, faces)
    print(f"Original volume estimate (triangles only): {orig_vol:,.2f} cubic yards (from {tri_count} triangles)")

    # Shift Z
    shifted_vertices = vertices.copy()
    shifted_vertices[:, 2] -= z_offset
    shifted_vol, _ = calculate_volume_estimate(shifted_vertices, faces)
    print(f"Shifted volume estimate (triangles only): {shifted_vol:,.2f} cubic yards")
    print(f"Volume between surfaces: {abs(orig_vol - shifted_vol):,.2f} cubic yards")

    # Write output
    if header_info['format_type'] == 'ascii':
        write_ascii_ply(output_file, header_lines, shifted_vertices, faces)
    else:
        write_binary_ply(output_file, header_lines, shifted_vertices, faces)
    print(f"Shifted surface written to {output_file}")

if __name__ == "__main__":
    main() 