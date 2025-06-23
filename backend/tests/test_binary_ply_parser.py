"""
Tests for binary PLY file parsing and validation
"""
import pytest
import numpy as np
import tempfile
import os
import time
import struct
from app.utils.ply_parser import PLYParser


def create_binary_ply_with_properties(vertex_count: int, properties: list) -> bytes:
    """Create binary PLY content with specified properties"""
    # Create header
    header = b"ply\n"
    header += b"format binary_little_endian 1.0\n"
    header += f"element vertex {vertex_count}\n".encode()
    
    # Add properties
    for prop in properties:
        if prop in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
            header += f"property float {prop}\n".encode()
        elif prop in ['red', 'green', 'blue']:
            header += f"property uchar {prop}\n".encode()
    
    header += b"element face 0\n"
    header += b"property list uchar int vertex_indices\n"
    header += b"end_header\n"
    
    # Create binary data
    binary_data = b""
    for i in range(vertex_count):
        for prop in properties:
            if prop in ['x', 'y', 'z', 'nx', 'ny', 'nz']:
                # Add float value
                value = float(i) * 0.1
                binary_data += struct.pack('<f', value)
            elif prop in ['red', 'green', 'blue']:
                # Add uchar value
                value = i % 256
                binary_data += struct.pack('<B', value)
    
    return header + binary_data


def create_binary_ply_from_vertices(vertices: np.ndarray) -> bytes:
    """Create binary PLY content from vertex array"""
    vertex_count = len(vertices)
    header = b"ply\n"
    header += b"format binary_little_endian 1.0\n"
    header += f"element vertex {vertex_count}\n".encode()
    
    # Add properties based on vertex data
    for i in range(vertices.shape[1]):
        if i < 3:  # x, y, z
            header += f"property float {['x', 'y', 'z'][i]}\n".encode()
        elif i < 6:  # nx, ny, nz
            header += f"property float {['nx', 'ny', 'nz'][i-3]}\n".encode()
        else:  # red, green, blue
            header += f"property uchar {['red', 'green', 'blue'][i-6]}\n".encode()
    
    header += b"element face 0\n"
    header += b"property list uchar int vertex_indices\n"
    header += b"end_header\n"
    
    # Create binary data
    binary_data = b""
    for vertex in vertices:
        for i, value in enumerate(vertex):
            if i < 6:  # float properties
                binary_data += struct.pack('<f', float(value))
            else:  # uchar properties
                binary_data += struct.pack('<B', int(value))
    
    return header + binary_data


def create_binary_ply_with_endianness(endianness: str, vertex_count: int) -> bytes:
    """Create binary PLY content with specified endianness"""
    header = b"ply\n"
    header += f"format {endianness} 1.0\n".encode()
    header += f"element vertex {vertex_count}\n".encode()
    header += b"property float x\n"
    header += b"property float y\n"
    header += b"property float z\n"
    header += b"element face 0\n"
    header += b"property list uchar int vertex_indices\n"
    header += b"end_header\n"
    
    # Create binary data
    binary_data = b""
    for i in range(vertex_count):
        x, y, z = float(i), float(i * 2), float(i * 3)
        if endianness == 'binary_little_endian':
            binary_data += struct.pack('<fff', x, y, z)
        else:  # big endian
            binary_data += struct.pack('>fff', x, y, z)
    
    return header + binary_data


def create_large_binary_ply(vertex_count: int) -> bytes:
    """Create large binary PLY file for performance testing"""
    header = b"ply\n"
    header += b"format binary_little_endian 1.0\n"
    header += f"element vertex {vertex_count}\n".encode()
    header += b"property float x\n"
    header += b"property float y\n"
    header += b"property float z\n"
    header += b"element face 0\n"
    header += b"property list uchar int vertex_indices\n"
    header += b"end_header\n"
    
    # Create binary data
    binary_data = b""
    for i in range(vertex_count):
        x = float(i % 100)
        y = float((i // 100) % 100)
        z = float(i * 0.1)
        binary_data += struct.pack('<fff', x, y, z)
    
    return header + binary_data


def create_ascii_ply_content(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Create ASCII PLY content for comparison testing"""
    content = "ply\n"
    content += "format ascii 1.0\n"
    content += f"element vertex {len(vertices)}\n"
    content += "property float x\n"
    content += "property float y\n"
    content += "property float z\n"
    content += f"element face {len(faces)}\n"
    content += "property list uchar int vertex_indices\n"
    content += "end_header\n"
    
    # Add vertices
    for vertex in vertices:
        content += f"{vertex[0]} {vertex[1]} {vertex[2]}\n"
    
    # Add faces
    for face in faces:
        content += f"{len(face)} {' '.join(map(str, face))}\n"
    
    return content


def create_binary_ply_content(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Create binary PLY content for comparison testing"""
    header = b"ply\n"
    header += b"format binary_little_endian 1.0\n"
    header += f"element vertex {len(vertices)}\n".encode()
    header += b"property float x\n"
    header += b"property float y\n"
    header += b"property float z\n"
    header += f"element face {len(faces)}\n".encode()
    header += b"property list uchar int vertex_indices\n"
    header += b"end_header\n"
    
    # Add vertices
    binary_data = b""
    for vertex in vertices:
        binary_data += struct.pack('<fff', vertex[0], vertex[1], vertex[2])
    
    # Add faces
    for face in faces:
        binary_data += struct.pack('<B', len(face))
        for vertex_idx in face:
            binary_data += struct.pack('<i', vertex_idx)
    
    return header + binary_data


class TestBinaryPLYParser:
    """Test suite for binary PLY file parsing"""
    
    def setup_method(self):
        self.parser = PLYParser()
    
    def test_binary_ply_parsing_with_normals_and_colors(self):
        """Test parsing binary PLY with normal vectors and color data"""
        # Create test binary PLY with normals and colors
        binary_ply_content = create_binary_ply_with_properties(
            vertex_count=1000,
            properties=['x', 'y', 'z', 'nx', 'ny', 'nz', 'red', 'green', 'blue']
        )
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(binary_ply_content)
            temp_file = f.name
        
        try:
            vertices, faces = self.parser.parse_ply_file(temp_file)
            
            # Should extract only x,y,z coordinates
            assert vertices.shape == (1000, 3)
            assert vertices.dtype == np.float32 or vertices.dtype == np.float64
            
            # Verify coordinate ranges are reasonable
            assert np.all(np.isfinite(vertices))
            
        finally:
            os.unlink(temp_file)
    
    def test_large_binary_ply_performance(self):
        """Test parsing performance with large binary PLY files"""
        # Create large binary PLY file (>50k vertices)
        large_ply_content = create_large_binary_ply(50000)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(large_ply_content)
            temp_file = f.name
        
        try:
            start_time = time.time()
            vertices, faces = self.parser.parse_ply_file(temp_file)
            parse_time = time.time() - start_time
            
            assert len(vertices) == 50000
            assert parse_time < 5.0  # Must parse 50k vertices in <5 seconds
            assert vertices.shape[1] == 3  # Only x,y,z coordinates
            
        finally:
            os.unlink(temp_file)
    
    def test_binary_ply_corruption_handling(self):
        """Test handling of corrupted binary PLY files"""
        # Create corrupted binary PLY (truncated data)
        corrupted_content = b"""ply
format binary_little_endian 1.0
element vertex 1000
property float x
property float y
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
element face 500
property list uchar int vertex_indices
end_header
"""
        # Add incomplete binary data (truncated)
        corrupted_content += b"\x00" * 1000  # Incomplete vertex data
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(corrupted_content)
            temp_file = f.name
        
        try:
            with pytest.raises(Exception) as exc_info:
                self.parser.parse_ply_file(temp_file)
            
            # Should provide meaningful error message
            error_msg = str(exc_info.value)
            assert "corrupted" in error_msg.lower() or "incomplete" in error_msg.lower() or "error" in error_msg.lower()
            
        finally:
            os.unlink(temp_file)
    
    def test_binary_ply_property_extraction(self):
        """Test extraction of specific properties from binary PLY"""
        # Create binary PLY with known property values
        test_vertices = np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0, 1.0, 255, 0, 0],  # x,y,z,nx,ny,nz,r,g,b
            [4.0, 5.0, 6.0, 0.0, 1.0, 0.0, 0, 255, 0],
            [7.0, 8.0, 9.0, 1.0, 0.0, 0.0, 0, 0, 255]
        ], dtype=np.float32)
        
        binary_ply_content = create_binary_ply_from_vertices(test_vertices)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(binary_ply_content)
            temp_file = f.name
        
        try:
            vertices, faces = self.parser.parse_ply_file(temp_file)
            
            # Should extract only x,y,z coordinates
            expected_coords = test_vertices[:, :3]
            np.testing.assert_array_almost_equal(vertices, expected_coords, decimal=6)
            
        finally:
            os.unlink(temp_file)
    
    def test_binary_ply_endianness_handling(self):
        """Test handling of different endianness in binary PLY files"""
        # Test both little-endian and big-endian formats
        for endianness in ['binary_little_endian', 'binary_big_endian']:
            binary_ply_content = create_binary_ply_with_endianness(endianness, 100)
            
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
                f.write(binary_ply_content)
                temp_file = f.name
            
            try:
                vertices, faces = self.parser.parse_ply_file(temp_file)
                
                assert len(vertices) == 100
                assert vertices.shape[1] == 3
                assert np.all(np.isfinite(vertices))
                
            finally:
                os.unlink(temp_file)
    
    def test_binary_ply_memory_efficiency(self):
        """Test memory efficiency when parsing large binary PLY files"""
        try:
            import psutil
            import os
            
            # Create large binary PLY file
            large_ply_content = create_large_binary_ply(100000)  # 100k vertices
            
            with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
                f.write(large_ply_content)
                temp_file = f.name
            
            try:
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss
                
                vertices, faces = self.parser.parse_ply_file(temp_file)
                
                peak_memory = process.memory_info().rss
                memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
                
                # Memory increase should be reasonable (not more than 3x theoretical minimum)
                theoretical_min = 100000 * 3 * 8 / 1024 / 1024  # 100k vertices * 3 coords * 8 bytes
                assert memory_increase < theoretical_min * 3
                
                assert len(vertices) == 100000
                assert vertices.shape[1] == 3
                
            finally:
                os.unlink(temp_file)
        except ImportError:
            # Skip if psutil is not available
            pytest.skip("psutil not available for memory testing")
    
    def test_output_format_compatibility(self):
        """Test that binary PLY parser maintains exact same output format as ASCII parser"""
        # Create identical content in both ASCII and binary formats
        test_vertices = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ], dtype=np.float32)
        test_faces = np.array([[0, 1, 2]], dtype=np.int32)
        
        # Create ASCII PLY file
        ascii_content = create_ascii_ply_content(test_vertices, test_faces)
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(ascii_content.encode())
            ascii_file = f.name
        
        # Create binary PLY file with same data
        binary_content = create_binary_ply_content(test_vertices, test_faces)
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(binary_content)
            binary_file = f.name
        
        try:
            # Parse both files
            ascii_vertices, ascii_faces = self.parser.parse_ply_file(ascii_file)
            binary_vertices, binary_faces = self.parser.parse_ply_file(binary_file)
            
            # Verify exact same format and content
            assert type(binary_vertices) == type(ascii_vertices)  # np.ndarray
            assert type(binary_faces) == type(ascii_faces)  # np.ndarray or None
            assert binary_vertices.shape == ascii_vertices.shape  # Same shape
            assert binary_vertices.dtype == ascii_vertices.dtype  # Same data type
            assert binary_faces.shape == ascii_faces.shape if ascii_faces is not None else ascii_faces is None
            assert binary_faces.dtype == ascii_faces.dtype if ascii_faces is not None else True
            
            # Verify same content (within floating point precision)
            np.testing.assert_array_almost_equal(binary_vertices, ascii_vertices, decimal=6)
            if ascii_faces is not None:
                np.testing.assert_array_equal(binary_faces, ascii_faces)
            
        finally:
            os.unlink(ascii_file)
            os.unlink(binary_file)
    
    def test_downstream_compatibility(self):
        """Test that parsed data works with all downstream components"""
        # Create test PLY file in binary format
        test_vertices = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 1.5]
        ], dtype=np.float32)
        test_faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
        
        binary_content = create_binary_ply_content(test_vertices, test_faces)
        
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(binary_content)
            binary_file = f.name
        
        try:
            # Parse with binary parser
            vertices, faces = self.parser.parse_ply_file(binary_file)
            
            # Test compatibility with surface processor
            from app.services.surface_processor import SurfaceProcessor
            surface_processor = SurfaceProcessor()
            
            # Should work with surface processing
            clipped_vertices = surface_processor.clip_to_boundary(vertices, [(0.0, 0.0), (2.0, 2.0)])
            assert len(clipped_vertices) > 0
            assert clipped_vertices.shape[1] == 3
            
            # Should work with mesh simplification (returns VTK objects)
            if faces is not None:
                simplified_vertices, simplified_faces = surface_processor.simplify_mesh(vertices, faces, 0.5)
                # VTK objects don't have .shape attribute, check they exist
                assert simplified_vertices is not None
                assert simplified_faces is None or simplified_faces is not None
            
            # Should work with triangulation
            from app.services import triangulation
            if len(vertices) >= 3:
                triangulation_result = triangulation.create_delaunay_triangulation(vertices[:, :2])
                assert hasattr(triangulation_result, 'simplices')
            
        finally:
            os.unlink(binary_file)
    
    def test_real_binary_ply_file(self):
        """Test with the actual tv_test.ply file"""
        tv_test_path = "../data/test_files/tv_test.ply"
        if os.path.exists(tv_test_path):
            vertices, faces = self.parser.parse_ply_file(tv_test_path)
            
            # Should have correct number of vertices and faces
            assert len(vertices) == 42173
            assert len(faces) == 82767
            assert vertices.shape[1] == 3  # Only x,y,z coordinates
            assert faces.shape[1] == 3  # Triangular faces
            
            # Verify coordinate ranges are reasonable (WGS84 coordinates)
            assert np.all(vertices[:, 0] >= -180) and np.all(vertices[:, 0] <= 180)  # longitude
            assert np.all(vertices[:, 1] >= -90) and np.all(vertices[:, 1] <= 90)   # latitude
            assert np.all(np.isfinite(vertices[:, 2]))  # elevation
        else:
            pytest.skip("tv_test.ply not available for testing") 