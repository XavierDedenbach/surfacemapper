"""
Tests for file validation logic
"""
import pytest
import tempfile
import os
from unittest.mock import mock_open, patch
from app.utils.file_validator import validate_file_extension, validate_file_size, validate_ply_format

class TestFileValidation:
    """Test cases for file validation functionality"""
    
    def test_validate_file_extension_valid(self):
        """Test validation of valid PLY file extensions"""
        valid_extensions = [
            "test.ply",
            "surface.PLY", 
            "file_with_underscores.ply",
            "file-with-dashes.ply",
            "file123.ply"
        ]
        
        for filename in valid_extensions:
            assert validate_file_extension(filename) is True
    
    def test_validate_file_extension_invalid(self):
        """Test validation of invalid file extensions"""
        invalid_extensions = [
            "",  # Empty
            ".ply",  # Only extension
            "file.txt",
            "file.obj", 
            "file.stl",
            "no_extension",
            "file.ply.txt",  # Double extension
            "PLY",  # No dot
        ]
        
        for filename in invalid_extensions:
            assert validate_file_extension(filename) is False
    
    def test_validate_file_size_acceptable(self):
        """Test file size validation for acceptable sizes"""
        acceptable_sizes = [
            1024,  # 1KB
            1024 * 1024,  # 1MB
            100 * 1024 * 1024,  # 100MB
            1024 * 1024 * 1024,  # 1GB
            2 * 1024 * 1024 * 1024  # 2GB (max)
        ]
        
        for size in acceptable_sizes:
            assert validate_file_size(size) is True
    
    def test_validate_file_size_too_large(self):
        """Test file size validation for oversized files"""
        oversized = [
            3 * 1024 * 1024 * 1024,  # 3GB
            5 * 1024 * 1024 * 1024,  # 5GB
            10 * 1024 * 1024 * 1024  # 10GB
        ]
        
        for size in oversized:
            assert validate_file_size(size) is False
    
    def test_validate_file_size_empty_or_zero(self):
        """Test file size validation for empty or zero files"""
        invalid_sizes = [0, -1, -1024]
        
        for size in invalid_sizes:
            assert validate_file_size(size) is False
    
    def test_validate_ply_format_valid_ascii(self):
        """Test validation of valid ASCII PLY format"""
        valid_ply_content = b"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 1.0
"""
        
        with patch('builtins.open', mock_open(read_data=valid_ply_content)):
            with open('test.ply', 'rb') as f:
                assert validate_ply_format(f) is True
    
    def test_validate_ply_format_valid_binary(self):
        """Test validation of valid binary PLY format"""
        binary_ply_content = b"""ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
end_header
"""
        # Add binary vertex data
        import struct
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        for x, y, z in vertices:
            binary_ply_content += struct.pack('<fff', x, y, z)
        
        with patch('builtins.open', mock_open(read_data=binary_ply_content)):
            with open('test.ply', 'rb') as f:
                assert validate_ply_format(f) is True
    
    def test_validate_ply_format_invalid_header(self):
        """Test validation of PLY files with invalid headers"""
        invalid_contents = [
            b"not a ply file",
            b"ply\ninvalid format",
            b"ply\nformat ascii 1.0\nno end_header",
            b"ply\nformat ascii 1.0\nelement face 3\nend_header",  # no vertex element
        ]
        
        for content in invalid_contents:
            with patch('builtins.open', mock_open(read_data=content)):
                with open('test.ply', 'rb') as f:
                    assert validate_ply_format(f) is False
    
    def test_validate_ply_format_corrupted(self):
        """Test validation of corrupted PLY files"""
        corrupted_content = b"""ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
end_header
0.0 0.0 0.0
1.0 0.0 0.0
# Missing third vertex - corrupted file
"""
        
        with patch('builtins.open', mock_open(read_data=corrupted_content)):
            with open('test.ply', 'rb') as f:
                # Should still pass format validation (content validation is separate)
                assert validate_ply_format(f) is True
    
    def test_validate_ply_format_empty_file(self):
        """Test validation of empty files"""
        empty_content = b""
        
        with patch('builtins.open', mock_open(read_data=empty_content)):
            with open('test.ply', 'rb') as f:
                assert validate_ply_format(f) is False
    
    def test_validate_ply_format_large_header(self):
        """Test validation of PLY files with large headers"""
        large_header = b"""ply
format ascii 1.0
comment This is a very long comment that goes on and on
comment Another comment line
comment Yet another comment
element vertex 3
property float x
property float y  
property float z
property float nx
property float ny
property float nz
property uchar red
property uchar green
property uchar blue
end_header
0.0 0.0 0.0 0.0 0.0 1.0 255 0 0
1.0 0.0 0.0 0.0 0.0 1.0 0 255 0
0.0 1.0 0.0 0.0 0.0 1.0 0 0 255
"""
        
        with patch('builtins.open', mock_open(read_data=large_header)):
            with open('test.ply', 'rb') as f:
                assert validate_ply_format(f) is True
    
    def test_validate_ply_format_mixed_case(self):
        """Test validation of PLY files with mixed case headers"""
        mixed_case_content = b"""PLY
FORMAT ASCII 1.0
ELEMENT VERTEX 3
PROPERTY FLOAT X
PROPERTY FLOAT Y
PROPERTY FLOAT Z
END_HEADER
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
"""
        
        with patch('builtins.open', mock_open(read_data=mixed_case_content)):
            with open('test.ply', 'rb') as f:
                # Should pass because validator is case-insensitive
                assert validate_ply_format(f) is True
    
    def test_validate_ply_format_with_faces(self):
        """Test validation of PLY files that include face data"""
        ply_with_faces = b"""ply
format ascii 1.0
element vertex 4
property float x
property float y
property float z
element face 2
property list uchar int vertex_indices
end_header
0.0 0.0 0.0
1.0 0.0 0.0
0.0 1.0 0.0
1.0 1.0 1.0
3 0 1 2
3 1 3 2
"""
        
        with patch('builtins.open', mock_open(read_data=ply_with_faces)):
            with open('test.ply', 'rb') as f:
                assert validate_ply_format(f) is True
    
    def test_validate_ply_format_binary_big_endian(self):
        """Test validation of binary big endian PLY format"""
        binary_big_endian = b"""ply
format binary_big_endian 1.0
element vertex 3
property float x
property float y
property float z
end_header
"""
        # Add binary vertex data (big endian)
        import struct
        vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
        for x, y, z in vertices:
            binary_big_endian += struct.pack('>fff', x, y, z)
        
        with patch('builtins.open', mock_open(read_data=binary_big_endian)):
            with open('test.ply', 'rb') as f:
                assert validate_ply_format(f) is True
    
    def test_validate_ply_format_file_io_error(self):
        """Test validation when file IO fails"""
        class FailingFile:
            def seek(self, *args, **kwargs):
                pass
            def readline(self, *args, **kwargs):
                raise IOError("File read error")
        f = FailingFile()
        assert validate_ply_format(f) is False
    
    def test_validate_ply_format_unicode_filename(self):
        """Test validation with unicode filenames"""
        unicode_filename = "测试文件.ply"
        assert validate_file_extension(unicode_filename) is True
    
    def test_validate_ply_format_special_characters_filename(self):
        """Test validation with special characters in filename"""
        special_filename = "file with spaces and @#$%.ply"
        assert validate_file_extension(special_filename) is True 