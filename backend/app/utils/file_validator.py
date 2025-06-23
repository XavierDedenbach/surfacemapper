"""
File validation utilities for PLY upload
"""
import os
from typing import BinaryIO

MAX_PLY_SIZE = 2 * 1024 * 1024 * 1024  # 2GB


def validate_file_extension(filename: str) -> bool:
    """Validate file extension is .ply"""
    return filename.lower().endswith('.ply') and filename.strip() != '.ply'


def validate_file_size(size_bytes: int) -> bool:
    """Validate file size is within limits (2GB max) and not empty"""
    return 0 < size_bytes <= MAX_PLY_SIZE


def validate_ply_format(file_or_bytes):
    """Validate that the file or bytes is a PLY file (ASCII or binary) by checking header structure and data integrity."""
    try:
        if hasattr(file_or_bytes, 'read'):
            # file-like
            file_or_bytes.seek(0)
            header = b""
            # Read up to 4096 bytes or until end_header
            for _ in range(100):
                line = file_or_bytes.readline()
                if not line:
                    break
                header += line
                if b'end_header' in line.lower():
                    break
            file_or_bytes.seek(0)
        elif isinstance(file_or_bytes, (bytes, bytearray)):
            header = file_or_bytes[:4096]
        else:
            return False
        
        header_str = header.decode(errors='ignore').lower()
        if not header_str.startswith('ply'):
            return False
        
        if ('format ascii' in header_str or 'format binary' in header_str) and \
           'end_header' in header_str and 'element vertex' in header_str:
            
            # Additional validation for binary PLY files
            if 'format binary' in header_str:
                return validate_binary_ply_integrity(file_or_bytes, header_str)
            
            return True
        return False
    except Exception:
        return False


def validate_binary_ply_integrity(file_or_bytes, header_str: str) -> bool:
    """Validate that binary PLY file has complete data after header."""
    try:
        # Parse header to get expected data size
        lines = header_str.split('\n')
        vertex_count = 0
        face_count = 0
        vertex_properties = 0
        face_properties = 0
        
        for line in lines:
            line = line.strip()
            if line.startswith('element vertex'):
                parts = line.split()
                if len(parts) >= 3:
                    vertex_count = int(parts[2])
            elif line.startswith('element face'):
                parts = line.split()
                if len(parts) >= 3:
                    face_count = int(parts[2])
            elif line.startswith('property') and 'element vertex' in header_str[:header_str.find(line)]:
                vertex_properties += 1
            elif line.startswith('property') and 'element face' in header_str[:header_str.find(line)]:
                face_properties += 1
        
        if vertex_count == 0:
            return False
        
        # Calculate expected data size
        # Each vertex property is typically 4 bytes (float) or 1 byte (uchar)
        vertex_data_size = vertex_count * vertex_properties * 4  # Assume float for now
        
        # Face data: each face has a count byte + vertex indices
        face_data_size = 0
        if face_count > 0:
            # Assume triangular faces (3 vertices per face)
            face_data_size = face_count * (1 + 3 * 4)  # 1 byte count + 3 int indices
        
        expected_data_size = vertex_data_size + face_data_size
        
        # Check if file has enough data
        if hasattr(file_or_bytes, 'read'):
            file_or_bytes.seek(0)
            # Find end_header position
            header_end = 0
            file_or_bytes.seek(0)
            while True:
                line = file_or_bytes.readline()
                if not line:
                    break
                header_end += len(line)
                if b'end_header' in line.lower():
                    break
            
            # Check if file has enough data after header
            file_or_bytes.seek(0, 2)  # Seek to end
            file_size = file_or_bytes.tell()
            data_size = file_size - header_end
            
            if data_size < expected_data_size * 0.5:  # Allow some tolerance
                return False
        elif isinstance(file_or_bytes, (bytes, bytearray)):
            # For bytes, check if we have reasonable amount of data
            header_end = header_str.find('end_header') + len('end_header')
            if header_end == -1:
                return False
            
            data_size = len(file_or_bytes) - header_end
            if data_size < expected_data_size * 0.5:  # Allow some tolerance
                return False
        
        return True
        
    except Exception:
        return False 