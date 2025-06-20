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
    """Validate that the file or bytes is a PLY file (ASCII or binary) by checking header structure."""
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
            return True
        return False
    except Exception:
        return False 