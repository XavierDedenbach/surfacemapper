#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from unittest.mock import mock_open, patch
from app.utils.file_validator import validate_ply_format

# Test content
content = b"""ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
end_header
"""

# Add some binary data
import struct
vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
for x, y, z in vertices:
    content += struct.pack('<fff', x, y, z)

print("Content length:", len(content))
print("Content preview:", content[:100])

# Test with mock
with patch('builtins.open', mock_open(read_data=content)) as mock_file:
    with open('test.ply', 'rb') as f:
        print("File type:", type(f))
        print("Has readline:", hasattr(f, 'readline'))
        print("Has seek:", hasattr(f, 'seek'))
        
        # Test readline
        f.seek(0)
        first_line = f.readline()
        print("First line:", first_line)
        
        # Test validation
        f.seek(0)
        result = validate_ply_format(f)
        print("Validation result:", result) 