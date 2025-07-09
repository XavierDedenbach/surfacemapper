"""
PLY file parsing utility using plyfile library
"""
import numpy as np
from plyfile import PlyData
from typing import Tuple, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class PLYParser:
    """
    Handles PLY file parsing and validation
    """
    
    def __init__(self):
        self.supported_formats = ['ascii', 'binary_little_endian', 'binary_big_endian']
    
    def parse_ply_file(self, file_path: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Parse a PLY file and return vertices and faces (if available) in UTM coordinates.
        
        Args:
            file_path: Path to the PLY file (assumed to be in UTM coordinates)
            
        Returns:
            Tuple of (vertices, faces) where vertices are in UTM meters and faces may be None
        """
        try:
            plydata = PlyData.read(file_path)
            # Debug: check what format is actually returned
            fmt = plydata.header.format
            logger.debug(f"PLY format detected: {fmt}, type: {type(fmt)}")
            
            # Handle different format representations from plyfile
            if hasattr(fmt, '__call__'):
                # If format is a method, call it to get the string
                fmt_str = fmt()
            else:
                fmt_str = str(fmt)
            
            # Check if format is supported (case-insensitive)
            fmt_lower = fmt_str.lower()
            if not any(supported in fmt_lower for supported in ['ascii', 'binary_little_endian', 'binary_big_endian']):
                raise ValueError(f"Unsupported PLY format: {fmt_str}")
            
            # Defensive: check for required elements
            if 'vertex' not in plydata:
                raise ValueError("PLY file does not contain vertex data")
            # Extract vertices (x, y, z only)
            vertices = self._extract_vertices(plydata)
            faces = self._extract_faces(plydata)
            logger.info(f"Successfully parsed PLY file: {file_path}")
            logger.info(f"Vertices: {len(vertices)}, Faces: {len(faces) if faces is not None else 0}")
            
            # Validate that coordinates are in UTM range (meters, not degrees)
            if len(vertices) > 0:
                x_coords = vertices[:, 0]
                y_coords = vertices[:, 1]
                
                # UTM coordinates should be in meters (typically 100,000 to 1,000,000 for X, 4,000,000 to 10,000,000 for Y)
                is_utm = np.all(x_coords > 10000) and np.all(y_coords > 1000000)
                if not is_utm:
                    logger.warning(f"PLY file coordinates may not be in UTM format. X range: {x_coords.min():.2f} to {x_coords.max():.2f}, Y range: {y_coords.min():.2f} to {y_coords.max():.2f}")
                    logger.info("Assuming coordinates are in UTM meters for processing")
                else:
                    logger.info(f"PLY file coordinates confirmed in UTM format. X range: {x_coords.min():.2f} to {x_coords.max():.2f}m, Y range: {y_coords.min():.2f} to {y_coords.max():.2f}m")
            
            # Output format checks
            if not isinstance(vertices, np.ndarray) or vertices.ndim != 2 or vertices.shape[1] != 3:
                raise ValueError(f"Parsed vertices array has invalid format: {type(vertices)}, shape {getattr(vertices, 'shape', None)}")
            if faces is not None and (not isinstance(faces, np.ndarray) or (faces.ndim != 2 and len(faces) > 0)):
                raise ValueError(f"Parsed faces array has invalid format: {type(faces)}, shape {getattr(faces, 'shape', None)}")
            return vertices, faces
        except Exception as e:
            logger.error(f"Error parsing PLY file {file_path}: {str(e)}")
            msg = str(e).lower()
            if "corrupted" in msg or "incomplete" in msg or "unexpected end" in msg or "truncated" in msg:
                raise ValueError(f"PLY file appears to be corrupted or incomplete: {str(e)}")
            elif "not found" in msg:
                raise FileNotFoundError(f"PLY file not found: {file_path}")
            elif "missing one or more of x, y, z" in msg:
                raise ValueError(f"PLY file missing required x, y, z vertex properties: {str(e)}")
            else:
                raise ValueError(f"Error parsing PLY file: {str(e)}")
    
    def _extract_vertices(self, plydata: PlyData) -> np.ndarray:
        """
        Extract only x, y, z vertex data from PLY file, ignoring other properties.
        Handles both ASCII and binary PLY files, and ignores extra properties (normals, colors, etc).
        """
        if 'vertex' not in plydata:
            raise ValueError("PLY file does not contain vertex data")
        vertex_data = plydata['vertex']
        names = vertex_data.data.dtype.names
        required = ('x', 'y', 'z')
        if not all(r in names for r in required):
            raise ValueError("PLY vertex data missing one or more of x, y, z properties")
        try:
            # Defensive: always extract as float32 for consistency
            x = np.asarray(vertex_data['x'], dtype=np.float32)
            y = np.asarray(vertex_data['y'], dtype=np.float32)
            z = np.asarray(vertex_data['z'], dtype=np.float32)
            vertices = np.column_stack([x, y, z])
        except Exception as e:
            logger.error(f"Error extracting x, y, z from vertex data: {e}")
            raise ValueError(f"Failed to extract x, y, z coordinates from PLY vertex data: {e}")
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"Extracted vertex array has invalid shape: {vertices.shape}")
        return vertices
    
    def _extract_faces(self, plydata: PlyData) -> Optional[np.ndarray]:
        """
        Extract face data from PLY file if available
        """
        if 'face' not in plydata:
            return None
        
        face_data = plydata['face']
        # Handle different face formats (triangles, quads, etc.)
        faces = np.array(face_data['vertex_indices'].tolist())
        
        return faces
    
    def validate_ply_file(self, file_path: str) -> Dict[str, Any]:
        """
        Validate PLY file format and content
        """
        try:
            plydata = PlyData.read(file_path)
            
            validation_result = {
                'is_valid': True,
                'format': plydata.header.format,
                'vertex_count': len(plydata['vertex']) if 'vertex' in plydata else 0,
                'face_count': len(plydata['face']) if 'face' in plydata else 0,
                'has_sufficient_data': False
            }
            
            # Check for sufficient data (at least vertices)
            if validation_result['vertex_count'] > 0:
                validation_result['has_sufficient_data'] = True
            
            return validation_result
            
        except Exception as e:
            return {
                'is_valid': False,
                'error': str(e),
                'has_sufficient_data': False
            }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic information about PLY file
        """
        try:
            plydata = PlyData.read(file_path)
            
            # Fix: plydata.elements is a tuple, not a dictionary
            # Extract element names from the tuple
            element_names = [elem.name for elem in plydata.elements]
            
            # Fix: properly extract format string
            fmt = plydata.header.format
            if hasattr(fmt, '__call__'):
                # If format is a method, call it to get the string
                format_str = fmt()
            else:
                format_str = str(fmt)
            
            return {
                'format': format_str,
                'elements': element_names,
                'vertex_count': len(plydata['vertex']) if 'vertex' in plydata else 0,
                'face_count': len(plydata['face']) if 'face' in plydata else 0,
                'properties': {
                    'vertex': list(plydata['vertex'].data.dtype.names) if 'vertex' in plydata else [],
                    'face': list(plydata['face'].data.dtype.names) if 'face' in plydata else []
                }
            }
            
        except Exception as e:
            return {'error': str(e)} 