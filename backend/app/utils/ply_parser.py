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
        Parse a PLY file and return vertices and faces (if available)
        
        Args:
            file_path: Path to the PLY file
            
        Returns:
            Tuple of (vertices, faces) where faces may be None
        """
        try:
            plydata = PlyData.read(file_path)
            vertices = self._extract_vertices(plydata)
            faces = self._extract_faces(plydata)
            
            logger.info(f"Successfully parsed PLY file: {file_path}")
            logger.info(f"Vertices: {len(vertices)}, Faces: {len(faces) if faces is not None else 0}")
            
            return vertices, faces
            
        except Exception as e:
            logger.error(f"Error parsing PLY file {file_path}: {str(e)}")
            # Provide more specific error messages for common issues
            if "corrupted" in str(e).lower() or "incomplete" in str(e).lower():
                raise ValueError(f"PLY file appears to be corrupted or incomplete: {str(e)}")
            elif "not found" in str(e).lower():
                raise FileNotFoundError(f"PLY file not found: {file_path}")
            else:
                raise ValueError(f"Error parsing PLY file: {str(e)}")
    
    def _extract_vertices(self, plydata: PlyData) -> np.ndarray:
        """
        Extract vertex data from PLY file
        """
        if 'vertex' not in plydata:
            raise ValueError("PLY file does not contain vertex data")
        
        vertex_data = plydata['vertex']
        vertices = np.column_stack([
            vertex_data['x'],
            vertex_data['y'],
            vertex_data['z']
        ])
        
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
            
            return {
                'format': plydata.header.format,
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