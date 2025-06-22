"""
Surface processing service for handling PLY files and surface operations
"""
import numpy as np
from typing import List, Tuple, Optional
from ..utils.ply_parser import PLYParser
import pyvista as pv
from . import triangulation, thickness_calculator
from .volume_calculator import VolumeCalculator
from ..utils.serialization import make_json_serializable

class SurfaceProcessor:
    """
    Handles PLY parsing, clipping, mesh simplification, and base surface generation
    """
    
    def __init__(self):
        self.parser = PLYParser()
    
    def parse_surface(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Parse a PLY file and return vertex and face data
        """
        vertices, faces = self.parser.parse_ply_file(file_path)
        return vertices, faces
    
    def clip_to_boundary(self, vertices: np.ndarray, boundary: List[Tuple[float, float]]) -> np.ndarray:
        """
        Clip surface vertices to the defined analysis boundary
        """
        # Validate input
        if vertices.shape[1] != 3:
            raise ValueError("Vertices must have 3 columns (x, y, z)")
        
        if len(boundary) != 2:
            raise ValueError("Boundary must be a list of 2 tuples defining rectangle corners")
        
        # Extract boundary coordinates
        min_x, min_y = boundary[0]
        max_x, max_y = boundary[1]
        
        # Filter vertices within boundary
        mask = (
            (vertices[:, 0] >= min_x) & (vertices[:, 0] <= max_x) &
            (vertices[:, 1] >= min_y) & (vertices[:, 1] <= max_y)
        )
        
        return vertices[mask]
    
    def generate_base_surface(self, reference_surface: np.ndarray, offset: float) -> np.ndarray:
        """
        Generate a flat base surface with specified vertical offset
        """
        # Validate input
        if len(reference_surface) == 0:
            raise ValueError("Reference surface cannot be empty")
        
        if offset < 0:
            raise ValueError("Offset must be non-negative")
        
        if reference_surface.shape[1] != 3:
            raise ValueError("Reference surface must have 3 columns (x, y, z)")
        
        # Find minimum Z coordinate
        min_z = np.min(reference_surface[:, 2])
        base_z = min_z - offset
        
        # Create base surface with same X,Y coordinates but flat Z
        base_surface = reference_surface.copy()
        base_surface[:, 2] = base_z
        
        return base_surface
    
    def validate_surface_overlap(self, surfaces: List[np.ndarray]) -> bool:
        """
        Check for substantial overlap between surfaces
        """
        if len(surfaces) == 0:
            raise ValueError("Surface list cannot be empty")
        
        if len(surfaces) == 1:
            return True  # Single surface always has overlap with itself
        
        # Calculate bounding boxes for all surfaces
        bounding_boxes = []
        for surface in surfaces:
            if len(surface) == 0:
                continue
            
            min_x, min_y = np.min(surface[:, :2], axis=0)
            max_x, max_y = np.max(surface[:, :2], axis=0)
            bounding_boxes.append((min_x, min_y, max_x, max_y))
        
        if len(bounding_boxes) < 2:
            return True
        
        # Check for overlap between all pairs of bounding boxes
        for i in range(len(bounding_boxes)):
            for j in range(i + 1, len(bounding_boxes)):
                box1 = bounding_boxes[i]
                box2 = bounding_boxes[j]
                
                # Check if boxes overlap
                overlap_x = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
                overlap_y = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                
                if overlap_x > 0 and overlap_y > 0:
                    # Calculate overlap area
                    overlap_area = overlap_x * overlap_y
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    
                    # Check if overlap is substantial (more than 10% of smaller area)
                    min_area = min(area1, area2)
                    if min_area > 0 and overlap_area / min_area > 0.1:
                        return True
        
        return False

    def simplify_mesh(self, vertices: np.ndarray, faces: Optional[np.ndarray], reduction: float = 0.5) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Simplify mesh using PyVista's decimate method.
        Args:
            vertices: Nx3 array of vertex positions
            faces: MxK array of face indices (K=3 for triangles)
            reduction: Fraction of reduction (0.0 = no reduction, 1.0 = full reduction)
        Returns:
            Tuple of (simplified_vertices, simplified_faces)
        """
        # Validate reduction factor
        if reduction < 0 or reduction > 1:
            raise ValueError("Reduction factor must be between 0.0 and 1.0")
        
        if faces is not None:
            # PyVista expects faces in a flat array: [n, v0, v1, v2, n, v0, v1, v2, ...]
            flat_faces = []
            for face in faces:
                flat_faces.append(len(face))
                flat_faces.extend(face)
            mesh = pv.PolyData(vertices, flat_faces)
            # Only decimate if all faces are triangles
            if mesh.n_cells > 0 and np.all([len(face) == 3 for face in faces]):
                try:
                    simplified = mesh.decimate(reduction)
                    simp_vertices = simplified.points
                    if simplified.n_cells > 0:
                        simp_faces = simplified.faces.reshape((-1, simplified.faces[0]+1))[:, 1:]
                    else:
                        simp_faces = np.empty((0, 3), dtype=np.int32)
                    return simp_vertices, simp_faces
                except Exception:
                    return mesh.points, faces
            else:
                return mesh.points, faces
        else:
            # No faces, cannot decimate
            return vertices, None

    def process_surfaces(self, surfaces_to_process, params):
        """
        Main processing pipeline for a list of surfaces.
        This orchestrates triangulation, thickness, and volume calculations.
        """
        
        volume_calculator = VolumeCalculator()

        # Step 1: Ensure all surfaces have a mesh (triangulate if needed)
        processed_surfaces = []
        for surface_data in surfaces_to_process:
            if 'faces' not in surface_data or surface_data['faces'] is None:
                vertices = surface_data['vertices']
                try:
                    # Use the triangulation function directly
                    delaunay_result = triangulation.create_delaunay_triangulation(vertices[:, :2])
                    faces = delaunay_result.simplices
                    processed_surfaces.append({**surface_data, 'faces': faces})
                except Exception as e:
                    print(f"Warning: Could not triangulate surface {surface_data.get('name', 'N/A')}. Error: {e}")
                    processed_surfaces.append({**surface_data, 'faces': np.empty((0, 3))})
            else:
                processed_surfaces.append(surface_data)

        # Step 2: Calculate thickness and volume between adjacent layers
        analysis_layers = []
        volume_results = []
        thickness_results = []
        compaction_results = []
        
        # Get tonnage data for compaction calculations
        tonnage_per_layer = params.get('tonnage_per_layer', [])
        tonnage_dict = {item['layer_index']: item['tonnage'] for item in tonnage_per_layer}
        
        for i in range(len(processed_surfaces) - 1):
            lower_surface = processed_surfaces[i]
            upper_surface = processed_surfaces[i+1]
            layer_name = f"{lower_surface.get('name', 'Layer ' + str(i))} to {upper_surface.get('name', 'Layer ' + str(i+1))}"

            # Calculate thickness
            thickness_data = thickness_calculator.calculate_thickness_between_surfaces(
                upper_surface['vertices'],
                lower_surface['vertices']
            )
            thickness_stats = thickness_calculator.calculate_thickness_statistics(thickness_data)
            
            # Calculate volume
            volume_result = volume_calculator.calculate_volume_difference(
                lower_surface['vertices'],
                upper_surface['vertices']
            )

            # Calculate compaction rate if tonnage is available
            compaction_rate = None
            tonnage_used = tonnage_dict.get(i, None)
            if tonnage_used and volume_result.volume_cubic_yards > 0:
                compaction_rate = tonnage_used / volume_result.volume_cubic_yards

            # Store results in the expected format
            volume_results.append({
                "layer_designation": layer_name,
                "volume_cubic_yards": volume_result.volume_cubic_yards,
                "confidence_interval": None,
                "uncertainty": None
            })
            
            thickness_results.append({
                "layer_designation": layer_name,
                "average_thickness_feet": thickness_stats.get('mean', 0),
                "min_thickness_feet": thickness_stats.get('min', 0),
                "max_thickness_feet": thickness_stats.get('max', 0),
                "std_dev_thickness_feet": thickness_stats.get('std_dev', 0),
                "confidence_interval": None
            })
            
            compaction_results.append({
                "layer_designation": layer_name,
                "compaction_rate_lbs_per_cubic_yard": compaction_rate,
                "tonnage_used": tonnage_used
            })

            analysis_layers.append({
                "layer_name": layer_name,
                "volume_cubic_yards": volume_result.volume_cubic_yards,
                "avg_thickness_feet": thickness_stats.get('mean'),
                "min_thickness_feet": thickness_stats.get('min'),
                "max_thickness_feet": thickness_stats.get('max'),
                "std_dev_thickness_feet": thickness_stats.get('std_dev')
            })

        # Convert numpy arrays to lists for JSON serialization
        for surface in processed_surfaces:
            if isinstance(surface.get('vertices'), np.ndarray):
                surface['vertices'] = surface['vertices'].tolist()
            if isinstance(surface.get('faces'), np.ndarray):
                surface['faces'] = surface['faces'].tolist()
        
        # Create the result structure and ensure it's fully serializable
        result = {
            "surfaces": processed_surfaces,
            "analysis_summary": analysis_layers,
            "volume_results": volume_results,
            "thickness_results": thickness_results,
            "compaction_results": compaction_results,
            "surface_tins": [surface.get('faces', []) for surface in processed_surfaces],
            "surface_names": [surface.get('name', f'Surface {i}') for i, surface in enumerate(processed_surfaces)],
            "georef": {
                "lat": 0.0,  # Default values - should be extracted from georeference_params
                "lon": 0.0,
                "orientation": 0.0,
                "scale": 1.0
            }
        }
        
        # Apply comprehensive serialization to ensure everything is JSON serializable
        return make_json_serializable(result) 