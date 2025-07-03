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
import logging

logger = logging.getLogger(__name__)

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
        
        Args:
            vertices: Nx3 numpy array of vertex coordinates
            boundary: Either:
                - List of 2 tuples defining rectangle corners: [(min_x, min_y), (max_x, max_y)]
                - List of 4 tuples defining polygon corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        if vertices is None or vertices.size == 0 or vertices.shape[1] != 3:
            raise ValueError("Vertices must be a non-empty Nx3 numpy array")
        
        if len(boundary) == 2:
            # Two-corner format: rectangular boundary
            min_x, min_y = boundary[0]
            max_x, max_y = boundary[1]
            
            # Filter vertices within boundary
            mask = (
                (vertices[:, 0] >= min_x) & (vertices[:, 0] <= max_x) &
                (vertices[:, 1] >= min_y) & (vertices[:, 1] <= max_y)
            )
            
            if mask is not None and mask.size > 0:
                return vertices[mask]
            else:
                return np.empty((0, 3), dtype=vertices.dtype)
        
        elif len(boundary) == 4:
            filtered = self._clip_to_polygon_boundary(vertices, boundary)
            if filtered is not None and filtered.size > 0:
                return filtered
            else:
                return np.empty((0, 3), dtype=vertices.dtype)
        
        else:
            raise ValueError("Boundary must be a list of 2 or 4 coordinate tuples")
    
    def _clip_to_polygon_boundary(self, vertices: np.ndarray, boundary: List[Tuple[float, float]]) -> np.ndarray:
        """
        Clip vertices to a polygon boundary defined by 4 corner points
        
        Args:
            vertices: Nx3 numpy array of vertex coordinates
            boundary: List of 4 tuples defining polygon corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            
        Returns:
            Filtered vertices that are inside the polygon boundary
        """
        # Convert boundary to numpy array for easier manipulation
        boundary_array = np.array(boundary)
        
        # Filter vertices using point-in-polygon test
        mask = np.zeros(len(vertices), dtype=bool)
        for i, vertex in enumerate(vertices):
            if self._is_point_in_polygon(vertex[:2], boundary_array):
                mask[i] = True
        
        return vertices[mask]
    
    def _is_point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Robust point-in-polygon test (even-odd rule, with edge/vertex inclusion)
        Args:
            point: 2D point [x, y]
            polygon: Polygon vertices [N, 2]
        Returns:
            True if point is inside or on the edge/vertex of the polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        eps = 1e-10
        for i in range(n):
            x1, y1 = polygon[i]
            x2, y2 = polygon[(i + 1) % n]
            # Check if point is exactly on a vertex
            if (abs(x - x1) < eps and abs(y - y1) < eps) or (abs(x - x2) < eps and abs(y - y2) < eps):
                return True
            # Check if point is exactly on an edge (including horizontal/vertical/diagonal)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < eps and abs(x - x1) < eps and min(y1, y2) - eps <= y <= max(y1, y2) + eps:
                return True  # Vertical edge
            if abs(dy) < eps and abs(y - y1) < eps and min(x1, x2) - eps <= x <= max(x1, x2) + eps:
                return True  # Horizontal edge
            # General edge
            if min(y1, y2) - eps < y < max(y1, y2) + eps:
                if abs(dy) > eps:
                    t = (y - y1) / dy
                    if 0.0 - eps <= t <= 1.0 + eps:
                        x_proj = x1 + t * dx
                        if abs(x - x_proj) < eps and min(x1, x2) - eps <= x <= max(x1, x2) + eps:
                            return True  # On edge
            # Ray casting (even-odd rule)
            if ((y1 > y) != (y2 > y)):
                x_intersect = (x2 - x1) * (y - y1) / (y2 - y1 + eps) + x1
                if x < x_intersect:
                    inside = not inside
        return inside
    
    def convert_boundary_to_rectangle(self, boundary: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Convert a four-corner boundary to a rectangular boundary (min/max format)
        
        Args:
            boundary: List of 4 coordinate tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            
        Returns:
            List of 2 coordinate tuples [(min_x, min_y), (max_x, max_y)]
        """
        if len(boundary) != 4:
            raise ValueError("Boundary must have exactly 4 coordinate pairs")
        
        # Convert to numpy array for easier manipulation
        boundary_array = np.array(boundary)
        
        # Calculate bounding box
        min_x, min_y = np.min(boundary_array, axis=0)
        max_x, max_y = np.max(boundary_array, axis=0)
        
        return [(min_x, min_y), (max_x, max_y)]
    
    def generate_base_surface(self, reference_surface: np.ndarray, offset: float) -> np.ndarray:
        """
        Generate a flat base surface with specified vertical offset
        """
        if len(reference_surface) == 0 or reference_surface is None:
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
            if surface is None or surface.size == 0:
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
        
        if faces is not None and faces.size > 0:
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
            if 'faces' not in surface_data or surface_data['faces'] is None or (isinstance(surface_data['faces'], np.ndarray) and surface_data['faces'].size == 0):
                vertices = surface_data['vertices']
                if vertices is None or vertices.size == 0:
                    processed_surfaces.append({**surface_data, 'faces': np.empty((0, 3))})
                    continue
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
            thickness_data, invalid_points = thickness_calculator.calculate_thickness_between_surfaces(
                upper_surface['vertices'],
                lower_surface['vertices']
            )
            thickness_stats = thickness_calculator.calculate_thickness_statistics(
                thickness_data, 
                invalid_points=invalid_points
            )
            
            # Calculate volume
            volume_result = volume_calculator.calculate_volume_difference(
                lower_surface,
                upper_surface
            )

            # Calculate compaction rate if tonnage is available
            compaction_rate = None
            tonnage_used = tonnage_dict.get(i)
            if tonnage_used and volume_result.volume_cubic_yards > 0:
                # Assuming tonnage is in US tons, material density in lbs/cubic yard
                compaction_rate = (tonnage_used * 2000) / volume_result.volume_cubic_yards

            # Store results in the expected format
            volume_results.append({
                "layer_designation": layer_name,
                "volume_cubic_yards": volume_result.volume_cubic_yards,
            })
            
            thickness_results.append({
                "layer_designation": layer_name,
                "average_thickness_feet": thickness_stats.get('average') if thickness_stats.get('average') is not None else 0,
                "min_thickness_feet": thickness_stats.get('min', 0),
                "max_thickness_feet": thickness_stats.get('max', 0),
            })
            
            compaction_results.append({
                "layer_designation": layer_name,
                "compaction_rate_lbs_per_cubic_yard": compaction_rate,
                "tonnage_used": tonnage_used
            })

            analysis_layers.append({
                "layer_designation": layer_name,
                "volume_cubic_yards": volume_result.volume_cubic_yards,
                "avg_thickness_feet": thickness_stats.get('average'),
                "min_thickness_feet": thickness_stats.get('min'),
                "max_thickness_feet": thickness_stats.get('max'),
            })

        # --- DETAILED LOGGING OF RESULTS ---
        logger.info("--- Analysis Results Summary ---")
        analysis_summary = []
        for i, layer in enumerate(analysis_layers):
            logger.info(f"  Layer {i}: {layer['layer_designation']}")
            logger.info(f"    - Volume: {layer['volume_cubic_yards']:.2f} cubic yards")
            
            avg_thickness = layer.get('avg_thickness_feet')
            if avg_thickness is not None:
                logger.info(f"    - Avg Thickness: {avg_thickness:.2f} feet")
            else:
                logger.info("    - Avg Thickness: N/A")
                
            logger.info(f"    - Min Thickness: {layer.get('min_thickness_feet', 0):.2f} feet")
            logger.info(f"    - Max Thickness: {layer.get('max_thickness_feet', 0):.2f} feet")
            analysis_summary.append({
                'layer_designation': layer['layer_designation'],
                'volume_cubic_yards': layer['volume_cubic_yards'],
                'avg_thickness_feet': avg_thickness,
                'min_thickness_feet': layer.get('min_thickness_feet'),
                'max_thickness_feet': layer.get('max_thickness_feet')
            })
        logger.info("---------------------------------")

        # Extract georeference parameters from the first entry in georeference_params if available
        georef_params_list = params.get('georeference_params', [])
        if georef_params_list and isinstance(georef_params_list, list) and len(georef_params_list) > 0:
            first_georef = georef_params_list[0]
            georef = {
                "lat": first_georef.get('wgs84_lat', 0.0),
                "lon": first_georef.get('wgs84_lon', 0.0),
                "orientation": first_georef.get('orientation_degrees', 0.0),
                "scale": first_georef.get('scaling_factor', 1.0)
            }
        else:
            georef = {"lat": 0.0, "lon": 0.0, "orientation": 0.0, "scale": 1.0}

        # Convert numpy arrays to lists for JSON serialization
        for surface in processed_surfaces:
            if isinstance(surface.get('vertices'), np.ndarray):
                surface['vertices'] = surface['vertices'].tolist()
            if isinstance(surface.get('faces'), np.ndarray):
                surface['faces'] = surface['faces'].tolist()
        
        # Create the result structure and ensure it's fully serializable
        result = {
            "surfaces": processed_surfaces,
            "analysis_summary": analysis_summary,
            "volume_results": volume_results,
            "thickness_results": thickness_results,
            "compaction_results": compaction_results,
            "surface_tins": [{s.get('name'): s.get('faces')} for s in processed_surfaces],
            "surface_names": [surface.get('name', f'Surface {i}') for i, surface in enumerate(processed_surfaces)],
            "georef": georef
        }
        
        # Apply comprehensive serialization to ensure everything is JSON serializable
        return make_json_serializable(result) 