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
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial import Delaunay
import triangle

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
    
    def clip_mesh_to_boundary(self, vertices: np.ndarray, faces: np.ndarray, boundary: list) -> tuple:
        """
        Clip a surface mesh (vertices, faces) to a polygon boundary.
        - Removes faces outside the boundary.
        - Splits faces that intersect the boundary, creating new vertices at intersection points.
        - Returns new vertices and faces strictly within the boundary.
        """
        import itertools
        from shapely.geometry import Polygon, Point, LineString
        import numpy as np
        import logging
        logger = logging.getLogger(__name__)

        if len(boundary) < 3 or vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0:
            return vertices, faces

        logger.info(f"Mesh clipping debug - Before clipping:")
        logger.info(f"  Original vertices: {len(vertices)}")
        logger.info(f"  Original faces: {len(faces)}")
        logger.info(f"  Vertex X range: {np.min(vertices[:, 0]):.6f} to {np.max(vertices[:, 0]):.6f}")
        logger.info(f"  Vertex Y range: {np.min(vertices[:, 1]):.6f} to {np.max(vertices[:, 1]):.6f}")
        logger.info(f"  Boundary: {boundary}")

        # FIXED: Improve geometry validation and precision handling
        try:
            # Round boundary coordinates to reduce precision issues
            boundary_rounded = [(round(p[0], 8), round(p[1], 8)) for p in boundary]
            
            # Validate and clean boundary polygon
            polygon = Polygon(boundary_rounded)
            if not polygon.is_valid:
                logger.warning("Boundary polygon is invalid, attempting to fix...")
                # Try multiple approaches to fix the polygon
                try:
                    polygon = polygon.buffer(0)  # Fix self-intersections
                except:
                    try:
                        polygon = polygon.simplify(0)  # Simplify to remove issues
                    except:
                        # Last resort: create convex hull
                        polygon = polygon.convex_hull
                
                if not polygon.is_valid:
                    logger.error("Could not fix boundary polygon, using bounding box")
                    # Fallback to bounding box
                    min_x = min(p[0] for p in boundary_rounded)
                    max_x = max(p[0] for p in boundary_rounded)
                    min_y = min(p[1] for p in boundary_rounded)
                    max_y = max(p[1] for p in boundary_rounded)
                    # Add small buffer to ensure we don't lose valid points
                    buffer_size = max((max_x - min_x), (max_y - min_y)) * 0.01
                    polygon = Polygon([
                        (min_x - buffer_size, min_y - buffer_size),
                        (max_x + buffer_size, min_y - buffer_size),
                        (max_x + buffer_size, max_y + buffer_size),
                        (min_x - buffer_size, max_y + buffer_size)
                    ])
        except Exception as e:
            logger.error(f"Error creating boundary polygon: {e}")
            # Fallback to bounding box
            min_x = min(p[0] for p in boundary)
            max_x = max(p[0] for p in boundary)
            min_y = min(p[1] for p in boundary)
            max_y = max(p[1] for p in boundary)
            buffer_size = max((max_x - min_x), (max_y - min_y)) * 0.01
            polygon = Polygon([
                (min_x - buffer_size, min_y - buffer_size),
                (max_x + buffer_size, min_y - buffer_size),
                (max_x + buffer_size, max_y + buffer_size),
                (min_x - buffer_size, max_y + buffer_size)
            ])
        
        # Fix 2: Start with empty vertex list and add only needed vertices
        new_vertices = []
        new_faces = []
        vertex_map = {}  # (tuple) -> index

        def get_vertex_index(pt):
            # Round to 8 decimal places to avoid precision issues
            key = tuple(np.round(pt, 8))
            if key in vertex_map:
                return vertex_map[key]
            idx = len(new_vertices)
            new_vertices.append(list(pt))
            vertex_map[key] = idx
            return idx

        def interpolate_z_barycentric(x, y, triangle_vertices):
            """Fix 3: Use barycentric coordinates for proper Z interpolation"""
            # Convert to 2D for barycentric calculation
            pts_2d = np.array([(v[0], v[1]) for v in triangle_vertices])
            target = np.array([x, y])
            
            # Calculate barycentric coordinates
            v0 = pts_2d[1] - pts_2d[0]
            v1 = pts_2d[2] - pts_2d[0]
            v2 = target - pts_2d[0]
            
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            d20 = np.dot(v2, v0)
            d21 = np.dot(v2, v1)
            
            denom = d00 * d11 - d01 * d01
            if abs(denom) < 1e-10:
                # Degenerate triangle, use average
                return np.mean([v[2] for v in triangle_vertices])
            
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w
            
            # Clamp to triangle
            u = max(0, min(1, u))
            v = max(0, min(1, v))
            w = max(0, min(1, w))
            
            # Normalize
            total = u + v + w
            u, v, w = u/total, v/total, w/total
            
            # Interpolate Z
            z_interp = u * triangle_vertices[0][2] + v * triangle_vertices[1][2] + w * triangle_vertices[2][2]
            return z_interp

        faces_kept = 0
        faces_discarded = 0
        faces_split = 0

        for face in faces:
            pts = [vertices[idx] for idx in face]
            # Round coordinates to reduce precision issues
            pts_rounded = [(round(pt[0], 8), round(pt[1], 8)) for pt in pts]
            inside = [polygon.contains(Point(p)) for p in pts_rounded]
            if all(inside):
                # All inside, keep face as is - but add vertices to new list
                new_face_indices = []
                for pt in pts:
                    new_face_indices.append(get_vertex_index(pt))
                new_faces.append(new_face_indices)
                faces_kept += 1
            elif not any(inside):
                # All outside, discard
                faces_discarded += 1
                continue
            else:
                # Intersecting: clip face to polygon
                faces_split += 1
                try:
                    # FIXED: Improve triangle validation and intersection handling
                    tri = Polygon(pts_rounded)
                    if not tri.is_valid:
                        logger.warning(f"Invalid triangle detected, attempting to fix: {pts_rounded}")
                        try:
                            tri = tri.buffer(0)
                            if tri.is_empty:
                                logger.warning(f"Triangle became empty after buffering, skipping")
                                faces_discarded += 1
                                continue
                        except:
                            logger.warning(f"Could not fix triangle, skipping")
                            faces_discarded += 1
                            continue
                    
                    # Use a small buffer to avoid precision issues
                    tri = tri.buffer(0, join_style=2, mitre_limit=1.0)
                    if tri.is_empty:
                        faces_discarded += 1
                        continue
                    
                    # Perform intersection with error handling
                    try:
                        intersection = tri.intersection(polygon)
                    except Exception as e:
                        logger.warning(f"Intersection failed: {e}, trying with simplified geometries")
                        try:
                            # Simplify both geometries and try again
                            tri_simple = tri.simplify(1e-8)
                            poly_simple = polygon.simplify(1e-8)
                            intersection = tri_simple.intersection(poly_simple)
                        except Exception as e2:
                            logger.warning(f"Simplified intersection also failed: {e2}, skipping face")
                            faces_discarded += 1
                            continue
                    
                    if intersection.is_empty:
                        faces_discarded += 1
                        continue
                    
                    # Fix 4: Better triangulation for intersections
                    polys = [intersection] if intersection.geom_type == 'Polygon' else list(intersection.geoms)
                    for poly in polys:
                        if not poly.is_valid:
                            logger.warning(f"Invalid intersection polygon, attempting to fix")
                            try:
                                poly = poly.buffer(0)
                                if poly.is_empty:
                                    continue
                            except:
                                logger.warning(f"Could not fix intersection polygon, skipping")
                                continue
                            
                        coords = list(poly.exterior.coords)[:-1]  # Remove closing point
                        if len(coords) < 3:
                            continue
                        
                        # Interpolate z for intersection points using barycentric coordinates
                        indices = []
                        for (x, y) in coords:
                            # Check if this point is one of the original triangle vertices
                            is_original_vertex = False
                            for i, pt in enumerate(pts):
                                if abs(pt[0] - x) < 1e-8 and abs(pt[1] - y) < 1e-8:
                                    indices.append(get_vertex_index(pt))
                                    is_original_vertex = True
                                    break
                            
                            if not is_original_vertex:
                                # This is a new intersection point, interpolate Z
                                z_val = interpolate_z_barycentric(x, y, pts)
                                indices.append(get_vertex_index((x, y, z_val)))
                        
                        # Use proper triangulation for the intersection polygon
                        if len(indices) >= 3:
                            # Convert to 2D points for triangulation
                            points_2d = [(new_vertices[idx][0], new_vertices[idx][1]) for idx in indices]
                            try:
                                tri_2d = Delaunay(points_2d)
                                for simplex in tri_2d.simplices:
                                    new_faces.append([indices[i] for i in simplex])
                            except Exception as e:
                                logger.warning(f"Triangulation failed for intersection: {e}")
                                # Fallback to fan triangulation
                                for i in range(1, len(indices) - 1):
                                    new_faces.append([indices[0], indices[i], indices[i+1]])
                                    
                except Exception as e:
                    logger.warning(f"Error processing intersecting face: {e}")
                    faces_discarded += 1
                    continue

        result_vertices = np.array(new_vertices)
        result_faces = np.array(new_faces, dtype=int)

        # Fix 1: Vertex cleanup after face filtering
        if len(result_faces) > 0:
            # Find all vertices that are actually used by faces
            used_vertices = set()
            for face in result_faces:
                used_vertices.update(face)
            
            # Create new vertex array with only used vertices
            cleaned_vertices = []
            old_to_new_map = {}
            for i, vertex in enumerate(result_vertices):
                if i in used_vertices:
                    old_to_new_map[i] = len(cleaned_vertices)
                    cleaned_vertices.append(vertex)
            
            # Update face indices to match new vertex array
            cleaned_faces = []
            for face in result_faces:
                cleaned_faces.append([old_to_new_map[idx] for idx in face])
            
            result_vertices = np.array(cleaned_vertices)
            result_faces = np.array(cleaned_faces, dtype=int)

        logger.info(f"Mesh clipping debug - After clipping:")
        logger.info(f"  Result vertices: {len(result_vertices)}")
        logger.info(f"  Result faces: {len(result_faces)}")
        logger.info(f"  Faces kept: {faces_kept}, discarded: {faces_discarded}, split: {faces_split}")
        if len(result_vertices) > 0:
            logger.info(f"  Result vertex X range: {np.min(result_vertices[:, 0]):.6f} to {np.max(result_vertices[:, 0]):.6f}")
            logger.info(f"  Result vertex Y range: {np.min(result_vertices[:, 1]):.6f} to {np.max(result_vertices[:, 1]):.6f}")
        else:
            logger.warning("  No vertices remain after clipping!")

        # Verify all faces are within the boundary
        all_inside = True
        for face in result_faces:
            pts = [result_vertices[idx] for idx in face]
            pts2d = [(pt[0], pt[1]) for pt in pts]
            if not all(polygon.contains(Point(p)) or polygon.touches(Point(p)) for p in pts2d):
                all_inside = False
                break
        logger.info(f"  All faces strictly within/touching boundary: {all_inside}")

        return result_vertices, result_faces

    def clip_to_boundary(self, vertices: np.ndarray, boundary: list, faces: np.ndarray = None) -> tuple:
        """
        Clip surface vertices (and faces if provided) to the defined analysis boundary.
        If faces are provided, returns (vertices, faces). Otherwise, returns vertices only.
        """
        if vertices is None or vertices.size == 0 or vertices.shape[1] != 3:
            raise ValueError("Vertices must be a non-empty Nx3 numpy array")
        if faces is not None and len(faces) > 0:
            return self.clip_mesh_to_boundary(vertices, faces, boundary)
        if len(boundary) == 2:
            min_x, min_y = boundary[0]
            max_x, max_y = boundary[1]
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
        Clip vertices to a polygon boundary defined by 4 corner points using Shapely
        
        Args:
            vertices: Nx3 numpy array of vertex coordinates
            boundary: List of 4 tuples defining polygon corners: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
            
        Returns:
            Filtered vertices that are inside the polygon boundary
        """
        boundary_array = np.array(boundary)
        logger.info(f"Polygon clipping debug:")
        logger.info(f"  Boundary coordinates: {boundary}")
        logger.info(f"  Boundary array shape: {boundary_array.shape}")
        logger.info(f"  Vertex count before trimming: {len(vertices)}")
        logger.info(f"  First 10 vertices before trimming: {vertices[:10].tolist()}")
        logger.info(f"  Vertex X range: {np.min(vertices[:, 0]):.2f} to {np.max(vertices[:, 0]):.2f}")
        logger.info(f"  Vertex Y range: {np.min(vertices[:, 1]):.2f} to {np.max(vertices[:, 1]):.2f}")
        logger.info(f"  Boundary X range: {np.min(boundary_array[:, 0]):.2f} to {np.max(boundary_array[:, 0]):.2f}")
        logger.info(f"  Boundary Y range: {np.min(boundary_array[:, 1]):.2f} to {np.max(boundary_array[:, 1]):.2f}")
        
        try:
            polygon = Polygon(boundary)
            if not polygon.is_valid:
                logger.warning(f"Invalid polygon created from boundary: {boundary}")
                polygon = polygon.buffer(0)
                if not polygon.is_valid:
                    logger.error(f"Could not fix invalid polygon, skipping clipping")
                    return vertices
        except Exception as e:
            logger.error(f"Error creating polygon from boundary: {e}")
            return vertices
        
        logger.info(f"  Polygon area: {polygon.area:.2f} square units")
        logger.info(f"  Polygon is valid: {polygon.is_valid}")
        
        mask = np.zeros(len(vertices), dtype=bool)
        inside_count = 0
        for i, vertex in enumerate(vertices):
            point = Point(vertex[0], vertex[1])
            if polygon.contains(point):
                mask[i] = True
                inside_count += 1
        logger.info(f"  Vertices inside polygon: {inside_count}/{len(vertices)}")
        result = vertices[mask]
        logger.info(f"  Clipped result: {len(result)} vertices")
        if len(result) > 0:
            logger.info(f"  First 10 vertices after trimming: {result[:10].tolist()}")
        else:
            logger.info(f"  No vertices remain after trimming.")
        return result
    
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
                "std_dev_thickness_feet": thickness_stats.get('std', 0),
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

    def create_constrained_triangulation(self, vertices: np.ndarray, boundary_polygon) -> tuple:
        """
        Create a triangulated mesh constrained to the boundary polygon using Triangle library.
        Returns (vertices, faces) where all faces are inside the boundary.
        """
        import triangle
        import numpy as np
        from shapely.geometry import Point
        import logging
        logger = logging.getLogger(__name__)

        print("DEBUG: Entered create_constrained_triangulation")
        print(f"DEBUG: Input vertices shape: {vertices.shape if vertices is not None else 'None'}")
        print(f"DEBUG: Input vertices type: {type(vertices)}")
        print(f"DEBUG: Boundary polygon type: {type(boundary_polygon)}")
        print(f"DEBUG: Boundary polygon area: {boundary_polygon.area if hasattr(boundary_polygon, 'area') else 'N/A'}")
        
        if vertices is None or len(vertices) < 3:
            print("DEBUG: Not enough input vertices for triangulation")
            print(f"DEBUG: vertices is None: {vertices is None}")
            print(f"DEBUG: len(vertices): {len(vertices) if vertices is not None else 'N/A'}")
            return vertices, np.empty((0, 3), dtype=int)

        logger.info(f"Creating constrained triangulation:")
        logger.info(f"  Input vertices: {len(vertices)}")
        logger.info(f"  Boundary polygon area: {boundary_polygon.area:.6f}")

        # Step 1: Filter vertices to only those inside or near the boundary
        print("DEBUG: Step 1 - Filtering vertices to boundary area")
        buffered_polygon = boundary_polygon.buffer(0.001)  # 0.001 degree buffer
        print(f"DEBUG: Created buffered polygon with area: {buffered_polygon.area}")
        
        filtered_vertices = []
        filtered_indices = []
        vertices_inside = 0
        vertices_outside = 0
        
        print(f"DEBUG: Checking {len(vertices)} vertices against boundary")
        for i, vertex in enumerate(vertices):
            if i % 1000 == 0:  # Progress indicator for large datasets
                print(f"DEBUG: Processing vertex {i}/{len(vertices)}")
            
            point = Point(vertex[0], vertex[1])
            if buffered_polygon.contains(point):
                filtered_vertices.append(vertex)
                filtered_indices.append(i)
                vertices_inside += 1
            else:
                vertices_outside += 1
        
        print(f"DEBUG: Vertex filtering results:")
        print(f"  Total vertices: {len(vertices)}")
        print(f"  Vertices inside boundary: {vertices_inside}")
        print(f"  Vertices outside boundary: {vertices_outside}")
        print(f"  Filtered vertices: {len(filtered_vertices)}")
        
        if len(filtered_vertices) < 3:
            print("DEBUG: Not enough filtered vertices for triangulation")
            print(f"DEBUG: Only {len(filtered_vertices)} vertices inside boundary, need at least 3")
            logger.warning("Not enough vertices inside boundary for triangulation")
            return np.array(filtered_vertices), np.empty((0, 3), dtype=int)

        filtered_vertices = np.array(filtered_vertices)
        print(f"DEBUG: Converted filtered vertices to numpy array: {filtered_vertices.shape}")
        logger.info(f"  Filtered vertices: {len(filtered_vertices)}")

        # Step 2: Prepare Triangle input
        print("DEBUG: Step 2 - Preparing Triangle input")
        boundary_coords = list(boundary_polygon.exterior.coords)[:-1]  # Remove closing point
        print(f"DEBUG: Boundary coordinates: {len(boundary_coords)} points")
        print(f"DEBUG: First few boundary coords: {boundary_coords[:3]}")
        
        tri_input = {
            'vertices': filtered_vertices[:, :2].tolist(),
            'segments': [],
            'holes': [],
            'regions': []
        }
        print(f"DEBUG: Triangle input vertices: {len(tri_input['vertices'])}")
        
        num_boundary_points = len(boundary_coords)
        print(f"DEBUG: Creating segments for {num_boundary_points} boundary points")
        
        for i in range(num_boundary_points):
            boundary_point = np.array(boundary_coords[i])
            distances = np.linalg.norm(filtered_vertices[:, :2] - boundary_point, axis=1)
            closest_idx = np.argmin(distances)
            next_i = (i + 1) % num_boundary_points
            next_boundary_point = np.array(boundary_coords[next_i])
            next_distances = np.linalg.norm(filtered_vertices[:, :2] - next_boundary_point, axis=1)
            next_closest_idx = np.argmin(next_distances)
            tri_input['segments'].append([closest_idx, next_closest_idx])
        
        print(f"DEBUG: Created {len(tri_input['segments'])} boundary segments")
        print(f"DEBUG: Triangle input structure:")
        print(f"  vertices: {len(tri_input['vertices'])}")
        print(f"  segments: {len(tri_input['segments'])}")
        print(f"  holes: {len(tri_input['holes'])}")
        print(f"  regions: {len(tri_input['regions'])}")
        
        # Step 3: Perform triangulation
        print("DEBUG: Step 3 - Performing Triangle triangulation")
        try:
            print("DEBUG: Calling triangle.triangulate with 'p' flag")
            tri_output = triangle.triangulate(tri_input, 'p')
            print("DEBUG: Triangle triangulation successful")
            print(f"DEBUG: Triangle output keys: {list(tri_output.keys())}")
            print(f"DEBUG: Triangle output vertices: {len(tri_output['vertices'])}")
            print(f"DEBUG: Triangle output triangles: {len(tri_output['triangles'])}")
            
            logger.info(f"  Triangle triangulation successful")
            logger.info(f"  Output vertices: {len(tri_output['vertices'])}")
            logger.info(f"  Output triangles: {len(tri_output['triangles'])}")
            
        except Exception as e:
            print(f"DEBUG: Triangle triangulation failed: {e}")
            print(f"DEBUG: Exception type: {type(e)}")
            logger.error(f"Triangle triangulation failed: {e}")
            
            print("DEBUG: Falling back to Delaunay triangulation")
            from scipy.spatial import Delaunay
            tri = Delaunay(filtered_vertices[:, :2])
            result_vertices = filtered_vertices
            result_faces = tri.simplices
            print(f"DEBUG: Delaunay fallback - vertices: {len(result_vertices)}, faces: {len(result_faces)}")
            logger.info(f"  Fallback to Delaunay triangulation")
            return result_vertices, result_faces
        
        # Step 4: Process Triangle output
        print("DEBUG: Step 4 - Processing Triangle output")
        result_vertices = []
        vertex_map = {}
        
        print(f"DEBUG: Processing {len(tri_output['vertices'])} triangle output vertices")
        for i, (x, y) in enumerate(tri_output['vertices']):
            if i % 1000 == 0:  # Progress indicator
                print(f"DEBUG: Processing triangle vertex {i}/{len(tri_output['vertices'])}")
            
            point_2d = np.array([x, y])
            distances = np.linalg.norm(filtered_vertices[:, :2] - point_2d, axis=1)
            closest_idx = np.argmin(distances)
            z_val = filtered_vertices[closest_idx, 2]
            result_vertices.append([x, y, z_val])
            vertex_map[i] = len(result_vertices) - 1
        
        print(f"DEBUG: Processed {len(result_vertices)} result vertices")
        
        result_faces = []
        print(f"DEBUG: Processing {len(tri_output['triangles'])} triangle output faces")
        for i, triangle in enumerate(tri_output['triangles']):
            if i % 1000 == 0:  # Progress indicator
                print(f"DEBUG: Processing triangle face {i}/{len(tri_output['triangles'])}")
            
            face = [vertex_map[idx] for idx in triangle]
            result_faces.append(face)
        
        print(f"DEBUG: Processed {len(result_faces)} result faces")
        
        result_vertices = np.array(result_vertices)
        result_faces = np.array(result_faces, dtype=int)
        
        print(f"DEBUG: Final result arrays:")
        print(f"  result_vertices shape: {result_vertices.shape}")
        print(f"  result_faces shape: {result_faces.shape}")
        
        # Step 5: Validate results
        print("DEBUG: Step 5 - Validating results")
        faces_inside = 0
        faces_outside = 0
        from shapely.geometry import Point as ShapelyPoint
        
        print(f"DEBUG: Checking {len(result_faces)} faces against boundary")
        for i, face in enumerate(result_faces):
            if i % 1000 == 0:  # Progress indicator
                print(f"DEBUG: Checking face {i}/{len(result_faces)}")
            
            face_vertices = [result_vertices[idx] for idx in face]
            face_center = np.mean(face_vertices, axis=0)
            center_point = ShapelyPoint(face_center[0], face_center[1])
            if boundary_polygon.contains(center_point):
                faces_inside += 1
            else:
                faces_outside += 1
        
        print(f"DEBUG: Face validation results:")
        print(f"  Total faces: {len(result_faces)}")
        print(f"  Faces inside boundary: {faces_inside}")
        print(f"  Faces outside boundary: {faces_outside}")
        print(f"  Inside percentage: {(faces_inside / len(result_faces) * 100):.1f}%" if len(result_faces) > 0 else "N/A")
        
        logger.info(f"  Final result vertices: {len(result_vertices)}")
        logger.info(f"  Final result faces: {len(result_faces)}")
        logger.info(f"  Faces inside boundary: {faces_inside}")
        logger.info(f"  Faces outside boundary: {faces_outside}")
        
        print(f"DEBUG: create_constrained_triangulation completed successfully")
        return result_vertices, result_faces

    def densify_and_triangulate_surface(self, vertices: np.ndarray) -> tuple:
        """
        Legacy function - now uses constrained triangulation.
        This is kept for backward compatibility but should be replaced with create_constrained_triangulation.
        """
        # For backward compatibility, create a simple boundary and use constrained triangulation
        from shapely.geometry import Polygon
        
        # Create a simple bounding box as boundary
        min_x, min_y = np.min(vertices[:, :2], axis=0)
        max_x, max_y = np.max(vertices[:, :2], axis=0)
        boundary_coords = [
            (min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)
        ]
        boundary_polygon = Polygon(boundary_coords)
        
        return self.create_constrained_triangulation(vertices, boundary_polygon) 