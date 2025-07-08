import time
import uuid
import logging
from typing import Dict, Any, Optional
from fastapi import HTTPException
from app.utils.serialization import make_json_serializable, validate_json_serializable, clean_floats_for_json
from app.services.surface_processor import SurfaceProcessor
from app.services.surface_cache import surface_cache
from app.utils.ply_parser import PLYParser
from app.utils.shp_parser import SHPParser
from app.services.coord_transformer import CoordinateTransformer, TransformationPipeline
import os
import numpy as np
from shapely.geometry import Polygon
from pyproj import Transformer
from scipy.spatial import Delaunay

# Performance optimization: Set environment variables for better performance
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')

# NumPy performance optimizations
np.set_printoptions(precision=6, suppress=True)
np.random.seed(42)  # For reproducible results

logger = logging.getLogger(__name__)

class AnalysisExecutor:
    """
    Manages analysis job execution with FastAPI background tasks.
    """
    def __init__(self):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._results_cache: Dict[str, Dict[str, Any]] = {}
        self.MAX_CONCURRENT_JOBS = 10
        self.surface_processor = SurfaceProcessor()
        self.coord_transformer = CoordinateTransformer()
        self.ply_parser = PLYParser()
        self.shp_parser = SHPParser()
        
        # Performance optimization: Pre-allocate memory pools
        self._chunk_size = 10000  # Process data in chunks for better memory management

    def run_analysis_sync(self, analysis_id: str, params: Optional[Dict[str, Any]] = None):
        """Synchronous analysis execution for background tasks"""
        logger.info(f"[{analysis_id}] Background task analysis started")
        
        try:
            # Update job status to running
            self._update_job_status(analysis_id, "running", 10.0, "loading_surfaces")
            
            # This will now set the result cache and determine the final status
            self._execute_analysis_logic(analysis_id, params)
            
            # Retrieve the final status from the result cache
            final_status = self._results_cache.get(analysis_id, {}).get("analysis_metadata", {}).get("status", "failed")
            
            # Update the main job status to completed or failed
            if final_status == "completed":
                self._update_job_status(analysis_id, "completed", 100.0, "finished")
                logger.info(f"[{analysis_id}] Background task analysis completed successfully")
            else:
                # If it's still processing, we let the polling continue. If it failed inside, it will be marked.
                logger.info(f"[{analysis_id}] Analysis logic finished, but results not ready. Final status: {final_status}")

            # Return the results from the cache
            return self._results_cache.get(analysis_id, {})

        except Exception as e:
            logger.error(f"[{analysis_id}] Background task analysis failed: {e}", exc_info=True)
            self._update_job_status(analysis_id, "failed", 0.0, "error", str(e))
            return {"analysis_metadata": {"status": "failed", "error": str(e)}}

    def _update_job_status(self, analysis_id: str, status: str, progress: float, step: str, error_msg: str = None):
        """Thread-safe job status update"""
        if analysis_id in self._jobs:
            self._jobs[analysis_id].update({
                "status": status,
                "progress_percent": progress,
                "current_step": step
            })
            if status in ["completed", "failed", "cancelled"]:
                self._jobs[analysis_id]["completion_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            if error_msg:
                self._jobs[analysis_id]["error_message"] = error_msg

    def _execute_analysis_logic(self, analysis_id: str, params: Optional[Dict[str, Any]] = None):
        """Extract analysis logic from current _run_analysis method"""
        # Copy the entire analysis logic from _run_analysis method
        # but remove all threading-specific code and locks
        # This includes surface loading, processing, and result caching
        
        print(f"DEBUG: Starting _execute_analysis_logic for analysis_id: {analysis_id}")
        
        # Handle None params
        if params is None:
            params = {}
        
        surface_ids = params.get('surface_ids', [])
        print(f"DEBUG: Surface IDs to process: {surface_ids}")
        logger.info(f"[{analysis_id}] Loading {len(surface_ids)} surfaces from cache")
        
        # Load surface data from cache with progress updates
        surfaces_to_process = []
        
        for i, sid in enumerate(surface_ids):
            print(f"DEBUG: Processing surface {i+1}/{len(surface_ids)}: {sid}")
            logger.info(f"[{analysis_id}] Loading surface {i+1}/{len(surface_ids)}: {sid}")
            
            cached_surface = surface_cache.get(sid)
            if not cached_surface or 'file_path' not in cached_surface:
                print(f"DEBUG: Surface {sid} not found in cache or is invalid")
                raise RuntimeError(f"Surface {sid} not found in cache or is invalid.")
            
            file_path = cached_surface['file_path']
            file_type = cached_surface.get('file_type', 'PLY')  # Default to PLY for backward compatibility
            print(f"DEBUG: Parsing {file_type} file: {file_path}")
            logger.info(f"[{analysis_id}] Parsing {file_type} file: {file_path}")
            
            # Parse file based on type
            if file_type.upper() == 'SHP':
                print(f"DEBUG: Processing SHP file")
                vertices, faces = self.shp_parser.process_shp_file(file_path)
                print(f"DEBUG: SHP file loaded - vertices: {len(vertices)}, faces: {len(faces) if faces is not None else 0}")
                logger.info(f"[{analysis_id}] Loaded {len(vertices)} vertices from SHP file (in WGS84)")
            else:
                # Default to PLY parsing
                print(f"DEBUG: Processing PLY file")
                vertices, faces = self.ply_parser.parse_ply_file(file_path)
                print(f"DEBUG: PLY file loaded - vertices: {len(vertices)}, faces: {len(faces) if faces is not None else 0}")
                logger.info(f"[{analysis_id}] Loaded {len(vertices)} vertices, {len(faces) if faces is not None else 0} faces from PLY file")

            # --- Apply Transformation Pipeline ---
            if file_type.upper() == 'SHP':
                # SHP files are in WGS84 coordinates, no transformation needed yet
                # They will be converted to UTM after clipping
                transformed_vertices = vertices
                print(f"DEBUG: SHP file - no transformation applied, vertices: {len(transformed_vertices)}")
                logger.info(f"[{analysis_id}] SHP file in WGS84 coordinates, will convert to UTM after clipping")
            else:
                # Apply transformation for PLY files
                print(f"DEBUG: Applying transformation to PLY file")
                georef_params_list = params.get('georeference_params', [])
                if georef_params_list and i < len(georef_params_list):
                    georef = georef_params_list[i]
                    anchor_lat = georef.get('wgs84_lat', 0.0)
                    anchor_lon = georef.get('wgs84_lon', 0.0)
                    rotation = georef.get('orientation_degrees', 0.0)
                    scale = georef.get('scaling_factor', 1.0)
                else:
                    anchor_lat = 0.0
                    anchor_lon = 0.0
                    rotation = 0.0
                    scale = 1.0
                print(f"DEBUG: Transformation params - lat: {anchor_lat}, lon: {anchor_lon}, rotation: {rotation}, scale: {scale}")
                pipeline = TransformationPipeline(
                    anchor_lat=anchor_lat,
                    anchor_lon=anchor_lon,
                    rotation_degrees=rotation,
                    scale_factor=scale,
                    center_surface=True
                )
                transformed_vertices = pipeline.transform_to_utm(vertices)
                print(f"DEBUG: PLY transformation completed - vertices: {len(transformed_vertices)}")
                # Log the final center in UTM and WGS84
                center_utm = np.mean(transformed_vertices, axis=0)
                center_lon, center_lat = pipeline.utm_to_wgs84_transformer.transform(center_utm[0], center_utm[1])
                logger.info(f"[{analysis_id}] Surface {i+1} center after transform: UTM=({center_utm[0]:.3f}, {center_utm[1]:.3f}), WGS84=({center_lat:.6f}, {center_lon:.6f})")
            
            surfaces_to_process.append({
                "id": sid,
                "name": cached_surface.get("filename", "Unknown"),
                "vertices": transformed_vertices,
                "faces": faces,
                "file_type": file_type
            })
            print(f"DEBUG: Surface {i+1} added to processing list - vertices: {len(transformed_vertices)}, faces: {len(faces) if faces is not None else 0}")
            
            # Update progress
            progress = 20.0 + (i / len(surface_ids)) * 30.0
            self._update_job_status(analysis_id, "running", progress, f"loaded_surface_{i+1}")
        
        print(f"DEBUG: All surfaces loaded, total surfaces to process: {len(surfaces_to_process)}")
        
        # Step 2: Apply boundary clipping if boundary is provided
        analysis_boundary = params.get('analysis_boundary', {})
        print(f"DEBUG: Analysis boundary: {analysis_boundary}")
        if analysis_boundary and 'wgs84_coordinates' in analysis_boundary:
            print(f"DEBUG: Boundary clipping will be applied")
            logger.info(f"[{analysis_id}] Processing boundary clipping in WGS84")
            self._update_job_status(analysis_id, "running", 40.0, "processing_boundary")
            
            # Extract WGS84 boundary coordinates
            wgs84_boundary = analysis_boundary['wgs84_coordinates']
            print(f"DEBUG: WGS84 boundary coordinates: {wgs84_boundary}")
            logger.info(f"[{analysis_id}] WGS84 boundary: {wgs84_boundary}")
            
            # Check if boundary area is too small (less than 0.000003 degrees in any dimension)
            wgs84_boundary_array = np.array(wgs84_boundary)
            # Boundary coordinates are in [lat, lon] format, so:
            # wgs84_boundary_array[:, 0] = latitudes
            # wgs84_boundary_array[:, 1] = longitudes
            lat_range = np.max(wgs84_boundary_array[:, 0]) - np.min(wgs84_boundary_array[:, 0])
            lon_range = np.max(wgs84_boundary_array[:, 1]) - np.min(wgs84_boundary_array[:, 1])
            
            print(f"DEBUG: Boundary area check:")
            print(f"  Boundary coordinates: {wgs84_boundary}")
            print(f"  Latitude range: {lat_range:.6f} degrees")
            print(f"  Longitude range: {lon_range:.6f} degrees")
            print(f"  Threshold: 0.000003 degrees")
            print(f"  Lat range < threshold: {lat_range < 0.000003}")
            print(f"  Lon range < threshold: {lon_range < 0.000003}")
            
            logger.info(f"[{analysis_id}] Boundary area check:")
            logger.info(f"  Boundary coordinates: {wgs84_boundary}")
            logger.info(f"  Latitude range: {lat_range:.6f} degrees")
            logger.info(f"  Longitude range: {lon_range:.6f} degrees")
            logger.info(f"  Threshold: 0.000003 degrees")
            logger.info(f"  Lat range < threshold: {lat_range < 0.000003}")
            logger.info(f"  Lon range < threshold: {lon_range < 0.000003}")
            
            if lat_range < 0.000003 or lon_range < 0.000003:
                print(f"DEBUG: Boundary clipping SKIPPED - area too small")
                logger.warning(f"[{analysis_id}] Boundary area too small ({lat_range:.6f}° lat x {lon_range:.6f}° lon), skipping clipping to preserve data")
                logger.info(f"[{analysis_id}] Skipping boundary clipping - area too small")
            else:
                print(f"DEBUG: Boundary clipping WILL PROCEED - area is large enough")
                print(f"DEBUG: Boundary coordinates: {wgs84_boundary}")
                print(f"DEBUG: Lat range: {lat_range:.6f}, Lon range: {lon_range:.6f}")
                # Apply boundary clipping to all surfaces in WGS84 coordinates
                for i, surface in enumerate(surfaces_to_process):
                    print(f"DEBUG: Starting boundary clipping for surface {i+1}")
                    original_vertex_count = len(surface['vertices'])
                    print(f"DEBUG: Surface {i+1} original vertices: {original_vertex_count}")
                    logger.info(f"[{analysis_id}] Clipping surface {i+1} from {original_vertex_count} vertices in WGS84")
                    
                    # Check if this is a SHP file (already in WGS84) or PLY file (needs conversion)
                    file_type = surface.get('file_type', 'PLY').upper()
                    print(f"DEBUG: Surface {i+1} file type: {file_type}")
                    
                    if file_type == 'SHP':
                        # SHP files are already in WGS84 coordinates, clip directly
                        print(f"DEBUG: Surface {i+1} is SHP file, clipping vertices to boundary first")
                        logger.info(f"[{analysis_id}] Surface {i+1} is SHP file, clipping vertices to boundary first")
                        
                        # Convert boundary from [lat, lon] to [lon, lat] format to match vertex format
                        # Ensure we have exactly 4 coordinates for polygon clipping
                        boundary_lon_lat = [[coord[1], coord[0]] for coord in wgs84_boundary[:4]]  # Take first 4 points
                        print(f"DEBUG: Converted boundary from [lat, lon] to [lon, lat]: {boundary_lon_lat}")
                        logger.info(f"[{analysis_id}] Converted boundary from [lat, lon] to [lon, lat]: {boundary_lon_lat}")
                        
                        # First, clip vertices to boundary to reduce count before UTM conversion
                        print(f"DEBUG: Clipping {len(surface['vertices'])} vertices to boundary")
                        # FIXED: Pass faces to preserve triangulation during clipping
                        if surface.get('faces') is not None and len(surface['faces']) > 0:
                            clipped_wgs84_vertices, clipped_wgs84_faces = self.surface_processor.clip_to_boundary(
                                surface['vertices'], boundary_lon_lat, surface['faces']
                            )
                            print(f"DEBUG: Clipping with faces - vertices: {len(surface['vertices'])} -> {len(clipped_wgs84_vertices)}")
                            print(f"DEBUG: Clipping with faces - faces: {len(surface['faces'])} -> {len(clipped_wgs84_faces)}")
                        else:
                            clipped_wgs84_vertices = self.surface_processor.clip_to_boundary(surface['vertices'], boundary_lon_lat)
                            clipped_wgs84_faces = None
                            print(f"DEBUG: Clipping without faces - vertices: {len(surface['vertices'])} -> {len(clipped_wgs84_vertices)}")
                        
                        clipped_vertex_count = len(clipped_wgs84_vertices)
                        print(f"DEBUG: Vertex clipping result: {len(surface['vertices'])} -> {clipped_vertex_count} vertices")
                        logger.info(f"[{analysis_id}] Surface {i+1} vertex clipping: {len(surface['vertices'])} -> {clipped_vertex_count} vertices")
                        
                        # Convert clipped vertices to UTM for downstream processing
                        if clipped_vertex_count > 0:
                            print(f"DEBUG: Converting {clipped_vertex_count} clipped vertices from WGS84 to UTM")
                            clipped_utm_vertices = self._convert_wgs84_to_utm(clipped_wgs84_vertices)
                            surface['vertices'] = clipped_utm_vertices
                            
                            # FIXED: Preserve original triangulation indices instead of creating new triangulation
                            # The original faces from WGS84 triangulation should be preserved
                            # since they reference the same vertex indices (just with different coordinates)
                            if clipped_wgs84_faces is not None and len(clipped_wgs84_faces) > 0:
                                print(f"DEBUG: Preserving clipped triangulation indices: {len(clipped_wgs84_faces)} faces")
                                logger.info(f"[{analysis_id}] Surface {i+1} preserving clipped triangulation: {len(clipped_wgs84_faces)} faces")
                                # Use the clipped faces - they reference the same vertex indices
                                surface['faces'] = clipped_wgs84_faces
                            elif surface.get('faces') is not None and len(surface['faces']) > 0:
                                print(f"DEBUG: Preserving original triangulation indices: {len(surface['faces'])} faces")
                                logger.info(f"[{analysis_id}] Surface {i+1} preserving original triangulation: {len(surface['faces'])} faces")
                                # Faces are already correct - they reference the same vertex indices
                                # No need to retriangulate since we're just transforming coordinates
                            else:
                                print(f"DEBUG: No original faces available, creating new triangulation in UTM")
                                try:
                                    # Only create new triangulation if no original faces exist
                                    tri = Delaunay(clipped_utm_vertices[:, :2])
                                    surface['faces'] = tri.simplices
                                    print(f"DEBUG: New triangulation completed - {len(surface['faces'])} faces")
                                    logger.info(f"[{analysis_id}] Surface {i+1} new triangulation in UTM: {len(surface['faces'])} faces")
                                except Exception as e:
                                    print(f"DEBUG: Triangulation failed: {e}")
                                    logger.warning(f"[{analysis_id}] Surface {i+1} triangulation failed: {e}")
                                    surface['faces'] = None
                            print(f"DEBUG: Conversion to UTM completed")
                            logger.info(f"[{analysis_id}] Surface {i+1} converted from WGS84 to UTM after clipping")
                        else:
                            print(f"DEBUG: No vertices after clipping, setting empty array")
                            surface['vertices'] = np.empty((0, 3), dtype=surface['vertices'].dtype)
                            surface['faces'] = np.empty((0, 3), dtype=int)
                            logger.warning(f"[{analysis_id}] Surface {i+1} has no vertices after clipping")
                    
                    else:
                        # PLY files are in UTM coordinates, need to convert to WGS84 for clipping
                        print(f"DEBUG: Surface {i+1} is PLY file, converting to WGS84 for clipping")
                        logger.info(f"[{analysis_id}] Surface {i+1} is PLY file, converting to WGS84 for clipping")
                        
                        # Convert surface vertices from UTM to WGS84 for clipping
                        wgs84_vertices = self._convert_utm_to_wgs84(surface['vertices'])
                        print(f"DEBUG: Converted PLY vertices to WGS84: {len(wgs84_vertices)}")
                        
                        # Clip vertices to boundary in WGS84
                        print(f"DEBUG: Calling clip_to_boundary for PLY surface")
                        # FIXED: Pass faces to preserve triangulation during clipping
                        if surface.get('faces') is not None and len(surface['faces']) > 0:
                            clipped_wgs84_vertices, clipped_wgs84_faces = self.surface_processor.clip_to_boundary(
                                wgs84_vertices, wgs84_boundary, surface['faces']
                            )
                            print(f"DEBUG: Clipping with faces - vertices: {len(wgs84_vertices)} -> {len(clipped_wgs84_vertices)}")
                            print(f"DEBUG: Clipping with faces - faces: {len(surface['faces'])} -> {len(clipped_wgs84_faces)}")
                        else:
                            clipped_wgs84_vertices = self.surface_processor.clip_to_boundary(wgs84_vertices, wgs84_boundary)
                            clipped_wgs84_faces = None
                            print(f"DEBUG: Clipping without faces - vertices: {len(wgs84_vertices)} -> {len(clipped_wgs84_vertices)}")
                        
                        clipped_vertex_count = len(clipped_wgs84_vertices)
                        print(f"DEBUG: clip_to_boundary returned {clipped_vertex_count} vertices")
                        
                        # Convert clipped vertices back to UTM for downstream processing
                        if clipped_vertex_count > 0:
                            print(f"DEBUG: Converting clipped vertices back to UTM")
                            clipped_utm_vertices = self._convert_wgs84_to_utm(clipped_wgs84_vertices)
                            surface['vertices'] = clipped_utm_vertices
                            
                            # FIXED: Preserve original triangulation indices instead of creating new triangulation
                            # The original faces from WGS84 triangulation should be preserved
                            # since they reference the same vertex indices (just with different coordinates)
                            if clipped_wgs84_faces is not None and len(clipped_wgs84_faces) > 0:
                                print(f"DEBUG: Preserving clipped triangulation indices: {len(clipped_wgs84_faces)} faces")
                                logger.info(f"[{analysis_id}] Surface {i+1} preserving clipped triangulation: {len(clipped_wgs84_faces)} faces")
                                # Use the clipped faces - they reference the same vertex indices
                                surface['faces'] = clipped_wgs84_faces
                            elif surface.get('faces') is not None and len(surface['faces']) > 0:
                                print(f"DEBUG: Preserving original triangulation indices: {len(surface['faces'])} faces")
                                logger.info(f"[{analysis_id}] Surface {i+1} preserving original triangulation: {len(surface['faces'])} faces")
                                # Faces are already correct - they reference the same vertex indices
                                # No need to retriangulate since we're just transforming coordinates
                            else:
                                print(f"DEBUG: No original faces available, creating new triangulation in UTM")
                                try:
                                    # Only create new triangulation if no original faces exist
                                    tri = Delaunay(clipped_utm_vertices[:, :2])
                                    surface['faces'] = tri.simplices
                                    print(f"DEBUG: New triangulation completed - {len(surface['faces'])} faces")
                                    logger.info(f"[{analysis_id}] Surface {i+1} new triangulation in UTM: {len(surface['faces'])} faces")
                                except Exception as e:
                                    print(f"DEBUG: Triangulation failed: {e}")
                                    logger.warning(f"[{analysis_id}] Surface {i+1} triangulation failed: {e}")
                                    surface['faces'] = None
                            print(f"DEBUG: Conversion to UTM completed")
                            logger.info(f"[{analysis_id}] Surface {i+1} converted from WGS84 to UTM after clipping")
                        else:
                            print(f"DEBUG: No vertices after clipping, setting empty array")
                            surface['vertices'] = np.empty((0, 3), dtype=surface['vertices'].dtype)
                            surface['faces'] = np.empty((0, 3), dtype=int)
                            logger.warning(f"[{analysis_id}] Surface {i+1} has no vertices after clipping")
                    
                    # Update faces if they exist (this is a simplified approach - in production you might want to re-triangulate)
                    faces = surface.get('faces')
                    if faces is not None and ((hasattr(faces, 'size') and faces.size > 0) or (isinstance(faces, list) and len(faces) > 0)):
                        print(f"DEBUG: Surface {i+1} has faces that may need re-triangulation after clipping")
                        # For now, we'll keep the faces but note that they may reference non-existent vertices
                        # In a production system, you'd want to re-triangulate the clipped surface
                        logger.warning(f"[{analysis_id}] Surface {i+1} has faces that may need re-triangulation after clipping")
                    
                    print(f"DEBUG: Surface {i+1} clipping completed: {original_vertex_count} -> {clipped_vertex_count} vertices")
                    logger.info(f"[{analysis_id}] Surface {i+1} clipped: {original_vertex_count} -> {clipped_vertex_count} vertices")
                
                print(f"DEBUG: Boundary clipping completed for all surfaces")
                logger.info(f"[{analysis_id}] Boundary clipping completed for all surfaces")
        else:
            print(f"DEBUG: No boundary provided, skipping boundary clipping")
        
        # Check if we need to generate a baseline surface
        generate_base_surface = params.get('generate_base_surface', False)
        print(f"DEBUG: Generate base surface flag: {generate_base_surface}")
        if generate_base_surface and len(surfaces_to_process) == 1:
            print(f"DEBUG: Generating baseline surface")
            logger.info(f"[{analysis_id}] Generating baseline surface 1 foot below minimum elevation")
            self._update_job_status(analysis_id, "running", 45.0, "generating_baseline")
            
            # Get the first (and only) surface
            first_surface = surfaces_to_process[0]
            vertices = first_surface['vertices']
            print(f"DEBUG: Baseline generation - using {len(vertices)} vertices from first surface")
            
            # Check if the first surface is in UTM coordinates (large values) or WGS84 (small values)
            x_range = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
            y_range = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
            
            print(f"DEBUG: First surface coordinate ranges - X: {x_range:.2f}, Y: {y_range:.2f}")
            
            # If coordinates are large (UTM), generate baseline in UTM
            # If coordinates are small (WGS84), generate baseline in WGS84
            is_utm = x_range > 1000 or y_range > 1000  # UTM coordinates are typically in thousands of meters
            
            if is_utm:
                print(f"DEBUG: First surface is in UTM coordinates, generating baseline in UTM")
                # Use the UTM extent of the first surface for baseline generation
                min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
                min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
                print(f"DEBUG: Using UTM extent for baseline: X({min_x:.2f} to {max_x:.2f}), Y({min_y:.2f} to {max_y:.2f})")
                logger.info(f"[{analysis_id}] Using UTM extent for baseline: X({min_x:.2f} to {max_x:.2f}), Y({min_y:.2f} to {max_y:.2f})")
            else:
                print(f"DEBUG: First surface is in WGS84 coordinates, generating baseline in WGS84")
                # Use boundary coordinates for baseline generation if available
                if analysis_boundary and 'wgs84_coordinates' in analysis_boundary:
                    # Convert boundary from [lat, lon] to [lon, lat] format
                    boundary_lon_lat = [[coord[1], coord[0]] for coord in analysis_boundary['wgs84_coordinates']]
                    boundary_array = np.array(boundary_lon_lat)
                    
                    # Use boundary extent for baseline generation
                    min_x, max_x = np.min(boundary_array[:, 0]), np.max(boundary_array[:, 0])  # longitude
                    min_y, max_y = np.min(boundary_array[:, 1]), np.max(boundary_array[:, 1])  # latitude
                    print(f"DEBUG: Using boundary extent for baseline: lon({min_x:.6f} to {max_x:.6f}), lat({min_y:.6f} to {max_y:.6f})")
                    logger.info(f"[{analysis_id}] Using boundary extent for baseline: lon({min_x:.6f} to {max_x:.6f}), lat({min_y:.6f} to {max_y:.6f})")
                else:
                    # Fallback to clipped surface extent
                    min_x, max_x = np.min(vertices[:, 0]), np.max(vertices[:, 0])
                    min_y, max_y = np.min(vertices[:, 1]), np.max(vertices[:, 1])
                    print(f"DEBUG: Using clipped surface extent for baseline: lon({min_x:.6f} to {max_x:.6f}), lat({min_y:.6f} to {max_y:.6f})")
                    logger.info(f"[{analysis_id}] Using clipped surface extent for baseline: lon({min_x:.6f} to {max_x:.6f}), lat({min_y:.6f} to {max_y:.6f})")
            
            min_z = np.min(vertices[:, 2])
            print(f"DEBUG: Minimum Z value: {min_z}")
            
            # Create a new grid with a similar number of points
            num_vertices = len(vertices)
            # Calculate grid dimensions to approximate original density
            aspect_ratio = (max_y - min_y) / (max_x - min_x) if (max_x - min_x) != 0 else 1
            num_x = int(np.sqrt(num_vertices / aspect_ratio))
            num_y = int(num_x * aspect_ratio)

            # Generate grid points
            x_coords = np.linspace(min_x, max_x, num_x)
            y_coords = np.linspace(min_y, max_y, num_y)
            grid_x, grid_y = np.meshgrid(x_coords, y_coords)
            
            # Set the baseline elevation
            if is_utm:
                # Convert 1 foot to meters for UTM coordinates
                baseline_offset = 0.3048  # 1 foot in meters
            else:
                # Keep in same units for WGS84 (assuming feet)
                baseline_offset = 1.0  # 1 foot
            
            baseline_z = min_z - baseline_offset
            print(f"DEBUG: Baseline Z value: {baseline_z}")

            # Create the new baseline vertices
            baseline_vertices = np.vstack([grid_x.ravel(), grid_y.ravel(), np.full(grid_x.size, baseline_z)]).T
            print(f"DEBUG: Generated baseline vertices: {len(baseline_vertices)}")
            
            # For a grid, we can't reuse the original faces. A simple point cloud is best here.
            baseline_surface = {
                "id": f"baseline_{first_surface['id']}",
                "name": "Baseline Surface (1ft below minimum)",
                "vertices": baseline_vertices,
                "faces": []  # Use an empty list instead of None for downstream compatibility
            }
            
            # Add baseline surface to the beginning of the list
            surfaces_to_process.insert(0, baseline_surface)
            print(f"DEBUG: Baseline surface added to processing list")
            logger.info(f"[{analysis_id}] Baseline surface generated with {len(baseline_vertices)} vertices")
        
        processing_params = params.get('params', {})
        print(f"DEBUG: Processing parameters: {processing_params}")
        
        print(f"DEBUG: Starting surface processing with {len(surfaces_to_process)} surfaces")
        logger.info(f"[{analysis_id}] Starting surface processing with {len(surfaces_to_process)} surfaces")
        self._update_job_status(analysis_id, "running", 50.0, "processing_surfaces")
        
        print(f"DEBUG: Calling surface_processor.process_surfaces")
        analysis_results = self.surface_processor.process_surfaces(surfaces_to_process, processing_params)
        print(f"DEBUG: surface_processor.process_surfaces completed")
        logger.info(f"[{analysis_id}] Surface processing completed successfully")
        
        # Ensure results are fully JSON serializable before caching
        print(f"DEBUG: Serializing results for caching")
        logger.info(f"[{analysis_id}] Serializing results for caching")
        self._update_job_status(analysis_id, "running", 90.0, "serializing_results")
        
        # Clean out-of-range floats for JSON compliance
        serializable_results = clean_floats_for_json(analysis_results)
        print(f"DEBUG: Results serialized successfully")
        logger.info(f"[{analysis_id}] Results serialized successfully")

        # Validate that results are actually JSON serializable
        if not validate_json_serializable(serializable_results):
            print(f"DEBUG: Results still not JSON serializable after conversion")
            logger.error(f"[{analysis_id}] Results still not JSON serializable after conversion")
            raise RuntimeError("Failed to serialize analysis results")

        # Determine if the analysis is considered complete
        is_volume_analysis = len(surfaces_to_process) > 1 or params.get('generate_base_surface')
        results_are_ready = not is_volume_analysis or ('volume_results' in serializable_results and serializable_results['volume_results'])
        print(f"DEBUG: Analysis type - is_volume_analysis: {is_volume_analysis}, results_are_ready: {results_are_ready}")

        final_status = "completed" if results_are_ready else "processing"
        print(f"DEBUG: Final status determined: {final_status}")

        logger.info(f"[{analysis_id}] Updating job status to {final_status}")

        # Add georeferencing metadata to the results
        # Use georeference_params[0] if present, fallback to defaults
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
        
        # Store results for visualization - ensure no threading primitives
        print(f"DEBUG: Caching results with status: {final_status}")
        self._results_cache[analysis_id] = {
            **serializable_results,
            "georef": georef,
            "analysis_metadata": {"status": final_status}
        }
        print(f"DEBUG: Results cached successfully")
        logger.info(f"[{analysis_id}] Results cached successfully. Analysis status: {final_status}.")

    def start_analysis_execution(self, analysis_id: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Start analysis execution with background task management"""
        if analysis_id in self._jobs:
            job = self._jobs[analysis_id]
            if job["status"] in ("running", "pending"):
                raise RuntimeError("Analysis already running")
        
        # Handle new frontend payload structure
        surface_count = len(params.get('surface_ids', [])) if params else 0
        
        # Ensure params are JSON serializable before storing
        serializable_params = make_json_serializable(params) if params else {}
        
        job = {
            "status": "pending",
            "progress_percent": 0.0,
            "current_step": "queued",
            "cancellable": True,
            "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": serializable_params
        }
        
        if len(self._jobs) >= self.MAX_CONCURRENT_JOBS:
            # Simple queueing mechanism
            oldest_job_id = min(self._jobs.keys(), key=lambda k: self._jobs[k]["start_time"])
            if self._jobs[oldest_job_id]["status"] not in ("completed", "failed", "cancelled"):
                 raise HTTPException(status_code=503, detail="Max concurrent jobs reached, please try again later.")
            else:
                del self._jobs[oldest_job_id]

        self._jobs[analysis_id] = job
        logger.info(f"[{analysis_id}] Job created with status: {job['status']}")
        
        return {
            "status": "started",
            "analysis_id": analysis_id,
            "message": f"Analysis started with {surface_count} surfaces"
        }

    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status without threading primitives"""
        if analysis_id not in self._jobs:
            raise KeyError("Analysis not found")
        
        job = self._jobs[analysis_id]
        
        status = {
            "analysis_id": analysis_id,
            "status": job["status"],
            "progress_percent": job["progress_percent"],
            "current_step": job["current_step"],
            "start_time": job.get("start_time"),
            "completion_time": job.get("completion_time"),
            "error_message": job.get("error_message")
        }
        
        logger.info(f"Returning status for {analysis_id}: {status}")
        
        # Ensure the status response is JSON serializable
        return make_json_serializable(status)

    def get_results(self, analysis_id: str, include: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get analysis results with optional filtering. Returns None if not available."""
        if analysis_id not in self._jobs:
            return None
        job = self._jobs[analysis_id]
        if job["status"] != "completed":
            return None
        results = self._results_cache.get(analysis_id)
        if not results:
            return None
        if include:
            filtered_results = {"analysis_metadata": results["analysis_metadata"]}
            if include == "volume":
                filtered_results["volume_results"] = results.get("volume_results", [])
            elif include == "thickness":
                filtered_results["thickness_results"] = results.get("thickness_results", [])
            elif include == "compaction":
                filtered_results["compaction_rates"] = results.get("compaction_rates", [])
            return filtered_results
        return results

    def cancel_analysis(self, analysis_id: str) -> Dict[str, Any]:
        job = self._jobs.get(analysis_id)
        if not job:
            raise KeyError("Analysis not found")
        if job["status"] == "completed":
            raise RuntimeError("Analysis already completed")
        if job["status"] == "cancelled":
            raise RuntimeError("Analysis already cancelled")
        job["cancelled"] = True
        job["status"] = "cancelled"
        job["cancellation_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return {"status": "cancelled", "analysis_id": analysis_id}

    @staticmethod
    def is_valid_status(status: str) -> bool:
        return status in ("pending", "running", "completed", "failed", "cancelled")

    @staticmethod
    def is_valid_priority(priority: str) -> bool:
        return priority in ("low", "normal", "high", "urgent")

    @staticmethod
    def validate_execution_parameters(params: Dict[str, Any]) -> bool:
        if not isinstance(params, dict):
            return False
        if "priority" in params and not AnalysisExecutor.is_valid_priority(params["priority"]):
            return False
        if "notify_on_completion" in params and not isinstance(params["notify_on_completion"], bool):
            return False
        if "save_intermediate_results" in params and not isinstance(params["save_intermediate_results"], bool):
            return False
        return True

    @staticmethod
    def generate_analysis_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def calculate_progress(current: int, total: int) -> float:
        if total <= 0:
            return 0.0
        return min(100.0, max(0.0, float(current) / total * 100))

    @staticmethod
    def calculate_estimated_duration(surface_count: int) -> float:
        return max(1.0, float(surface_count) * 1.0)

    @staticmethod
    def can_cancel(status: str) -> bool:
        return status in ("pending", "running")

    @staticmethod
    def format_error_message(msg: str, step: str) -> str:
        return f"Error during {step}: {msg}"

    @staticmethod
    def classify_error(msg: str) -> str:
        if "not found" in msg.lower():
            return "input_error"
        if "memory" in msg.lower():
            return "resource_error"
        return "processing_error"

    def _determine_utm_zone_from_coordinates(self, utm_x: float, utm_y: float) -> int:
        """
        Determine UTM zone EPSG code from UTM coordinates
        
        Args:
            utm_x: UTM X coordinate (meters)
            utm_y: UTM Y coordinate (meters)
            
        Returns:
            UTM zone EPSG code (e.g., 32618 for UTM Zone 18N)
        """
        # This is a simplified approach - in production you might want to store the UTM zone
        # For now, we'll use a reasonable default based on the coordinates
        if utm_y > 5000000:  # Northern hemisphere
            # Estimate zone from X coordinate (rough approximation)
            # UTM zones are 6 degrees wide, so roughly 667,000 meters per zone
            estimated_zone = int((utm_x / 667000) + 1)
            return 32600 + estimated_zone  # Northern hemisphere
        else:
            # Southern hemisphere
            estimated_zone = int((utm_x / 667000) + 1)
            return 32700 + estimated_zone  # Southern hemisphere

    def _convert_utm_to_wgs84(self, vertices: np.ndarray) -> np.ndarray:
        """
        Convert UTM vertices back to WGS84 coordinates
        
        Args:
            vertices: Nx3 numpy array of UTM coordinates (meters)
            
        Returns:
            Nx3 numpy array of WGS84 coordinates (lat, lon, elevation)
        """
        if vertices is None or vertices.size == 0:
            return vertices
        
        # Convert each vertex from UTM to WGS84
        wgs84_vertices = []
        for vertex in vertices:
            utm_x, utm_y = vertex[0], vertex[1]
            elevation = vertex[2] if len(vertex) > 2 else 0.0
            
            # Use coordinate transformer to convert UTM to WGS84
            # Note: We need to determine the UTM zone from the coordinates
            # For now, we'll use a simple approach - determine zone from the first vertex
            if len(wgs84_vertices) == 0:
                # Determine UTM zone from the first vertex
                # This is a simplified approach - in production you might want to store the UTM zone
                utm_zone = self._determine_utm_zone_from_coordinates(utm_x, utm_y)
            
            try:
                # Create transformer for this specific UTM zone
                transformer = Transformer.from_crs(f"EPSG:{utm_zone}", "EPSG:4326", always_xy=True)
                lon, lat = transformer.transform(utm_x, utm_y)
                wgs84_vertices.append([lat, lon, elevation])
            except Exception as e:
                logger.warning(f"Failed to convert UTM vertex {vertex} to WGS84: {e}")
                # Fallback: use original coordinates
                wgs84_vertices.append([0.0, 0.0, elevation])
        
        return np.array(wgs84_vertices)

    def _convert_wgs84_to_utm(self, vertices: np.ndarray) -> np.ndarray:
        """
        Convert WGS84 vertices to UTM coordinates
        
        Args:
            vertices: Nx3 numpy array of WGS84 coordinates (lat, lon, elevation)
            
        Returns:
            Nx3 numpy array of UTM coordinates (meters)
        """
        if vertices is None or vertices.size == 0:
            return vertices
        
        # Determine UTM zone from the first vertex for efficiency
        if len(vertices) > 0:
            first_vertex = vertices[0]
            lat, lon = first_vertex[0], first_vertex[1]
            # Use a reasonable default UTM zone for the area (Zone 17N for this region)
            utm_zone = 32617  # UTM Zone 17N
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_zone}", always_xy=True)
        else:
            return vertices
        
        # Convert vertices in batches for better performance
        batch_size = 10000  # Process 10k vertices at a time
        utm_vertices = []
        
        for i in range(0, len(vertices), batch_size):
            batch = vertices[i:i + batch_size]
            batch_utm = []
            
            for vertex in batch:
                lat, lon = vertex[0], vertex[1]
                elevation = vertex[2] if len(vertex) > 2 else 0.0
                
                try:
                    utm_x, utm_y = transformer.transform(lon, lat)
                    batch_utm.append([utm_x, utm_y, elevation])
                except Exception as e:
                    logger.warning(f"Failed to convert WGS84 vertex {vertex} to UTM: {e}")
                    # Fallback: use original coordinates
                    batch_utm.append([0.0, 0.0, elevation])
            
            utm_vertices.extend(batch_utm)
            
            # Log progress for large conversions
            if len(vertices) > 100000:
                progress = (i + batch_size) / len(vertices) * 100
                print(f"DEBUG: UTM conversion progress: {min(progress, 100):.1f}%")
        
        return np.array(utm_vertices) 