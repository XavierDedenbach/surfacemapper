from fastapi import APIRouter, HTTPException, Request, Query, Body, BackgroundTasks, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional
from app.services.analysis_executor import AnalysisExecutor
from app.services import thickness_calculator
from app.services.coord_transformer import CoordinateTransformer, TransformationPipeline
import numpy as np
from app.services.statistical_analysis import StatisticalAnalyzer
from app.models.data_models import StatisticalAnalysis, ProcessingRequest
from fastapi import status
from app.services.data_export import DataExporter
import tempfile
import os
from datetime import datetime
import logging
import traceback
import io
import csv
from app.services import triangulation

router = APIRouter(tags=["analysis"])
executor = AnalysisExecutor()
stat_analyzer = StatisticalAnalyzer()
data_exporter = DataExporter()
logger = logging.getLogger(__name__)

@router.post("/start", status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(request: ProcessingRequest, background_tasks: BackgroundTasks):
    """
    Starts a new analysis job using FastAPI background tasks.
    """
    try:
        analysis_id = executor.generate_analysis_id()
        # The request body is now a Pydantic model, so we convert it to a dict
        params = request.model_dump()
        
        # Start the analysis using background tasks
        background_tasks.add_task(executor.run_analysis_sync, analysis_id, params)
        
        result = executor.start_analysis_execution(analysis_id, params)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to start analysis: " + str(e))

@router.post("/{analysis_id}/execute")
async def execute_analysis(analysis_id: str, request: Request, background_tasks: BackgroundTasks):
    try:
        params = await request.json() if request.headers.get("content-type", "").startswith("application/json") else None
        if params and not executor.validate_execution_parameters(params):
            raise HTTPException(status_code=400, detail="Invalid execution parameters")
        
        # Start the analysis using background tasks
        background_tasks.add_task(executor.run_analysis_sync, analysis_id, params)
        
        result = executor.start_analysis_execution(analysis_id, params)
        return JSONResponse(status_code=202, content=result)
    except RuntimeError as e:
        if "already running" in str(e):
            raise HTTPException(status_code=409, detail="Analysis already running")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
    except Exception as e:
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail="Analysis not found")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

@router.get("/{analysis_id}/status")
async def get_analysis_status(analysis_id: str):
    try:
        logger.info(f"Getting status for analysis {analysis_id}")
        status = executor.get_analysis_status(analysis_id)
        logger.info(f"Status retrieved successfully: {status}")
        
        # Test serialization before returning
        import json
        try:
            json.dumps(status)
            logger.info("Status object is JSON serializable")
        except Exception as e:
            logger.error(f"Status object is NOT JSON serializable: {e}")
            logger.error(f"Status object type: {type(status)}")
            logger.error(f"Status object content: {repr(status)}")
            raise HTTPException(status_code=500, detail=f"Serialization error: {str(e)}")
        
        return status
    except KeyError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        logger.error(f"Error in get_analysis_status: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

@router.get("/{analysis_id}/results")
async def get_analysis_results(analysis_id: str, include: Optional[str] = Query(None, description="Filter results to include only specific components: volume, thickness, compaction")):
    try:
        # First, check the status of the analysis
        status_info = executor.get_analysis_status(analysis_id)
        current_status = status_info.get("status")

        if current_status in ("pending", "running", "processing"):
            return JSONResponse(
                status_code=status.HTTP_202_ACCEPTED,
                content={"status": current_status, "progress": status_info.get("progress_percent", 0)}
            )
        
        if current_status == "failed":
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "analysis_metadata": {
                        "analysis_id": analysis_id,
                        "status": "failed",
                        "error_message": status_info.get("error_message", "Processing failed without a specific error."),
                    }
                }
            )

        # If completed, get the results
        if current_status == "completed":
            results = executor.get_results(analysis_id, include)
            if results:
                logger.info(f"--- Serving Results for Analysis ID: {analysis_id} ---")
                # Log a summary of what's being sent
                volume_summary = "present" if "volume_results" in results and results["volume_results"] else "absent"
                thickness_summary = "present" if "thickness_results" in results and results["thickness_results"] else "absent"
                logger.info(f"  - Volume Results: {volume_summary}")
                logger.info(f"  - Thickness Results: {thickness_summary}")
                logger.info("-------------------------------------------------")
                # Add analysis_id at the top level
                return {"analysis_id": analysis_id, **results}
            else:
                # This case might happen in a race condition.
                # The job is complete, but results are not yet in the cache.
                # Return a 202 to encourage the client to poll again.
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={"status": "completed_caching", "message": "Results are being cached."}
                )
        
        # Handle other statuses or unexpected scenarios
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Analysis in unexpected state: {current_status}")

    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    except Exception as e:
        logger.error(f"Failed to get analysis results for {analysis_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error")

@router.post("/{analysis_id}/cancel")
async def cancel_analysis(analysis_id: str):
    try:
        result = executor.cancel_analysis(analysis_id)
        return result
    except KeyError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except RuntimeError as e:
        if "already completed" in str(e):
            raise HTTPException(status_code=400, detail="Analysis already completed")
        if "already cancelled" in str(e):
            raise HTTPException(status_code=400, detail="Analysis already cancelled")
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e)) 

@router.post("/{analysis_id}/point_query")
async def point_query(
    analysis_id: str,
    payload: dict = Body(...)
):
    """Return thickness for each layer at a single query point."""
    # Validate input
    x = payload.get("x")
    y = payload.get("y")
    coordinate_system = payload.get("coordinate_system", "utm")
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise HTTPException(status_code=400, detail="Invalid coordinates")
    if coordinate_system not in ("utm", "wgs84"):
        raise HTTPException(status_code=400, detail="Unsupported coordinate system")

    # Retrieve analysis results
    results = executor.get_results(analysis_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Analysis not found or not completed")
    if results.get("analysis_metadata", {}).get("status") != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    # Get TINs and metadata
    surface_tins = results.get("surface_tins")
    surface_names = results.get("surface_names")
    if surface_tins is None or len(surface_tins) == 0 or surface_names is None or len(surface_names) == 0:
        raise HTTPException(status_code=500, detail="Surface TINs not available")

    # Get georeference metadata
    georef = results.get("georef")
    if not georef:
        raise HTTPException(status_code=500, detail="Georeference metadata missing from analysis results")
    if georef["lat"] == 0.0 and georef["lon"] == 0.0:
        print("[WARNING] Anchor point is (0.0, 0.0). This may indicate a misconfiguration or missing georeference data.")
    pipeline = TransformationPipeline(
        anchor_lat=georef["lat"],
        anchor_lon=georef["lon"],
        rotation_degrees=georef["orientation"],
        scale_factor=georef["scale"]
    )
    utm_to_wgs84 = pipeline.utm_to_wgs84_transformer
    # Debug logging for anchor and UTM zone
    print(f"[DEBUG] Anchor lat: {georef['lat']}, lon: {georef['lon']}, UTM zone: {pipeline.utm_zone}, anchor_utm_x: {pipeline.anchor_utm_x}, anchor_utm_y: {pipeline.anchor_utm_y}")

    # Transform coordinates if needed (assume UTM is default system)
    point = np.array([x, y])
    if coordinate_system == "wgs84":
        point = pipeline.transform_to_utm(np.array([[x, y, 0.0]]))[0][:2]

    # Compute lat/lon for the query point
    print(f"[DEBUG] UTM input: x={point[0]}, y={point[1]}")
    lon, lat = utm_to_wgs84.transform(point[0], point[1])
    print(f"[DEBUG] WGS84 output: lon={lon}, lat={lat}")

    # For each layer, calculate thickness at this point
    thickness_layers = []
    for i in range(len(surface_tins) - 1):
        upper_tin = surface_tins[i+1]
        lower_tin = surface_tins[i]
        # Each TIN must have z_values attribute
        upper_z = thickness_calculator._interpolate_z_at_point(point, upper_tin)
        lower_z = thickness_calculator._interpolate_z_at_point(point, lower_tin)
        if np.isnan(upper_z) or np.isnan(lower_z):
            thickness = None
        else:
            thickness = upper_z - lower_z
        thickness_layers.append({
            "layer_designation": f"{surface_names[i]} to {surface_names[i+1]}",
            "thickness_feet": thickness
        })

    return {
        "thickness_layers": thickness_layers,
        "query_point": {
            "x": float(point[0]),
            "y": float(point[1]),
            "lat": lat,
            "lon": lon,
            "coordinate_system": "utm"
        },
        "interpolation_method": "linear"
    }

@router.post("/{analysis_id}/batch_point_query")
async def batch_point_query(
    analysis_id: str,
    payload: dict = Body(...)
):
    """Return thickness for each layer at multiple query points."""
    points = payload.get("points")
    if not isinstance(points, list) or not all(isinstance(p, dict) for p in points):
        raise HTTPException(status_code=400, detail="Invalid format for points")
    if len(points) > 1000:
        raise HTTPException(status_code=400, detail="Too many points (max 1000)")

    # Retrieve analysis results
    results = executor.get_results(analysis_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Analysis not found or not completed")
    if results.get("analysis_metadata", {}).get("status") != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    surface_tins = results.get("surface_tins")
    surface_names = results.get("surface_names")
    if surface_tins is None or len(surface_tins) == 0 or surface_names is None or len(surface_names) == 0:
        raise HTTPException(status_code=500, detail="Surface TINs not available")

    # Prepare points array (transform if needed)
    batch_points = []
    for p in points:
        x = p.get("x")
        y = p.get("y")
        coordinate_system = p.get("coordinate_system", "utm")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            continue  # skip invalid
        pt = np.array([x, y])
        if coordinate_system == "wgs84":
            transformer = CoordinateTransformer(
                anchor_lat=results["georef"]["lat"],
                anchor_lon=results["georef"]["lon"],
                rotation_degrees=results["georef"]["orientation"],
                scale_factor=results["georef"]["scale"]
            )
            pt = transformer.transform_to_utm(np.array([[x, y]]))[0]
        batch_points.append(pt)
    batch_points = np.array(batch_points)

    # For each layer, calculate thickness at all points
    results_list = []
    for idx, pt in enumerate(batch_points):
        thickness_layers = []
        for i in range(len(surface_tins) - 1):
            upper_tin = surface_tins[i+1]
            lower_tin = surface_tins[i]
            upper_z = thickness_calculator._interpolate_z_at_point(pt, upper_tin)
            lower_z = thickness_calculator._interpolate_z_at_point(pt, lower_tin)
            if np.isnan(upper_z) or np.isnan(lower_z):
                thickness = None
            else:
                thickness = upper_z - lower_z
            thickness_layers.append({
                "layer_designation": f"{surface_names[i]} to {surface_names[i+1]}",
                "thickness_feet": thickness
            })
        results_list.append({
            "point": {"x": float(pt[0]), "y": float(pt[1]), "coordinate_system": "utm"},
            "thickness_layers": thickness_layers
        })

    return {
        "results": results_list,
        "total_points": len(results_list)
    }

@router.get("/{analysis_id}/surface/{surface_id}/mesh")
async def get_3d_visualization_data(
    analysis_id: str,
    surface_id: int,
    level_of_detail: str = Query("medium", description="Level of detail: low, medium, high"),
    max_vertices: Optional[int] = Query(None, description="Maximum number of vertices for mesh simplification"),
    preserve_boundaries: bool = Query(True, description="Preserve boundary edges during simplification")
):
    """
    Get 3D mesh data for visualization with optional simplification
    """
    try:
        # Validate surface_id
        if surface_id < 0:
            raise HTTPException(status_code=400, detail="Surface ID must be non-negative")
        
        # Validate level of detail
        if level_of_detail not in ["low", "medium", "high"]:
            raise HTTPException(status_code=400, detail="Level of detail must be low, medium, or high")
        
        # Get analysis results
        results = executor.get_results(analysis_id)
        if results is None:
            raise HTTPException(status_code=404, detail="Analysis not found or not completed")
        
        if results.get("analysis_metadata", {}).get("status") != "completed":
            raise HTTPException(status_code=400, detail="Analysis not completed")
        
        # Get surface data
        surface_tins = results.get("surface_tins")
        if surface_tins is None or len(surface_tins) == 0 or surface_id >= len(surface_tins):
            raise HTTPException(status_code=404, detail="Surface not found")
        
        tin = surface_tins[surface_id]
        if not hasattr(tin, 'points') or not hasattr(tin, 'simplices'):
            raise HTTPException(status_code=500, detail="Invalid surface data")
        
        # Extract mesh data
        vertices = tin.points.tolist()
        faces = tin.simplices.tolist()
        
        # Calculate bounds
        if len(vertices) > 0:
            vertices_array = np.array(vertices)
            bounds = {
                "x_min": float(np.min(vertices_array[:, 0])),
                "x_max": float(np.max(vertices_array[:, 0])),
                "y_min": float(np.min(vertices_array[:, 1])),
                "y_max": float(np.max(vertices_array[:, 1])),
                "z_min": float(np.min(vertices_array[:, 2])),
                "z_max": float(np.max(vertices_array[:, 2]))
            }
        else:
            bounds = {
                "x_min": 0.0, "x_max": 0.0,
                "y_min": 0.0, "y_max": 0.0,
                "z_min": 0.0, "z_max": 0.0
            }
        
        # Apply mesh simplification based on level of detail
        simplified_vertices, simplified_faces = _simplify_mesh(
            vertices, faces, level_of_detail, max_vertices, preserve_boundaries
        )
        
        # Recalculate bounds for simplified mesh
        if len(simplified_vertices) > 0:
            simplified_array = np.array(simplified_vertices)
            bounds = {
                "x_min": float(np.min(simplified_array[:, 0])),
                "x_max": float(np.max(simplified_array[:, 0])),
                "y_min": float(np.min(simplified_array[:, 1])),
                "y_max": float(np.max(simplified_array[:, 1])),
                "z_min": float(np.min(simplified_array[:, 2])),
                "z_max": float(np.max(simplified_array[:, 2]))
            }
        
        return {
            "vertices": simplified_vertices,
            "faces": simplified_faces,
            "bounds": bounds,
            "metadata": {
                "analysis_id": analysis_id,
                "surface_id": surface_id,
                "level_of_detail": level_of_detail,
                "original_vertex_count": len(vertices),
                "simplified_vertex_count": len(simplified_vertices),
                "original_face_count": len(faces),
                "simplified_face_count": len(simplified_faces),
                "simplification_ratio": len(simplified_vertices) / len(vertices) if len(vertices) > 0 else 1.0
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def _simplify_mesh(vertices, faces, level_of_detail, max_vertices=None, preserve_boundaries=True):
    """
    Simplify mesh based on level of detail and parameters
    """
    # Patch: Use explicit checks for numpy arrays
    if vertices is None or faces is None:
        return vertices, faces
    if isinstance(vertices, np.ndarray) and vertices.size == 0:
        return vertices, faces
    if isinstance(faces, np.ndarray) and faces.size == 0:
        return vertices, faces
    if isinstance(vertices, list) and len(vertices) == 0:
        return vertices, faces
    if isinstance(faces, list) and len(faces) == 0:
        return vertices, faces
    
    # Determine target vertex count based on level of detail
    original_vertex_count = len(vertices)
    
    if max_vertices is not None:
        target_vertices = min(max_vertices, original_vertex_count)
    else:
        if level_of_detail == "high":
            target_vertices = original_vertex_count
        elif level_of_detail == "medium":
            target_vertices = max(original_vertex_count // 2, 100)
        else:  # low
            target_vertices = max(original_vertex_count // 4, 50)
    
    # If no simplification needed, return original
    if target_vertices >= original_vertex_count:
        return vertices, faces
    
    # Simple mesh simplification using vertex clustering
    # In production, use more sophisticated algorithms like Quadric Error Metrics
    simplified_vertices, simplified_faces = _cluster_based_simplification(
        vertices, faces, target_vertices, preserve_boundaries
    )
    
    return simplified_vertices, simplified_faces


def _cluster_based_simplification(vertices, faces, target_vertices, preserve_boundaries):
    """
    Simple vertex clustering-based mesh simplification
    """
    if len(vertices) <= target_vertices:
        return vertices, faces
    
    # Convert to numpy arrays for easier manipulation
    vertices_array = np.array(vertices)
    faces_array = np.array(faces)
    
    # Calculate bounding box
    min_coords = np.min(vertices_array, axis=0)
    max_coords = np.max(vertices_array, axis=0)
    
    # Create grid for clustering
    grid_size = int(np.ceil(np.power(target_vertices, 1/3)))
    cell_size = (max_coords - min_coords) / grid_size
    
    # Assign vertices to grid cells
    vertex_clusters = {}
    cluster_centers = {}
    
    for i, vertex in enumerate(vertices_array):
        # Calculate grid cell coordinates
        cell_coords = ((vertex - min_coords) / cell_size).astype(int)
        cell_coords = np.clip(cell_coords, 0, grid_size - 1)
        cell_key = tuple(cell_coords)
        
        if cell_key not in vertex_clusters:
            vertex_clusters[cell_key] = []
            cluster_centers[cell_key] = []
        
        vertex_clusters[cell_key].append(i)
        cluster_centers[cell_key].append(vertex)
    
    # Create vertex mapping and new vertices
    vertex_mapping = {}
    new_vertices = []
    
    for cell_key, vertex_indices in vertex_clusters.items():
        if vertex_indices:
            # Use centroid of cluster as new vertex
            cluster_center = np.mean(cluster_centers[cell_key], axis=0)
            new_vertex_id = len(new_vertices)
            new_vertices.append(cluster_center.tolist())
            
            # Map all vertices in cluster to new vertex
            for old_vertex_id in vertex_indices:
                vertex_mapping[old_vertex_id] = new_vertex_id
    
    # Create new faces
    new_faces = []
    seen_faces = set()
    
    for face in faces_array:
        # Map face vertices to new vertices
        new_face = [vertex_mapping.get(v, v) for v in face]
        
        # Remove degenerate faces (all vertices the same)
        if len(set(new_face)) == 1:
            continue
        
        # Remove duplicate faces
        face_key = tuple(sorted(new_face))
        if face_key in seen_faces:
            continue
        
        seen_faces.add(face_key)
        new_faces.append(new_face)
    
    return new_vertices, new_faces

@router.post("/statistics", response_model=StatisticalAnalysis)
async def statistical_analysis(payload: dict = Body(...)):
    values = payload.get("values")
    if not isinstance(values, list):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing or invalid 'values' key")
    try:
        stats = stat_analyzer.calculate_statistics(values)
        return stats
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@router.post("/export")
async def export_data(payload: dict = Body(...)):
    """Export data to various formats"""
    # Validate required fields
    data_type = payload.get("data_type")
    data = payload.get("data")
    export_format = payload.get("format")
    
    if not data_type:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing 'data_type'")
    if not data:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing 'data'")
    if not export_format:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Missing 'format'")
    
    # Validate format support for data type
    if data_type == "analysis_results":
        if export_format not in ["csv", "json", "excel"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Format '{export_format}' not supported for analysis results")
    elif data_type == "statistical_data":
        if export_format not in ["csv", "json"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Format '{export_format}' not supported for statistical data")
    elif data_type == "surface_data":
        if export_format not in ["ply", "obj", "stl", "xyz"]:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Format '{export_format}' not supported for surface data")
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unsupported data type: {data_type}")
    
    try:
        # Generate filename
        filename = payload.get("filename")
        ext = export_format
        if export_format == "excel":
            ext = "xlsx"
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{data_type}_{timestamp}.{ext}"
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False) as tmp_file:
            file_path = tmp_file.name
        
        # Export data based on type
        if data_type == "analysis_results":
            from app.models.data_models import AnalysisResults
            analysis_results = AnalysisResults(**data)
            exported_path = data_exporter.export_analysis_results(
                analysis_results, export_format, file_path, payload.get("metadata")
            )
        elif data_type == "statistical_data":
            from app.models.data_models import StatisticalAnalysis
            statistical_data = StatisticalAnalysis(**data)
            exported_path = data_exporter.export_statistical_data(
                statistical_data, export_format, file_path
            )
        elif data_type == "surface_data":
            exported_path = data_exporter.export_surface_data(
                data, export_format, file_path
            )
        
        return {
            "file_path": exported_path,
            "format": export_format,
            "data_type": data_type,
            "filename": filename,
            "export_timestamp": datetime.now().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Export failed: {str(e)}")

@router.get("/{analysis_id}/thickness_grid_csv")
async def thickness_grid_csv(
    analysis_id: str,
    spacing: float = Query(1.0, description="Grid spacing in feet (default 1.0)")
):
    """Stream a CSV of thickness at each grid point for all layers."""
    # Retrieve analysis results
    results = executor.get_results(analysis_id)
    if results is None:
        raise HTTPException(status_code=404, detail="Analysis not found or not completed")
    if results.get("analysis_metadata", {}).get("status") != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    surfaces = results.get("surfaces")
    if not surfaces or len(surfaces) < 2:
        raise HTTPException(status_code=400, detail="At least two surfaces required for thickness grid export")

    # Convert vertices back to numpy arrays
    surface_points = [np.array(s["vertices"]) for s in surfaces]
    layer_names = [f"{surfaces[i]['name']} to {surfaces[i+1]['name']}" for i in range(len(surfaces)-1)]

    # Compute bounding box for all surfaces
    all_xy = np.vstack([pts[:, :2] for pts in surface_points])
    x_min, y_min = np.min(all_xy, axis=0)
    x_max, y_max = np.max(all_xy, axis=0)

    # Generate grid points
    x_coords = np.arange(x_min, x_max + spacing, spacing)
    y_coords = np.arange(y_min, y_max + spacing, spacing)
    grid_points = np.array([[x, y] for x in x_coords for y in y_coords])

    # Build TINs for each surface
    tins = []
    for pts in surface_points:
        tin = triangulation.create_delaunay_triangulation(pts[:, :2])
        setattr(tin, 'z_values', pts[:, 2])
        tins.append(tin)

    # Prepare transformation pipeline for UTM->WGS84
    georef = results.get("georef")
    if not georef:
        raise HTTPException(status_code=500, detail="Georeference metadata missing from analysis results")
    pipeline = TransformationPipeline(
        anchor_lat=georef["lat"],
        anchor_lon=georef["lon"],
        rotation_degrees=georef["orientation"],
        scale_factor=georef["scale"]
    )
    utm_to_wgs84 = pipeline.utm_to_wgs84_transformer

    # Prepare CSV streaming
    def csv_generator():
        output = io.StringIO()
        writer = csv.writer(output)
        # Write header with local x, y, lat, lon
        writer.writerow(["local_x_feet", "local_y_feet", "lat", "lon"] + layer_names)
        yield output.getvalue()
        output.seek(0)
        output.truncate(0)
        for pt in grid_points:
            # Compute lat/lon from UTM (x, y)
            lon, lat = utm_to_wgs84.transform(pt[0], pt[1])
            # Inverse transform: UTM -> local feet
            utm_point = np.array([[pt[0], pt[1], 0.0]])
            local_point_m = pipeline.inverse_transform(utm_point)[0]
            FEET_PER_METER = 3.280839895
            local_x_feet = local_point_m[0] * FEET_PER_METER
            local_y_feet = local_point_m[1] * FEET_PER_METER
            row = [local_x_feet, local_y_feet, lat, lon]
            for i in range(len(tins)-1):
                upper_z = thickness_calculator._interpolate_z_at_point(pt, tins[i+1])
                lower_z = thickness_calculator._interpolate_z_at_point(pt, tins[i])
                if not np.isnan(upper_z) and not np.isnan(lower_z):
                    thickness = upper_z - lower_z
                else:
                    thickness = ''
                row.append(thickness)
            writer.writerow(row)
            yield output.getvalue()
            output.seek(0)
            output.truncate(0)

    return StreamingResponse(csv_generator(), media_type="text/csv", headers={
        "Content-Disposition": f"attachment; filename=thickness_grid_{analysis_id}.csv"
    }) 