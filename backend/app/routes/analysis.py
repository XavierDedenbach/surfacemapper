from fastapi import APIRouter, HTTPException, Request, Query, Body
from fastapi.responses import JSONResponse
from typing import Optional
from app.services.analysis_executor import AnalysisExecutor
from app.services import thickness_calculator
from app.services.coord_transformer import CoordTransformer
import numpy as np

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
executor = AnalysisExecutor()

@router.post("/{analysis_id}/execute")
async def execute_analysis(analysis_id: str, request: Request):
    try:
        params = await request.json() if request.headers.get("content-type", "").startswith("application/json") else None
        if params and not executor.validate_execution_parameters(params):
            raise HTTPException(status_code=400, detail="Invalid execution parameters")
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
        status = executor.get_analysis_status(analysis_id)
        return status
    except KeyError:
        raise HTTPException(status_code=404, detail="Analysis not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

@router.get("/{analysis_id}/results")
async def get_analysis_results(analysis_id: str, include: Optional[str] = Query(None, description="Filter results to include only specific components: volume, thickness, compaction")):
    try:
        # Get results from executor
        results = executor.get_results(analysis_id, include)
        
        # If no results and analysis doesn't exist
        if results is None:
            # Check if analysis exists
            try:
                status = executor.get_analysis_status(analysis_id)
                # Analysis exists but not completed
                if status["status"] == "processing":
                    return JSONResponse(
                        status_code=202,
                        content={
                            "status": "processing",
                            "progress": status["progress_percent"] / 100.0,
                            "estimated_completion": "2024-12-20T11:00:00Z"  # Mock estimation
                        }
                    )
                elif status["status"] == "failed":
                    return JSONResponse(
                        status_code=200,
                        content={
                            "analysis_metadata": {
                                "analysis_id": analysis_id,
                                "status": "failed",
                                "failure_time": status.get("completion_time"),
                                "error_message": status.get("error_message", "Processing failed"),
                                "partial_results_available": False
                            }
                        }
                    )
                elif status["status"] == "cancelled":
                    return JSONResponse(
                        status_code=200,
                        content={
                            "analysis_metadata": {
                                "analysis_id": analysis_id,
                                "status": "cancelled",
                                "cancellation_time": status.get("completion_time"),
                                "partial_results_available": False
                            }
                        }
                    )
            except KeyError:
                # Analysis doesn't exist
                raise HTTPException(status_code=404, detail="Analysis not found")
        
        # Return results
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error: " + str(e))

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
    if results.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    # Get TINs and metadata
    surface_tins = results.get("surface_tins")
    surface_names = results.get("surface_names")
    if not surface_tins or not surface_names:
        raise HTTPException(status_code=500, detail="Surface TINs not available")

    # Transform coordinates if needed (assume UTM is default system)
    point = np.array([x, y])
    if coordinate_system == "wgs84":
        # Use metadata to get transformation (mocked here)
        # In production, use actual transformation logic
        transformer = CoordTransformer(
            anchor_lat=results["georef"]["lat"],
            anchor_lon=results["georef"]["lon"],
            rotation_degrees=results["georef"]["orientation"],
            scale_factor=results["georef"]["scale"]
        )
        point = transformer.transform_to_utm(np.array([[x, y]]))[0]

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
            "layer_name": f"{surface_names[i]} to {surface_names[i+1]}",
            "thickness_feet": thickness
        })

    return {
        "thickness_layers": thickness_layers,
        "query_point": {"x": float(point[0]), "y": float(point[1]), "coordinate_system": "utm"},
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
    if results.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Analysis not completed")

    surface_tins = results.get("surface_tins")
    surface_names = results.get("surface_names")
    if not surface_tins or not surface_names:
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
            transformer = CoordTransformer(
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
                "layer_name": f"{surface_names[i]} to {surface_names[i+1]}",
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