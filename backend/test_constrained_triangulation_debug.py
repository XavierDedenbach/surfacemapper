#!/usr/bin/env python3
"""
Debug script to test constrained triangulation and WGS84 clipping area accuracy
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from shapely.geometry import Polygon, Point, MultiPoint, shape
from app.services.surface_processor import SurfaceProcessor
from app.services.analysis_executor import AnalysisExecutor
from app.services.surface_cache import surface_cache
from app.utils.shp_parser import SHPParser
import json
from pyproj import CRS, Transformer
import fiona

def get_utm_crs(lon, lat):
    # Determine UTM zone and EPSG code
    zone_number = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    if is_northern:
        epsg_code = 32600 + zone_number  # Northern hemisphere
    else:
        epsg_code = 32700 + zone_number  # Southern hemisphere
    return epsg_code

def project_polygon_to_utm(polygon: Polygon):
    # Hardcode EPSG code for this specific location (UTM Zone 17N)
    epsg_code = 32617  # UTM Zone 17N
    utm_crs = CRS.from_epsg(epsg_code)
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
    utm_coords = [transformer.transform(x, y) for x, y in polygon.exterior.coords]
    return Polygon(utm_coords)

def project_vertices_to_utm(vertices: np.ndarray):
    # Hardcode EPSG code for this specific location (UTM Zone 17N)
    epsg_code = 32617  # UTM Zone 17N
    utm_crs = CRS.from_epsg(epsg_code)
    wgs84_crs = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84_crs, utm_crs, always_xy=True)
    
    utm_vertices = np.zeros_like(vertices)
    for i, vertex in enumerate(vertices):
        x, y, z = vertex
        utm_x, utm_y = transformer.transform(x, y)
        utm_vertices[i] = [utm_x, utm_y, z]
    
    return utm_vertices

def mesh_area(vertices, faces):
    # Project mesh vertices to UTM
    from shapely.geometry import Polygon as ShapelyPolygon
    # Use centroid to determine UTM zone
    lon, lat = np.mean(vertices[:,0]), np.mean(vertices[:,1])
    zone_number = int((lon + 180) / 6) + 1
    is_northern = lat >= 0
    utm_crs = CRS.from_dict({
        'proj': 'utm',
        'zone': zone_number,
        'datum': 'WGS84',
        'south': not is_northern
    })
    transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
    verts_utm = np.array([transformer.transform(x, y) for x, y in vertices[:, :2]])
    area = 0.0
    for tri in faces:
        pts = verts_utm[tri]
        poly = ShapelyPolygon(pts)
        area += poly.area
    return area

def calculate_mesh_elevation_stats(vertices, faces):
    """Calculate elevation statistics for the mesh"""
    if len(vertices) == 0 or len(faces) == 0:
        return None
    
    # Get all Z coordinates
    z_coords = vertices[:, 2]
    
    # Calculate elevation statistics
    min_elevation = np.min(z_coords)
    max_elevation = np.max(z_coords)
    elevation_range = max_elevation - min_elevation
    mean_elevation = np.mean(z_coords)
    
    # Calculate true surface area (accounting for slope)
    true_surface_area = 0.0
    for face in faces:
        if len(face) >= 3:
            v1, v2, v3 = vertices[face[:3]]
            
            # Calculate vectors
            vec1 = v2 - v1
            vec2 = v3 - v1
            
            # Cross product to get normal vector
            normal = np.cross(vec1, vec2)
            
            # Area is half the magnitude of the cross product
            area = 0.5 * np.linalg.norm(normal)
            true_surface_area += area
    
    # Calculate average slope (simplified)
    # For each triangle, calculate the slope based on elevation difference
    total_slope = 0.0
    triangle_count = 0
    
    for face in faces:
        if len(face) >= 3:
            v1, v2, v3 = vertices[face[:3]]
            
            # Calculate horizontal distance and elevation difference
            # Use the longest edge for slope calculation
            edges = [
                (np.linalg.norm(v2[:2] - v1[:2]), abs(v2[2] - v1[2])),
                (np.linalg.norm(v3[:2] - v1[:2]), abs(v3[2] - v1[2])),
                (np.linalg.norm(v3[:2] - v2[:2]), abs(v3[2] - v2[2]))
            ]
            
            max_edge = max(edges, key=lambda x: x[0])
            if max_edge[0] > 0:
                slope = max_edge[1] / max_edge[0]  # rise/run
                total_slope += slope
                triangle_count += 1
    
    avg_slope = total_slope / triangle_count if triangle_count > 0 else 0.0
    
    return {
        'min_elevation': min_elevation,
        'max_elevation': max_elevation,
        'elevation_range': elevation_range,
        'mean_elevation': mean_elevation,
        'true_surface_area': true_surface_area,
        'avg_slope': avg_slope,
        'triangle_count': triangle_count
    }

def test_constrained_triangulation_with_real_shp():
    """Test constrained triangulation with real SHP file data"""
    print("=== Testing Constrained Triangulation with Real SHP Data ===")
    
    # Path to the real SHP file
    shp_file_path = "../drone_surfaces/27June2025_0550AM_emptyworkingface/27June2025_0550AM_emptyworkingface.shp"
    
    if not os.path.exists(shp_file_path):
        print(f"ERROR: SHP file not found at {shp_file_path}")
        return
    
    # Load and process the SHP file
    shp_parser = SHPParser()
    
    try:
        # Process the SHP file to get vertices and faces
        vertices, faces = shp_parser.process_shp_file(shp_file_path)
        
        print(f"DEBUG: Original mesh - Vertices: {len(vertices)}, Faces: {len(faces) if faces is not None else 0}")
        
        # Load boundary from the same directory
        boundary_file = "../drone_surfaces/surface_location_config1.json"
        if os.path.exists(boundary_file):
            with open(boundary_file, 'r') as f:
                boundary_config = json.load(f)
            
            boundary_coords = boundary_config.get('boundary_coordinates', [])
            if boundary_coords:
                # Create boundary polygon
                boundary_polygon = Polygon(boundary_coords)
                
                print(f"DEBUG: Boundary polygon area (WGS84 degrees²): {boundary_polygon.area}")
                
                # Project to UTM for area calculation
                utm_polygon = project_polygon_to_utm(boundary_polygon)
                boundary_area_m2 = utm_polygon.area
                print(f"DEBUG: Boundary polygon area (UTM, m²): {boundary_area_m2:.2f}")
                
                # Test constrained triangulation
                surface_processor = SurfaceProcessor()
                clipped_vertices, clipped_faces = surface_processor.create_constrained_triangulation(
                    vertices, boundary_polygon
                )
                
                print(f"DEBUG: Clipped mesh - Vertices: {len(clipped_vertices)}, Faces: {len(clipped_faces) if clipped_faces is not None else 0}")
                
                # Calculate clipped mesh area in UTM
                if len(clipped_vertices) > 0 and clipped_faces is not None and len(clipped_faces) > 0:
                    # Project clipped vertices to UTM
                    utm_vertices = project_vertices_to_utm(clipped_vertices)
                    
                    # Calculate flat projected area
                    clipped_area_m2 = calculate_mesh_area_utm(utm_vertices, clipped_faces)
                    print(f"DEBUG: Clipped mesh area (UTM, m²): {clipped_area_m2:.2f}")
                    
                    # Calculate elevation statistics and true surface area
                    elevation_stats = calculate_mesh_elevation_stats(clipped_vertices, clipped_faces)
                    if elevation_stats:
                        print(f"DEBUG: Elevation range: {elevation_stats['elevation_range']:.2f} units")
                        print(f"DEBUG: Average slope: {elevation_stats['avg_slope']:.4f}")
                        print(f"DEBUG: True surface area: {elevation_stats['true_surface_area']:.2f} square units")
                        
                        # Calculate percent difference
                        percent_diff = ((clipped_area_m2 - boundary_area_m2) / boundary_area_m2) * 100
                        print(f"DEBUG: Percent difference (flat projection): {percent_diff:.2f}%")
                        
                        # Calculate area ratio
                        area_ratio = clipped_area_m2 / boundary_area_m2
                        print(f"DEBUG: Area ratio (clipped/boundary): {area_ratio:.2f}")
                        
                        # Calculate true surface area in UTM coordinates
                        # Convert the true surface area from WGS84 units to UTM square meters
                        # First, get the scale factor for the area conversion
                        # Approximate: 1 degree ≈ 111,000 meters at the equator
                        # For this latitude (40.15°), the scale factor is approximately cos(40.15°) * 111000
                        lat_rad = np.radians(40.15)  # Approximate latitude from the data
                        scale_factor = np.cos(lat_rad) * 111000  # meters per degree at this latitude
                        scale_factor_squared = scale_factor ** 2  # for area conversion
                        
                        true_surface_area_m2 = elevation_stats['true_surface_area'] * scale_factor_squared
                        print(f"DEBUG: True surface area (UTM, m²): {true_surface_area_m2:.2f}")
                        
                        # Calculate percent difference for true surface area
                        true_percent_diff = ((true_surface_area_m2 - boundary_area_m2) / boundary_area_m2) * 100
                        print(f"DEBUG: Percent difference (true surface): {true_percent_diff:.2f}%")
                        
                        # Calculate true area ratio
                        true_area_ratio = true_surface_area_m2 / boundary_area_m2
                        print(f"DEBUG: True area ratio (true/boundary): {true_area_ratio:.2f}")
                        
                        # If elevation range is significant, this might explain the area difference
                        if elevation_stats['elevation_range'] > 1.0:  # More than 1 unit of elevation change
                            print(f"DEBUG: Significant elevation change detected ({elevation_stats['elevation_range']:.2f} units)")
                            print(f"DEBUG: This explains the larger surface area due to slope effects")
                            print(f"DEBUG: Flat projection area: {clipped_area_m2:.2f} m²")
                            print(f"DEBUG: True surface area: {true_surface_area_m2:.2f} m²")
                            print(f"DEBUG: Area increase due to slope: {((true_surface_area_m2 - clipped_area_m2) / clipped_area_m2 * 100):.2f}%")
                        else:
                            print(f"DEBUG: Minimal elevation change ({elevation_stats['elevation_range']:.2f} units)")
                            print(f"DEBUG: Area difference may be due to clipping or triangulation issues")
                    else:
                        print("DEBUG: Could not calculate elevation statistics")
                else:
                    print("DEBUG: No clipped mesh generated")
            else:
                print("DEBUG: No boundary coordinates found in config file")
        else:
            print(f"DEBUG: Boundary config file not found at {boundary_file}")
            
    except Exception as e:
        print(f"ERROR: Failed to process SHP file: {e}")
        import traceback
        traceback.print_exc()

def test_analysis_executor_with_real_shp():
    """Test analysis executor with real SHP file and boundary clipping"""
    print("\n=== Testing Analysis Executor with Real SHP Data ===")
    
    # Path to the real SHP file
    shp_file_path = "../drone_surfaces/27June2025_0550AM_emptyworkingface/27June2025_0550AM_emptyworkingface.shp"
    
    if not os.path.exists(shp_file_path):
        print(f"Error: SHP file not found at {shp_file_path}")
        return
    
    # Create test surface ID
    test_surface_id = "test_real_surface_123"
    
    # Create analysis parameters with boundary
    # Use a smaller boundary to test clipping
    analysis_params = {
        'surface_ids': [test_surface_id],
        'analysis_boundary': {
            'wgs84_coordinates': [
                [40.145, -79.855],  # lat, lon format
                [40.145, -79.835],
                [40.165, -79.835], 
                [40.165, -79.855],
                [40.145, -79.855]
            ]
        }
    }
    
    # Create analysis executor
    executor = AnalysisExecutor()
    analysis_id = "test_real_analysis_123"
    
    print(f"Analysis ID: {analysis_id}")
    print(f"Surface IDs: {analysis_params['surface_ids']}")
    print(f"Boundary: {analysis_params['analysis_boundary']}")
    
    try:
        # Add to surface cache using the proper method
        # First, let's check what methods are available on surface_cache
        print(f"Surface cache type: {type(surface_cache)}")
        print(f"Surface cache methods: {[m for m in dir(surface_cache) if not m.startswith('_')]}")
        
        # Try to add the surface to cache
        if hasattr(surface_cache, 'add_surface'):
            surface_cache.add_surface(test_surface_id, shp_file_path, 'SHP', '27June2025_0550AM_emptyworkingface.shp')
        else:
            # If no add_surface method, try direct assignment
            surface_cache._cache[test_surface_id] = {
                'file_path': shp_file_path,
                'file_type': 'SHP',
                'filename': '27June2025_0550AM_emptyworkingface.shp'
            }
        
        print("\n=== Executing Analysis ===")
        executor._execute_analysis_logic(analysis_id, analysis_params)
        
    except Exception as e:
        print(f"Error in analysis execution: {e}")
        import traceback
        traceback.print_exc()

def main():
    shp_file_path = os.path.join("../drone_surfaces/27June2025_0550AM_emptyworkingface/27June2025_0550AM_emptyworkingface.shp")
    parser = SHPParser()
    geometries, crs = parser.parse_shp_file(shp_file_path)
    linestrings = [g for g in geometries if g.geom_type == 'LineString']
    boundary_polygon = parser.create_polygon_boundary_from_contours(linestrings)
    boundary_utm = project_polygon_to_utm(boundary_polygon)
    boundary_area = boundary_utm.area
    print(f"Boundary polygon area (UTM, m^2): {boundary_area:.2f}")

    # Generate mesh and faces
    vertices, faces = parser.generate_surface_mesh_from_linestrings(linestrings, spacing_feet=1.0)
    mesh_area_val = mesh_area(vertices, faces)
    print(f"Clipped mesh area (UTM, m^2): {mesh_area_val:.2f}")
    percent_diff = 100.0 * abs(mesh_area_val - boundary_area) / boundary_area
    print(f"Percent difference: {percent_diff:.2f}%")

if __name__ == "__main__":
    test_constrained_triangulation_with_real_shp()
    test_analysis_executor_with_real_shp()
    main() 