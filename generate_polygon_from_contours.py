import fiona
import shapely.geometry as sgeom
from shapely.geometry import LineString, Polygon, Point, MultiLineString
from shapely.ops import unary_union, linemerge
from shapely.geometry import box
import numpy as np
from pyproj import Transformer
import tempfile
import os

def densify_linestring(linestring, max_distance_feet=1.0):
    """
    Densify a LineString to have points no more than max_distance_feet apart.
    Assumes input coordinates are in WGS84 degrees.
    """
    coords = list(linestring.coords)
    if len(coords) < 2:
        return linestring
    
    # Convert to UTM for accurate distance calculations
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32617", always_xy=True)  # UTM Zone 17N
    
    densified_coords = []
    for i in range(len(coords) - 1):
        start_coord = coords[i]
        end_coord = coords[i + 1]
        
        # Add start point
        densified_coords.append(start_coord)
        
        # Calculate distance in UTM coordinates
        start_utm = transformer.transform(start_coord[0], start_coord[1])
        end_utm = transformer.transform(end_coord[0], end_coord[1])
        
        distance_meters = np.sqrt((end_utm[0] - start_utm[0])**2 + (end_utm[1] - start_utm[1])**2)
        distance_feet = distance_meters * 3.28084  # Convert meters to feet
        
        if distance_feet > max_distance_feet:
            # Calculate number of intermediate points needed
            num_segments = int(np.ceil(distance_feet / max_distance_feet))
            
            for j in range(1, num_segments):
                t = j / num_segments
                
                # Interpolate in WGS84 coordinates
                lat = start_coord[1] + t * (end_coord[1] - start_coord[1])
                lon = start_coord[0] + t * (end_coord[0] - start_coord[0])
                
                # Interpolate elevation if available
                if len(start_coord) > 2 and len(end_coord) > 2:
                    elev = start_coord[2] + t * (end_coord[2] - start_coord[2])
                    densified_coords.append((lon, lat, elev))
                else:
                    densified_coords.append((lon, lat))
    
    # Add final point
    densified_coords.append(coords[-1])
    
    return LineString(densified_coords)

def create_surface_polygon_from_contours(shp_file_path, output_shp_path, max_distance_feet=1.0):
    """
    Create a surface polygon from LineString contours using multiple approaches.
    """
    print(f"Processing contours from: {shp_file_path}")
    
    # Read all LineStrings from the SHP file
    linestrings = []
    all_points = []
    elevations = []
    
    with fiona.open(shp_file_path, 'r') as src:
        print(f"Found {len(src)} contour features")
        
        for feature in src:
            geom = sgeom.shape(feature['geometry'])
            if isinstance(geom, LineString):
                # Densify the LineString
                densified = densify_linestring(geom, max_distance_feet)
                linestrings.append(densified)
                
                # Collect all points for boundary creation
                coords = list(densified.coords)
                for coord in coords:
                    all_points.append(Point(coord[0], coord[1]))  # Use only x,y for boundary
                
                elevations.append(feature['properties'].get('ELEVATION', 0))
    
    if not linestrings:
        print("No LineString features found!")
        return None
    
    print(f"Processing {len(linestrings)} densified contours...")
    print(f"Total points collected: {len(all_points)}")
    
    # Method 1: Try to create polygon from merged lines
    try:
        print("Attempting to create polygon from merged lines...")
        merged_lines = linemerge(linestrings)
        
        if isinstance(merged_lines, MultiLineString):
            print("Multiple line segments found, trying to create boundary...")
            # Create a boundary polygon from all line segments
            boundary_lines = unary_union(linestrings)
            if hasattr(boundary_lines, 'boundary'):
                polygon = Polygon(boundary_lines.boundary.coords)
            else:
                raise Exception("Cannot create boundary from line segments")
        else:
            # Single line - try to close it
            coords = list(merged_lines.coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])  # Close the ring
            polygon = Polygon(coords)
        
        if polygon.is_valid:
            print(f"Successfully created polygon from merged lines with area: {polygon.area:.6f} square degrees")
            method = "merged_lines"
        else:
            raise Exception("Invalid polygon from merged lines")
            
    except Exception as e:
        print(f"Method 1 failed: {e}")
        
        # Method 2: Create convex hull from all points
        try:
            print("Attempting to create convex hull from all points...")
            if len(all_points) >= 3:
                # Create a MultiPoint and get its convex hull
                multipoint = sgeom.MultiPoint(all_points)
                polygon = multipoint.convex_hull
                
                if polygon.is_valid and polygon.area > 0:
                    print(f"Successfully created convex hull with area: {polygon.area:.6f} square degrees")
                    method = "convex_hull"
                else:
                    raise Exception("Invalid convex hull")
            else:
                raise Exception("Not enough points for convex hull")
                
        except Exception as e2:
            print(f"Method 2 failed: {e2}")
            
            # Method 3: Create bounding box with buffer
            try:
                print("Attempting to create bounding box with buffer...")
                if len(all_points) > 0:
                    # Get bounding box of all points
                    x_coords = [p.x for p in all_points]
                    y_coords = [p.y for p in all_points]
                    
                    min_x, max_x = min(x_coords), max(x_coords)
                    min_y, max_y = min(y_coords), max(y_coords)
                    
                    # Create bounding box with small buffer
                    bbox = box(min_x, min_y, max_x, max_y)
                    polygon = bbox.buffer(0.0001)  # Small buffer in degrees
                    
                    print(f"Successfully created bounding box with area: {polygon.area:.6f} square degrees")
                    method = "bounding_box"
                else:
                    raise Exception("No points available")
                    
            except Exception as e3:
                print(f"Method 3 failed: {e3}")
                return None
    
    # Save the polygon to a new SHP file
    try:
        schema = {
            'geometry': 'Polygon',
            'properties': {
                'ID': 'int',
                'AREA_SQ_DEG': 'float',
                'METHOD': 'str',
                'SOURCE': 'str'
            }
        }
        
        with fiona.open(output_shp_path, 'w', 
                      driver='ESRI Shapefile',
                      schema=schema,
                      crs='EPSG:4326') as dst:
            dst.write({
                'geometry': sgeom.mapping(polygon),
                'properties': {
                    'ID': 1,
                    'AREA_SQ_DEG': polygon.area,
                    'METHOD': method,
                    'SOURCE': 'Generated from contours'
                }
            })
        
        print(f"Polygon saved to: {output_shp_path}")
        print(f"Method used: {method}")
        return polygon
        
    except Exception as e:
        print(f"Error saving polygon: {e}")
        return None

def analyze_contour_statistics(shp_file_path):
    """
    Analyze the contour data to understand the elevation range and distribution.
    """
    print(f"Analyzing contours in: {shp_file_path}")
    
    elevations = []
    line_lengths = []
    
    with fiona.open(shp_file_path, 'r') as src:
        for feature in src:
            geom = sgeom.shape(feature['geometry'])
            if isinstance(geom, LineString):
                elevations.append(feature['properties'].get('ELEVATION', 0))
                line_lengths.append(geom.length)
    
    if elevations:
        print(f"Elevation range: {min(elevations):.2f} to {max(elevations):.2f} feet")
        print(f"Average elevation: {np.mean(elevations):.2f} feet")
        print(f"Total contour length: {sum(line_lengths):.2f} degrees")
        print(f"Number of contours: {len(elevations)}")

if __name__ == "__main__":
    # Input SHP file
    input_shp = "drone_surfaces/27June20250541PM1619tonspartialcover/27June20250541PM1619tonspartialcover.shp"
    
    # Output SHP file for the generated polygon
    output_shp = "drone_surfaces/generated_surface_boundary.shp"
    
    # Analyze the contours first
    analyze_contour_statistics(input_shp)
    
    # Generate polygon with 1-foot spacing
    polygon = create_surface_polygon_from_contours(input_shp, output_shp, max_distance_feet=1.0)
    
    if polygon:
        print("Successfully generated surface boundary polygon!")
        print(f"Polygon bounds: {polygon.bounds}")
    else:
        print("Failed to generate surface boundary polygon") 