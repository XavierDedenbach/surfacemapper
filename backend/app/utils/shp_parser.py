import fiona
import shapely.geometry as sgeom
import shapely.ops as sops
import numpy as np
from pyproj import CRS, Transformer
from typing import List, Tuple, Optional, Union
import logging
from shapely.geometry import LineString, Polygon, Point, MultiPoint, MultiLineString
from shapely.ops import unary_union, linemerge
from shapely.geometry import box

logger = logging.getLogger(__name__)

class SHPParser:
    """
    Handles SHP file parsing, LineString densification, polygon boundary creation, 
    WGS84 boundary clipping, and CRS validation.
    """
    def __init__(self):
        self.required_crs = CRS.from_epsg(4326)  # WGS84

    def parse_shp_file(self, file_path: str) -> Tuple[List[sgeom.base.BaseGeometry], CRS]:
        """
        Parse a SHP file and return a list of Shapely geometries and the CRS.
        
        Args:
            file_path: Path to the SHP file
            
        Returns:
            Tuple of (geometries, crs) where geometries is a list of Shapely geometries
            and crs is a pyproj CRS object
            
        Raises:
            ValueError: If CRS is not WGS84 or geometry type is unsupported
            RuntimeError: If file cannot be read or parsed
        """
        try:
            with fiona.open(file_path, 'r') as src:
                # Validate CRS
                if src.crs:
                    crs = CRS(src.crs)
                    if not self._is_wgs84_crs(crs):
                        raise ValueError(f"CRS must be WGS84 (EPSG:4326), got {crs}")
                else:
                    raise ValueError("SHP file must have a defined CRS")
                
                # Parse geometries
                geometries = []
                for feature in src:
                    geom = sgeom.shape(feature['geometry'])
                    if geom.is_empty:
                        continue
                    
                    # Handle different geometry types
                    if isinstance(geom, (sgeom.Point, sgeom.MultiPoint)):
                        geometries.extend(self._extract_points_from_geometry(geom))
                    elif isinstance(geom, sgeom.Polygon):
                        geometries.append(geom)
                    elif isinstance(geom, sgeom.MultiPolygon):
                        geometries.extend(geom.geoms)
                    elif isinstance(geom, sgeom.LineString):
                        geometries.append(geom)
                    elif isinstance(geom, sgeom.MultiLineString):
                        geometries.extend(geom.geoms)
                    else:
                        raise ValueError(f"Unsupported geometry type: {type(geom)}")
                
                logger.info(f"Parsed {len(geometries)} geometries from {file_path}")
                return geometries, crs
                
        except Exception as e:
            raise RuntimeError(f"Failed to parse SHP file {file_path}: {str(e)}")

    def densify_linestring(self, linestring: LineString, max_distance_feet: float = 1.0) -> LineString:
        """
        Densify a LineString to have points no more than max_distance_feet apart.
        Assumes input coordinates are in WGS84 degrees.
        
        Args:
            linestring: Input LineString geometry
            max_distance_feet: Maximum distance between points in feet
            
        Returns:
            Densified LineString with points no more than max_distance_feet apart
        """
        coords = list(linestring.coords)
        if len(coords) < 2:
            return linestring
        
        # Convert to UTM for accurate distance calculations
        # Determine UTM zone from the first coordinate
        first_lon, first_lat = coords[0][0], coords[0][1]
        utm_zone = self._get_utm_zone(first_lat, first_lon)
        utm_crs = f"EPSG:{utm_zone}"
        
        transformer = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True)
        
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

    def create_polygon_boundary_from_contours(self, linestrings: List[LineString]) -> Polygon:
        """
        Create a polygon boundary from LineString contours using multiple approaches.
        
        Args:
            linestrings: List of LineString contours
            
        Returns:
            Polygon boundary that encompasses all contours
            
        Raises:
            ValueError: If no valid polygon can be created
        """
        if not linestrings:
            raise ValueError("No LineString contours provided")
        
        # Collect all points from all LineStrings
        all_points = []
        for linestring in linestrings:
            coords = list(linestring.coords)
            for coord in coords:
                all_points.append(Point(coord[0], coord[1]))  # Use only x,y for boundary
        
        if len(all_points) < 3:
            raise ValueError("Not enough points to create a polygon boundary")
        
        # Method 1: Try to create polygon from merged lines
        try:
            logger.info("Attempting to create polygon from merged lines...")
            merged_lines = linemerge(linestrings)
            
            if isinstance(merged_lines, MultiLineString):
                logger.info("Multiple line segments found, trying to create boundary...")
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
                logger.info(f"Successfully created polygon from merged lines with area: {polygon.area:.6f} square degrees")
                return polygon
            else:
                raise Exception("Invalid polygon from merged lines")
                
        except Exception as e:
            logger.info(f"Method 1 failed: {e}")
            
            # Method 2: Create convex hull from all points
            try:
                logger.info("Attempting to create convex hull from all points...")
                if len(all_points) >= 3:
                    # Create a MultiPoint and get its convex hull
                    multipoint = sgeom.MultiPoint(all_points)
                    polygon = multipoint.convex_hull
                    
                    if polygon.is_valid and polygon.area > 0:
                        logger.info(f"Successfully created convex hull with area: {polygon.area:.6f} square degrees")
                        return polygon
                    else:
                        raise Exception("Invalid convex hull")
                else:
                    raise Exception("Not enough points for convex hull")
                    
            except Exception as e2:
                logger.info(f"Method 2 failed: {e2}")
                
                # Method 3: Create bounding box with buffer
                try:
                    logger.info("Attempting to create bounding box with buffer...")
                    if len(all_points) > 0:
                        # Get bounding box of all points
                        x_coords = [p.x for p in all_points]
                        y_coords = [p.y for p in all_points]
                        
                        min_x, max_x = min(x_coords), max(x_coords)
                        min_y, max_y = min(y_coords), max(y_coords)
                        
                        # Create bounding box with small buffer
                        bbox = box(min_x, min_y, max_x, max_y)
                        polygon = bbox.buffer(0.0001)  # Small buffer in degrees
                        
                        logger.info(f"Successfully created bounding box with area: {polygon.area:.6f} square degrees")
                        return polygon
                    else:
                        raise Exception("No points available")
                        
                except Exception as e3:
                    logger.error(f"Method 3 failed: {e3}")
                    raise ValueError("Failed to create polygon boundary from contours")

    def process_shp_file(self, file_path: str, max_distance_feet: float = 1.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a SHP file: parse, densify contours, create polygon boundary, and output as numpy arrays.
        
        Args:
            file_path: Path to the SHP file
            max_distance_feet: Maximum distance between points for densification
            
        Returns:
            Tuple of (vertices, faces) where vertices is a numpy array of shape (N, 3)
            and faces is None (since we're creating a boundary polygon)
        """
        # Parse the SHP file
        geometries, crs = self.parse_shp_file(file_path)
        
        # Separate LineStrings from other geometries
        linestrings = [g for g in geometries if isinstance(g, LineString)]
        other_geometries = [g for g in geometries if not isinstance(g, LineString)]
        
        if linestrings:
            # Densify LineString contours
            logger.info(f"Densifying {len(linestrings)} LineString contours...")
            densified_linestrings = []
            for linestring in linestrings:
                densified = self.densify_linestring(linestring, max_distance_feet)
                densified_linestrings.append(densified)
            
            # Create polygon boundary from densified contours
            logger.info("Creating polygon boundary from densified contours...")
            boundary_polygon = self.create_polygon_boundary_from_contours(densified_linestrings)
            
            # Convert polygon to vertices
            coords = list(boundary_polygon.exterior.coords)
            vertices = np.array(coords, dtype=np.float64)
            
            # Add z-coordinate if not present
            if vertices.shape[1] == 2:
                # Add z=0 for all points
                z_coords = np.zeros((vertices.shape[0], 1))
                vertices = np.hstack([vertices, z_coords])
            
            logger.info(f"Created boundary polygon with {len(vertices)} vertices")
            return vertices, None
        
        elif other_geometries:
            # Handle other geometry types (points, polygons)
            all_vertices = []
            for geom in other_geometries:
                if isinstance(geom, Point):
                    coords = list(geom.coords)
                    all_vertices.extend(coords)
                elif isinstance(geom, Polygon):
                    coords = list(geom.exterior.coords)
                    all_vertices.extend(coords)
            
            if all_vertices:
                vertices = np.array(all_vertices, dtype=np.float64)
                # Add z-coordinate if not present
                if vertices.shape[1] == 2:
                    z_coords = np.zeros((vertices.shape[0], 1))
                    vertices = np.hstack([vertices, z_coords])
                return vertices, None
        
        # If no valid geometries found
        raise ValueError("No valid geometries found in SHP file")

    def clip_to_boundary(self, geometry: sgeom.base.BaseGeometry, boundary: sgeom.base.BaseGeometry) -> sgeom.base.BaseGeometry:
        """
        Clip a geometry to a boundary.
        
        Args:
            geometry: Geometry to clip
            boundary: Boundary geometry to clip against
            
        Returns:
            Clipped geometry
        """
        try:
            clipped = geometry.intersection(boundary)
            if clipped.is_empty:
                return None
            return clipped
        except Exception as e:
            logger.error(f"Error clipping geometry: {e}")
            return None

    def validate_shp_file(self, file_path: str) -> bool:
        """
        Validate a SHP file for processing.
        
        Args:
            file_path: Path to the SHP file
            
        Returns:
            True if file is valid, False otherwise
        """
        try:
            with fiona.open(file_path, 'r') as src:
                # Check CRS
                if not src.crs:
                    logger.error("SHP file must have a defined CRS")
                    return False
                
                crs = CRS(src.crs)
                if not self._is_wgs84_crs(crs):
                    logger.error(f"CRS must be WGS84 (EPSG:4326), got {crs}")
                    return False
                
                # Check for valid geometries
                geometry_types = set()
                for feature in src:
                    geom = sgeom.shape(feature['geometry'])
                    if not geom.is_empty:
                        geometry_types.add(type(geom))
                
                # Check for supported geometry types
                supported_types = {sgeom.Point, sgeom.MultiPoint, sgeom.Polygon, 
                                 sgeom.MultiPolygon, sgeom.LineString, sgeom.MultiLineString}
                
                unsupported_types = geometry_types - supported_types
                if unsupported_types:
                    logger.error(f"SHP file contains unsupported geometry types: {unsupported_types}")
                    return False
                
                return True
                
        except Exception as e:
            logger.error(f"Error validating SHP file: {e}")
            return False

    def _is_wgs84_crs(self, crs: CRS) -> bool:
        """Check if CRS is WGS84."""
        try:
            return crs.to_epsg() == 4326
        except:
            return False

    def _extract_points_from_geometry(self, geom: sgeom.base.BaseGeometry) -> List[sgeom.Point]:
        """Extract Point geometries from Point or MultiPoint."""
        if isinstance(geom, sgeom.Point):
            return [geom]
        elif isinstance(geom, sgeom.MultiPoint):
            return list(geom.geoms)
        else:
            return []

    def _get_utm_zone(self, lat: float, lon: float) -> int:
        """Get UTM zone number from latitude and longitude."""
        zone_number = int((lon + 180) / 6) + 1
        
        # Handle special cases
        if lat >= 84.0:
            if lon >= 0:
                return 32601  # UTM Zone 1N
            else:
                return 32701  # UTM Zone 1S
        elif lat <= -80.0:
            if lon >= 0:
                return 32601  # UTM Zone 1N
            else:
                return 32701  # UTM Zone 1S
        
        # Determine hemisphere
        if lat >= 0:
            return 32600 + zone_number  # Northern hemisphere
        else:
            return 32700 + zone_number  # Southern hemisphere 