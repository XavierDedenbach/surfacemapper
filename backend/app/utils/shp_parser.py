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

    # Minor Task 10.2.2: SHP to UTM Projection and Preparation Methods
    
    def _project_to_utm(self, wgs84_coords: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Project WGS84 coordinates to UTM coordinates.
        
        Args:
            wgs84_coords: List of (lon, lat, z) coordinates in WGS84 degrees
            
        Returns:
            numpy array of shape (N, 3) with (x, y, z) coordinates in UTM meters
            
        Raises:
            ValueError: If coordinates are invalid or span multiple UTM zones
        """
        if not wgs84_coords:
            raise ValueError("No coordinates provided")
        
        # Validate input coordinates
        for i, (lon, lat, z) in enumerate(wgs84_coords):
            if not isinstance(lon, (int, float)) or not isinstance(lat, (int, float)) or not isinstance(z, (int, float)):
                raise ValueError(f"Coordinates must be numeric, got {type(lon)}, {type(lat)}, {type(z)} at index {i}")
            
            if not np.isfinite(lon) or not np.isfinite(lat) or not np.isfinite(z):
                raise ValueError(f"Coordinates must be finite, got {lon}, {lat}, {z} at index {i}")
            
            if not (-180 <= lon <= 180):
                raise ValueError(f"Longitude must be between -180 and 180, got {lon} at index {i}")
            
            if not (-90 <= lat <= 90):
                raise ValueError(f"Latitude must be between -90 and 90, got {lat} at index {i}")
        
        # Determine UTM zone from first coordinate
        first_lon, first_lat = wgs84_coords[0][0], wgs84_coords[0][1]
        utm_zone = self._get_utm_zone(first_lat, first_lon)
        
        # Validate all coordinates are in the same UTM zone
        for lon, lat, z in wgs84_coords:
            coord_zone = self._get_utm_zone(lat, lon)
            if coord_zone != utm_zone:
                raise ValueError(f"All coordinates must be in the same UTM zone. Expected {utm_zone}, got {coord_zone} for ({lon}, {lat})")
        
        # Create transformer
        try:
            transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{utm_zone}", always_xy=True)
        except Exception as e:
            raise ValueError(f"Failed to create coordinate transformer: {e}")
        
        # Project coordinates
        utm_coords = []
        for lon, lat, z in wgs84_coords:
            try:
                x, y = transformer.transform(lon, lat)
                utm_coords.append([x, y, z])
            except Exception as e:
                raise ValueError(f"Failed to project coordinate ({lon}, {lat}, {z}): {e}")
        
        return np.array(utm_coords, dtype=np.float64)

    def _project_to_wgs84(self, utm_coords: np.ndarray, utm_zone: int = None) -> np.ndarray:
        """
        Project UTM coordinates back to WGS84 coordinates.
        
        Args:
            utm_coords: numpy array of shape (N, 3) with (x, y, z) coordinates in UTM meters
            utm_zone: UTM zone EPSG code (e.g., 32617 for UTM Zone 17N). If None, will be inferred.
            
        Returns:
            numpy array of shape (N, 3) with (lon, lat, z) coordinates in WGS84 degrees
            
        Raises:
            ValueError: If coordinates are invalid or UTM zone cannot be determined
        """
        if utm_coords is None or len(utm_coords) == 0:
            raise ValueError("No coordinates provided")
        
        if not isinstance(utm_coords, np.ndarray) or utm_coords.shape[1] != 3:
            raise ValueError("UTM coordinates must be numpy array with shape (N, 3)")
        
        # Validate UTM coordinates
        if not self._validate_utm_coordinates(utm_coords):
            raise ValueError("Invalid UTM coordinates provided")
        
        # Determine UTM zone if not provided
        if utm_zone is None:
            # Try to infer from coordinate ranges
            x_coords = utm_coords[:, 0]
            y_coords = utm_coords[:, 1]
            
            # Rough estimation based on coordinate ranges
            if np.all(y_coords > 5000000):  # Northern hemisphere
                if np.all(x_coords > 500000):  # Typical UTM X range
                    # This is a rough estimate - in practice, we'd need more context
                    utm_zone = 32617  # Default to UTM Zone 17N
                else:
                    raise ValueError("Cannot determine UTM zone from coordinates")
            else:
                raise ValueError("Cannot determine UTM zone from coordinates")
        
        # Create transformer
        try:
            transformer = Transformer.from_crs(f"EPSG:{utm_zone}", "EPSG:4326", always_xy=True)
        except Exception as e:
            raise ValueError(f"Failed to create coordinate transformer: {e}")
        
        # Project coordinates
        wgs84_coords = []
        for x, y, z in utm_coords:
            try:
                lon, lat = transformer.transform(x, y)
                wgs84_coords.append([lon, lat, z])
            except Exception as e:
                raise ValueError(f"Failed to project coordinate ({x}, {y}, {z}): {e}")
        
        return np.array(wgs84_coords, dtype=np.float64)

    def _validate_utm_coordinates(self, utm_coords: np.ndarray) -> bool:
        """
        Validate UTM coordinates.
        
        Args:
            utm_coords: numpy array of shape (N, 3) with (x, y, z) coordinates
            
        Returns:
            True if coordinates are valid UTM coordinates, False otherwise
        """
        if utm_coords is None or len(utm_coords) == 0:
            return False
        
        if not isinstance(utm_coords, np.ndarray) or utm_coords.shape[1] != 3:
            return False
        
        # Check for finite values
        if not np.all(np.isfinite(utm_coords)):
            return False
        
        # Check UTM coordinate ranges
        x_coords = utm_coords[:, 0]
        y_coords = utm_coords[:, 1]
        z_coords = utm_coords[:, 2]
        
        # UTM X coordinates should be positive and reasonable (typically 100,000 to 1,000,000)
        if not np.all(x_coords > 10000):
            return False
        
        # UTM Y coordinates should be positive and reasonable (typically 4,000,000 to 10,000,000)
        if not np.all(y_coords > 1000000):
            return False
        
        # Z coordinates should be reasonable (typically -1000 to 10000 meters)
        if not np.all(z_coords > -10000) or not np.all(z_coords < 100000):
            return False
        
        return True

    def process_shp_file_to_utm(self, file_path: str, max_distance_feet: float = 1.0) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a SHP file and project to UTM coordinates for analysis.
        
        Args:
            file_path: Path to the SHP file
            max_distance_feet: Maximum distance between points for densification
            
        Returns:
            Tuple of (vertices, faces) where vertices is a numpy array of shape (N, 3)
            in UTM meters and faces is None (since we're creating a boundary polygon)
        """
        # Process SHP file to get WGS84 coordinates
        wgs84_vertices, faces = self.process_shp_file(file_path, max_distance_feet)
        
        # Convert numpy array to list of tuples for projection
        wgs84_coords = [(row[0], row[1], row[2]) for row in wgs84_vertices]
        
        # Project to UTM
        utm_vertices = self._project_to_utm(wgs84_coords)
        
        logger.info(f"Projected {len(utm_vertices)} vertices from WGS84 to UTM coordinates")
        logger.info(f"UTM coordinate ranges: X={utm_vertices[:, 0].min():.1f}-{utm_vertices[:, 0].max():.1f}, "
                   f"Y={utm_vertices[:, 1].min():.1f}-{utm_vertices[:, 1].max():.1f}, "
                   f"Z={utm_vertices[:, 2].min():.1f}-{utm_vertices[:, 2].max():.1f}")
        
        return utm_vertices, faces 