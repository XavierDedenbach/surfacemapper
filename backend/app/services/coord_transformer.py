"""
Coordinate transformation service using pyproj
"""
import numpy as np
import pyproj
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class GeoreferenceParams:
    """Georeferencing parameters for a surface"""
    wgs84_lat: float
    wgs84_lon: float
    orientation_degrees: float
    scaling_factor: float

class CoordinateTransformer:
    """
    Handles coordinate transformations using pyproj
    """
    
    def __init__(self):
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.utm_zone = None  # Will be determined based on longitude
    
    def determine_utm_zone(self, longitude: float) -> str:
        """
        Determine UTM zone based on longitude
        Note: This method only determines the zone number, not the hemisphere.
        For complete UTM zone determination, use determine_utm_zone_with_hemisphere().
        """
        # Validate longitude
        if not isinstance(longitude, (int, float)) or np.isnan(longitude) or np.isinf(longitude):
            raise ValueError("Longitude must be a finite number")
        
        if longitude < -180 or longitude > 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        
        # Calculate zone number (1-60)
        # UTM zones are 6 degrees wide, starting at 180°W (zone 1)
        zone_number = int((longitude + 180) / 6) + 1
        
        # Ensure zone number is within valid range
        if zone_number < 1:
            zone_number = 1
        elif zone_number > 60:
            zone_number = 60
        
        # Return zone number only (hemisphere will be determined by latitude)
        return zone_number
    
    def determine_utm_zone_with_hemisphere(self, latitude: float, longitude: float) -> str:
        """
        Determine complete UTM zone (including hemisphere) based on coordinates
        """
        zone_number = self.determine_utm_zone(longitude)
        hemisphere = "N" if latitude >= 0 else "S"
        
        # Return EPSG code: 326xx for Northern, 327xx for Southern
        return f"EPSG:32{6 if hemisphere == 'N' else 7}{zone_number:02d}"
    
    def transform_wgs84_to_utm(self, latitude: float, longitude: float) -> Tuple[float, float]:
        """
        Transform WGS84 coordinates to UTM coordinates
        
        Args:
            latitude: WGS84 latitude in degrees
            longitude: WGS84 longitude in degrees
            
        Returns:
            Tuple of (utm_x, utm_y) coordinates in meters
        """
        # Validate inputs
        if not isinstance(latitude, (int, float)) or np.isnan(latitude) or np.isinf(latitude):
            raise ValueError("Latitude must be a finite number")
        
        if not isinstance(longitude, (int, float)) or np.isnan(longitude) or np.isinf(longitude):
            raise ValueError("Longitude must be a finite number")
        
        if latitude < -90 or latitude > 90:
            raise ValueError("Latitude must be between -90 and 90 degrees")
        
        if longitude < -180 or longitude > 180:
            raise ValueError("Longitude must be between -180 and 180 degrees")
        
        # Handle edge cases
        if abs(latitude) == 90:
            raise ValueError("Pole coordinates are not supported for UTM transformation")
        
        # Determine UTM zone and hemisphere
        zone_number = int((longitude + 180) / 6) + 1
        hemisphere = "N" if latitude >= 0 else "S"
        
        # Create UTM CRS
        utm_epsg = f"EPSG:32{6 if hemisphere == 'N' else 7}{zone_number:02d}"
        utm_crs = pyproj.CRS(utm_epsg)
        
        # Create transformer
        transformer = pyproj.Transformer.from_crs(self.wgs84, utm_crs, always_xy=True)
        
        # Transform coordinates
        utm_x, utm_y = transformer.transform(longitude, latitude)
        
        return (utm_x, utm_y)
    
    def transform_wgs84_to_utm_batch(self, coordinates: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Transform a batch of WGS84 coordinates to UTM coordinates
        
        Args:
            coordinates: List of (latitude, longitude) tuples
            
        Returns:
            List of (utm_x, utm_y) coordinate tuples
        """
        if not coordinates:
            raise ValueError("Coordinates list cannot be empty")
        
        # Validate all coordinates
        for i, coord in enumerate(coordinates):
            if not isinstance(coord, tuple) or len(coord) != 2:
                raise ValueError(f"Coordinate {i} must be a tuple of (lat, lon)")
            
            lat, lon = coord
            if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
                raise ValueError(f"Coordinate {i} must contain numeric values")
        
        # Group coordinates by UTM zone for efficiency
        zone_groups = {}
        
        for lat, lon in coordinates:
            # Validate coordinates
            if lat < -90 or lat > 90:
                raise ValueError(f"Latitude {lat} must be between -90 and 90 degrees")
            if lon < -180 or lon > 180:
                raise ValueError(f"Longitude {lon} must be between -180 and 180 degrees")
            if abs(lat) == 90:
                raise ValueError("Pole coordinates are not supported for UTM transformation")
            
            # Determine zone
            zone_number = int((lon + 180) / 6) + 1
            hemisphere = "N" if lat >= 0 else "S"
            utm_epsg = f"EPSG:32{6 if hemisphere == 'N' else 7}{zone_number:02d}"
            
            if utm_epsg not in zone_groups:
                zone_groups[utm_epsg] = []
            zone_groups[utm_epsg].append((lat, lon))
        
        # Transform each zone group
        results = []
        for utm_epsg, zone_coords in zone_groups.items():
            utm_crs = pyproj.CRS(utm_epsg)
            transformer = pyproj.Transformer.from_crs(self.wgs84, utm_crs, always_xy=True)
            
            # Extract lat/lon arrays
            lats = [coord[0] for coord in zone_coords]
            lons = [coord[1] for coord in zone_coords]
            
            # Transform batch
            utm_xs, utm_ys = transformer.transform(lons, lats)
            
            # Add to results
            for x, y in zip(utm_xs, utm_ys):
                results.append((x, y))
        
        return results
    
    def transform_surface_coordinates(
        self, 
        vertices: np.ndarray, 
        params: GeoreferenceParams
    ) -> np.ndarray:
        """
        Transform surface coordinates from local to geo-referenced system
        """
        # TODO: Implement coordinate transformation logic
        # 1. Apply scaling
        # 2. Apply rotation
        # 3. Transform to UTM using pyproj
        return vertices
    
    def transform_boundary_coordinates(
        self, 
        wgs84_boundary: List[Tuple[float, float]]
    ) -> List[Tuple[float, float]]:
        """
        Transform WGS84 boundary coordinates to UTM
        """
        # TODO: Implement boundary coordinate transformation
        return wgs84_boundary
    
    def validate_transformation_accuracy(self, original: np.ndarray, transformed: np.ndarray) -> bool:
        """
        Validate transformation accuracy (<0.1m for surveying applications)
        """
        # TODO: Implement accuracy validation
        return True 
    
    def get_rotation_matrix_z(self, angle_degrees: float) -> np.ndarray:
        """
        Get 3D rotation matrix around Z-axis
        
        Args:
            angle_degrees: Rotation angle in degrees (clockwise from North as per business requirements)
            
        Returns:
            3x3 rotation matrix as numpy array
        """
        # Validate input
        if not isinstance(angle_degrees, (int, float)) or np.isnan(angle_degrees) or np.isinf(angle_degrees):
            raise ValueError("Angle must be a finite number")
        
        # Convert degrees to radians
        # Note: Business requirement specifies "clockwise from North"
        # For standard mathematical rotation matrices, we use counterclockwise
        # So we negate the angle to convert from clockwise to counterclockwise
        angle_rad = np.radians(-angle_degrees)
        
        # Create rotation matrix around Z-axis
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotation_matrix = np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,    1]
        ])
        
        return rotation_matrix
    
    def get_scaling_matrix(self, scale_factor: float) -> np.ndarray:
        """
        Get 3D uniform scaling matrix
        
        Args:
            scale_factor: Uniform scaling factor (must be positive)
            
        Returns:
            3x3 scaling matrix as numpy array
        """
        # Validate input
        if not isinstance(scale_factor, (int, float)) or np.isnan(scale_factor) or np.isinf(scale_factor):
            raise ValueError("Scale factor must be a finite number")
        
        if scale_factor < 0:
            raise ValueError("Scale factor must be non-negative")
        
        # Create uniform scaling matrix
        scaling_matrix = np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, scale_factor]
        ])
        
        return scaling_matrix
    
    def apply_rotation_z(self, points: np.ndarray, angle_degrees: float) -> np.ndarray:
        """
        Apply 3D rotation around Z-axis to points
        
        Args:
            points: Nx3 numpy array of 3D points
            angle_degrees: Rotation angle in degrees (positive = counterclockwise)
            
        Returns:
            Nx3 numpy array of rotated points
        """
        # Validate inputs
        if points is None:
            raise ValueError("Points array cannot be None")
        
        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array")
        
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a 2D array with 3 columns (Nx3)")
        
        if not isinstance(angle_degrees, (int, float)) or np.isnan(angle_degrees) or np.isinf(angle_degrees):
            raise ValueError("Angle must be a finite number")
        
        # Handle empty array
        if points.shape[0] == 0:
            return points.copy()
        
        # Get rotation matrix
        rotation_matrix = self.get_rotation_matrix_z(angle_degrees)
        
        # Apply rotation: points @ rotation_matrix.T
        rotated_points = np.dot(points, rotation_matrix.T)
        
        return rotated_points
    
    def apply_scaling(self, points: np.ndarray, scale_factor: float) -> np.ndarray:
        """
        Apply uniform scaling to points
        
        Args:
            points: Nx3 numpy array of 3D points
            scale_factor: Uniform scaling factor (must be positive)
            
        Returns:
            Nx3 numpy array of scaled points
        """
        # Validate inputs
        if points is None:
            raise ValueError("Points array cannot be None")
        
        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array")
        
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a 2D array with 3 columns (Nx3)")
        
        if not isinstance(scale_factor, (int, float)) or np.isnan(scale_factor) or np.isinf(scale_factor):
            raise ValueError("Scale factor must be a finite number")
        
        if scale_factor < 0:
            raise ValueError("Scale factor must be non-negative")
        
        # Handle empty array
        if points.shape[0] == 0:
            return points.copy()
        
        # Apply scaling: points * scale_factor
        scaled_points = points * scale_factor
        
        return scaled_points 