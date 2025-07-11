"""
Coordinate transformation service using pyproj
"""
import numpy as np
import pyproj
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from scipy.spatial import cKDTree

@dataclass
class GeoreferenceParams:
    """Georeferencing parameters for a surface"""
    wgs84_lat: float
    wgs84_lon: float
    orientation_degrees: float
    scaling_factor: float


class TransformationPipeline:
    """
    Complete coordinate transformation pipeline combining scaling, rotation, and UTM transformation
    """
    
    def __init__(self, anchor_lat: float, anchor_lon: float, rotation_degrees: float, scale_factor: float, center_surface: bool = True):
        """
        Initialize transformation pipeline
        
        Args:
            anchor_lat: WGS84 latitude of anchor point
            anchor_lon: WGS84 longitude of anchor point
            rotation_degrees: Rotation angle in degrees (clockwise from North)
            scale_factor: Uniform scaling factor
            center_surface: Whether to center the surface at the anchor point (default: True)
        """
        # Validate parameters
        if not isinstance(anchor_lat, (int, float)) or np.isnan(anchor_lat) or np.isinf(anchor_lat):
            raise ValueError("Anchor latitude must be a finite number")
        if not isinstance(anchor_lon, (int, float)) or np.isnan(anchor_lon) or np.isinf(anchor_lon):
            raise ValueError("Anchor longitude must be a finite number")
        if not isinstance(rotation_degrees, (int, float)) or np.isnan(rotation_degrees) or np.isinf(rotation_degrees):
            raise ValueError("Rotation degrees must be a finite number")
        if not isinstance(scale_factor, (int, float)) or np.isnan(scale_factor) or np.isinf(scale_factor):
            raise ValueError("Scale factor must be a finite number")
        
        if anchor_lat < -90 or anchor_lat > 90:
            raise ValueError("Anchor latitude must be between -90 and 90 degrees")
        if anchor_lon < -180 or anchor_lon > 180:
            raise ValueError("Anchor longitude must be between -180 and 180 degrees")
        if scale_factor < 0:
            raise ValueError("Scale factor must be non-negative")
        
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon
        self.rotation_degrees = rotation_degrees
        self.scale_factor = scale_factor
        self.center_surface = center_surface
        
        # Initialize coordinate systems
        self.wgs84 = pyproj.CRS.from_epsg(4326)
        self.utm_zone = self._determine_utm_zone()
        self.utm_crs = self._create_utm_crs()
        
        # Create transformers
        self.wgs84_to_utm_transformer = pyproj.Transformer.from_crs(
            self.wgs84, self.utm_crs, always_xy=True
        )
        self.utm_to_wgs84_transformer = pyproj.Transformer.from_crs(
            self.utm_crs, self.wgs84, always_xy=True
        )
        
        # Get anchor point in UTM coordinates
        self.anchor_utm_x, self.anchor_utm_y = self.wgs84_to_utm_transformer.transform(
            self.anchor_lon, self.anchor_lat
        )
        
        # Create transformation matrices
        self.rotation_matrix = self._get_rotation_matrix_z(rotation_degrees)
        self.scaling_matrix = self._get_scaling_matrix(scale_factor)
        
        # Combined transformation matrix (scale then rotate)
        self.combined_matrix = np.dot(self.rotation_matrix, self.scaling_matrix)
        
        # Initialize surface center storage for inverse transformation
        self._surface_center = None
    
    def _determine_utm_zone(self) -> str:
        """Determine UTM zone for the anchor point"""
        zone_number = int((self.anchor_lon + 180) / 6) + 1
        hemisphere = "N" if self.anchor_lat >= 0 else "S"
        return f"EPSG:32{6 if hemisphere == 'N' else 7}{zone_number:02d}"
    
    def _create_utm_crs(self) -> pyproj.CRS:
        """Create UTM CRS for the anchor point"""
        zone_number = int((self.anchor_lon + 180) / 6) + 1
        hemisphere = "N" if self.anchor_lat >= 0 else "S"
        try:
            return pyproj.CRS(f"EPSG:32{6 if hemisphere == 'N' else 7}{zone_number:02d}")
        except (TypeError, AttributeError):
            # Fallback to proj string
            return pyproj.CRS(f"+proj=utm +zone={zone_number} +{'north' if hemisphere == 'N' else 'south'} +datum=WGS84")
    
    def _get_rotation_matrix_z(self, angle_degrees: float) -> np.ndarray:
        """Get 3D rotation matrix around Z-axis"""
        angle_rad = np.radians(-angle_degrees)  # Negative for clockwise rotation
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        return np.array([
            [cos_a, -sin_a, 0],
            [sin_a,  cos_a, 0],
            [0,      0,    1]
        ])
    
    def _get_scaling_matrix(self, scale_factor: float) -> np.ndarray:
        """Get 3D uniform scaling matrix"""
        return np.array([
            [scale_factor, 0, 0],
            [0, scale_factor, 0],
            [0, 0, scale_factor]
        ])
    
    def transform_to_utm(self, points: np.ndarray) -> np.ndarray:
        """
        Transform local PLY coordinates to UTM coordinates
        
        Args:
            points: Nx3 numpy array of local coordinates (x, y, z)
            
        Returns:
            Nx3 numpy array of UTM coordinates (x, y, z)
        """
        # Validate input
        if points is None:
            raise ValueError("Points array cannot be None")
        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a 2D array with 3 columns (Nx3)")
        
        # Handle empty array
        if points.shape[0] == 0:
            return points.copy()
        
        # --- FEET TO METER CONVERSION ---
        FEET_TO_METER = 0.3048
        points = points * FEET_TO_METER
        
        # Store original points for potential inverse transformation
        original_points = points.copy()
        
        # Apply surface centering if enabled
        if self.center_surface:
            # Calculate surface center
            surface_center = np.mean(points, axis=0)
            # Center the surface at origin
            points = points - surface_center
            # Store surface center for inverse transformation
            self._surface_center = surface_center
        else:
            # For backward compatibility, don't center
            self._surface_center = None
        
        # Apply local transformations (scale then rotate)
        transformed_points = np.dot(points, self.combined_matrix.T)
        
        # Translate to UTM coordinates
        utm_points = transformed_points.copy()
        utm_points[:, 0] += self.anchor_utm_x
        utm_points[:, 1] += self.anchor_utm_y
        
        # For Z coordinate: if surface centering was enabled, restore the original Z coordinates
        # from the surface center, otherwise keep as is
        if self.center_surface and self._surface_center is not None:
            # Add back the original Z coordinate from the surface center
            utm_points[:, 2] += self._surface_center[2]
        
        return utm_points
    
    def transform_local_only(self, points: np.ndarray) -> np.ndarray:
        """
        Apply only local transformations (scaling and rotation) without UTM translation
        
        Args:
            points: Nx3 numpy array of local coordinates
            
        Returns:
            Nx3 numpy array of transformed local coordinates
        """
        # Validate input
        if points is None:
            raise ValueError("Points array cannot be None")
        if not isinstance(points, np.ndarray):
            raise ValueError("Points must be a numpy array")
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a 2D array with 3 columns (Nx3)")
        
        # Handle empty array
        if points.shape[0] == 0:
            return points.copy()
        
        # Apply local transformations only
        return np.dot(points, self.combined_matrix.T)
    
    def inverse_transform(self, utm_points: np.ndarray) -> np.ndarray:
        """
        Transform UTM coordinates back to local PLY coordinates
        
        Args:
            utm_points: Nx3 numpy array of UTM coordinates (x, y, z)
            
        Returns:
            Nx3 numpy array of local coordinates (x, y, z)
        """
        # Validate input
        if utm_points is None:
            raise ValueError("UTM points array cannot be None")
        if not isinstance(utm_points, np.ndarray):
            raise ValueError("UTM points must be a numpy array")
        if utm_points.ndim != 2 or utm_points.shape[1] != 3:
            raise ValueError("UTM points must be a 2D array with 3 columns (Nx3)")
        
        # Handle empty array
        if utm_points.shape[0] == 0:
            return utm_points.copy()
        
        # Remove UTM translation
        local_transformed = utm_points.copy()
        local_transformed[:, 0] -= self.anchor_utm_x
        local_transformed[:, 1] -= self.anchor_utm_y
        
        # For Z coordinate: if surface centering was enabled, remove the surface center Z coordinate
        if self.center_surface and self._surface_center is not None:
            local_transformed[:, 2] -= self._surface_center[2]
        
        # Apply inverse local transformations
        inverse_matrix = np.linalg.inv(self.combined_matrix)
        original_points = np.dot(local_transformed, inverse_matrix.T)
        
        # Restore surface center if it was centered during transformation
        if self.center_surface and self._surface_center is not None:
            original_points = original_points + self._surface_center
        
        return original_points
    
    def get_transformation_metadata(self) -> Dict[str, Any]:
        """
        Get metadata about the transformation pipeline
        
        Returns:
            Dictionary containing transformation parameters and metadata
        """
        metadata = {
            'anchor_lat': self.anchor_lat,
            'anchor_lon': self.anchor_lon,
            'rotation_degrees': self.rotation_degrees,
            'scale_factor': self.scale_factor,
            'center_surface': self.center_surface,
            'utm_zone': self.utm_zone,
            'anchor_utm_x': self.anchor_utm_x,
            'anchor_utm_y': self.anchor_utm_y,
            'transformation_order': ['center', 'scale', 'rotate', 'translate'] if self.center_surface else ['scale', 'rotate', 'translate'],
            'coordinate_systems': {
                'source': 'local_ply',
                'target': 'utm',
                'wgs84_epsg': 'EPSG:4326',
                'utm_epsg': self.utm_zone
            }
        }
        
        # Add surface center information if available
        if self._surface_center is not None:
            metadata['surface_center'] = self._surface_center.tolist()
        
        return metadata


class CoordinateTransformer:
    """
    Handles coordinate transformations using pyproj
    """
    
    def __init__(self):
        try:
            # Try the newer pyproj API first
            self.wgs84 = pyproj.CRS.from_epsg(4326)
        except (TypeError, AttributeError):
            try:
                # Fallback to older pyproj API
                self.wgs84 = pyproj.CRS("EPSG:4326")
            except Exception as e:
                # Final fallback
                self.wgs84 = pyproj.CRS("+proj=longlat +datum=WGS84 +no_defs")
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
            Tuple of (utm_x, utm_y) in meters
        """
        # Validate input coordinates
        if not (isinstance(latitude, (int, float)) or np.issubdtype(type(latitude), np.number)) or \
           not (isinstance(longitude, (int, float)) or np.issubdtype(type(longitude), np.number)):
            raise ValueError(f"Coordinates must be numeric, got {type(latitude)}, {type(longitude)}")
        
        if not np.isfinite(latitude) or not np.isfinite(longitude):
            raise ValueError(f"Coordinates must be finite, got {latitude}, {longitude}")
        
        if not (-90 <= latitude <= 90):
            raise ValueError(f"Latitude must be between -90 and 90 degrees, got {latitude}")
        
        if not (-180 <= longitude <= 180):
            raise ValueError(f"Longitude must be between -180 and 180 degrees, got {longitude}")
        
        # Determine UTM zone and hemisphere
        utm_zone = self.determine_utm_zone_with_hemisphere(latitude, longitude)
        
        # Create UTM CRS
        utm_epsg = f"EPSG:32{6 if utm_zone.endswith('N') else 7}{int(utm_zone[:-1]):02d}"
        try:
            utm_crs = pyproj.CRS(utm_epsg)
        except (TypeError, AttributeError):
            # Fallback to proj string
            zone_number = int(utm_zone[:-1])
            hemisphere = utm_zone[-1]
            utm_crs = pyproj.CRS(f"+proj=utm +zone={zone_number} +{'north' if hemisphere == 'N' else 'south'} +datum=WGS84")
        
        # Create transformer
        transformer = pyproj.Transformer.from_crs(self.wgs84, utm_crs, always_xy=True)
        
        # Transform coordinates
        utm_x, utm_y = transformer.transform(longitude, latitude)
        
        return utm_x, utm_y

    def transform_utm_to_wgs84(self, utm_x: float, utm_y: float, utm_zone: str) -> Tuple[float, float]:
        """
        Transform UTM coordinates to WGS84 coordinates
        
        Args:
            utm_x: UTM X coordinate in meters
            utm_y: UTM Y coordinate in meters
            utm_zone: UTM zone string (e.g., "18N")
            
        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        # Validate input coordinates
        if not (isinstance(utm_x, (int, float)) or np.issubdtype(type(utm_x), np.number)) or \
           not (isinstance(utm_y, (int, float)) or np.issubdtype(type(utm_y), np.number)):
            raise ValueError(f"Coordinates must be numeric, got {type(utm_x)}, {type(utm_y)}")
        
        if not np.isfinite(utm_x) or not np.isfinite(utm_y):
            raise ValueError(f"Coordinates must be finite, got {utm_x}, {utm_y}")
        
        # Validate UTM zone format
        if not isinstance(utm_zone, str) or len(utm_zone) < 2:
            raise ValueError(f"UTM zone must be a string like '18N', got {utm_zone}")
        
        # Create UTM CRS
        utm_epsg = f"EPSG:32{6 if utm_zone.endswith('N') else 7}{int(utm_zone[:-1]):02d}"
        try:
            utm_crs = pyproj.CRS(utm_epsg)
        except (TypeError, AttributeError):
            # Fallback to proj string
            zone_number = int(utm_zone[:-1])
            hemisphere = utm_zone[-1]
            utm_crs = pyproj.CRS(f"+proj=utm +zone={zone_number} +{'north' if hemisphere == 'N' else 'south'} +datum=WGS84")
        
        # Create transformer (reverse direction)
        transformer = pyproj.Transformer.from_crs(utm_crs, self.wgs84, always_xy=True)
        
        # Transform coordinates
        longitude, latitude = transformer.transform(utm_x, utm_y)
        
        return latitude, longitude
    
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
            try:
                utm_crs = pyproj.CRS(utm_epsg)
            except (TypeError, AttributeError):
                # Extract zone and hemisphere from EPSG code
                zone_number = int(utm_epsg[-2:])
                hemisphere = "N" if utm_epsg[-3] == "6" else "S"
                utm_crs = pyproj.CRS(f"+proj=utm +zone={zone_number} +{'north' if hemisphere == 'N' else 'south'} +datum=WGS84")
            
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
        
        Args:
            wgs84_boundary: List of 4 WGS84 coordinate pairs [(lat1, lon1), (lat2, lon2), (lat3, lon3), (lat4, lon4)]
            
        Returns:
            List of 4 UTM coordinate pairs [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        if len(wgs84_boundary) != 4:
            raise ValueError("Boundary must have exactly 4 coordinate pairs")
        
        # Transform each coordinate pair from WGS84 to UTM
        utm_boundary = []
        for lat, lon in wgs84_boundary:
            utm_x, utm_y = self.transform_wgs84_to_utm(lat, lon)
            utm_boundary.append((utm_x, utm_y))
        
        return utm_boundary
    
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
    
    def create_transformation_pipeline(
        self, 
        anchor_lat: float, 
        anchor_lon: float, 
        rotation_degrees: float, 
        scale_factor: float,
        center_surface: bool = True
    ) -> TransformationPipeline:
        """
        Create a complete transformation pipeline
        
        Args:
            anchor_lat: WGS84 latitude of anchor point
            anchor_lon: WGS84 longitude of anchor point
            rotation_degrees: Rotation angle in degrees (clockwise from North)
            scale_factor: Uniform scaling factor
            center_surface: Whether to center the surface at the anchor point (default: True)
            
        Returns:
            TransformationPipeline instance
        """
        return TransformationPipeline(
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            rotation_degrees=rotation_degrees,
            scale_factor=scale_factor,
            center_surface=center_surface
        )


class SurfaceAlignment:
    """
    Surface alignment and registration algorithms (reference point, ICP, outlier rejection)
    """
    def align_surfaces(self, source: np.ndarray, target: np.ndarray, method='icp', return_metrics=False, reject_outliers=False):
        if method == 'point':
            src_centroid = np.mean(source, axis=0)
            tgt_centroid = np.mean(target, axis=0)
            offset = tgt_centroid - src_centroid
            src_rms = np.sqrt(np.mean(np.sum((source - src_centroid)**2, axis=1)))
            tgt_rms = np.sqrt(np.mean(np.sum((target - tgt_centroid)**2, axis=1)))
            scale = tgt_rms / src_rms if src_rms > 0 else 1.0
            rotation = 0.0
            result = {'rotation': rotation, 'scale': scale, 'offset': offset}
            if return_metrics:
                result['rmse'] = np.sqrt(np.mean(np.sum((source * scale + offset - target)**2, axis=1)))
                result['inlier_ratio'] = 1.0
            return result
        elif method == 'icp':
            src = source.copy()
            tgt = target.copy()
            max_iter = 50
            tol = 1e-6
            prev_error = None
            mask = np.ones(src.shape[0], dtype=bool)
            
            for i in range(max_iter):
                tree = cKDTree(tgt)
                dists, idx = tree.query(src)
                tgt_corr = tgt[idx]
                
                if reject_outliers:
                    # Use a more robust outlier rejection based on distance percentiles
                    threshold = np.percentile(dists, 85)  # Keep 85% of points
                    mask = dists < threshold
                    # Ensure we have enough points for alignment
                    if np.sum(mask) < 10:
                        mask = dists < np.percentile(dists, 95)
                else:
                    mask = np.ones(src.shape[0], dtype=bool)
                
                src_corr = src[mask]
                tgt_corr = tgt_corr[mask]
                
                if len(src_corr) < 3:
                    # Fallback to simple centroid alignment if too few points
                    src_centroid = np.mean(source, axis=0)
                    tgt_centroid = np.mean(target, axis=0)
                    offset = tgt_centroid - src_centroid
                    return {'rotation': 0.0, 'scale': 1.0, 'offset': offset}
                
                src_xy = src_corr[:, :2]
                tgt_xy = tgt_corr[:, :2]
                src_centroid = np.mean(src_xy, axis=0)
                tgt_centroid = np.mean(tgt_xy, axis=0)
                src_centered = src_xy - src_centroid
                tgt_centered = tgt_xy - tgt_centroid
                src_norm = np.sqrt(np.sum(src_centered**2))
                tgt_norm = np.sqrt(np.sum(tgt_centered**2))
                scale = tgt_norm / src_norm if src_norm > 0 else 1.0
                H = (src_centered * scale).T @ tgt_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[1, :] *= -1
                    R = Vt.T @ U.T
                # Apply transform to all src points
                src_xy_full = src[:, :2] - src_centroid
                src_xy_full = (src_xy_full * scale) @ R.T + tgt_centroid
                src_new = src.copy()
                src_new[:, :2] = src_xy_full
                z_offset = np.mean(tgt_corr[:, 2] - src_new[mask, 2])
                src_new[:, 2] += z_offset
                error = np.mean(np.linalg.norm(src_new[mask] - tgt_corr, axis=1))
                if prev_error is not None and abs(prev_error - error) < tol:
                    src = src_new
                    break
                prev_error = error
                src = src_new
            
            # After convergence, compute best-fit similarity from original to target
            # Use only inlier points for final transformation calculation
            if reject_outliers:
                # Recompute distances with final aligned source
                tree = cKDTree(tgt)
                dists, idx = tree.query(src)
                threshold = np.percentile(dists, 85)
                final_mask = dists < threshold
                if np.sum(final_mask) < 10:
                    final_mask = dists < np.percentile(dists, 95)
                
                src_inliers = source[final_mask]
                tgt_inliers = tgt[final_mask]
            else:
                src_inliers = source
                tgt_inliers = target
            
            # Compute final transformation using only inliers
            src0_xy = src_inliers[:, :2]
            tgt_xy = tgt_inliers[:, :2]
            
            # Step 1: Fit translation (centroid difference)
            src0_centroid = np.mean(src0_xy, axis=0)
            tgt_centroid = np.mean(tgt_xy, axis=0)
            translation = tgt_centroid - src0_centroid
            
            # Step 2: Fit rotation (after removing translation)
            src0_centered = src0_xy - src0_centroid
            tgt_centered = tgt_xy - tgt_centroid
            H = src0_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R_final = Vt.T @ U.T
            if np.linalg.det(R_final) < 0:
                Vt[1, :] *= -1
                R_final = Vt.T @ U.T
            angle_rad = np.arctan2(R_final[1, 0], R_final[0, 0])
            rotation_final = np.degrees(angle_rad)
            
            # Step 3: Fit scale (after rotation)
            src0_rotated = src0_centered @ R_final.T
            src0_norm = np.sqrt(np.sum(src0_rotated**2))
            tgt_norm = np.sqrt(np.sum(tgt_centered**2))
            scale_final = tgt_norm / src0_norm if src0_norm > 0 else 1.0
            
            # Final offset combines translation and any remaining difference
            xy_offset = translation
            z_offset = np.mean(tgt_inliers[:, 2] - src_inliers[:, 2])
            offset_final = np.array([xy_offset[0], xy_offset[1], z_offset])
            
            result = {'rotation': rotation_final, 'scale': scale_final, 'offset': offset_final}
            if return_metrics:
                if reject_outliers:
                    rmse = np.sqrt(np.mean(np.sum((src[final_mask] - tgt_inliers) ** 2, axis=1)))
                    inlier_ratio = float(np.sum(final_mask)) / len(source)
                else:
                    rmse = np.sqrt(np.mean(np.sum((src - tgt) ** 2, axis=1)))
                    inlier_ratio = 1.0
                result['rmse'] = rmse
                result['inlier_ratio'] = inlier_ratio
            return result
        else:
            raise ValueError(f"Unknown alignment method: {method}") 