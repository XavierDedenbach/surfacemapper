"""
Volume and thickness calculation service using PyVista algorithms
"""
import numpy as np
import pyvista as pv
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class VolumeResult:
    """Volume calculation result"""
    volume_cubic_yards: float
    confidence_interval: Tuple[float, float]
    uncertainty: float

@dataclass
class ThicknessResult:
    """Thickness calculation result"""
    average_thickness_feet: float
    min_thickness_feet: float
    max_thickness_feet: float
    std_dev_thickness_feet: float
    confidence_interval: Tuple[float, float]

class VolumeCalculator:
    """
    Handles volume and thickness calculations using PyVista
    """
    
    def __init__(self):
        self.tolerance = 0.01  # 1% tolerance for cross-validation
    
    def calculate_volume_difference(self, surface1_data: Dict, surface2_data: Dict) -> VolumeResult:
        """
        Calculate volume difference between two surfaces using PyVista Delaunay triangulation
        """
        try:
            surface1 = surface1_data.get('vertices')
            surface2 = surface2_data.get('vertices')

            if surface1 is None or surface2 is None:
                raise ValueError("Surface data is missing 'vertices'.")

            # Defer to the more robust calculation method
            volume_diff_cubic_meters = calculate_volume_between_surfaces(
                surface1,
                surface2,
                method="pyvista"
            )
            
            # Convert from cubic meters to cubic yards
            # 1 cubic meter = 1.30795 cubic yards
            volume_diff_cubic_yards = volume_diff_cubic_meters * 1.30795
            
            # Calculate confidence interval (simplified for now)
            uncertainty = volume_diff_cubic_yards * 0.05  # 5% uncertainty
            confidence_interval = (
                volume_diff_cubic_yards - uncertainty,
                volume_diff_cubic_yards + uncertainty
            )
            
            return VolumeResult(
                volume_cubic_yards=volume_diff_cubic_yards,
                confidence_interval=confidence_interval,
                uncertainty=uncertainty
            )
        except Exception as e:
            logger.error(f"Volume calculation failed: {e}", exc_info=True)
            # Return zero volume if calculation fails
            return VolumeResult(
                volume_cubic_yards=0.0,
                confidence_interval=(0.0, 0.0),
                uncertainty=0.0
            )
    
    def calculate_layer_thickness(self, upper_surface: np.ndarray, lower_surface: np.ndarray) -> ThicknessResult:
        """
        Calculate layer thickness between two surfaces using PyVista
        """
        try:
            # Convert numpy arrays to PyVista point clouds
            pc_upper = pv.PolyData(upper_surface)
            pc_lower = pv.PolyData(lower_surface)
            
            # Create meshes using PyVista's triangulation
            mesh_upper = pc_upper.delaunay_2d()
            mesh_lower = pc_lower.delaunay_2d()
            
            # Calculate thickness as Z-difference between corresponding points
            # For simplicity, we'll use the average Z-difference
            z_upper = upper_surface[:, 2]
            z_lower = lower_surface[:, 2]
            
            thicknesses = z_upper - z_lower
            thicknesses = thicknesses[thicknesses > 0]  # Only positive thicknesses
            
            if thicknesses is None or thicknesses.size == 0:
                return ThicknessResult(
                    average_thickness_feet=0.0,
                    min_thickness_feet=0.0,
                    max_thickness_feet=0.0,
                    std_dev_thickness_feet=0.0,
                    confidence_interval=(0.0, 0.0)
                )
            
            # Convert from meters to feet (UTM coordinates are in meters)
            # 1 meter = 3.28084 feet
            METERS_TO_FEET = 3.28084
            thicknesses_feet = thicknesses * METERS_TO_FEET
            
            average_thickness = np.mean(thicknesses_feet)
            min_thickness = np.min(thicknesses_feet)
            max_thickness = np.max(thicknesses_feet)
            
            # Calculate confidence interval
            std_thickness = np.std(thicknesses_feet)
            confidence_interval = (
                average_thickness - std_thickness,
                average_thickness + std_thickness
            )
            
            return ThicknessResult(
                average_thickness_feet=average_thickness,
                min_thickness_feet=min_thickness,
                max_thickness_feet=max_thickness,
                std_dev_thickness_feet=std_thickness,
                confidence_interval=confidence_interval
            )
        except Exception as e:
            # Return zero thickness if calculation fails
            return ThicknessResult(
                average_thickness_feet=0.0,
                min_thickness_feet=0.0,
                max_thickness_feet=0.0,
                std_dev_thickness_feet=0.0,
                confidence_interval=(0.0, 0.0)
            )
    
    def calculate_compaction_rate(self, volume_cubic_yards: float, tonnage: float) -> float:
        """
        Calculate compaction rate in lbs/cubic yard
        """
        if volume_cubic_yards <= 0:
            return 0.0
        return (tonnage * 2000) / volume_cubic_yards 

def _validate_utm_coordinates(surface_points: np.ndarray, context: str = ""):
    """
    Validate that surface points are in UTM (meters). If points appear to be in WGS84 (degrees), log a warning.
    """
    if surface_points is None or len(surface_points) == 0:
        return
    x = surface_points[:, 0]
    y = surface_points[:, 1]
    # If all x in [-180, 180] and all y in [-90, 90], likely WGS84
    if np.all(np.abs(x) <= 180) and np.all(np.abs(y) <= 90):
        logger.warning(f"{context}: Input coordinates appear to be in WGS84 degrees, not UTM meters. Calculations may be invalid. All mesh operations require UTM.")

def calculate_volume_between_surfaces(
    bottom_surface: np.ndarray, 
    top_surface: np.ndarray,
    method: str = "pyvista"
) -> float:
    """
    Calculate volume between two surfaces using PyVista mesh operations in UTM coordinates.
    Warn if coordinates appear to be in WGS84 (degrees).
    Args:
        bottom_surface: 3D points representing bottom surface [N, 3] in UTM (meters)
        top_surface: 3D points representing top surface [M, 3] in UTM (meters)
        method: Calculation method ("pyvista" or "prism")
    Returns:
        Volume in cubic meters (UTM coordinates are in meters)
    Raises:
        ValueError: If surfaces are invalid
    """
    _validate_utm_coordinates(bottom_surface, context="calculate_volume_between_surfaces: bottom_surface")
    _validate_utm_coordinates(top_surface, context="calculate_volume_between_surfaces: top_surface")
    if bottom_surface is None or bottom_surface.size == 0 or top_surface is None or top_surface.size == 0:
        raise ValueError("Bottom and top surfaces must be non-empty numpy arrays")
    if len(bottom_surface) < 3 or len(top_surface) < 3:
        logger.warning("Surfaces have fewer than 3 points, returning 0 volume")
        return 0.0
    
    # Handle different numbers of points by using interpolation
    if len(bottom_surface) != len(top_surface):
        logger.info(f"Surfaces have different point counts ({len(bottom_surface)} vs {len(top_surface)}), using interpolation")
        return _calculate_volume_with_interpolation(bottom_surface, top_surface, method)
    
    if method == "pyvista":
        return _calculate_volume_pyvista(bottom_surface, top_surface)
    elif method == "prism":
        return _calculate_volume_prism_method(bottom_surface, top_surface)
    else:
        raise ValueError(f"Unknown method: {method}")


def _calculate_volume_with_interpolation(bottom_surface: np.ndarray, top_surface: np.ndarray, method: str) -> float:
    """
    Calculate volume between surfaces with different point counts using interpolation.
    
    Args:
        bottom_surface: 3D points representing bottom surface [N, 3]
        top_surface: 3D points representing top surface [M, 3]
        method: Calculation method
        
    Returns:
        Volume in cubic units
    """
    try:
        # Use the surface with more points as the reference for triangulation
        if len(bottom_surface) >= len(top_surface):
            reference_surface = bottom_surface
            other_surface = top_surface
            reference_is_bottom = True
        else:
            reference_surface = top_surface
            other_surface = bottom_surface
            reference_is_bottom = False
        
        # Create triangulation from reference surface
        cloud = pv.PolyData(reference_surface.astype(np.float32))
        mesh = cloud.delaunay_2d()
        
        if mesh.n_cells == 0:
            logger.warning("Triangulation failed, falling back to prism method")
            return _calculate_volume_prism_method(bottom_surface, top_surface)
        
        total_volume = 0.0
        
        # For each triangle in the reference surface
        for i in range(mesh.n_cells):
            triangle = mesh.get_cell(i)
            triangle_points = triangle.points
            
            # Get Z values for this triangle from both surfaces
            triangle_z_bottom = np.zeros(3)
            triangle_z_top = np.zeros(3)
            
            for j, point in enumerate(triangle_points):
                # Find closest point in bottom surface
                distances_bottom = np.sqrt(
                    (bottom_surface[:, 0] - point[0])**2 + 
                    (bottom_surface[:, 1] - point[1])**2
                )
                closest_bottom_idx = np.argmin(distances_bottom)
                triangle_z_bottom[j] = bottom_surface[closest_bottom_idx, 2]
                
                # Find closest point in top surface
                distances_top = np.sqrt(
                    (top_surface[:, 0] - point[0])**2 + 
                    (top_surface[:, 1] - point[1])**2
                )
                closest_top_idx = np.argmin(distances_top)
                triangle_z_top[j] = top_surface[closest_top_idx, 2]
            
            # Calculate triangle area
            v1 = triangle_points[1] - triangle_points[0]
            v2 = triangle_points[2] - triangle_points[0]
            cross_product = np.cross(v1, v2)
            triangle_area = 0.5 * np.linalg.norm(cross_product)
            
            # Calculate average thickness at triangle vertices
            avg_thickness = np.mean(np.abs(triangle_z_top - triangle_z_bottom))
            
            # Volume = area * average thickness
            triangle_volume = triangle_area * avg_thickness
            total_volume += triangle_volume
        
        return total_volume
        
    except Exception as e:
        logger.error(f"Interpolation-based volume calculation failed: {e}")
        return _calculate_volume_prism_method(bottom_surface, top_surface)


def _calculate_volume_pyvista(bottom_surface: np.ndarray, top_surface: np.ndarray) -> float:
    """
    Calculate volume using a triangle-based prism method via PyVista.
    This is more accurate than the simple prism method as it correctly
    calculates the area of the base of each prism.
    
    Args:
        bottom_surface: 3D points representing bottom surface [N, 3]
        top_surface: 3D points representing top surface [N, 3]
        
    Returns:
        Volume in cubic units
    """
    try:
        # Check for pyramid-like cases first (all top points at same X,Y location)
        if _is_pyramid_case(bottom_surface, top_surface):
            logger.info("Pyramid case detected, using analytical calculation")
            return _calculate_pyramid_volume(bottom_surface, top_surface)
        
        # Check for other degenerate cases
        if _is_degenerate_surface(bottom_surface) or _is_degenerate_surface(top_surface):
            logger.info("Degenerate surface detected, falling back to simple prism method")
            return _calculate_volume_prism_method(bottom_surface, top_surface)
        
        # This is the core of the improved method.
        return _calculate_volume_triangle_based(bottom_surface, top_surface)
        
    except Exception as e:
        logger.error(f"PyVista volume calculation failed: {e}", exc_info=True)
        logger.info("Falling back to simple prism method")
        return _calculate_volume_prism_method(bottom_surface, top_surface)


def _is_degenerate_surface(surface_points: np.ndarray) -> bool:
    """
    Check if surface is degenerate (all points at same X,Y location).
    
    Args:
        surface_points: 3D points representing surface
        
    Returns:
        True if surface is degenerate
    """
    if len(surface_points) < 3:
        return True
    
    # Check if all X,Y coordinates are the same
    x_unique = np.unique(surface_points[:, 0])
    y_unique = np.unique(surface_points[:, 1])
    
    return len(x_unique) == 1 and len(y_unique) == 1


def _is_pyramid_case(bottom_surface: np.ndarray, top_surface: np.ndarray) -> bool:
    """
    Check if this is a pyramid case (all top points at same X,Y location).
    
    Args:
        bottom_surface: 3D points representing bottom surface
        top_surface: 3D points representing top surface
        
    Returns:
        True if this is a pyramid case
    """
    # Check if all top surface points have the same X,Y coordinates
    x_unique = np.unique(top_surface[:, 0])
    y_unique = np.unique(top_surface[:, 1])
    
    is_pyramid = len(x_unique) == 1 and len(y_unique) == 1
    
    if is_pyramid:
        logger.info(f"Pyramid case detected: apex at ({x_unique[0]}, {y_unique[0]})")
    else:
        logger.debug(f"Not a pyramid case: {len(x_unique)} unique X values, {len(y_unique)} unique Y values")
    
    return is_pyramid


def _calculate_pyramid_volume(bottom_surface: np.ndarray, top_surface: np.ndarray) -> float:
    """
    Calculate volume for pyramid case using analytical formula.
    
    Args:
        bottom_surface: 3D points representing bottom surface
        top_surface: 3D points representing top surface
        
    Returns:
        Volume in cubic units
    """
    # Calculate base area using triangulation
    try:
        cloud = pv.PolyData(bottom_surface.astype(np.float32))
        mesh = cloud.delaunay_2d()
        
        if mesh.n_cells == 0:
            # Fallback to bounding box area
            x_range = np.max(bottom_surface[:, 0]) - np.min(bottom_surface[:, 0])
            y_range = np.max(bottom_surface[:, 1]) - np.min(bottom_surface[:, 1])
            base_area = x_range * y_range
        else:
            # Calculate total area of triangulated base
            base_area = mesh.compute_cell_sizes()['Area'].sum()
        
        # Calculate height (distance from base to apex)
        apex_z = top_surface[0, 2]  # All top points have same Z
        base_z = np.mean(bottom_surface[:, 2])  # Average base Z
        height = abs(apex_z - base_z)
        
        # Pyramid volume = (1/3) * base_area * height
        volume = (1/3) * base_area * height
        
        return volume
        
    except Exception as e:
        logger.warning(f"Pyramid volume calculation failed: {e}")
        return _calculate_volume_prism_method(bottom_surface, top_surface)


def _calculate_volume_triangle_based(bottom_surface: np.ndarray, top_surface: np.ndarray) -> float:
    """
    Calculate volume using triangle-based method for regular grids.
    
    Args:
        bottom_surface: 3D points representing bottom surface [N, 3]
        top_surface: 3D points representing top surface [N, 3]
        
    Returns:
        Volume in cubic units
    """
    try:
        # Create PyVista point cloud, clean it, and triangulate
        cloud = pv.PolyData(bottom_surface.astype(np.float32)).clean()
        mesh = cloud.delaunay_2d()
        
        if mesh.n_cells == 0:
            # Fallback to simple prism method
            logger.warning("Triangle-based method failed: Delaunay triangulation resulted in 0 cells. Falling back to simple prism method.")
            return _calculate_volume_prism_method(bottom_surface, top_surface)
        
        total_volume = 0.0
        
        # For each triangle, calculate the prism volume
        for i in range(mesh.n_cells):
            # Get triangle vertices
            triangle = mesh.get_cell(i)
            triangle_points = triangle.points
            
            # Find corresponding points in the surface arrays
            bottom_triangle_points = np.zeros_like(triangle_points)
            top_triangle_points = np.zeros_like(triangle_points)
            
            for j, point in enumerate(triangle_points):
                # Find closest point in bottom surface
                distances = np.sqrt(
                    (bottom_surface[:, 0] - point[0])**2 + 
                    (bottom_surface[:, 1] - point[1])**2
                )
                closest_idx = np.argmin(distances)
                bottom_triangle_points[j] = bottom_surface[closest_idx]
                top_triangle_points[j] = top_surface[closest_idx]
            
            # Calculate triangle area
            v1 = bottom_triangle_points[1] - bottom_triangle_points[0]
            v2 = bottom_triangle_points[2] - bottom_triangle_points[0]
            cross_product = np.cross(v1, v2)
            triangle_area = 0.5 * np.linalg.norm(cross_product)
            
            # Calculate average thickness at triangle vertices
            avg_thickness = np.mean(np.abs(
                top_triangle_points[:, 2] - bottom_triangle_points[:, 2]
            ))
            
            # Volume = area * average thickness
            triangle_volume = triangle_area * avg_thickness
            total_volume += triangle_volume
        
        return total_volume
        
    except Exception as e:
        logger.warning(f"Triangle-based volume calculation failed: {e}")
        return _calculate_volume_prism_method(bottom_surface, top_surface)


def _is_regular_grid(surface_points: np.ndarray) -> bool:
    """
    Check if surface points form a regular grid.
    
    Args:
        surface_points: 3D points representing surface
        
    Returns:
        True if points form a regular grid
    """
    if len(surface_points) < 4:
        return False
    
    # Extract X and Y coordinates
    x_coords = surface_points[:, 0]
    y_coords = surface_points[:, 1]
    
    # Check if X and Y coordinates are evenly spaced
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    
    if len(x_unique) < 2 or len(y_unique) < 2:
        return False
    
    # Check if spacing is uniform
    x_spacing = np.diff(x_unique)
    y_spacing = np.diff(y_unique)
    
    x_uniform = np.allclose(x_spacing, x_spacing[0], atol=1e-10)
    y_uniform = np.allclose(y_spacing, y_spacing[0], atol=1e-10)
    
    return x_uniform and y_uniform


def _calculate_regular_grid_area_per_point(surface_points: np.ndarray) -> float:
    """
    Calculate area per point for regular grid surfaces.
    
    Args:
        surface_points: 3D points representing regular grid surface
        
    Returns:
        Area per point
    """
    x_coords = surface_points[:, 0]
    y_coords = surface_points[:, 1]
    
    x_unique = np.unique(x_coords)
    y_unique = np.unique(y_coords)
    
    # Calculate spacing
    x_spacing = x_unique[1] - x_unique[0] if len(x_unique) > 1 else 1.0
    y_spacing = y_unique[1] - y_unique[0] if len(y_unique) > 1 else 1.0
    
    return x_spacing * y_spacing


def _calculate_irregular_grid_area_per_point(surface_points: np.ndarray) -> float:
    """
    Calculate area per point for irregular grid surfaces using triangulation.
    
    Args:
        surface_points: 3D points representing irregular surface
        
    Returns:
        Area per point
    """
    try:
        # Create PyVista point cloud and triangulate
        cloud = pv.PolyData(surface_points.astype(np.float32))
        mesh = cloud.delaunay_2d()
        
        if mesh.n_cells == 0:
            # Fallback to bounding box method
            return _calculate_bounding_box_area_per_point(surface_points)
        
        # Calculate total area of triangulated surface
        total_area = mesh.compute_cell_sizes()['Area'].sum()
        
        # Distribute area evenly among points
        return total_area / len(surface_points)
        
    except Exception as e:
        logger.warning(f"Triangulation-based area calculation failed: {e}")
        return _calculate_bounding_box_area_per_point(surface_points)


def _calculate_bounding_box_area_per_point(surface_points: np.ndarray) -> float:
    """
    Calculate area per point using bounding box method.
    
    Args:
        surface_points: 3D points representing surface
        
    Returns:
        Area per point
    """
    x_coords = surface_points[:, 0]
    y_coords = surface_points[:, 1]
    
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    total_area = x_range * y_range
    
    return total_area / len(surface_points)


def _calculate_volume_prism_method(bottom_surface: np.ndarray, top_surface: np.ndarray) -> float:
    """
    Calculate volume using vertical prism method.
    
    Args:
        bottom_surface: 3D points representing bottom surface [N, 3]
        top_surface: 3D points representing top surface [N, 3]
        
    Returns:
        Volume in cubic units
    """
    # Calculate vertical distances (thickness) at each point
    thicknesses = np.abs(top_surface[:, 2] - bottom_surface[:, 2])
    
    # For regular grids, calculate area per point and multiply by thickness
    # This is an approximation that works well for dense, regular grids
    if len(bottom_surface) > 1:
        # Estimate area per point based on bounding box
        x_range = np.max(bottom_surface[:, 0]) - np.min(bottom_surface[:, 0])
        y_range = np.max(bottom_surface[:, 1]) - np.min(bottom_surface[:, 1])
        total_area = x_range * y_range
        area_per_point = total_area / len(bottom_surface)
        
        # Calculate volume as sum of prism volumes
        volume = np.sum(thicknesses * area_per_point)
    else:
        # Single point case
        volume = thicknesses[0] if len(thicknesses) > 0 else 0.0
    
    return volume


def create_square_grid(size_x: int, size_y: int, z: float = 0.0) -> np.ndarray:
    """
    Create a square grid of points at specified Z level.
    
    Args:
        size_x: Number of points in X direction
        size_y: Number of points in Y direction
        z: Z coordinate for all points
        
    Returns:
        Array of 3D points [N, 3]
    """
    x = np.linspace(0, size_x, size_x + 1)
    y = np.linspace(0, size_y, size_y + 1)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    return points


def create_rectangular_grid(length: int, width: int, z: float = 0.0) -> np.ndarray:
    """
    Create a rectangular grid of points at specified Z level.
    
    Args:
        length: Length of rectangle
        width: Width of rectangle
        z: Z coordinate for all points
        
    Returns:
        Array of 3D points [N, 3]
    """
    x = np.linspace(0, length, length + 1)
    y = np.linspace(0, width, width + 1)
    X, Y = np.meshgrid(x, y)
    points = np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, z)])
    return points


def create_sine_wave_surface(amplitude: float = 2.0, wavelength: float = 10.0, size: int = 20) -> np.ndarray:
    """
    Create a surface with sine wave variation.
    
    Args:
        amplitude: Amplitude of sine wave
        wavelength: Wavelength of sine wave
        size: Size of the surface grid
        
    Returns:
        Array of 3D points [N, 3]
    """
    x = np.linspace(0, size, size + 1)
    y = np.linspace(0, size, size + 1)
    X, Y = np.meshgrid(x, y)
    Z = amplitude * np.sin(2 * np.pi * X / wavelength)
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return points


def calculate_surface_area(surface_points: np.ndarray) -> float:
    """
    Calculate surface area for a set of 3D points in UTM coordinates.
    Warn if coordinates appear to be in WGS84 (degrees).
    Args:
        surface_points: 3D points [N, 3] in UTM (meters)
    Returns:
        Area in square meters
    """
    _validate_utm_coordinates(surface_points, context="calculate_surface_area")
    if len(surface_points) < 3:
        return 0.0
    
    # Simple approximation: bounding box area
    x_coords = surface_points[:, 0]
    y_coords = surface_points[:, 1]
    x_range = np.max(x_coords) - np.min(x_coords)
    y_range = np.max(y_coords) - np.min(y_coords)
    return x_range * y_range


def validate_surface_data(surface_points: np.ndarray) -> bool:
    """
    Validate surface point data.
    
    Args:
        surface_points: 3D points to validate
        
    Returns:
        True if data is valid, False otherwise
    """
    if surface_points is None or len(surface_points) == 0:
        return False
    
    if surface_points.ndim != 2 or surface_points.shape[1] != 3:
        return False
    
    if np.any(np.isnan(surface_points)) or np.any(np.isinf(surface_points)):
        return False
    
    return True


def optimize_mesh_quality(mesh: pv.PolyData, target_cells: Optional[int] = None) -> pv.PolyData:
    """
    Optimize mesh quality for volume calculation.
    
    Args:
        mesh: PyVista mesh to optimize
        target_cells: Target number of cells (None for auto)
        
    Returns:
        Optimized mesh
    """
    try:
        if target_cells is not None and mesh.n_cells > target_cells:
            # Decimate mesh to target cell count
            mesh = mesh.decimate(target=target_cells / mesh.n_cells)
        
        # Clean mesh
        mesh = mesh.clean()
        
        # Fill holes if any
        if mesh.n_holes > 0:
            mesh = mesh.fill_holes()
        
        return mesh
        
    except Exception as e:
        logger.warning(f"Mesh optimization failed: {e}")
        return mesh 

def calculate_real_surface_area(surface_points: np.ndarray) -> float:
    """
    Calculate real surface area using triangulation in UTM coordinates.
    
    Args:
        surface_points: 3D points representing surface [N, 3] in UTM (meters)
        
    Returns:
        Real surface area in square meters (UTM coordinates are in meters)
    """
    if len(surface_points) < 3:
        return 0.0
    
    try:
        # Create PyVista point cloud and triangulate
        cloud = pv.PolyData(surface_points.astype(np.float32))
        mesh = cloud.delaunay_2d()
        
        if mesh.n_cells == 0:
            # Fallback to bounding box area
            return calculate_surface_area(surface_points)
        
        # Calculate total area of triangulated surface
        total_area = mesh.compute_cell_sizes()['Area'].sum()
        
        return total_area
        
    except Exception as e:
        logger.warning(f"Real surface area calculation failed: {e}")
        return calculate_surface_area(surface_points)


def convert_cubic_feet_to_yards(volume_cubic_feet: float, allow_negative: bool = False) -> float:
    """
    Convert volume from cubic feet to cubic yards.
    
    Args:
        volume_cubic_feet: Volume in cubic feet
        allow_negative: Whether to allow negative values (default: False)
        
    Returns:
        Volume in cubic yards
        
    Raises:
        ValueError: If volume is negative and allow_negative is False
    """
    if not allow_negative and volume_cubic_feet < 0:
        raise ValueError("Volume cannot be negative")
    
    # Conversion factor: 27 cubic feet = 1 cubic yard
    return volume_cubic_feet / 27.0


def convert_cubic_yards_to_feet(volume_cubic_yards: float, allow_negative: bool = False) -> float:
    """
    Convert volume from cubic yards to cubic feet.
    
    Args:
        volume_cubic_yards: Volume in cubic yards
        allow_negative: Whether to allow negative values (default: False)
        
    Returns:
        Volume in cubic feet
        
    Raises:
        ValueError: If volume is negative and allow_negative is False
    """
    if not allow_negative and volume_cubic_yards < 0:
        raise ValueError("Volume cannot be negative")
    
    # Conversion factor: 1 cubic yard = 27 cubic feet
    return volume_cubic_yards * 27.0


def validate_volume_units(volume: float, unit: str) -> bool:
    """
    Validate volume value and unit.
    
    Args:
        volume: Volume value
        unit: Unit string ('cubic_feet', 'cubic_yards', etc.)
        
    Returns:
        True if valid, False otherwise
    """
    if not isinstance(volume, (int, float)):
        return False
    
    if np.isnan(volume) or np.isinf(volume):
        return False
    
    valid_units = ['cubic_feet', 'cubic_yards', 'cubic_meters']
    if unit not in valid_units:
        return False
    
    return True 