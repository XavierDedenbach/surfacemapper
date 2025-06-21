"""
Thickness calculation service using TIN interpolation
"""
import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import Delaunay
import logging

logger = logging.getLogger(__name__)


def calculate_point_to_surface_distance(point: np.ndarray, surface_tin: Delaunay) -> float:
    """
    Calculate vertical distance from a point to a TIN surface.
    
    Args:
        point: 3D point [x, y, z]
        surface_tin: Delaunay triangulation of surface with z_values attribute
        
    Returns:
        Vertical distance (positive if point is above surface, negative if below)
        Returns NaN if point is outside surface boundary
    """
    try:
        # Check if point is within surface boundary
        if not _is_point_in_triangulation(point[:2], surface_tin):
            return np.nan
        
        # Find the triangle containing the point
        triangle_index = surface_tin.find_simplex(point[:2])
        
        if triangle_index == -1:
            # Point is outside triangulation
            return np.nan
        
        # Get triangle vertices
        triangle_vertices = surface_tin.points[surface_tin.simplices[triangle_index]]
        triangle_z_values = surface_tin.z_values[surface_tin.simplices[triangle_index]]
        
        # Calculate barycentric coordinates
        barycentric_coords = _calculate_barycentric_coordinates(
            point[:2], triangle_vertices
        )
        
        # Interpolate Z value at point
        interpolated_z = np.sum(barycentric_coords * triangle_z_values)
        
        # Calculate vertical distance
        distance = point[2] - interpolated_z
        
        return distance
        
    except Exception as e:
        logger.error(f"Error calculating point-to-surface distance: {e}")
        return np.nan


def calculate_batch_point_to_surface_distances(points: np.ndarray, surface_tin: Delaunay) -> np.ndarray:
    """
    Calculate vertical distances for multiple points to a TIN surface.
    
    Args:
        points: Array of 3D points [N, 3]
        surface_tin: Delaunay triangulation of surface with z_values attribute
        
    Returns:
        Array of vertical distances [N]
    """
    try:
        distances = np.zeros(len(points))
        
        for i, point in enumerate(points):
            distances[i] = calculate_point_to_surface_distance(point, surface_tin)
        
        return distances
        
    except Exception as e:
        logger.error(f"Error calculating batch point-to-surface distances: {e}")
        return np.full(len(points), np.nan)


def _is_point_in_triangulation(point_2d: np.ndarray, triangulation: Delaunay) -> bool:
    """
    Check if a 2D point is within the triangulation boundary.
    
    Args:
        point_2d: 2D point [x, y]
        triangulation: Delaunay triangulation
        
    Returns:
        True if point is within triangulation boundary
    """
    try:
        # Find simplex containing the point
        simplex_index = triangulation.find_simplex(point_2d)
        return simplex_index != -1
        
    except Exception:
        return False


def _calculate_barycentric_coordinates(point_2d: np.ndarray, triangle_vertices: np.ndarray) -> np.ndarray:
    """
    Calculate barycentric coordinates of a point within a triangle.
    
    Args:
        point_2d: 2D point [x, y]
        triangle_vertices: Triangle vertices [3, 2]
        
    Returns:
        Barycentric coordinates [3]
    """
    try:
        # Calculate areas using cross products
        v0 = triangle_vertices[1] - triangle_vertices[0]
        v1 = triangle_vertices[2] - triangle_vertices[0]
        v2 = point_2d - triangle_vertices[0]
        
        # Calculate dot products
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        
        # Calculate barycentric coordinates
        denom = d00 * d11 - d01 * d01
        if abs(denom) < 1e-12:
            # Degenerate triangle, use equal weights
            return np.array([1/3, 1/3, 1/3])
        
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        
        return np.array([u, v, w])
        
    except Exception as e:
        logger.error(f"Error calculating barycentric coordinates: {e}")
        # Return equal weights as fallback
        return np.array([1/3, 1/3, 1/3])


def calculate_thickness_between_surfaces(
    upper_surface_points: np.ndarray,
    lower_surface_points: np.ndarray,
    sample_points: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Calculate thickness between two surfaces at sample points.
    
    Args:
        upper_surface_points: Points defining upper surface [N, 3]
        lower_surface_points: Points defining lower surface [M, 3]
        sample_points: Points where thickness is calculated [K, 2] (optional)
        
    Returns:
        Thickness values at sample points [K]
    """
    try:
        # Create TINs for both surfaces
        from services.triangulation import create_delaunay_triangulation
        
        upper_tin = create_delaunay_triangulation(upper_surface_points[:, :2])
        upper_tin.z_values = upper_surface_points[:, 2]
        
        lower_tin = create_delaunay_triangulation(lower_surface_points[:, :2])
        lower_tin.z_values = lower_surface_points[:, 2]
        
        # Generate sample points if not provided
        if sample_points is None:
            sample_points = _generate_uniform_sample_points(upper_surface_points, lower_surface_points)
        
        # Calculate thickness at each sample point
        thicknesses = np.zeros(len(sample_points))
        
        for i, sample_point_2d in enumerate(sample_points):
            # Find upper surface Z at this point
            upper_z = _interpolate_z_at_point(sample_point_2d, upper_tin)
            lower_z = _interpolate_z_at_point(sample_point_2d, lower_tin)
            
            if not np.isnan(upper_z) and not np.isnan(lower_z):
                thicknesses[i] = upper_z - lower_z
            else:
                thicknesses[i] = np.nan
        
        return thicknesses
        
    except Exception as e:
        logger.error(f"Error calculating thickness between surfaces: {e}")
        return np.full(len(sample_points) if sample_points is not None else 100, np.nan)


def _interpolate_z_at_point(point_2d: np.ndarray, surface_tin: Delaunay) -> float:
    """
    Interpolate Z value at a 2D point using TIN.
    
    Args:
        point_2d: 2D point [x, y]
        surface_tin: Delaunay triangulation with z_values
        
    Returns:
        Interpolated Z value or NaN if outside boundary
    """
    try:
        # Check if point is within triangulation
        if not _is_point_in_triangulation(point_2d, surface_tin):
            return np.nan
        
        # Find containing triangle
        triangle_index = surface_tin.find_simplex(point_2d)
        
        if triangle_index == -1:
            return np.nan
        
        # Get triangle data
        triangle_vertices = surface_tin.points[surface_tin.simplices[triangle_index]]
        triangle_z_values = surface_tin.z_values[surface_tin.simplices[triangle_index]]
        
        # Calculate barycentric coordinates
        barycentric_coords = _calculate_barycentric_coordinates(point_2d, triangle_vertices)
        
        # Interpolate Z value
        interpolated_z = np.sum(barycentric_coords * triangle_z_values)
        
        return interpolated_z
        
    except Exception as e:
        logger.error(f"Error interpolating Z value: {e}")
        return np.nan


def generate_uniform_sample_points(boundary: np.ndarray, sample_spacing: float) -> np.ndarray:
    """
    Generate uniform grid sampling points within boundary.
    
    Args:
        boundary: Boundary polygon vertices [N, 2]
        sample_spacing: Spacing between sample points
        
    Returns:
        Sample points [K, 2]
    """
    try:
        if len(boundary) < 3:
            # Single point or line - return boundary points
            return boundary
        
        # Calculate bounding box
        x_min, y_min = np.min(boundary, axis=0)
        x_max, y_max = np.max(boundary, axis=0)
        
        # Generate uniform grid including boundary points
        x_coords = np.arange(x_min, x_max + sample_spacing, sample_spacing)
        y_coords = np.arange(y_min, y_max + sample_spacing, sample_spacing)
        
        X, Y = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack([X.ravel(), Y.ravel()])
        
        # For rectangular boundaries, include all grid points
        # For irregular boundaries, filter points inside
        if _is_rectangular_boundary(boundary):
            return grid_points
        else:
            # Filter points inside boundary
            sample_points = []
            for point in grid_points:
                if _is_point_in_polygon(point, boundary):
                    sample_points.append(point)
            
            return np.array(sample_points) if sample_points else boundary[:1]
        
    except Exception as e:
        logger.error(f"Error generating uniform sample points: {e}")
        return boundary[:1]  # Return first boundary point as fallback


def _is_rectangular_boundary(boundary: np.ndarray) -> bool:
    """
    Check if boundary is approximately rectangular.
    
    Args:
        boundary: Boundary polygon vertices [N, 2]
        
    Returns:
        True if boundary is rectangular
    """
    try:
        if len(boundary) != 4:
            return False
        
        # Check if boundary forms a rectangle
        x_coords = boundary[:, 0]
        y_coords = boundary[:, 1]
        
        # Should have exactly 2 unique x and y values
        unique_x = np.unique(x_coords)
        unique_y = np.unique(y_coords)
        
        return len(unique_x) == 2 and len(unique_y) == 2
        
    except Exception:
        return False


def generate_adaptive_sample_points(
    surface_points: np.ndarray,
    boundary: np.ndarray,
    max_spacing: float,
    min_spacing: float = 0.1
) -> np.ndarray:
    """
    Generate adaptive sampling points based on surface complexity.
    
    Args:
        surface_points: Surface points [N, 3]
        boundary: Boundary polygon vertices [M, 2]
        max_spacing: Maximum spacing between points
        min_spacing: Minimum spacing between points
        
    Returns:
        Sample points [K, 2]
    """
    try:
        if len(surface_points) < 3:
            return generate_uniform_sample_points(boundary, max_spacing)
        
        # Calculate surface complexity (roughness)
        complexity = _calculate_surface_complexity(surface_points)
        
        # Adjust spacing based on complexity
        # Higher complexity = smaller spacing
        adjusted_spacing = max_spacing - (max_spacing - min_spacing) * complexity
        adjusted_spacing = max(min_spacing, min(max_spacing, adjusted_spacing))
        
        # Generate uniform grid with adjusted spacing
        return generate_uniform_sample_points(boundary, adjusted_spacing)
        
    except Exception as e:
        logger.error(f"Error generating adaptive sample points: {e}")
        return generate_uniform_sample_points(boundary, max_spacing)


def generate_boundary_aware_sample_points(boundary: np.ndarray, sample_spacing: float) -> np.ndarray:
    """
    Generate sample points that respect irregular boundaries.
    
    Args:
        boundary: Boundary polygon vertices [N, 2]
        sample_spacing: Spacing between sample points
        
    Returns:
        Sample points [K, 2]
    """
    try:
        if len(boundary) < 3:
            return boundary
        
        # Use uniform sampling with boundary filtering
        return generate_uniform_sample_points(boundary, sample_spacing)
        
    except Exception as e:
        logger.error(f"Error generating boundary-aware sample points: {e}")
        return boundary[:1]


def _calculate_surface_complexity(surface_points: np.ndarray) -> float:
    """
    Calculate surface complexity/roughness measure.
    
    Args:
        surface_points: Surface points [N, 3]
        
    Returns:
        Complexity measure between 0 (smooth) and 1 (rough)
    """
    try:
        if len(surface_points) < 4:
            return 0.0
        
        # Create TIN for surface
        from services.triangulation import create_delaunay_triangulation
        
        tin = create_delaunay_triangulation(surface_points[:, :2])
        tin.z_values = surface_points[:, 2]
        
        if tin.simplices.size == 0:
            return 0.0
        
        # Calculate surface roughness using triangle gradients
        gradients = []
        for simplex in tin.simplices:
            vertices = surface_points[simplex]
            if len(vertices) == 3:
                # Calculate triangle normal
                v1 = vertices[1] - vertices[0]
                v2 = vertices[2] - vertices[0]
                normal = np.cross(v1, v2)
                normal_magnitude = np.linalg.norm(normal)
                
                if normal_magnitude > 0:
                    # Gradient magnitude (steepness)
                    gradient = normal_magnitude / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    gradients.append(gradient)
        
        if not gradients:
            return 0.0
        
        # Normalize complexity measure
        avg_gradient = np.mean(gradients)
        max_gradient = np.max(gradients)
        
        # Return normalized complexity (0-1)
        complexity = min(1.0, avg_gradient / max_gradient if max_gradient > 0 else 0.0)
        
        return complexity
        
    except Exception as e:
        logger.error(f"Error calculating surface complexity: {e}")
        return 0.5  # Return medium complexity as fallback


def _is_point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: 2D point [x, y]
        polygon: Polygon vertices [N, 2]
        
    Returns:
        True if point is inside polygon
    """
    try:
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
        
    except Exception as e:
        logger.error(f"Error checking point in polygon: {e}")
        return False


def optimize_sample_density(
    surface_points: np.ndarray,
    boundary: np.ndarray,
    target_point_count: int,
    max_spacing: float = 10.0,
    min_spacing: float = 0.1
) -> float:
    """
    Optimize sample spacing to achieve target point count.
    
    Args:
        surface_points: Surface points [N, 3]
        boundary: Boundary polygon vertices [M, 2]
        target_point_count: Desired number of sample points
        max_spacing: Maximum allowed spacing
        min_spacing: Minimum allowed spacing
        
    Returns:
        Optimized spacing value
    """
    try:
        # Binary search for optimal spacing
        left = min_spacing
        right = max_spacing
        
        while right - left > 0.01:  # 1cm tolerance
            mid = (left + right) / 2
            sample_points = generate_uniform_sample_points(boundary, mid)
            
            if len(sample_points) >= target_point_count:
                left = mid
            else:
                right = mid
        
        return left
        
    except Exception as e:
        logger.error(f"Error optimizing sample density: {e}")
        return max_spacing


def calculate_thickness_statistics(thicknesses: np.ndarray) -> dict:
    """
    Calculate statistical measures of thickness distribution.
    
    Args:
        thicknesses: Array of thickness values
        
    Returns:
        Dictionary with thickness statistics
    """
    try:
        # Remove NaN values
        valid_thicknesses = thicknesses[~np.isnan(thicknesses)]
        
        if len(valid_thicknesses) == 0:
            return {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'count': 0
            }
        
        stats = {
            'min': np.min(valid_thicknesses),
            'max': np.max(valid_thicknesses),
            'mean': np.mean(valid_thicknesses),
            'median': np.median(valid_thicknesses),
            'std': np.std(valid_thicknesses),
            'count': len(valid_thicknesses)
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating thickness statistics: {e}")
        return {
            'min': np.nan,
            'max': np.nan,
            'mean': np.nan,
            'median': np.nan,
            'std': np.nan,
            'count': 0
        } 