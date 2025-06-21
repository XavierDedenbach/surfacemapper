import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial._qhull import QhullError


def create_delaunay_triangulation(points_2d: np.ndarray) -> Delaunay:
    """
    Create a Delaunay triangulation from 2D points.
    Removes duplicate points and checks for sufficient dimensionality.
    Raises QhullError for collinear or insufficient points.
    """
    if len(points_2d) == 0:
        raise ValueError("Cannot create triangulation from empty point set")
    
    # Remove duplicate points
    unique_points = np.unique(points_2d, axis=0)
    if unique_points.shape[0] < 3:
        raise QhullError("At least 3 unique points are required for triangulation.")
    try:
        tri = Delaunay(unique_points)
    except QhullError as e:
        raise QhullError(f"Triangulation failed: {e}")
    return tri


def validate_triangulation_quality(triangulation: Delaunay) -> float:
    """
    Calculate a simple quality metric for a Delaunay triangulation.
    Returns the mean aspect ratio (min side / max side) for all triangles (0-1, higher is better).
    """
    if len(triangulation.simplices) == 0:
        return 0.0
    points = triangulation.points
    simplices = triangulation.simplices
    aspect_ratios = []
    for simplex in simplices:
        triangle_points = points[simplex]
        sides = [np.linalg.norm(triangle_points[i] - triangle_points[(i+1)%3]) for i in range(3)]
        if max(sides) > 0:
            aspect_ratios.append(min(sides) / max(sides))
    return float(np.mean(aspect_ratios)) if aspect_ratios else 0.0


def create_tin_from_points(points_3d: np.ndarray) -> Delaunay:
    """
    Create a TIN (Triangulated Irregular Network) from 3D points.
    Uses only X,Y coordinates for triangulation, preserving Z values for interpolation.
    """
    if len(points_3d) == 0:
        raise ValueError("Cannot create TIN from empty point set")
    
    # Extract X,Y coordinates for triangulation
    points_2d = points_3d[:, :2]
    return create_delaunay_triangulation(points_2d)


def interpolate_z_from_tin(tin: Delaunay, points_3d: np.ndarray, query_point: np.ndarray) -> float:
    """
    Interpolate Z value from TIN using barycentric coordinates.
    
    Args:
        tin: Delaunay triangulation object
        points_3d: Original 3D points used to create the TIN
        query_point: 2D point [x, y] for which to interpolate Z value
        
    Returns:
        Interpolated Z value, or NaN if point is outside convex hull
    """
    if len(tin.simplices) == 0:
        return np.nan
    
    # Find which triangle contains the query point
    triangle_index = tin.find_simplex(query_point)
    
    # If exact location fails, try with small tolerance for edge cases
    if triangle_index == -1:
        # Check if point is very close to any triangle
        for i, simplex in enumerate(tin.simplices):
            vertices = tin.points[simplex]
            barycentric = calculate_barycentric_coordinates(vertices, query_point)
            
            # If any barycentric coordinate is very close to 0 or 1, consider this triangle
            if np.any(np.abs(barycentric) < 1e-12) or np.any(np.abs(barycentric - 1) < 1e-12):
                # Check if point is actually close to triangle
                min_distance = min(np.linalg.norm(query_point - vertex) for vertex in vertices)
                if min_distance < 1e-10:
                    triangle_index = i
                    break
    
    if triangle_index == -1:
        return np.nan  # Point outside convex hull
    
    # Get the triangle vertices
    simplex = tin.simplices[triangle_index]
    triangle_points_2d = tin.points[simplex]
    triangle_points_3d = points_3d[simplex]
    
    # Calculate barycentric coordinates
    barycentric_coords = calculate_barycentric_coordinates(
        triangle_points_2d, query_point
    )
    
    # Interpolate Z value using barycentric coordinates
    interpolated_z = np.sum(barycentric_coords * triangle_points_3d[:, 2])
    
    return float(interpolated_z)


def calculate_barycentric_coordinates(triangle_points: np.ndarray, query_point: np.ndarray) -> np.ndarray:
    """
    Calculate barycentric coordinates of a point within a triangle.
    
    Args:
        triangle_points: 3x2 array of triangle vertex coordinates
        query_point: 2D point [x, y] to find barycentric coordinates for
    
    Returns:
        Array of 3 barycentric coordinates that sum to 1
    """
    # Use area-based calculation for barycentric coordinates
    return calculate_barycentric_by_areas(triangle_points, query_point)


def calculate_barycentric_by_areas(triangle_points: np.ndarray, query_point: np.ndarray) -> np.ndarray:
    """
    Calculate barycentric coordinates using area ratios.
    This method is more robust than matrix-based approaches.
    """
    # Calculate areas of sub-triangles
    areas = []
    for i in range(3):
        # Create sub-triangle with query point and two vertices
        sub_triangle = np.vstack([
            query_point,
            triangle_points[(i + 1) % 3],
            triangle_points[(i + 2) % 3]
        ])
        area = calculate_triangle_area(sub_triangle)
        areas.append(area)
    
    # Calculate total triangle area
    total_area = calculate_triangle_area(triangle_points)
    
    if total_area == 0:
        # Degenerate triangle, return equal weights
        return np.array([1/3, 1/3, 1/3])
    
    # Barycentric coordinates are area ratios
    barycentric = np.array(areas) / total_area
    return barycentric


def calculate_triangle_area(triangle_points: np.ndarray) -> float:
    """
    Calculate the area of a triangle using the shoelace formula.
    """
    if len(triangle_points) != 3:
        return 0.0
    
    x = triangle_points[:, 0]
    y = triangle_points[:, 1]
    
    # Shoelace formula: A = 1/2 * |x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2|
    area = 0.5 * abs(
        x[0] * y[1] + x[1] * y[2] + x[2] * y[0] - 
        x[0] * y[2] - x[1] * y[0] - x[2] * y[1]
    )
    
    return float(area)


def interpolate_z_batch(tin: Delaunay, points_3d: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """
    Interpolate Z values for multiple query points using vectorized operations.
    
    Args:
        tin: Delaunay triangulation object
        points_3d: Original 3D points used to create the TIN
        query_points: Nx2 array of query points [x, y]
    
    Returns:
        Array of interpolated Z values, with NaN for points outside convex hull
    """
    if len(query_points) == 0:
        return np.array([])
    
    # Find which triangle contains each query point
    triangle_indices = tin.find_simplex(query_points)
    
    # Initialize result array
    interpolated_z = np.full(len(query_points), np.nan)
    
    # Process points that are inside the convex hull
    for idx, (triangle_index, query_point) in enumerate(zip(triangle_indices, query_points)):
        if triangle_index == -1:
            continue  # Leave as NaN
        simplex = tin.simplices[triangle_index]
        triangle_points_2d = tin.points[simplex]
        triangle_points_3d = points_3d[simplex]
        barycentric_coords = calculate_barycentric_coordinates(triangle_points_2d, query_point)
        interpolated_z[idx] = np.sum(barycentric_coords * triangle_points_3d[:, 2])
    return interpolated_z


def find_containing_triangle(triangulation, point, tolerance=1e-10):
    """
    Find the triangle containing a point with tolerance for edge cases.
    
    Args:
        triangulation: SciPy Delaunay triangulation
        point: 2D point [x, y]
        tolerance: Tolerance for edge/vertex cases
        
    Returns:
        Triangle index or None if point is outside all triangles
    """
    # First try exact location
    triangle_idx = triangulation.find_simplex(point)
    
    if triangle_idx >= 0:
        return triangle_idx
    
    # If exact location fails, try with tolerance
    # Check all triangles and find the one with minimum distance to point
    min_distance = float('inf')
    best_triangle = None
    
    for i, simplex in enumerate(triangulation.simplices):
        # Get triangle vertices
        vertices = triangulation.points[simplex]
        
        # Calculate distance from point to triangle
        # For points inside or very close to triangle, distance should be small
        barycentric = calculate_barycentric_coordinates(vertices, point)
        
        # If any barycentric coordinate is within tolerance of 0 or 1,
        # consider this triangle as a candidate
        if np.any(np.abs(barycentric) < tolerance) or np.any(np.abs(barycentric - 1) < tolerance):
            # Calculate distance to triangle
            distance = point_to_triangle_distance(vertices, point)
            if distance < min_distance:
                min_distance = distance
                best_triangle = i
    
    return best_triangle


def point_to_triangle_distance(vertices, point):
    """
    Calculate the minimum distance from a point to a triangle.
    
    Args:
        vertices: Triangle vertices as 3x2 array
        point: 2D point [x, y]
        
    Returns:
        Minimum distance from point to triangle
    """
    # Calculate distances to each edge and vertex
    distances = []
    
    # Distance to vertices
    for vertex in vertices:
        distances.append(np.linalg.norm(point - vertex))
    
    # Distance to edges
    for i in range(3):
        v1 = vertices[i]
        v2 = vertices[(i + 1) % 3]
        
        # Vector from v1 to v2
        edge_vec = v2 - v1
        # Vector from v1 to point
        point_vec = point - v1
        
        # Project point onto edge
        edge_length = np.linalg.norm(edge_vec)
        if edge_length > 0:
            t = np.dot(point_vec, edge_vec) / (edge_length ** 2)
            t = max(0, min(1, t))  # Clamp to edge
            
            # Closest point on edge
            closest = v1 + t * edge_vec
            distances.append(np.linalg.norm(point - closest))
    
    return min(distances) 