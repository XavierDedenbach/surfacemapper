import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial._qhull import QhullError


def create_delaunay_triangulation(points_2d: np.ndarray) -> Delaunay:
    """
    Create a Delaunay triangulation from 2D points.
    Removes duplicate points and checks for sufficient dimensionality.
    Raises QhullError for collinear or insufficient points.
    """
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