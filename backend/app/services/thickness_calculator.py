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
        from .triangulation import create_delaunay_triangulation
        
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
        
        # Calculate elevation variance as a measure of complexity
        z_values = surface_points[:, 2]
        z_mean = np.mean(z_values)
        z_variance = np.var(z_values)
        
        # Normalize variance to 0-1 range
        # Use a reasonable threshold for "high complexity"
        max_expected_variance = 10.0  # Lower threshold for more sensitivity
        complexity = min(1.0, z_variance / max_expected_variance)
        
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
        total_count = len(thicknesses)
        # Remove NaN values
        valid_thicknesses = thicknesses[~np.isnan(thicknesses)]
        valid_count = len(valid_thicknesses)
        
        if valid_count == 0:
            return {
                'min': np.nan,
                'max': np.nan,
                'mean': np.nan,
                'median': np.nan,
                'std': np.nan,
                'count': total_count,
                'valid_count': valid_count
            }
        
        stats = {
            'min': np.min(valid_thicknesses),
            'max': np.max(valid_thicknesses),
            'mean': np.mean(valid_thicknesses),
            'median': np.median(valid_thicknesses),
            'std': np.std(valid_thicknesses),
            'count': total_count,
            'valid_count': valid_count
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
            'count': 0,
            'valid_count': 0
        }


def analyze_thickness_distribution(thicknesses: np.ndarray) -> dict:
    """
    Analyze thickness distribution patterns and characteristics.
    
    Args:
        thicknesses: Array of thickness values
        
    Returns:
        Dictionary with distribution analysis results
    """
    try:
        # Remove NaN values
        valid_thicknesses = thicknesses[~np.isnan(thicknesses)]
        valid_count = len(valid_thicknesses)
        
        if valid_count == 0:
            return {
                'distribution_type': 'unknown',
                'valid_count': 0,
                'statistics': {},
                'peaks': [],
                'skewness': np.nan,
                'kurtosis': np.nan,
                'variance': np.nan
            }
        
        if valid_count == 1:
            return {
                'distribution_type': 'single_value',
                'valid_count': valid_count,
                'statistics': {'value': valid_thicknesses[0]},
                'peaks': [valid_thicknesses[0]],
                'skewness': 0.0,
                'kurtosis': 0.0,
                'variance': 0.0
            }
        
        # Calculate basic statistics
        mean_val = np.mean(valid_thicknesses)
        std_val = np.std(valid_thicknesses)
        variance = np.var(valid_thicknesses)
        
        # Calculate skewness and kurtosis
        skewness = _calculate_skewness(valid_thicknesses)
        kurtosis = _calculate_kurtosis(valid_thicknesses)
        
        # Determine distribution type
        distribution_type = _classify_distribution(valid_thicknesses, skewness, kurtosis, variance)
        
        # Find peaks for multimodal distributions
        peaks = _find_distribution_peaks(valid_thicknesses) if distribution_type == 'multimodal' else []
        
        return {
            'distribution_type': distribution_type,
            'valid_count': valid_count,
            'statistics': {
                'mean': mean_val,
                'std': std_val,
                'variance': variance,
                'min': np.min(valid_thicknesses),
                'max': np.max(valid_thicknesses),
                'median': np.median(valid_thicknesses)
            },
            'peaks': peaks,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'variance': variance
        }
        
    except Exception as e:
        logger.error(f"Error analyzing thickness distribution: {e}")
        return {
            'distribution_type': 'error',
            'valid_count': 0,
            'statistics': {},
            'peaks': [],
            'skewness': np.nan,
            'kurtosis': np.nan,
            'variance': np.nan
        }


def detect_thickness_anomalies(thicknesses: np.ndarray, method: str = 'iqr') -> dict:
    """
    Detect anomalies and outliers in thickness data.
    
    Args:
        thicknesses: Array of thickness values
        method: Detection method ('iqr', 'zscore', 'isolation_forest')
        
    Returns:
        Dictionary with anomaly detection results
    """
    try:
        # Remove NaN values
        valid_thicknesses = thicknesses[~np.isnan(thicknesses)]
        
        if len(valid_thicknesses) < 3:
            return {
                'anomalies': [],
                'outliers': [],
                'anomaly_count': 0,
                'anomaly_percentage': 0.0,
                'method': method
            }
        
        if method == 'iqr':
            anomalies, outliers = _detect_anomalies_iqr(valid_thicknesses)
        elif method == 'zscore':
            anomalies, outliers = _detect_anomalies_zscore(valid_thicknesses)
        else:
            # Default to IQR method
            anomalies, outliers = _detect_anomalies_iqr(valid_thicknesses)
        
        anomaly_count = len(anomalies)
        anomaly_percentage = (anomaly_count / len(valid_thicknesses)) * 100
        
        return {
            'anomalies': anomalies.tolist() if len(anomalies) > 0 else [],
            'outliers': outliers.tolist() if len(outliers) > 0 else [],
            'anomaly_count': anomaly_count,
            'anomaly_percentage': anomaly_percentage,
            'method': method
        }
        
    except Exception as e:
        logger.error(f"Error detecting thickness anomalies: {e}")
        return {
            'anomalies': [],
            'outliers': [],
            'anomaly_count': 0,
            'anomaly_percentage': 0.0,
            'method': method
        }


def analyze_thickness_clusters(thicknesses: np.ndarray, max_clusters: int = 5) -> dict:
    """
    Perform clustering analysis on thickness data.
    
    Args:
        thicknesses: Array of thickness values
        max_clusters: Maximum number of clusters to consider
        
    Returns:
        Dictionary with clustering results
    """
    try:
        # Remove NaN values
        valid_thicknesses = thicknesses[~np.isnan(thicknesses)]
        
        if len(valid_thicknesses) < 3:
            return {
                'clusters': [],
                'cluster_centers': [],
                'cluster_sizes': [],
                'optimal_clusters': 1
            }
        
        # Reshape for clustering
        data = valid_thicknesses.reshape(-1, 1)
        
        # Use K-means clustering
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        # Find optimal number of clusters
        best_score = -1
        optimal_clusters = 1
        
        for n_clusters in range(2, min(max_clusters + 1, len(valid_thicknesses))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(data)
            
            if len(np.unique(cluster_labels)) > 1:
                score = silhouette_score(data, cluster_labels)
                if score > best_score:
                    best_score = score
                    optimal_clusters = n_clusters
        
        # Perform clustering with optimal number
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(data)
        
        # Extract cluster information
        clusters = []
        cluster_centers = kmeans.cluster_centers_.flatten().tolist()
        cluster_sizes = []
        
        for i in range(optimal_clusters):
            cluster_mask = cluster_labels == i
            cluster_data = valid_thicknesses[cluster_mask]
            clusters.append(cluster_data.tolist())
            cluster_sizes.append(len(cluster_data))
        
        return {
            'clusters': clusters,
            'cluster_centers': cluster_centers,
            'cluster_sizes': cluster_sizes,
            'optimal_clusters': optimal_clusters,
            'silhouette_score': best_score
        }
        
    except Exception as e:
        logger.error(f"Error analyzing thickness clusters: {e}")
        return {
            'clusters': [],
            'cluster_centers': [],
            'cluster_sizes': [],
            'optimal_clusters': 1
        }


def analyze_thickness_spatial_patterns(spatial_data: np.ndarray) -> dict:
    """
    Analyze spatial patterns in thickness data.
    
    Args:
        spatial_data: Array of [x, y, thickness] coordinates
        
    Returns:
        Dictionary with spatial analysis results
    """
    try:
        if len(spatial_data) < 3:
            return {
                'spatial_trends': {},
                'spatial_correlation': {},
                'spatial_variability': {}
            }
        
        x_coords = spatial_data[:, 0]
        y_coords = spatial_data[:, 1]
        thicknesses = spatial_data[:, 2]
        
        # Remove NaN values
        valid_mask = ~np.isnan(thicknesses)
        x_valid = x_coords[valid_mask]
        y_valid = y_coords[valid_mask]
        thicknesses_valid = thicknesses[valid_mask]
        
        if len(thicknesses_valid) < 3:
            return {
                'spatial_trends': {},
                'spatial_correlation': {},
                'spatial_variability': {}
            }
        
        # Calculate spatial trends
        x_trend = np.polyfit(x_valid, thicknesses_valid, 1)[0] if len(x_valid) > 1 else 0
        y_trend = np.polyfit(y_valid, thicknesses_valid, 1)[0] if len(y_valid) > 1 else 0
        
        # Calculate spatial variability
        spatial_variance = np.var(thicknesses_valid)
        spatial_range = np.max(thicknesses_valid) - np.min(thicknesses_valid)
        
        # Calculate simple spatial correlation (distance-based)
        spatial_correlation = _calculate_spatial_correlation(x_valid, y_valid, thicknesses_valid)
        
        return {
            'spatial_trends': {
                'x_direction': x_trend,
                'y_direction': y_trend,
                'overall_trend': np.sqrt(x_trend**2 + y_trend**2)
            },
            'spatial_correlation': {
                'correlation_coefficient': spatial_correlation,
                'spatial_dependency': 'high' if abs(spatial_correlation) > 0.5 else 'low'
            },
            'spatial_variability': {
                'variance': spatial_variance,
                'range': spatial_range,
                'coefficient_of_variation': spatial_variance / np.mean(thicknesses_valid) if np.mean(thicknesses_valid) > 0 else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error analyzing spatial patterns: {e}")
        return {
            'spatial_trends': {},
            'spatial_correlation': {},
            'spatial_variability': {}
        }


def generate_thickness_insights(thicknesses: np.ndarray) -> dict:
    """
    Generate comprehensive insights about thickness distribution.
    
    Args:
        thicknesses: Array of thickness values
        
    Returns:
        Dictionary with insights and recommendations
    """
    try:
        # Get distribution analysis
        distribution = analyze_thickness_distribution(thicknesses)
        
        # Get anomaly detection
        anomalies = detect_thickness_anomalies(thicknesses)
        
        # Get clustering analysis
        clusters = analyze_thickness_clusters(thicknesses)
        
        # Generate quality assessment
        quality_assessment = _assess_data_quality(thicknesses, distribution, anomalies)
        
        # Generate recommendations
        recommendations = _generate_recommendations(distribution, anomalies, clusters)
        
        # Identify risk factors
        risk_factors = _identify_risk_factors(distribution, anomalies)
        
        return {
            'distribution_summary': {
                'type': distribution['distribution_type'],
                'statistics': distribution['statistics'],
                'characteristics': {
                    'skewness': distribution['skewness'],
                    'kurtosis': distribution['kurtosis'],
                    'variance': distribution['variance']
                }
            },
            'quality_assessment': quality_assessment,
            'recommendations': recommendations,
            'risk_factors': risk_factors,
            'anomaly_summary': {
                'count': anomalies['anomaly_count'],
                'percentage': anomalies['anomaly_percentage'],
                'method': anomalies['method']
            },
            'clustering_summary': {
                'optimal_clusters': clusters['optimal_clusters'],
                'cluster_centers': clusters['cluster_centers']
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating thickness insights: {e}")
        return {
            'distribution_summary': {},
            'quality_assessment': {},
            'recommendations': ['Error occurred during analysis'],
            'risk_factors': []
        }


# Helper functions for distribution analysis

def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data."""
    try:
        if len(data) < 3:
            return 0.0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        skewness = np.mean(((data - mean_val) / std_val) ** 3)
        return skewness
    except:
        return 0.0


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data."""
    try:
        if len(data) < 4:
            return 0.0
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        kurtosis = np.mean(((data - mean_val) / std_val) ** 4) - 3
        return kurtosis
    except:
        return 0.0


def _classify_distribution(data: np.ndarray, skewness: float, kurtosis: float, variance: float) -> str:
    """Classify the type of distribution."""
    try:
        if variance == 0:
            return 'uniform'
        
        # Check for multimodal distribution using histogram with more conservative approach
        hist, bins = np.histogram(data, bins=min(20, len(data)//10))
        peaks = _find_peaks_in_histogram(hist)
        
        # Only classify as multimodal if we have clear, well-separated peaks
        if len(peaks) > 1:
            # Check if peaks are well-separated
            peak_values = [bins[p] for p in peaks]
            peak_separation = min(abs(peak_values[i] - peak_values[i-1]) for i in range(1, len(peak_values)))
            data_range = np.max(data) - np.min(data)
            
            # Require peaks to be separated by at least 20% of data range
            if peak_separation > 0.2 * data_range:
                return 'multimodal'
        
        # Classify based on skewness and kurtosis
        if abs(skewness) < 0.5 and abs(kurtosis) < 1.0:
            return 'unimodal'
        elif abs(skewness) > 1.0:
            return 'skewed'
        elif abs(kurtosis) > 2.0:
            return 'heavy_tailed'
        else:
            return 'unimodal'
            
    except:
        return 'unknown'


def _find_peaks_in_histogram(hist: np.ndarray) -> list:
    """Find peaks in histogram."""
    try:
        peaks = []
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                peaks.append(i)
        return peaks
    except:
        return []


def _find_distribution_peaks(data: np.ndarray) -> list:
    """Find peaks in distribution."""
    try:
        hist, bins = np.histogram(data, bins=min(20, len(data)//5))
        peaks = _find_peaks_in_histogram(hist)
        peak_values = [bins[p] for p in peaks]
        return peak_values
    except:
        return []


def _detect_anomalies_iqr(data: np.ndarray) -> tuple:
    """Detect anomalies using IQR method."""
    try:
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = data[(data < lower_bound) | (data > upper_bound)]
        outliers = data[(data < q1 - 3 * iqr) | (data > q3 + 3 * iqr)]
        
        return anomalies, outliers
    except:
        return np.array([]), np.array([])


def _detect_anomalies_zscore(data: np.ndarray, threshold: float = 3.0) -> tuple:
    """Detect anomalies using Z-score method."""
    try:
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return np.array([]), np.array([])
        
        z_scores = np.abs((data - mean_val) / std_val)
        anomalies = data[z_scores > threshold]
        outliers = data[z_scores > threshold * 2]
        
        return anomalies, outliers
    except:
        return np.array([]), np.array([])


def _calculate_spatial_correlation(x: np.ndarray, y: np.ndarray, values: np.ndarray) -> float:
    """Calculate simple spatial correlation."""
    try:
        if len(values) < 3:
            return 0.0
        
        # Calculate distances and value differences
        distances = []
        value_diffs = []
        
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                dist = np.sqrt((x[i] - x[j])**2 + (y[i] - y[j])**2)
                value_diff = abs(values[i] - values[j])
                
                if dist > 0:
                    distances.append(dist)
                    value_diffs.append(value_diff)
        
        if len(distances) < 2:
            return 0.0
        
        # Calculate correlation
        correlation = np.corrcoef(distances, value_diffs)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
        
    except:
        return 0.0


def _assess_data_quality(thicknesses: np.ndarray, distribution: dict, anomalies: dict) -> dict:
    """Assess data quality."""
    try:
        valid_count = distribution['valid_count']
        total_count = len(thicknesses)
        anomaly_percentage = anomalies['anomaly_percentage']
        
        # Assess data quality
        if valid_count / total_count > 0.95:
            data_quality = 'excellent'
        elif valid_count / total_count > 0.9:
            data_quality = 'good'
        elif valid_count / total_count > 0.8:
            data_quality = 'fair'
        else:
            data_quality = 'poor'
        
        # Assess coverage
        if valid_count > 1000:
            coverage_adequacy = 'excellent'
        elif valid_count > 500:
            coverage_adequacy = 'good'
        elif valid_count > 100:
            coverage_adequacy = 'fair'
        else:
            coverage_adequacy = 'poor'
        
        # Assess sampling density
        if anomaly_percentage < 5:
            sampling_density = 'excellent'
        elif anomaly_percentage < 10:
            sampling_density = 'good'
        elif anomaly_percentage < 20:
            sampling_density = 'fair'
        else:
            sampling_density = 'poor'
        
        return {
            'data_quality': data_quality,
            'coverage_adequacy': coverage_adequacy,
            'sampling_density': sampling_density,
            'valid_percentage': (valid_count / total_count) * 100,
            'anomaly_percentage': anomaly_percentage
        }
        
    except:
        return {
            'data_quality': 'unknown',
            'coverage_adequacy': 'unknown',
            'sampling_density': 'unknown',
            'valid_percentage': 0.0,
            'anomaly_percentage': 0.0
        }


def _generate_recommendations(distribution: dict, anomalies: dict, clusters: dict) -> list:
    """Generate recommendations based on analysis."""
    recommendations = []
    
    try:
        # Distribution-based recommendations
        if distribution['distribution_type'] == 'multimodal':
            recommendations.append("Consider analyzing different regions separately due to multimodal distribution")
        
        if abs(distribution['skewness']) > 1.0:
            recommendations.append("Distribution is skewed - consider log transformation for analysis")
        
        # Anomaly-based recommendations
        if anomalies['anomaly_percentage'] > 10:
            recommendations.append("High anomaly rate detected - review data collection methods")
        
        if anomalies['anomaly_count'] > 0:
            recommendations.append("Investigate anomalies for potential measurement errors or geological features")
        
        # Clustering-based recommendations
        if clusters['optimal_clusters'] > 1:
            recommendations.append(f"Data shows {clusters['optimal_clusters']} distinct thickness regions")
        
        # Quality-based recommendations
        if distribution['valid_count'] < 100:
            recommendations.append("Consider increasing sampling density for better statistical reliability")
        
        if not recommendations:
            recommendations.append("Data quality appears good - proceed with standard analysis")
            
    except:
        recommendations.append("Unable to generate recommendations due to analysis error")
    
    return recommendations


def _identify_risk_factors(distribution: dict, anomalies: dict) -> list:
    """Identify potential risk factors."""
    risk_factors = []
    
    try:
        # High variance indicates potential instability
        if distribution['variance'] > 10:
            risk_factors.append("High thickness variability may indicate unstable conditions")
        
        # High anomaly rate
        if anomalies['anomaly_percentage'] > 15:
            risk_factors.append("High anomaly rate suggests potential measurement or geological issues")
        
        # Extreme values
        if distribution['statistics']['max'] > 20 or distribution['statistics']['min'] < 0.1:
            risk_factors.append("Extreme thickness values detected - verify measurements")
        
        # Insufficient data
        if distribution['valid_count'] < 50:
            risk_factors.append("Limited data may affect analysis reliability")
            
    except:
        risk_factors.append("Unable to assess risk factors due to analysis error")
    
    return risk_factors


def validate_thickness_data(thicknesses: np.ndarray) -> dict:
    """
    Validate thickness data for quality and consistency.
    
    Args:
        thicknesses: Array of thickness values
        
    Returns:
        Dictionary with validation results
    """
    if len(thicknesses) == 0:
        return {
            'is_valid': False,
            'error_count': 0,
            'errors': ['empty_data'],
            'warnings': []
        }
    
    errors = []
    warnings = []
    error_count = 0
    
    # Check for negative values
    negative_mask = thicknesses < 0
    if np.any(negative_mask):
        errors.append('negative_values')
        error_count += np.sum(negative_mask)
    
    # Check for NaN values
    nan_mask = np.isnan(thicknesses)
    if np.any(nan_mask):
        errors.append('nan_values')
        error_count += np.sum(nan_mask)
    
    # Check for extreme values (outliers)
    valid_mask = ~(negative_mask | nan_mask)
    if np.any(valid_mask):
        valid_thicknesses = thicknesses[valid_mask]
        q1, q3 = np.percentile(valid_thicknesses, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 3 * iqr
        upper_bound = q3 + 3 * iqr
        
        extreme_mask = (valid_thicknesses < lower_bound) | (valid_thicknesses > upper_bound)
        if np.any(extreme_mask):
            errors.append('extreme_values')
            error_count += np.sum(extreme_mask)
    
    # Check for zero values
    zero_mask = thicknesses == 0
    if np.any(zero_mask):
        warnings.append('zero_values')
    
    # Check for lack of variation
    if len(thicknesses) > 1:
        if np.std(thicknesses) < 1e-6:
            warnings.append('no_variation')
    
    # Check for too many identical values
    unique_ratio = len(np.unique(thicknesses)) / len(thicknesses)
    if unique_ratio < 0.1:  # Less than 10% unique values
        warnings.append('low_precision')
    
    is_valid = len(errors) == 0
    
    return {
        'is_valid': is_valid,
        'error_count': error_count,
        'errors': errors,
        'warnings': warnings,
        'total_points': len(thicknesses),
        'valid_points': len(thicknesses) - error_count
    }


def detect_measurement_errors(thicknesses: np.ndarray) -> dict:
    """
    Detect measurement errors and inconsistencies in thickness data.
    
    Args:
        thicknesses: Array of thickness values
        
    Returns:
        Dictionary with error detection results
    """
    if len(thicknesses) < 10:
        return {
            'systematic_errors_detected': False,
            'random_errors_detected': False,
            'precision_issues_detected': False,
            'error_count': 0,
            'error_regions': [],
            'precision_score': 1.0
        }
    
    # Remove invalid values for analysis
    valid_mask = (thicknesses > 0) & ~np.isnan(thicknesses)
    valid_thicknesses = thicknesses[valid_mask]
    
    if len(valid_thicknesses) < 10:
        return {
            'systematic_errors_detected': False,
            'random_errors_detected': False,
            'precision_issues_detected': False,
            'error_count': len(thicknesses) - len(valid_thicknesses),
            'error_regions': [],
            'precision_score': 0.0
        }
    
    # Detect systematic errors using change point detection
    systematic_errors_detected = False
    error_regions = []
    
    # Simple change point detection using rolling statistics
    window_size = min(50, len(valid_thicknesses) // 4)
    if window_size > 10:
        rolling_mean = np.convolve(valid_thicknesses, np.ones(window_size)/window_size, mode='valid')
        
        # Detect significant changes in mean
        mean_threshold = 1.5 * np.std(rolling_mean)  # Reduced threshold for better sensitivity
        change_points = np.where(np.abs(np.diff(rolling_mean)) > mean_threshold)[0]
        
        if len(change_points) > 0:
            systematic_errors_detected = True
            for cp in change_points:
                error_regions.append({
                    'start_index': cp,
                    'end_index': cp + window_size,
                    'change_magnitude': np.abs(np.diff(rolling_mean)[cp])
                })
    
    # Alternative detection: check for significant mean differences between segments
    if not systematic_errors_detected and len(valid_thicknesses) > 20:
        # Split data into segments and compare means
        segment_size = len(valid_thicknesses) // 4
        segments = [valid_thicknesses[i:i+segment_size] for i in range(0, len(valid_thicknesses), segment_size)]
        
        if len(segments) >= 2:
            segment_means = [np.mean(seg) for seg in segments if len(seg) > 0]
            overall_mean = np.mean(valid_thicknesses)
            overall_std = np.std(valid_thicknesses)
            
            # Check if any segment mean deviates significantly
            for i, seg_mean in enumerate(segment_means):
                if abs(seg_mean - overall_mean) > 1.0 * overall_std:  # 1 standard deviation threshold
                    systematic_errors_detected = True
                    error_regions.append({
                        'start_index': i * segment_size,
                        'end_index': min((i + 1) * segment_size, len(valid_thicknesses)),
                        'change_magnitude': abs(seg_mean - overall_mean)
                    })
    
    # Detect random errors using outlier detection
    random_errors_detected = False
    error_count = 0
    
    q1, q3 = np.percentile(valid_thicknesses, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 2.5 * iqr
    upper_bound = q3 + 2.5 * iqr
    
    outlier_mask = (valid_thicknesses < lower_bound) | (valid_thicknesses > upper_bound)
    if np.any(outlier_mask):
        random_errors_detected = True
        error_count = np.sum(outlier_mask)
    
    # Detect precision issues
    precision_issues_detected = False
    precision_score = 1.0
    
    # Check for too many identical values
    unique_ratio = len(np.unique(valid_thicknesses)) / len(valid_thicknesses)
    if unique_ratio < 0.7:  # Less than 70% unique values (more sensitive)
        precision_issues_detected = True
        precision_score = unique_ratio
    
    # Check for rounding patterns
    rounded_values_0 = np.round(valid_thicknesses, 0)
    rounding_ratio_0 = np.sum(rounded_values_0 == valid_thicknesses) / len(valid_thicknesses)
    if rounding_ratio_0 > 0.15:  # More than 15% are rounded to integer
        precision_issues_detected = True
        precision_score = min(precision_score, 0.5)
    
    rounded_values_1 = np.round(valid_thicknesses, 1)
    rounding_ratio_1 = np.sum(rounded_values_1 == valid_thicknesses) / len(valid_thicknesses)
    if rounding_ratio_1 > 0.5:  # More than 50% are rounded to 1 decimal
        precision_issues_detected = True
        precision_score = min(precision_score, 0.7)
    
    # Additional: check for high proportion of exactly integer values
    int_mask = (valid_thicknesses == np.floor(valid_thicknesses))
    if np.sum(int_mask) / len(valid_thicknesses) > 0.15:
        precision_issues_detected = True
        precision_score = min(precision_score, 0.5)
    
    total_errors = error_count + len(thicknesses) - len(valid_thicknesses)
    
    return {
        'systematic_errors_detected': systematic_errors_detected,
        'random_errors_detected': random_errors_detected,
        'precision_issues_detected': precision_issues_detected,
        'error_count': total_errors,
        'error_regions': error_regions,
        'precision_score': precision_score,
        'outlier_count': error_count,
        'invalid_count': len(thicknesses) - len(valid_thicknesses),
        'total_errors': total_errors
    }


def calculate_thickness_quality_metrics(thicknesses: np.ndarray) -> dict:
    """
    Calculate comprehensive quality metrics for thickness data.
    
    Args:
        thicknesses: Array of thickness values
        
    Returns:
        Dictionary with quality metrics
    """
    if len(thicknesses) == 0:
        return {
            'overall_quality_score': 0.0,
            'precision_score': 0.0,
            'consistency_score': 0.0,
            'coverage_score': 0.0,
            'variance_score': 0.0,
            'outlier_score': 0.0,
            'distribution_score': 0.0
        }
    
    # Remove invalid values
    valid_mask = (thicknesses > 0) & ~np.isnan(thicknesses)
    valid_thicknesses = thicknesses[valid_mask]
    
    if len(valid_thicknesses) == 0:
        return {
            'overall_quality_score': 0.0,
            'precision_score': 0.0,
            'consistency_score': 0.0,
            'coverage_score': 0.0,
            'variance_score': 0.0,
            'outlier_score': 0.0,
            'distribution_score': 0.0
        }
    
    # Coverage score (percentage of valid data)
    coverage_score = len(valid_thicknesses) / len(thicknesses)
    
    # Precision score (based on unique values and rounding)
    unique_ratio = len(np.unique(valid_thicknesses)) / len(valid_thicknesses)
    rounded_values = np.round(valid_thicknesses, 2)
    rounding_ratio = np.sum(rounded_values == valid_thicknesses) / len(valid_thicknesses)
    precision_score = unique_ratio * (1 - rounding_ratio * 0.5)
    
    # Consistency score (based on variance and outliers)
    mean_thickness = np.mean(valid_thicknesses)
    std_thickness = np.std(valid_thicknesses)
    cv = std_thickness / mean_thickness if mean_thickness > 0 else 0
    
    # Lower CV is better for consistency
    consistency_score = max(0, 1 - cv)
    
    # Variance score (appropriate amount of variation)
    # Too little variance suggests measurement issues, too much suggests errors
    optimal_cv = 0.2  # 20% coefficient of variation
    cv_diff = abs(cv - optimal_cv)
    variance_score = max(0, 1 - cv_diff / optimal_cv)
    
    # Outlier score (fewer outliers is better)
    q1, q3 = np.percentile(valid_thicknesses, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (valid_thicknesses < lower_bound) | (valid_thicknesses > upper_bound)
    outlier_ratio = np.sum(outlier_mask) / len(valid_thicknesses)
    outlier_score = max(0, 1 - 2 * outlier_ratio)  # Penalize outliers more strongly
    
    # Distribution score (how well it follows expected distribution)
    # Check if distribution is roughly normal or log-normal
    from scipy import stats
    
    # Test for normality
    try:
        _, normality_p = stats.normaltest(valid_thicknesses)
        normality_score = min(1.0, normality_p * 10)
    except:
        normality_score = 0.5
    
    # Test for log-normality
    try:
        log_thicknesses = np.log(valid_thicknesses)
        _, log_normality_p = stats.normaltest(log_thicknesses)
        log_normality_score = min(1.0, log_normality_p * 10)
    except:
        log_normality_score = 0.5
    
    distribution_score = max(normality_score, log_normality_score)
    
    # Overall quality score (balanced weighted average)
    weights = {
        'coverage': 0.20,
        'precision': 0.25,
        'consistency': 0.20,
        'variance': 0.15,
        'outlier': 0.15,
        'distribution': 0.05
    }
    
    overall_quality_score = (
        coverage_score * weights['coverage'] +
        precision_score * weights['precision'] +
        consistency_score * weights['consistency'] +
        variance_score * weights['variance'] +
        outlier_score * weights['outlier'] +
        distribution_score * weights['distribution']
    )
    
    # Clamp score to 0-1
    overall_quality_score = max(0, min(1, overall_quality_score))
    
    return {
        'overall_quality_score': overall_quality_score,
        'precision_score': precision_score,
        'consistency_score': consistency_score,
        'coverage_score': coverage_score,
        'variance_score': variance_score,
        'outlier_score': outlier_score,
        'distribution_score': distribution_score,
        'coefficient_of_variation': cv,
        'outlier_ratio': outlier_ratio,
        'unique_ratio': unique_ratio
    }


def perform_thickness_quality_assurance(thicknesses: np.ndarray) -> dict:
    """
    Perform comprehensive quality assurance on thickness data.
    
    Args:
        thicknesses: Array of thickness values
        
    Returns:
        Dictionary with QA results and recommendations
    """
    # Run all quality checks
    validation_results = validate_thickness_data(thicknesses)
    error_detection_results = detect_measurement_errors(thicknesses)
    quality_metrics = calculate_thickness_quality_metrics(thicknesses)
    
    # Assess overall quality
    overall_score = quality_metrics['overall_quality_score']
    
    if overall_score >= 0.8:
        quality_assessment = 'excellent'
        passes_quality_check = True
    elif overall_score >= 0.6:
        quality_assessment = 'good'
        passes_quality_check = True
    elif overall_score >= 0.4:
        quality_assessment = 'fair'
        passes_quality_check = False
    else:
        quality_assessment = 'poor'
        passes_quality_check = False
    
    # Generate recommendations
    recommendations = []
    
    if validation_results['error_count'] > 0:
        recommendations.append(f"Remove {validation_results['error_count']} invalid data points")
    
    if error_detection_results['systematic_errors_detected']:
        recommendations.append("Investigate systematic measurement errors")
    
    if error_detection_results['random_errors_detected']:
        recommendations.append("Review outlier detection and measurement procedures")
    
    if error_detection_results['precision_issues_detected']:
        recommendations.append("Improve measurement precision and resolution")
    
    if quality_metrics['coverage_score'] < 0.9:
        recommendations.append("Increase data coverage and reduce missing values")
    
    if quality_metrics['consistency_score'] < 0.7:
        recommendations.append("Improve measurement consistency and calibration")
    
    if quality_metrics['variance_score'] < 0.6:
        recommendations.append("Review measurement variability and sampling strategy")
    
    if not recommendations:
        recommendations.append("Data quality is acceptable for analysis")
    
    return {
        'validation_results': validation_results,
        'error_detection_results': error_detection_results,
        'quality_metrics': quality_metrics,
        'quality_assessment': quality_assessment,
        'overall_quality_score': overall_score,
        'passes_quality_check': passes_quality_check,
        'recommendations': recommendations
    }


def clean_thickness_data(thicknesses: np.ndarray, 
                        remove_negative: bool = True,
                        remove_nan: bool = True,
                        remove_extreme: bool = True,
                        extreme_threshold: float = 3.0) -> dict:
    """
    Clean thickness data by removing or correcting invalid values.
    
    Args:
        thicknesses: Array of thickness values
        remove_negative: Whether to remove negative values
        remove_nan: Whether to remove NaN values
        remove_extreme: Whether to remove extreme outliers
        extreme_threshold: IQR multiplier for extreme value detection
        
    Returns:
        Dictionary with cleaned data and cleaning summary
    """
    original_count = len(thicknesses)
    cleaned_data = thicknesses.copy()
    
    cleaning_summary = {
        'negative_values_removed': 0,
        'nan_values_removed': 0,
        'extreme_values_removed': 0,
        'zero_values_removed': 0
    }
    
    # Remove negative values
    if remove_negative:
        negative_mask = cleaned_data < 0
        cleaning_summary['negative_values_removed'] = np.sum(negative_mask)
        cleaned_data = cleaned_data[~negative_mask]
    
    # Remove NaN values
    if remove_nan:
        nan_mask = np.isnan(cleaned_data)
        cleaning_summary['nan_values_removed'] = np.sum(nan_mask)
        cleaned_data = cleaned_data[~nan_mask]
    
    # Remove extreme values
    if remove_extreme and len(cleaned_data) > 10:
        q1, q3 = np.percentile(cleaned_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - extreme_threshold * iqr
        upper_bound = q3 + extreme_threshold * iqr
        
        extreme_mask = (cleaned_data < lower_bound) | (cleaned_data > upper_bound)
        cleaning_summary['extreme_values_removed'] = np.sum(extreme_mask)
        cleaned_data = cleaned_data[~extreme_mask]
    
    # Remove zero values (optional)
    zero_mask = cleaned_data == 0
    if np.any(zero_mask):
        cleaning_summary['zero_values_removed'] = np.sum(zero_mask)
        cleaned_data = cleaned_data[~zero_mask]
    
    removed_count = original_count - len(cleaned_data)
    
    return {
        'cleaned_data': cleaned_data,
        'cleaning_summary': cleaning_summary,
        'removed_count': removed_count,
        'original_count': original_count,
        'retention_rate': len(cleaned_data) / original_count
    } 