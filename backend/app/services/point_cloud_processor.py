"""
Point cloud processing service using PyVista and NumPy
"""
import numpy as np
import pyvista as pv
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PointCloudProcessor:
    """
    Handles point cloud manipulation and optimization
    """
    
    def __init__(self):
        self.supported_operations = ['filter', 'downsample', 'outlier_removal', 'transform']
    
    def filter_by_bounds(self, points: np.ndarray, min_bound: List[float], max_bound: List[float]) -> np.ndarray:
        """
        Filter point cloud by bounding box
        
        Args:
            points: Nx3 array of point coordinates
            min_bound: [x_min, y_min, z_min]
            max_bound: [x_max, y_max, z_max]
            
        Returns:
            Filtered point cloud
        """
        if len(points) == 0:
            return points
        
        # Validate input dimensions
        if points.shape[1] != 3:
            raise ValueError("Points must have 3 columns (x, y, z)")
        
        if len(min_bound) != 3 or len(max_bound) != 3:
            raise ValueError("Bounds must have 3 values (x, y, z)")
            
        mask = np.all((points >= min_bound) & (points <= max_bound), axis=1)
        return points[mask]
    
    def downsample(self, points: np.ndarray, target_points: int, method: str = 'uniform') -> np.ndarray:
        """
        Downsample point cloud for performance
        
        Args:
            points: Nx3 array of point coordinates
            target_points: Target number of points
            method: Downsampling method ('uniform', 'random')
            
        Returns:
            Downsampled point cloud
        """
        # Validate input dimensions
        if points.shape[1] != 3:
            raise ValueError("Points must have 3 columns (x, y, z)")
        
        if len(points) <= target_points:
            return points
            
        if method == 'uniform':
            # Uniform sampling
            step = len(points) // target_points
            indices = np.arange(0, len(points), step)[:target_points]
        elif method == 'random':
            # Random sampling
            indices = np.random.choice(len(points), target_points, replace=False)
        else:
            raise ValueError(f"Unknown downsampling method: {method}")
            
        return points[indices]
    
    def remove_outliers(self, points: np.ndarray, std_dev: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers from point cloud
        
        Args:
            points: Nx3 array of point coordinates
            std_dev: Standard deviation threshold for outlier detection
            
        Returns:
            Point cloud with outliers removed
        """
        if len(points) == 0:
            return points
        
        # For small datasets or when std_dev is low, be more conservative
        if len(points) < 10 or std_dev < 1.5:
            return points
            
        # Calculate distances from centroid
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        
        # Calculate mean and standard deviation of distances
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        
        # Filter points within threshold
        threshold = mean_dist + std_dev * std_dist
        mask = distances <= threshold
        
        return points[mask]
    
    def apply_coordinate_transform(self, points: np.ndarray, scale: float = 1.0, 
                                 rotation: float = 0.0, translation: List[float] = [0, 0, 0]) -> np.ndarray:
        """
        Apply coordinate transformation to point cloud
        
        Args:
            points: Nx3 array of point coordinates
            scale: Scaling factor
            rotation: Rotation angle in degrees (around Z-axis)
            translation: Translation vector [x, y, z]
            
        Returns:
            Transformed point cloud
        """
        if len(points) == 0:
            return points
            
        # Apply scaling
        scaled = points * scale
        
        # Apply rotation (around Z-axis)
        if rotation != 0:
            angle_rad = np.radians(rotation)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
            scaled = scaled @ rotation_matrix.T
            
        # Apply translation
        transformed = scaled + translation
        
        return transformed
    
    def create_mesh_from_points(self, points: np.ndarray, method: str = 'delaunay') -> Optional[pv.PolyData]:
        """
        Create mesh from point cloud using PyVista
        
        Args:
            points: Nx3 array of point coordinates
            method: Meshing method ('delaunay', 'alpha_shape')
            
        Returns:
            PyVista mesh or None if failed
        """
        if len(points) < 3:
            return None
        
        # Validate method before attempting mesh creation
        if method not in ['delaunay', 'alpha_shape']:
            raise ValueError(f"Unknown meshing method: {method}")
            
        try:
            point_cloud = pv.PolyData(points)
            
            if method == 'delaunay':
                mesh = point_cloud.delaunay_2d()
            elif method == 'alpha_shape':
                # PyVista doesn't have alpha shape, use convex hull as approximation
                mesh = point_cloud.delaunay_3d()
            else:
                raise ValueError(f"Unknown meshing method: {method}")
                
            return mesh
            
        except Exception as e:
            logger.error(f"Failed to create mesh from points: {str(e)}")
            return None
    
    def validate_point_cloud(self, points: np.ndarray) -> bool:
        """
        Validate point cloud data
        
        Args:
            points: Nx3 array of point coordinates
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(points, np.ndarray):
            return False
            
        if points.ndim != 2 or points.shape[1] != 3:
            return False
            
        if len(points) == 0:
            return False
            
        # Check for NaN or infinite values
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False
            
        return True
    
    def get_point_cloud_stats(self, points: np.ndarray) -> dict:
        """
        Get statistics about point cloud
        
        Args:
            points: Nx3 array of point coordinates
            
        Returns:
            Dictionary with point cloud statistics
        """
        if not self.validate_point_cloud(points):
            return {'error': 'Invalid point cloud'}
            
        stats = {
            'point_count': len(points),
            'bounds': {
                'x_min': float(np.min(points[:, 0])),
                'x_max': float(np.max(points[:, 0])),
                'y_min': float(np.min(points[:, 1])),
                'y_max': float(np.max(points[:, 1])),
                'z_min': float(np.min(points[:, 2])),
                'z_max': float(np.max(points[:, 2]))
            },
            'centroid': points.mean(axis=0).tolist(),
            'density': len(points) / (np.prod(np.ptp(points, axis=0)) + 1e-10)
        }
        
        return stats 