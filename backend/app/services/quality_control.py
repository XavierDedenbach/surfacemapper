"""
Quality control service for surface data analysis
"""
import numpy as np
import time
from typing import Dict, List, Any, Optional
from app.models.data_models import QualityMetrics
import logging

logger = logging.getLogger(__name__)

class QualityControl:
    """
    Quality control service for surface data validation and metrics calculation
    """
    
    def __init__(self):
        self.min_points_for_triangulation = 3
        
    def validate_surface_data(self, surface_data: np.ndarray) -> Dict[str, Any]:
        """
        Validate surface data for quality and integrity
        
        Args:
            surface_data: 3D surface points [N, 3]
            
        Returns:
            Dictionary with validation results
        """
        if surface_data is None:
            raise ValueError("Surface data cannot be None")
            
        if not isinstance(surface_data, np.ndarray):
            raise ValueError("Surface data must be a numpy array")
            
        if surface_data.ndim != 2 or surface_data.shape[1] != 3:
            raise ValueError("Surface data must be 2D array with 3 columns (x, y, z)")
            
        errors = []
        
        # Check for empty surface
        if len(surface_data) == 0:
            errors.append("Surface data is empty")
            
        # Check for insufficient points
        if len(surface_data) < self.min_points_for_triangulation:
            errors.append(f"Insufficient points for triangulation (need at least {self.min_points_for_triangulation})")
            
        # Check for NaN values
        if np.any(np.isnan(surface_data)):
            errors.append("Surface data contains NaN values")
            
        # Check for infinite values
        if np.any(np.isinf(surface_data)):
            errors.append("Surface data contains infinite values")
            
        # Check for duplicate points
        unique_points = np.unique(surface_data, axis=0)
        if len(unique_points) < len(surface_data):
            errors.append("Surface data contains duplicate points")
            
        result = {'is_valid': len(errors) == 0}
        if errors:
            result['errors'] = errors
        return result
        
    def detect_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Detect outliers in data using Z-score method
        
        Args:
            data: Input data array
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Boolean array indicating outlier positions
        """
        if len(data) == 0:
            return np.array([], dtype=bool)
            
        # Handle case where all values are the same (no variance)
        if np.std(data) == 0:
            return np.zeros(len(data), dtype=bool)
            
        z_scores = np.abs((data - np.mean(data)) / np.std(data))
        return z_scores > threshold
        
    def remove_outliers(self, data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
        """
        Remove outliers from data
        
        Args:
            data: Input data array
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Data array with outliers removed
        """
        outlier_mask = self.detect_outliers(data, threshold)
        return data[~outlier_mask]
        
    def calculate_point_density(self, surface_data: np.ndarray, area: float) -> float:
        """
        Calculate point density per unit area
        
        Args:
            surface_data: 3D surface points [N, 3]
            area: Total area in square units
            
        Returns:
            Point density (points per unit area)
        """
        if area <= 0:
            return 0.0
        return len(surface_data) / area
        
    def calculate_surface_coverage(self, surface_data: np.ndarray, boundary_area: float) -> float:
        """
        Calculate surface coverage ratio
        
        Args:
            surface_data: 3D surface points [N, 3]
            boundary_area: Total boundary area
            
        Returns:
            Coverage ratio (0-1)
        """
        if boundary_area <= 0 or len(surface_data) == 0:
            return 0.0
            
        # Estimate surface area using convex hull or bounding box
        try:
            # Simple bounding box approach
            x_range = np.max(surface_data[:, 0]) - np.min(surface_data[:, 0])
            y_range = np.max(surface_data[:, 1]) - np.min(surface_data[:, 1])
            surface_area = x_range * y_range
            return min(surface_area / boundary_area, 1.0)
        except:
            return 0.0
            
    def calculate_data_completeness(self, surface_data: np.ndarray) -> float:
        """
        Calculate data completeness ratio
        
        Args:
            surface_data: 3D surface points [N, 3]
            
        Returns:
            Completeness ratio (0-1)
        """
        if len(surface_data) == 0:
            return 0.0
            
        # Count valid (non-NaN) points
        valid_points = np.sum(~np.isnan(surface_data).any(axis=1))
        return valid_points / len(surface_data)
        
    def estimate_noise_level(self, surface_data: np.ndarray) -> float:
        """
        Estimate noise level in surface data
        
        Args:
            surface_data: 3D surface points [N, 3]
            
        Returns:
            Noise level estimate (0-1)
        """
        if len(surface_data) < 3:
            return 0.0
            
        # Calculate local variance as noise estimate
        try:
            # Use Z-coordinate variance as noise indicator
            z_variance = np.var(surface_data[:, 2])
            # Normalize to 0-1 range (assuming reasonable noise levels)
            noise_level = min(z_variance / 10.0, 1.0)  # Normalize by expected variance
            return max(noise_level, 0.0)
        except:
            return 0.0
            
    def estimate_accuracy(self, surface_data: np.ndarray) -> float:
        """
        Estimate accuracy of surface data
        
        Args:
            surface_data: 3D surface points [N, 3]
            
        Returns:
            Accuracy estimate (0-1)
        """
        if len(surface_data) < 3:
            return 0.0
            
        # Simple accuracy estimate based on point distribution
        try:
            # Calculate point spacing consistency
            distances = []
            for i in range(min(100, len(surface_data))):  # Sample for performance
                for j in range(i+1, min(i+10, len(surface_data))):
                    dist = np.linalg.norm(surface_data[i] - surface_data[j])
                    if dist > 0:
                        distances.append(dist)
                        
            if len(distances) == 0:
                return 0.5  # Default accuracy
                
            # Consistency in spacing indicates accuracy
            spacing_std = np.std(distances)
            spacing_mean = np.mean(distances)
            if spacing_mean > 0:
                consistency = 1.0 - min(spacing_std / spacing_mean, 1.0)
                return max(consistency, 0.0)
            else:
                return 0.5
        except:
            return 0.5
            
    def estimate_precision(self, surface_data: np.ndarray) -> float:
        """
        Estimate precision of surface data
        
        Args:
            surface_data: 3D surface points [N, 3]
            
        Returns:
            Precision estimate (0-1)
        """
        if len(surface_data) < 3:
            return 0.0
            
        # Precision based on point density and distribution
        try:
            # Calculate point density
            x_range = np.max(surface_data[:, 0]) - np.min(surface_data[:, 0])
            y_range = np.max(surface_data[:, 1]) - np.min(surface_data[:, 1])
            area = x_range * y_range
            
            if area <= 0:
                return 0.5
                
            density = len(surface_data) / area
            # Normalize density to 0-1 (assuming reasonable densities)
            precision = min(density / 100.0, 1.0)  # Normalize by expected density
            return max(precision, 0.0)
        except:
            return 0.5
            
    def calculate_reliability_score(self, surface_data: np.ndarray) -> float:
        """
        Calculate overall reliability score
        
        Args:
            surface_data: 3D surface points [N, 3]
            
        Returns:
            Reliability score (0-1)
        """
        if len(surface_data) < 3:
            return 0.0
            
        # Combine multiple factors for reliability
        completeness = self.calculate_data_completeness(surface_data)
        accuracy = self.estimate_accuracy(surface_data)
        precision = self.estimate_precision(surface_data)
        
        # Weighted average
        reliability = (completeness * 0.4 + accuracy * 0.3 + precision * 0.3)
        return max(min(reliability, 1.0), 0.0)
        
    def assess_quality_flags(self, surface_data: np.ndarray) -> Dict[str, bool]:
        """
        Assess quality flags for surface data
        
        Args:
            surface_data: 3D surface points [N, 3]
            
        Returns:
            Dictionary of quality flags
        """
        validation = self.validate_surface_data(surface_data)
        
        return {
            'has_sufficient_points': bool(len(surface_data) >= self.min_points_for_triangulation),
            'no_nan_values': bool(not np.any(np.isnan(surface_data)) if len(surface_data) > 0 else True),
            'no_infinite_values': bool(not np.any(np.isinf(surface_data)) if len(surface_data) > 0 else True),
            'no_duplicates': bool(len(np.unique(surface_data, axis=0)) == len(surface_data) if len(surface_data) > 0 else True),
            'is_valid': bool(validation['is_valid']),
            'has_good_coverage': bool(self.calculate_surface_coverage(surface_data, 1.0) > 0.1 if len(surface_data) > 0 else False),
            'has_good_completeness': bool(self.calculate_data_completeness(surface_data) > 0.9 if len(surface_data) > 0 else False)
        }
        
    def calculate_quality_metrics(self, surface_data: np.ndarray, boundary_area: float = 1.0) -> QualityMetrics:
        """
        Calculate comprehensive quality metrics
        
        Args:
            surface_data: 3D surface points [N, 3]
            boundary_area: Total boundary area
            
        Returns:
            QualityMetrics object
        """
        if len(surface_data) == 0:
            return QualityMetrics(
                point_density=0.0,
                surface_coverage=0.0,
                data_completeness=0.0,
                noise_level=0.0,
                accuracy_estimate=0.0,
                precision_estimate=0.0,
                reliability_score=0.0,
                quality_flags={}
            )
            
        point_density = self.calculate_point_density(surface_data, boundary_area)
        surface_coverage = self.calculate_surface_coverage(surface_data, boundary_area)
        data_completeness = self.calculate_data_completeness(surface_data)
        noise_level = self.estimate_noise_level(surface_data)
        accuracy_estimate = self.estimate_accuracy(surface_data)
        precision_estimate = self.estimate_precision(surface_data)
        reliability_score = self.calculate_reliability_score(surface_data)
        quality_flags = self.assess_quality_flags(surface_data)
        
        return QualityMetrics(
            point_density=point_density,
            surface_coverage=surface_coverage,
            data_completeness=data_completeness,
            noise_level=noise_level,
            accuracy_estimate=accuracy_estimate,
            precision_estimate=precision_estimate,
            reliability_score=reliability_score,
            quality_flags=quality_flags
        )
        
    def _get_time(self) -> float:
        """Get current time for performance testing"""
        return time.time() 