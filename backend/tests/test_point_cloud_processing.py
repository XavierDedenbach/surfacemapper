"""
Comprehensive point cloud processing tests for Major Task 6.1.3
"""
import pytest
import numpy as np
import time
from unittest.mock import Mock, patch
from app.services.point_cloud_processor import PointCloudProcessor


class TestPointCloudProcessing:
    """Comprehensive test cases for PointCloudProcessor"""
    
    @pytest.fixture
    def point_cloud_processor(self):
        return PointCloudProcessor()
    
    @pytest.fixture
    def sample_points(self):
        """Sample point cloud data for testing"""
        return np.array([
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0]
        ], dtype=np.float32)
    
    @pytest.fixture
    def large_point_cloud(self):
        """Large point cloud for performance testing"""
        return np.random.rand(10000, 3).astype(np.float32)
    
    def test_filter_by_bounds_all_points_inside(self, point_cloud_processor, sample_points):
        """Test filtering when all points are inside bounds"""
        min_bound = [0.0, 0.0, 0.0]
        max_bound = [5.0, 5.0, 5.0]
        
        result = point_cloud_processor.filter_by_bounds(sample_points, min_bound, max_bound)
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_points)  # All points should remain
        assert result.shape[1] == 3
    
    def test_filter_by_bounds_some_points_outside(self, point_cloud_processor, sample_points):
        """Test filtering when some points are outside bounds"""
        min_bound = [1.5, 1.5, 1.5]
        max_bound = [3.5, 3.5, 3.5]
        
        result = point_cloud_processor.filter_by_bounds(sample_points, min_bound, max_bound)
        assert isinstance(result, np.ndarray)
        assert len(result) == 2  # Only points [2,2,2] and [3,3,3] should remain
        assert np.allclose(result[0], [2.0, 2.0, 2.0])
        assert np.allclose(result[1], [3.0, 3.0, 3.0])
    
    def test_filter_by_bounds_no_points_inside(self, point_cloud_processor, sample_points):
        """Test filtering when no points are inside bounds"""
        min_bound = [10.0, 10.0, 10.0]
        max_bound = [15.0, 15.0, 15.0]
        
        result = point_cloud_processor.filter_by_bounds(sample_points, min_bound, max_bound)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0  # No points should remain
    
    def test_filter_by_bounds_empty_point_cloud(self, point_cloud_processor):
        """Test filtering with empty point cloud"""
        empty_points = np.empty((0, 3), dtype=np.float32)
        min_bound = [0.0, 0.0, 0.0]
        max_bound = [1.0, 1.0, 1.0]
        
        result = point_cloud_processor.filter_by_bounds(empty_points, min_bound, max_bound)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_filter_by_bounds_invalid_bounds(self, point_cloud_processor, sample_points):
        """Test filtering with invalid bounds"""
        min_bound = [5.0, 5.0, 5.0]  # min > max
        max_bound = [1.0, 1.0, 1.0]
        
        result = point_cloud_processor.filter_by_bounds(sample_points, min_bound, max_bound)
        assert isinstance(result, np.ndarray)
        assert len(result) == 0  # No points should be inside invalid bounds
    
    def test_downsample_uniform_method(self, point_cloud_processor, large_point_cloud):
        """Test uniform downsampling"""
        target_points = 1000
        result = point_cloud_processor.downsample(large_point_cloud, target_points, method='uniform')
        
        assert isinstance(result, np.ndarray)
        assert len(result) == target_points
        assert result.shape[1] == 3
    
    def test_downsample_random_method(self, point_cloud_processor, large_point_cloud):
        """Test random downsampling"""
        target_points = 1000
        result = point_cloud_processor.downsample(large_point_cloud, target_points, method='random')
        
        assert isinstance(result, np.ndarray)
        assert len(result) == target_points
        assert result.shape[1] == 3
    
    def test_downsample_no_reduction_needed(self, point_cloud_processor, sample_points):
        """Test downsampling when target is larger than input"""
        target_points = 10
        result = point_cloud_processor.downsample(sample_points, target_points)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_points)  # Should return original points
        assert np.array_equal(result, sample_points)
    
    def test_downsample_invalid_method(self, point_cloud_processor, sample_points):
        """Test downsampling with invalid method"""
        with pytest.raises(ValueError):
            point_cloud_processor.downsample(sample_points, 3, method='invalid_method')
    
    def test_remove_outliers_normal_data(self, point_cloud_processor):
        """Test outlier removal with normal data"""
        # Create normal data with a few outliers
        normal_points = np.random.randn(100, 3) * 10
        outliers = np.array([[1000, 1000, 1000], [-1000, -1000, -1000]])
        all_points = np.vstack([normal_points, outliers])
        
        result = point_cloud_processor.remove_outliers(all_points, std_dev=2.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) < len(all_points)  # Should remove some points
        assert len(result) >= len(normal_points) * 0.8  # Should keep most normal points
    
    def test_remove_outliers_no_outliers(self, point_cloud_processor, sample_points):
        """Test outlier removal when no outliers exist"""
        result = point_cloud_processor.remove_outliers(sample_points, std_dev=1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(sample_points)  # Should keep all points
        assert np.array_equal(result, sample_points)
    
    def test_remove_outliers_all_outliers(self, point_cloud_processor):
        """Test outlier removal when all points are outliers"""
        outlier_points = np.array([[1000, 1000, 1000], [-1000, -1000, -1000]])
        
        result = point_cloud_processor.remove_outliers(outlier_points, std_dev=1.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) <= len(outlier_points)  # Should remove some or all points
    
    def test_remove_outliers_empty_point_cloud(self, point_cloud_processor):
        """Test outlier removal with empty point cloud"""
        empty_points = np.empty((0, 3), dtype=np.float32)
        
        result = point_cloud_processor.remove_outliers(empty_points, std_dev=2.0)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_apply_coordinate_transform_scale_only(self, point_cloud_processor, sample_points):
        """Test coordinate transformation with scaling only"""
        scale = 2.0
        result = point_cloud_processor.apply_coordinate_transform(sample_points, scale=scale)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_points.shape
        assert np.allclose(result, sample_points * scale)
    
    def test_apply_coordinate_transform_rotation_only(self, point_cloud_processor):
        """Test coordinate transformation with rotation only"""
        points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        rotation = 90.0  # 90 degrees around Z-axis
        
        result = point_cloud_processor.apply_coordinate_transform(points, rotation=rotation)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == points.shape
        
        # Check that [1,0,0] rotated 90° becomes approximately [0,1,0]
        expected_first = np.array([0.0, 1.0, 0.0])
        assert np.allclose(result[0], expected_first, atol=0.1)
    
    def test_apply_coordinate_transform_translation_only(self, point_cloud_processor, sample_points):
        """Test coordinate transformation with translation only"""
        translation = [10.0, 20.0, 30.0]
        result = point_cloud_processor.apply_coordinate_transform(sample_points, translation=translation)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_points.shape
        assert np.allclose(result, sample_points + translation)
    
    def test_apply_coordinate_transform_combined(self, point_cloud_processor, sample_points):
        """Test coordinate transformation with all transformations combined"""
        scale = 2.0
        rotation = 45.0
        translation = [1.0, 2.0, 3.0]
        
        result = point_cloud_processor.apply_coordinate_transform(
            sample_points, scale=scale, rotation=rotation, translation=translation
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_points.shape
        assert not np.array_equal(result, sample_points)  # Should be transformed
    
    def test_apply_coordinate_transform_empty_point_cloud(self, point_cloud_processor):
        """Test coordinate transformation with empty point cloud"""
        empty_points = np.empty((0, 3), dtype=np.float32)
        
        result = point_cloud_processor.apply_coordinate_transform(empty_points)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 0
    
    def test_create_mesh_from_points_delaunay_method(self, point_cloud_processor, sample_points):
        """Test mesh creation using Delaunay method"""
        result = point_cloud_processor.create_mesh_from_points(sample_points, method='delaunay')
        
        assert result is not None
        assert hasattr(result, 'n_points')
        assert hasattr(result, 'n_cells')
    
    def test_create_mesh_from_points_alpha_shape_method(self, point_cloud_processor, sample_points):
        """Test mesh creation using alpha shape method"""
        result = point_cloud_processor.create_mesh_from_points(sample_points, method='alpha_shape')
        
        assert result is not None
        assert hasattr(result, 'n_points')
        assert hasattr(result, 'n_cells')
    
    def test_create_mesh_from_points_invalid_method(self, point_cloud_processor, sample_points):
        """Test mesh creation with invalid method"""
        with pytest.raises(ValueError):
            point_cloud_processor.create_mesh_from_points(sample_points, method='invalid_method')
    
    def test_create_mesh_from_points_insufficient_points(self, point_cloud_processor):
        """Test mesh creation with insufficient points"""
        insufficient_points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])  # Only 2 points
        
        result = point_cloud_processor.create_mesh_from_points(insufficient_points)
        
        assert result is None  # Should fail with insufficient points
    
    def test_create_mesh_from_points_empty_point_cloud(self, point_cloud_processor):
        """Test mesh creation with empty point cloud"""
        empty_points = np.empty((0, 3), dtype=np.float32)
        
        result = point_cloud_processor.create_mesh_from_points(empty_points)
        
        assert result is None  # Should fail with empty point cloud
    
    def test_validate_point_cloud_valid_data(self, point_cloud_processor, sample_points):
        """Test validation with valid point cloud data"""
        result = point_cloud_processor.validate_point_cloud(sample_points)
        assert result is True
    
    def test_validate_point_cloud_invalid_type(self, point_cloud_processor):
        """Test validation with invalid data type"""
        invalid_data = [[0, 0, 0], [1, 1, 1]]  # List instead of numpy array
        
        result = point_cloud_processor.validate_point_cloud(invalid_data)
        assert result is False
    
    def test_validate_point_cloud_wrong_dimensions(self, point_cloud_processor):
        """Test validation with wrong dimensions"""
        wrong_dim_data = np.array([[0, 0], [1, 1]])  # 2D instead of 3D
        
        result = point_cloud_processor.validate_point_cloud(wrong_dim_data)
        assert result is False
    
    def test_validate_point_cloud_empty_data(self, point_cloud_processor):
        """Test validation with empty data"""
        empty_data = np.empty((0, 3), dtype=np.float32)
        
        result = point_cloud_processor.validate_point_cloud(empty_data)
        assert result is False
    
    def test_validate_point_cloud_nan_values(self, point_cloud_processor):
        """Test validation with NaN values"""
        nan_data = np.array([[0, 0, 0], [1, 1, np.nan], [2, 2, 2]])
        
        result = point_cloud_processor.validate_point_cloud(nan_data)
        assert result is False
    
    def test_validate_point_cloud_infinite_values(self, point_cloud_processor):
        """Test validation with infinite values"""
        inf_data = np.array([[0, 0, 0], [1, 1, np.inf], [2, 2, 2]])
        
        result = point_cloud_processor.validate_point_cloud(inf_data)
        assert result is False
    
    def test_get_point_cloud_stats_valid_data(self, point_cloud_processor, sample_points):
        """Test statistics calculation with valid data"""
        result = point_cloud_processor.get_point_cloud_stats(sample_points)
        
        assert isinstance(result, dict)
        assert 'point_count' in result
        assert 'bounds' in result
        assert 'centroid' in result
        assert 'density' in result
        
        assert result['point_count'] == 5
        assert result['bounds']['x_min'] == 0.0
        assert result['bounds']['x_max'] == 4.0
        assert result['bounds']['y_min'] == 0.0
        assert result['bounds']['y_max'] == 4.0
        assert result['bounds']['z_min'] == 0.0
        assert result['bounds']['z_max'] == 4.0
    
    def test_get_point_cloud_stats_invalid_data(self, point_cloud_processor):
        """Test statistics calculation with invalid data"""
        invalid_data = np.array([[0, 0], [1, 1]])  # Wrong dimensions
        
        result = point_cloud_processor.get_point_cloud_stats(invalid_data)
        
        assert isinstance(result, dict)
        assert 'error' in result
        assert result['error'] == 'Invalid point cloud'
    
    def test_point_cloud_processing_workflow(self, point_cloud_processor, large_point_cloud):
        """Test complete point cloud processing workflow"""
        # Step 1: Filter by bounds
        filtered = point_cloud_processor.filter_by_bounds(
            large_point_cloud, min_bound=[0.0, 0.0, 0.0], max_bound=[0.5, 0.5, 0.5]
        )
        assert len(filtered) <= len(large_point_cloud)
        
        # Step 2: Downsample
        downsampled = point_cloud_processor.downsample(filtered, target_points=1000)
        assert len(downsampled) <= 1000
        
        # Step 3: Remove outliers
        cleaned = point_cloud_processor.remove_outliers(downsampled, std_dev=2.0)
        assert len(cleaned) <= len(downsampled)
        
        # Step 4: Apply transformation
        transformed = point_cloud_processor.apply_coordinate_transform(
            cleaned, scale=2.0, rotation=45.0, translation=[1.0, 2.0, 3.0]
        )
        assert transformed.shape == cleaned.shape
        
        # Step 5: Create mesh
        mesh = point_cloud_processor.create_mesh_from_points(transformed)
        if mesh is not None:
            assert hasattr(mesh, 'n_points')
            assert hasattr(mesh, 'n_cells')
        
        # Step 6: Get statistics
        stats = point_cloud_processor.get_point_cloud_stats(transformed)
        assert isinstance(stats, dict)
        assert 'point_count' in stats
    
    def test_point_cloud_processing_error_handling(self, point_cloud_processor):
        """Test error handling in point cloud processing"""
        # Test with invalid input data
        invalid_points = np.array([[1.0, 2.0], [3.0, 4.0]])  # Missing Z coordinate
        
        # These should raise ValueError for invalid data
        with pytest.raises(ValueError):
            point_cloud_processor.filter_by_bounds(invalid_points, [0, 0, 0], [1, 1, 1])
        
        with pytest.raises(ValueError):
            point_cloud_processor.downsample(invalid_points, 10)
        
        # Validation should catch invalid data
        is_valid = point_cloud_processor.validate_point_cloud(invalid_points)
        assert is_valid is False
    
    def test_point_cloud_processing_performance(self, point_cloud_processor):
        """Test point cloud processing performance with large datasets"""
        # Create large test dataset
        num_points = 50000
        points = np.random.rand(num_points, 3).astype(np.float32)
        
        # Test filtering performance
        start_time = time.time()
        filtered = point_cloud_processor.filter_by_bounds(points, [0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        filtering_time = time.time() - start_time
        
        # Test downsampling performance
        start_time = time.time()
        downsampled = point_cloud_processor.downsample(filtered, 5000)
        downsampling_time = time.time() - start_time
        
        # Test outlier removal performance
        start_time = time.time()
        cleaned = point_cloud_processor.remove_outliers(downsampled, std_dev=2.0)
        outlier_removal_time = time.time() - start_time
        
        # Performance assertions (should complete within reasonable time)
        assert filtering_time < 1.0  # Should complete within 1 second
        assert downsampling_time < 1.0  # Should complete within 1 second
        assert outlier_removal_time < 2.0  # Should complete within 2 seconds
        
        assert len(filtered) <= len(points)
        assert len(downsampled) <= len(filtered)
        assert len(cleaned) <= len(downsampled) 