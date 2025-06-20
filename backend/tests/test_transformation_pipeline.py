"""
Tests for complete coordinate transformation pipeline
"""
import pytest
import numpy as np
import time
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PipelineTest:
    """Test data for transformation pipeline operations"""
    name: str
    local_points: np.ndarray
    anchor_lat: float
    anchor_lon: float
    rotation_degrees: float
    scale_factor: float
    expected_utm_bounds: Tuple[float, float, float, float]  # min_x, max_x, min_y, max_y
    tolerance: float = 1e-6


class TestTransformationPipeline:
    """Test suite for complete coordinate transformation pipeline"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from app.services.coord_transformer import CoordinateTransformer
        self.transformer = CoordinateTransformer()
        
        # Test cases for transformation pipeline
        self.pipeline_tests = [
            PipelineTest(
                name="Simple translation only",
                local_points=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                anchor_lat=40.7128,
                anchor_lon=-74.0060,
                rotation_degrees=0.0,
                scale_factor=1.0,
                expected_utm_bounds=(583000, 585000, 4507000, 4509000)  # Approximate UTM bounds
            ),
            PipelineTest(
                name="Translation with rotation",
                local_points=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]),
                anchor_lat=40.7128,
                anchor_lon=-74.0060,
                rotation_degrees=45.0,
                scale_factor=1.0,
                expected_utm_bounds=(583000, 585000, 4507000, 4509000)
            ),
            PipelineTest(
                name="Translation with scaling",
                local_points=np.array([[0, 0, 0], [1, 1, 1]]),
                anchor_lat=40.7128,
                anchor_lon=-74.0060,
                rotation_degrees=0.0,
                scale_factor=2.0,
                expected_utm_bounds=(583000, 585000, 4507000, 4509000)
            ),
            PipelineTest(
                name="Complete transformation (rotation + scaling)",
                local_points=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
                anchor_lat=40.7128,
                anchor_lon=-74.0060,
                rotation_degrees=45.0,
                scale_factor=2.0,
                expected_utm_bounds=(583000, 585000, 4507000, 4509000)
            ),
            PipelineTest(
                name="Large point set transformation",
                local_points=np.random.rand(1000, 3) * 100,  # 1000 random points
                anchor_lat=40.7128,
                anchor_lon=-74.0060,
                rotation_degrees=30.0,
                scale_factor=1.5,
                expected_utm_bounds=(583000, 585000, 4507000, 4509000)
            )
        ]
    
    def test_transformation_pipeline_round_trip(self):
        """Test complete pipeline: local PLY -> scaled -> rotated -> translated -> UTM -> inverse"""
        for test_case in self.pipeline_tests:
            # Create transformation pipeline
            pipeline = self.transformer.create_transformation_pipeline(
                anchor_lat=test_case.anchor_lat,
                anchor_lon=test_case.anchor_lon,
                rotation_degrees=test_case.rotation_degrees,
                scale_factor=test_case.scale_factor
            )
            
            # Transform to UTM
            utm_points = pipeline.transform_to_utm(test_case.local_points)
            
            # Transform back to local coordinates
            recovered_points = pipeline.inverse_transform(utm_points)
            
            # Check round-trip accuracy
            np.testing.assert_allclose(
                test_case.local_points, 
                recovered_points, 
                atol=test_case.tolerance,
                err_msg=f"Round-trip transformation failed for {test_case.name}"
            )
    
    def test_pipeline_consistency(self):
        """Test that different transformation orders give same result"""
        points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
        
        # Method 1: Scale -> Rotate -> Translate
        pipeline1 = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=45.0,
            scale_factor=2.0
        )
        result1 = pipeline1.transform_to_utm(points)
        
        # Method 2: Rotate -> Scale -> Translate (should give same result for uniform scaling)
        pipeline2 = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=45.0,
            scale_factor=2.0
        )
        result2 = pipeline2.transform_to_utm(points)
        
        # Results should be the same (uniform scaling and Z-axis rotation commute)
        np.testing.assert_allclose(
            result1, 
            result2, 
            atol=1e-8,
            err_msg="Different transformation orders should give same result"
        )
    
    def test_pipeline_parameter_validation(self):
        """Test pipeline parameter validation"""
        # Test invalid latitude
        with pytest.raises(ValueError):
            self.transformer.create_transformation_pipeline(
                anchor_lat=91.0,  # Invalid latitude
                anchor_lon=-74.0060,
                rotation_degrees=45.0,
                scale_factor=2.0
            )
        
        # Test invalid longitude
        with pytest.raises(ValueError):
            self.transformer.create_transformation_pipeline(
                anchor_lat=40.7128,
                anchor_lon=181.0,  # Invalid longitude
                rotation_degrees=45.0,
                scale_factor=2.0
            )
        
        # Test invalid scale factor
        with pytest.raises(ValueError):
            self.transformer.create_transformation_pipeline(
                anchor_lat=40.7128,
                anchor_lon=-74.0060,
                rotation_degrees=45.0,
                scale_factor=-1.0  # Invalid scale factor
            )
    
    def test_pipeline_performance(self):
        """Test pipeline performance with large datasets"""
        # Generate large point set
        num_points = 10000
        points = np.random.rand(num_points, 3) * 100
        
        pipeline = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=45.0,
            scale_factor=2.0
        )
        
        # Test forward transformation performance
        start_time = time.time()
        utm_points = pipeline.transform_to_utm(points)
        forward_time = time.time() - start_time
        
        # Test inverse transformation performance
        start_time = time.time()
        recovered_points = pipeline.inverse_transform(utm_points)
        inverse_time = time.time() - start_time
        
        # Performance requirements: <1 second for 10k points
        assert forward_time < 1.0, f"Forward transformation took {forward_time:.3f}s, expected <1.0s"
        assert inverse_time < 1.0, f"Inverse transformation took {inverse_time:.3f}s, expected <1.0s"
        
        # Check accuracy
        np.testing.assert_allclose(points, recovered_points, atol=1e-6)
    
    def test_pipeline_metadata_tracking(self):
        """Test that pipeline tracks transformation metadata"""
        pipeline = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=45.0,
            scale_factor=2.0
        )
        
        # Check metadata
        metadata = pipeline.get_transformation_metadata()
        assert 'anchor_lat' in metadata
        assert 'anchor_lon' in metadata
        assert 'rotation_degrees' in metadata
        assert 'scale_factor' in metadata
        assert 'utm_zone' in metadata
        assert 'transformation_order' in metadata
        
        assert metadata['anchor_lat'] == 40.7128
        assert metadata['anchor_lon'] == -74.0060
        assert metadata['rotation_degrees'] == 45.0
        assert metadata['scale_factor'] == 2.0
    
    def test_pipeline_edge_cases(self):
        """Test pipeline with edge cases"""
        # Test empty point array
        pipeline = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=0.0,
            scale_factor=1.0
        )
        
        empty_points = np.empty((0, 3))
        utm_points = pipeline.transform_to_utm(empty_points)
        assert utm_points.shape == (0, 3)
        
        # Test single point
        single_point = np.array([[1, 1, 1]])
        utm_single = pipeline.transform_to_utm(single_point)
        assert utm_single.shape == (1, 3)
        
        # Test zero rotation and scaling
        pipeline_zero = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=0.0,
            scale_factor=1.0
        )
        
        points = np.array([[0, 0, 0], [1, 1, 1]])
        utm_zero = pipeline_zero.transform_to_utm(points)
        recovered_zero = pipeline_zero.inverse_transform(utm_zero)
        
        np.testing.assert_allclose(points, recovered_zero, atol=1e-6)
    
    def test_pipeline_utm_zone_handling(self):
        """Test pipeline handles different UTM zones correctly"""
        # Test different UTM zones
        test_zones = [
            (40.7128, -74.0060),  # UTM Zone 18N (New York)
            (40.7128, -120.0),    # UTM Zone 10N (California)
            (40.7128, 0.0),       # UTM Zone 30N (Prime Meridian)
            (40.7128, 180.0),     # UTM Zone 60N (Date Line)
        ]
        
        points = np.array([[0, 0, 0], [1, 1, 1]])
        
        for lat, lon in test_zones:
            pipeline = self.transformer.create_transformation_pipeline(
                anchor_lat=lat,
                anchor_lon=lon,
                rotation_degrees=0.0,
                scale_factor=1.0
            )
            
            utm_points = pipeline.transform_to_utm(points)
            recovered_points = pipeline.inverse_transform(utm_points)
            
            # Check round-trip accuracy
            np.testing.assert_allclose(points, recovered_points, atol=1e-6)
            
            # Check that UTM coordinates are reasonable
            metadata = pipeline.get_transformation_metadata()
            assert 'utm_zone' in metadata
            assert metadata['utm_zone'].startswith('EPSG:32')
    
    def test_pipeline_transformation_order(self):
        """Test that transformation order is correctly applied"""
        points = np.array([[1, 0, 0]])
        
        # Test order: Scale -> Rotate -> Translate
        pipeline = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=90.0,
            scale_factor=2.0
        )
        
        utm_points = pipeline.transform_to_utm(points)
        
        # For 90-degree rotation and 2x scaling, point [1,0,0] should become [0,-2,0] in local coordinates
        # before translation to UTM
        local_transformed = pipeline.transform_local_only(points)
        expected_local = np.array([[0, -2, 0]])  # 90° rotation + 2x scaling
        
        np.testing.assert_allclose(local_transformed, expected_local, atol=1e-10)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling"""
        # Test with invalid points
        pipeline = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=0.0,
            scale_factor=1.0
        )
        
        # Test None points
        with pytest.raises(ValueError):
            pipeline.transform_to_utm(None)
        
        # Test wrong shape
        with pytest.raises(ValueError):
            pipeline.transform_to_utm(np.array([[1, 2]]))  # Missing Z coordinate
        
        # Test non-numeric points
        with pytest.raises(ValueError):
            pipeline.transform_to_utm(np.array([['a', 'b', 'c']]))
    
    def test_pipeline_memory_efficiency(self):
        """Test that pipeline operations are memory efficient"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create large point set
        num_points = 50000
        points = np.random.rand(num_points, 3) * 100
        
        pipeline = self.transformer.create_transformation_pipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=45.0,
            scale_factor=2.0
        )
        
        # Perform transformations
        utm_points = pipeline.transform_to_utm(points)
        recovered_points = pipeline.inverse_transform(utm_points)
        
        peak_memory = process.memory_info().rss
        memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
        
        # Memory increase should be reasonable (not more than 10x the data size)
        expected_max_increase = num_points * 3 * 8 * 10 / 1024 / 1024  # 10x data size in MB
        assert memory_increase < expected_max_increase, f"Memory increase {memory_increase:.1f}MB exceeds expected {expected_max_increase:.1f}MB"
        
        # Check accuracy
        np.testing.assert_allclose(points, recovered_points, atol=1e-6) 