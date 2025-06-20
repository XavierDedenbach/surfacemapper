"""
Tests for 3D rotation and scaling transformations
"""
import pytest
import numpy as np
import time
from typing import List, Tuple
from dataclasses import dataclass


class RotationScalingTest:
    """Test data for rotation and scaling operations"""
    def __init__(self, name: str, points: np.ndarray, expected_result: np.ndarray, 
                 rotation_degrees: float = 0.0, scale_factor: float = 1.0, 
                 tolerance: float = 1e-10):
        self.name = name
        self.points = points
        self.expected_result = expected_result
        self.rotation_degrees = rotation_degrees
        self.scale_factor = scale_factor
        self.tolerance = tolerance


class TestRotationAndScaling:
    """Test suite for 3D rotation and scaling transformations"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from app.services.coord_transformer import CoordinateTransformer
        self.transformer = CoordinateTransformer()
        
        # Test cases for rotation transformations
        self.rotation_tests = [
            RotationScalingTest(
                name="90 degree rotation around Z-axis",
                points=np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]]),
                expected_result=np.array([[0, 1, 0], [-1, 0, 0], [-1, 1, 1]]),
                rotation_degrees=90.0
            ),
            RotationScalingTest(
                name="180 degree rotation around Z-axis",
                points=np.array([[1, 0, 0], [0, 1, 0]]),
                expected_result=np.array([[-1, 0, 0], [0, -1, 0]]),
                rotation_degrees=180.0
            ),
            RotationScalingTest(
                name="270 degree rotation around Z-axis",
                points=np.array([[1, 0, 0], [0, 1, 0]]),
                expected_result=np.array([[0, -1, 0], [1, 0, 0]]),
                rotation_degrees=270.0
            ),
            RotationScalingTest(
                name="45 degree rotation around Z-axis",
                points=np.array([[1, 0, 0]]),
                expected_result=np.array([[0.7071067811865476, 0.7071067811865476, 0]]),
                rotation_degrees=45.0
            ),
            RotationScalingTest(
                name="0 degree rotation (identity)",
                points=np.array([[1, 1, 1], [2, 2, 2]]),
                expected_result=np.array([[1, 1, 1], [2, 2, 2]]),
                rotation_degrees=0.0
            ),
            RotationScalingTest(
                name="359 degree rotation (near identity)",
                points=np.array([[1, 0, 0]]),
                expected_result=np.array([[0.9998476951563913, -0.01745240643728351, 0]]),
                rotation_degrees=359.0
            )
        ]
        
        # Test cases for scaling transformations
        self.scaling_tests = [
            RotationScalingTest(
                name="Uniform scaling factor 2.0",
                points=np.array([[1, 1, 1], [2, 2, 2]]),
                expected_result=np.array([[2, 2, 2], [4, 4, 4]]),
                scale_factor=2.0
            ),
            RotationScalingTest(
                name="Uniform scaling factor 0.5",
                points=np.array([[2, 2, 2], [4, 4, 4]]),
                expected_result=np.array([[1, 1, 1], [2, 2, 2]]),
                scale_factor=0.5
            ),
            RotationScalingTest(
                name="Uniform scaling factor 0.1",
                points=np.array([[10, 10, 10]]),
                expected_result=np.array([[1, 1, 1]]),
                scale_factor=0.1
            ),
            RotationScalingTest(
                name="Uniform scaling factor 10.0",
                points=np.array([[1, 1, 1]]),
                expected_result=np.array([[10, 10, 10]]),
                scale_factor=10.0
            ),
            RotationScalingTest(
                name="Identity scaling (factor 1.0)",
                points=np.array([[1, 2, 3], [4, 5, 6]]),
                expected_result=np.array([[1, 2, 3], [4, 5, 6]]),
                scale_factor=1.0
            )
        ]
        
        # Test cases for combined transformations
        self.combined_tests = [
            RotationScalingTest(
                name="90 degree rotation + 2x scaling",
                points=np.array([[1, 0, 0]]),
                expected_result=np.array([[0, -2, 0]]),
                rotation_degrees=90.0,
                scale_factor=2.0
            ),
            RotationScalingTest(
                name="45 degree rotation + 0.5x scaling",
                points=np.array([[2, 0, 0]]),
                expected_result=np.array([[0.7071067811865476, -0.7071067811865476, 0]]),
                rotation_degrees=45.0,
                scale_factor=0.5
            ),
            RotationScalingTest(
                name="180 degree rotation + 3x scaling",
                points=np.array([[1, 1, 1]]),
                expected_result=np.array([[-3, -3, 3]]),
                rotation_degrees=180.0,
                scale_factor=3.0
            )
        ]
    
    def test_rotation_transformation(self):
        """Test basic rotation transformations with known results"""
        test_cases = [
            RotationScalingTest(
                name="90 degree clockwise rotation",
                points=np.array([[1, 0, 0], [0, 1, 0]]),
                expected_result=np.array([[0, -1, 0], [1, 0, 0]]),
                rotation_degrees=90.0,
                tolerance=1e-10
            ),
            RotationScalingTest(
                name="180 degree clockwise rotation",
                points=np.array([[1, 0, 0], [0, 1, 0]]),
                expected_result=np.array([[-1, 0, 0], [0, -1, 0]]),
                rotation_degrees=180.0,
                tolerance=1e-10
            ),
            RotationScalingTest(
                name="270 degree clockwise rotation",
                points=np.array([[1, 0, 0], [0, 1, 0]]),
                expected_result=np.array([[0, 1, 0], [-1, 0, 0]]),
                rotation_degrees=270.0,
                tolerance=1e-10
            ),
            RotationScalingTest(
                name="45 degree clockwise rotation",
                points=np.array([[1, 0, 0]]),
                expected_result=np.array([[0.7071067811865476, -0.7071067811865476, 0]]),
                rotation_degrees=45.0,
                tolerance=1e-10
            )
        ]
        
        for test_case in test_cases:
            result = self.transformer.apply_rotation_z(
                test_case.points, 
                test_case.rotation_degrees
            )
            
            np.testing.assert_allclose(
                result, 
                test_case.expected_result, 
                atol=test_case.tolerance,
                err_msg=f"Rotation test failed for {test_case.name}"
            )
    
    def test_scaling_transformation(self):
        """Test uniform scaling transformations"""
        for test_case in self.scaling_tests:
            # Apply scaling transformation
            scaled = self.transformer.apply_scaling(
                test_case.points, 
                test_case.scale_factor
            )
            
            # Check accuracy within tolerance
            np.testing.assert_allclose(
                scaled, 
                test_case.expected_result, 
                atol=test_case.tolerance,
                err_msg=f"Scaling test failed for {test_case.name}"
            )
    
    def test_combined_transformations(self):
        """Test combined rotation and scaling transformations"""
        for test_case in self.combined_tests:
            # Apply combined transformation (scale then rotate)
            scaled = self.transformer.apply_scaling(
                test_case.points, 
                test_case.scale_factor
            )
            result = self.transformer.apply_rotation_z(
                scaled, 
                test_case.rotation_degrees
            )
            
            # Check accuracy within tolerance
            np.testing.assert_allclose(
                result, 
                test_case.expected_result, 
                atol=test_case.tolerance,
                err_msg=f"Combined transformation test failed for {test_case.name}"
            )
    
    def test_transformation_order_consistency(self):
        """Test that transformation order is consistent and reversible"""
        points = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 1]])
        rotation_degrees = 45.0
        scale_factor = 2.0
        
        # Method 1: Scale then rotate
        scaled_then_rotated = self.transformer.apply_rotation_z(
            self.transformer.apply_scaling(points, scale_factor),
            rotation_degrees
        )
        
        # Method 2: Rotate then scale
        rotated_then_scaled = self.transformer.apply_scaling(
            self.transformer.apply_rotation_z(points, rotation_degrees),
            scale_factor
        )
        
        # Results should be the same (uniform scaling and Z-axis rotation commute)
        np.testing.assert_allclose(scaled_then_rotated, rotated_then_scaled, atol=1e-10,
                                 err_msg="Uniform scaling and Z-axis rotation should commute")
    
    def test_rotation_matrix_properties(self):
        """Test that rotation matrices have correct mathematical properties"""
        angles = [0, 45, 90, 180, 270, 360]
        
        for angle in angles:
            # Get rotation matrix
            rotation_matrix = self.transformer.get_rotation_matrix_z(angle)
            
            # Test properties:
            # 1. Determinant should be 1 (preserves volume)
            det = np.linalg.det(rotation_matrix)
            assert abs(det - 1.0) < 1e-10, f"Rotation matrix determinant should be 1 for angle {angle}"
            
            # 2. Matrix should be orthogonal (R * R^T = I)
            identity = np.eye(3)
            orthogonal_check = np.dot(rotation_matrix, rotation_matrix.T)
            np.testing.assert_allclose(orthogonal_check, identity, atol=1e-10,
                                     err_msg=f"Rotation matrix should be orthogonal for angle {angle}")
            
            # 3. Z-axis should remain unchanged for Z-axis rotation
            z_axis = np.array([0, 0, 1])
            rotated_z = np.dot(rotation_matrix, z_axis)
            np.testing.assert_allclose(rotated_z, z_axis, atol=1e-10,
                                     err_msg=f"Z-axis should remain unchanged for Z-axis rotation {angle}")
    
    def test_scaling_matrix_properties(self):
        """Test that scaling matrices have correct mathematical properties"""
        scale_factors = [0.1, 0.5, 1.0, 2.0, 10.0]
        
        for scale in scale_factors:
            # Get scaling matrix
            scaling_matrix = self.transformer.get_scaling_matrix(scale)
            
            # Test properties:
            # 1. Determinant should be scale^3 (for 3D uniform scaling)
            det = np.linalg.det(scaling_matrix)
            expected_det = scale ** 3
            assert abs(det - expected_det) < 1e-10, f"Scaling matrix determinant should be {expected_det} for scale {scale}"
            
            # 2. Matrix should be diagonal
            diagonal = np.diag([scale, scale, scale])
            np.testing.assert_allclose(scaling_matrix, diagonal, atol=1e-10,
                                     err_msg=f"Scaling matrix should be diagonal for scale {scale}")
    
    def test_large_point_set_performance(self):
        """Test performance with large point sets"""
        # Generate large point set
        num_points = 10000
        points = np.random.rand(num_points, 3) * 100  # Random points in 100x100x100 cube
        
        # Test rotation performance
        start_time = time.time()
        rotated = self.transformer.apply_rotation_z(points, 45.0)
        rotation_time = time.time() - start_time
        
        # Test scaling performance
        start_time = time.time()
        scaled = self.transformer.apply_scaling(points, 2.0)
        scaling_time = time.time() - start_time
        
        # Performance requirements: should complete in <1 second each
        assert rotation_time < 1.0, f"Rotation took {rotation_time:.2f}s, expected <1.0s"
        assert scaling_time < 1.0, f"Scaling took {scaling_time:.2f}s, expected <1.0s"
        
        # Verify output dimensions
        assert rotated.shape == points.shape, "Rotation should preserve point count"
        assert scaled.shape == points.shape, "Scaling should preserve point count"
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with single point
        single_point = np.array([[1, 1, 1]])
        rotated_single = self.transformer.apply_rotation_z(single_point, 90.0)
        scaled_single = self.transformer.apply_scaling(single_point, 2.0)
        
        assert rotated_single.shape == (1, 3), "Single point rotation should preserve shape"
        assert scaled_single.shape == (1, 3), "Single point scaling should preserve shape"
        
        # Test with empty array
        empty_points = np.array([]).reshape(0, 3)
        rotated_empty = self.transformer.apply_rotation_z(empty_points, 45.0)
        scaled_empty = self.transformer.apply_scaling(empty_points, 2.0)
        
        assert rotated_empty.shape == (0, 3), "Empty array rotation should preserve shape"
        assert scaled_empty.shape == (0, 3), "Empty array scaling should preserve shape"
        
        # Test with zero scaling factor
        points = np.array([[1, 1, 1], [2, 2, 2]])
        zero_scaled = self.transformer.apply_scaling(points, 0.0)
        expected_zero = np.zeros_like(points)
        
        np.testing.assert_allclose(zero_scaled, expected_zero, atol=1e-10,
                                 err_msg="Zero scaling should result in zero points")
    
    def test_numerical_precision(self):
        """Test numerical precision and stability"""
        # Test with very small angles
        points = np.array([[1, 0, 0]])
        small_angle = 1e-10
        rotated_small = self.transformer.apply_rotation_z(points, small_angle)
        
        # Should be very close to original (within numerical precision)
        np.testing.assert_allclose(rotated_small, points, atol=1e-8,
                                 err_msg="Very small rotation should be close to identity")
        
        # Test with very small scaling factors
        small_scale = 1e-10
        scaled_small = self.transformer.apply_scaling(points, small_scale)
        expected_small = points * small_scale
        
        np.testing.assert_allclose(scaled_small, expected_small, atol=1e-10,
                                 err_msg="Very small scaling should work correctly")
        
        # Test with very large scaling factors
        large_scale = 1e10
        scaled_large = self.transformer.apply_scaling(points, large_scale)
        expected_large = points * large_scale
        
        np.testing.assert_allclose(scaled_large, expected_large, atol=1e-10,
                                 err_msg="Very large scaling should work correctly")
    
    def test_inverse_transformations(self):
        """Test that transformations can be inverted"""
        points = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        rotation_angle = 45.0
        scale_factor = 2.0
        
        # Apply forward transformation
        rotated = self.transformer.apply_rotation_z(points, rotation_angle)
        scaled = self.transformer.apply_scaling(points, scale_factor)
        
        # Apply inverse transformation
        inverse_rotated = self.transformer.apply_rotation_z(rotated, -rotation_angle)
        inverse_scaled = self.transformer.apply_scaling(scaled, 1.0 / scale_factor)
        
        # Should recover original points
        np.testing.assert_allclose(inverse_rotated, points, atol=1e-10,
                                 err_msg="Rotation inverse should recover original points")
        np.testing.assert_allclose(inverse_scaled, points, atol=1e-10,
                                 err_msg="Scaling inverse should recover original points")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        points = np.array([[1, 1, 1]])
        
        # Test invalid rotation angles
        with pytest.raises(ValueError):
            self.transformer.apply_rotation_z(points, float('inf'))
        
        with pytest.raises(ValueError):
            self.transformer.apply_rotation_z(points, float('nan'))
        
        # Test invalid scaling factors
        with pytest.raises(ValueError):
            self.transformer.apply_scaling(points, float('inf'))
        
        with pytest.raises(ValueError):
            self.transformer.apply_scaling(points, float('nan'))
        
        with pytest.raises(ValueError):
            self.transformer.apply_scaling(points, -1.0)  # Negative scaling not allowed
        
        # Test invalid point arrays
        with pytest.raises(ValueError):
            self.transformer.apply_rotation_z(None, 45.0)
        
        with pytest.raises(ValueError):
            self.transformer.apply_scaling(None, 2.0)
        
        # Test points with wrong dimensions
        invalid_points = np.array([1, 2, 3])  # 1D array
        with pytest.raises(ValueError):
            self.transformer.apply_rotation_z(invalid_points, 45.0)
        
        with pytest.raises(ValueError):
            self.transformer.apply_scaling(invalid_points, 2.0)


if __name__ == "__main__":
    pytest.main([__file__]) 