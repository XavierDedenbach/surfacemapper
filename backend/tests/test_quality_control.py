import pytest
import numpy as np
from app.models.data_models import QualityMetrics
from app.services.quality_control import QualityControl

class TestQualityControl:
    @pytest.fixture
    def quality_control(self):
        return QualityControl()

    def test_data_validation_valid_surface(self, quality_control):
        """Test data validation with valid surface data"""
        valid_surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.float32)
        result = quality_control.validate_surface_data(valid_surface)
        assert result['is_valid'] is True
        assert 'errors' not in result

    def test_data_validation_invalid_surface(self, quality_control):
        """Test data validation with invalid surface data"""
        # Surface with NaN values
        invalid_surface = np.array([[0, 0, 0], [1, 0, np.nan], [0, 1, 0]], dtype=np.float32)
        result = quality_control.validate_surface_data(invalid_surface)
        assert result['is_valid'] is False
        assert 'errors' in result
        assert len(result['errors']) > 0

    def test_data_validation_empty_surface(self, quality_control):
        """Test data validation with empty surface"""
        empty_surface = np.empty((0, 3))
        result = quality_control.validate_surface_data(empty_surface)
        assert result['is_valid'] is False
        assert 'errors' in result

    def test_data_validation_insufficient_points(self, quality_control):
        """Test data validation with insufficient points for triangulation"""
        insufficient_surface = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        result = quality_control.validate_surface_data(insufficient_surface)
        assert result['is_valid'] is False
        assert 'errors' in result

    def test_outlier_detection_normal_data(self, quality_control):
        """Test outlier detection with normal data"""
        normal_data = np.random.normal(0, 1, 1000)
        outliers = quality_control.detect_outliers(normal_data, threshold=3.0)
        # Should detect very few outliers in normal data
        outlier_count = np.sum(outliers)  # Count True values
        assert outlier_count < len(normal_data) * 0.01

    def test_outlier_detection_with_outliers(self, quality_control):
        """Test outlier detection with known outliers"""
        data = np.concatenate([
            np.random.normal(0, 1, 100),
            np.array([100, -100])  # Known outliers
        ])
        outliers = quality_control.detect_outliers(data, threshold=3.0)
        outlier_count = np.sum(outliers)  # Count True values
        assert outlier_count >= 2
        assert 100 in data[outliers]
        assert -100 in data[outliers]

    def test_outlier_removal(self, quality_control):
        """Test outlier removal functionality"""
        data = np.concatenate([
            np.random.normal(0, 1, 100),
            np.array([100, -100])  # Outliers
        ])
        cleaned_data = quality_control.remove_outliers(data, threshold=3.0)
        assert len(cleaned_data) < len(data)
        assert 100 not in cleaned_data
        assert -100 not in cleaned_data

    def test_point_density_calculation(self, quality_control):
        """Test point density calculation"""
        # Create a 10x10 grid of points
        x = np.linspace(0, 10, 11)
        y = np.linspace(0, 10, 11)
        X, Y = np.meshgrid(x, y)
        surface = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])
        
        density = quality_control.calculate_point_density(surface, area=100.0)
        expected_density = len(surface) / 100.0  # points per unit area
        assert abs(density - expected_density) < 0.01

    def test_surface_coverage_calculation(self, quality_control):
        """Test surface coverage calculation"""
        # Create surface with known coverage
        surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        boundary_area = 4.0  # 2x2 boundary
        coverage = quality_control.calculate_surface_coverage(surface, boundary_area)
        assert 0 <= coverage <= 1
        assert coverage > 0

    def test_data_completeness_calculation(self, quality_control):
        """Test data completeness calculation"""
        # Create surface with some missing data (NaN values)
        surface = np.array([[0, 0, 0], [1, 0, np.nan], [0, 1, 0], [1, 1, 0]])
        completeness = quality_control.calculate_data_completeness(surface)
        assert 0 <= completeness <= 1
        assert completeness < 1.0  # Should be less than 1 due to NaN

    def test_noise_level_estimation(self, quality_control):
        """Test noise level estimation"""
        # Create surface with known noise
        clean_surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        noise = np.random.normal(0, 0.1, clean_surface.shape)
        noisy_surface = clean_surface + noise
        
        noise_level = quality_control.estimate_noise_level(noisy_surface)
        assert noise_level >= 0
        assert noise_level < 1.0  # Should be reasonable noise level

    def test_accuracy_estimation(self, quality_control):
        """Test accuracy estimation"""
        # Create test data with known accuracy
        surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        accuracy = quality_control.estimate_accuracy(surface)
        assert accuracy >= 0
        assert accuracy <= 1

    def test_precision_estimation(self, quality_control):
        """Test precision estimation"""
        # Create test data with known precision
        surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        precision = quality_control.estimate_precision(surface)
        assert precision >= 0
        assert precision <= 1

    def test_reliability_score_calculation(self, quality_control):
        """Test reliability score calculation"""
        # Create test surface
        surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        reliability = quality_control.calculate_reliability_score(surface)
        assert 0 <= reliability <= 1

    def test_quality_flags_assessment(self, quality_control):
        """Test quality flags assessment"""
        # Create test surface
        surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        flags = quality_control.assess_quality_flags(surface)
        assert isinstance(flags, dict)
        assert all(isinstance(flag, bool) for flag in flags.values())

    def test_comprehensive_quality_metrics(self, quality_control):
        """Test comprehensive quality metrics calculation"""
        # Create test surface
        surface = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        metrics = quality_control.calculate_quality_metrics(surface, boundary_area=4.0)
        
        assert isinstance(metrics, QualityMetrics)
        assert metrics.point_density >= 0
        assert 0 <= metrics.surface_coverage <= 1
        assert 0 <= metrics.data_completeness <= 1
        assert metrics.noise_level >= 0
        assert metrics.accuracy_estimate >= 0
        assert metrics.precision_estimate >= 0
        assert 0 <= metrics.reliability_score <= 1
        assert isinstance(metrics.quality_flags, dict)

    def test_quality_control_edge_cases(self, quality_control):
        """Test quality control with edge cases"""
        # Single point surface
        single_point = np.array([[0, 0, 0]])
        result = quality_control.validate_surface_data(single_point)
        assert result['is_valid'] is False
        
        # Surface with infinite values
        infinite_surface = np.array([[0, 0, 0], [1, 0, np.inf], [0, 1, 0]])
        result = quality_control.validate_surface_data(infinite_surface)
        assert result['is_valid'] is False

    def test_quality_control_performance(self, quality_control):
        """Test quality control performance with large datasets"""
        # Create large surface
        large_surface = np.random.rand(10000, 3)
        start_time = quality_control._get_time()
        metrics = quality_control.calculate_quality_metrics(large_surface, boundary_area=100.0)
        end_time = quality_control._get_time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0  # Less than 5 seconds
        assert isinstance(metrics, QualityMetrics)

    def test_quality_control_error_handling(self, quality_control):
        """Test quality control error handling"""
        # Test with None input
        with pytest.raises(ValueError):
            quality_control.validate_surface_data(None)
        
        # Test with wrong shape
        wrong_shape = np.array([[0, 0], [1, 1]])  # 2D instead of 3D
        with pytest.raises(ValueError):
            quality_control.validate_surface_data(wrong_shape) 