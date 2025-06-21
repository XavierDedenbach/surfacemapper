import pytest
import numpy as np
from app.models.data_models import StatisticalAnalysis
from app.services.statistical_analysis import StatisticalAnalyzer

class TestStatisticalAnalysis:
    @pytest.fixture
    def statistical_analyzer(self):
        return StatisticalAnalyzer()

    def test_basic_statistics_normal_data(self, statistical_analyzer):
        """Test basic statistics with normal distribution data"""
        data = np.random.normal(10, 2, 1000)
        stats = statistical_analyzer.calculate_statistics(data)
        
        assert isinstance(stats, StatisticalAnalysis)
        assert abs(stats.mean_value - 10) < 0.2  # Should be close to true mean
        assert abs(stats.median_value - 10) < 0.2  # Should be close to true median
        assert abs(stats.standard_deviation - 2) < 0.2  # Should be close to true std
        assert abs(stats.variance - 4) < 0.5  # Should be close to true variance
        assert stats.sample_count == 1000

    def test_skewness_calculation(self, statistical_analyzer):
        """Test skewness calculation"""
        # Create skewed data (positive skew)
        data = np.concatenate([np.random.normal(0, 1, 900), np.random.normal(5, 1, 100)])
        stats = statistical_analyzer.calculate_statistics(data)
        
        assert stats.skewness > 0  # Should be positively skewed
        assert abs(stats.skewness) < 10  # Should be reasonable value

    def test_kurtosis_calculation(self, statistical_analyzer):
        """Test kurtosis calculation"""
        # Create data with different kurtosis
        data = np.random.normal(0, 1, 1000)
        stats = statistical_analyzer.calculate_statistics(data)
        
        # Kurtosis should be around 0 for normal distribution
        assert abs(stats.kurtosis) < 5  # Should be reasonable value

    def test_confidence_interval_calculation(self, statistical_analyzer):
        """Test 95% confidence interval calculation"""
        data = np.random.normal(10, 2, 1000)
        stats = statistical_analyzer.calculate_statistics(data)
        
        assert len(stats.confidence_interval_95) == 2
        assert stats.confidence_interval_95[0] < stats.confidence_interval_95[1]
        assert stats.confidence_interval_95[0] < stats.mean_value < stats.confidence_interval_95[1]

    def test_percentiles_calculation(self, statistical_analyzer):
        """Test percentiles calculation"""
        data = np.random.normal(0, 1, 1000)
        stats = statistical_analyzer.calculate_statistics(data)
        
        assert isinstance(stats.percentiles, dict)
        assert '25' in stats.percentiles
        assert '50' in stats.percentiles
        assert '75' in stats.percentiles
        assert '90' in stats.percentiles
        assert '95' in stats.percentiles
        assert '99' in stats.percentiles
        
        # Check percentile ordering
        assert stats.percentiles['25'] < stats.percentiles['50'] < stats.percentiles['75']

    def test_edge_case_empty_data(self, statistical_analyzer):
        """Test statistical analysis with empty data"""
        empty_data = np.array([])
        
        with pytest.raises(ValueError):
            statistical_analyzer.calculate_statistics(empty_data)

    def test_edge_case_single_value(self, statistical_analyzer):
        """Test statistical analysis with single value"""
        single_data = np.array([5.0])
        stats = statistical_analyzer.calculate_statistics(single_data)
        
        assert stats.mean_value == 5.0
        assert stats.median_value == 5.0
        assert stats.standard_deviation == 0.0
        assert stats.variance == 0.0
        assert stats.sample_count == 1
        # Skewness and kurtosis may be undefined for single value

    def test_edge_case_all_same_values(self, statistical_analyzer):
        """Test statistical analysis with all same values"""
        same_data = np.full(100, 5.0)
        stats = statistical_analyzer.calculate_statistics(same_data)
        
        assert stats.mean_value == 5.0
        assert stats.median_value == 5.0
        assert stats.standard_deviation == 0.0
        assert stats.variance == 0.0
        assert stats.sample_count == 100

    def test_edge_case_nan_values(self, statistical_analyzer):
        """Test statistical analysis with NaN values"""
        data_with_nan = np.array([1, 2, np.nan, 4, 5])
        
        with pytest.raises(ValueError):
            statistical_analyzer.calculate_statistics(data_with_nan)

    def test_edge_case_infinite_values(self, statistical_analyzer):
        """Test statistical analysis with infinite values"""
        data_with_inf = np.array([1, 2, np.inf, 4, 5])
        
        with pytest.raises(ValueError):
            statistical_analyzer.calculate_statistics(data_with_inf)

    def test_negative_values(self, statistical_analyzer):
        """Test statistical analysis with negative values"""
        negative_data = np.array([-5, -3, -1, 0, 1, 3, 5])
        stats = statistical_analyzer.calculate_statistics(negative_data)
        
        assert stats.mean_value == 0.0
        assert stats.median_value == 0.0
        assert stats.standard_deviation > 0
        assert stats.variance > 0

    def test_large_values(self, statistical_analyzer):
        """Test statistical analysis with large values"""
        large_data = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
        stats = statistical_analyzer.calculate_statistics(large_data)
        
        assert stats.mean_value == 3e6
        assert stats.median_value == 3e6
        assert stats.standard_deviation > 0

    def test_mixed_data_types(self, statistical_analyzer):
        """Test statistical analysis with mixed data types"""
        mixed_data = np.array([1.0, 2, 3.5, 4, 5.0])
        stats = statistical_analyzer.calculate_statistics(mixed_data)
        
        assert isinstance(stats.mean_value, float)
        assert isinstance(stats.median_value, float)
        assert isinstance(stats.standard_deviation, float)

    def test_performance_large_dataset(self, statistical_analyzer):
        """Test performance with large dataset"""
        large_data = np.random.normal(0, 1, 100000)
        
        import time
        start_time = time.time()
        stats = statistical_analyzer.calculate_statistics(large_data)
        end_time = time.time()
        
        # Should complete in reasonable time
        assert (end_time - start_time) < 5.0  # Less than 5 seconds
        assert stats.sample_count == 100000

    def test_skewness_edge_cases(self, statistical_analyzer):
        """Test skewness with edge cases"""
        # Symmetric data should have skewness close to 0
        symmetric_data = np.array([-3, -2, -1, 0, 1, 2, 3])
        stats = statistical_analyzer.calculate_statistics(symmetric_data)
        assert abs(stats.skewness) < 0.1

    def test_kurtosis_edge_cases(self, statistical_analyzer):
        """Test kurtosis with edge cases"""
        # Normal distribution should have kurtosis close to 0
        normal_data = np.random.normal(0, 1, 1000)
        stats = statistical_analyzer.calculate_statistics(normal_data)
        assert abs(stats.kurtosis) < 2

    def test_confidence_interval_edge_cases(self, statistical_analyzer):
        """Test confidence interval with edge cases"""
        # Small sample size
        small_data = np.array([1, 2, 3, 4, 5])
        stats = statistical_analyzer.calculate_statistics(small_data)
        
        assert len(stats.confidence_interval_95) == 2
        assert stats.confidence_interval_95[0] < stats.confidence_interval_95[1]

    def test_percentiles_edge_cases(self, statistical_analyzer):
        """Test percentiles with edge cases"""
        # Test with exactly 100 values
        data = np.arange(100)
        stats = statistical_analyzer.calculate_statistics(data)
        
        assert stats.percentiles['25'] == pytest.approx(24.75)
        assert stats.percentiles['50'] == pytest.approx(49.5)
        assert stats.percentiles['75'] == pytest.approx(74.25)

    def test_error_handling_invalid_input(self, statistical_analyzer):
        """Test error handling with invalid input"""
        # Test with None
        with pytest.raises(ValueError):
            statistical_analyzer.calculate_statistics(None)
        
        # Test with non-numeric data
        with pytest.raises(ValueError):
            statistical_analyzer.calculate_statistics(['a', 'b', 'c'])

    def test_statistical_consistency(self, statistical_analyzer):
        """Test statistical consistency across multiple runs"""
        data = np.random.normal(0, 1, 1000)
        
        # Run multiple times with same data
        stats1 = statistical_analyzer.calculate_statistics(data)
        stats2 = statistical_analyzer.calculate_statistics(data)
        
        # Results should be identical
        assert stats1.mean_value == stats2.mean_value
        assert stats1.median_value == stats2.median_value
        assert stats1.standard_deviation == stats2.standard_deviation
        assert stats1.variance == stats2.variance 