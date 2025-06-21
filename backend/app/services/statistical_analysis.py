import numpy as np
from scipy import stats
from typing import Union, List
from app.models.data_models import StatisticalAnalysis


class StatisticalAnalyzer:
    """Service for performing statistical analysis on numerical data"""
    
    def calculate_statistics(self, data: Union[np.ndarray, List[float]]) -> StatisticalAnalysis:
        """
        Calculate comprehensive statistical analysis of the input data.
        
        Args:
            data: Input numerical data as numpy array or list
            
        Returns:
            StatisticalAnalysis object with all calculated statistics
            
        Raises:
            ValueError: If data is invalid (empty, contains NaN/inf, or non-numeric)
        """
        # Validate input data first
        if data is None:
            raise ValueError("Data cannot be None")
        
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)
        
        # Validate input data
        self._validate_data(data)
        
        # Calculate basic statistics
        mean_value = float(np.mean(data))
        median_value = float(np.median(data))
        sample_count = int(len(data))
        
        # Handle edge cases for standard deviation and variance
        if sample_count == 1 or np.all(data == data[0]):
            standard_deviation = 0.0
            variance = 0.0
        else:
            standard_deviation = float(np.std(data, ddof=1))  # Sample standard deviation
            variance = float(np.var(data, ddof=1))  # Sample variance
        
        # Calculate higher moments (skewness and kurtosis)
        if sample_count == 1 or np.all(data == data[0]):
            # For single value or all same values, skewness and kurtosis are undefined
            skewness = 0.0
            kurtosis = 0.0
        else:
            skewness = float(stats.skew(data))
            kurtosis = float(stats.kurtosis(data))
        
        # Calculate 95% confidence interval
        confidence_interval_95 = self._calculate_confidence_interval(data, confidence_level=0.95)
        
        # Calculate percentiles
        percentiles = self._calculate_percentiles(data)
        
        return StatisticalAnalysis(
            mean_value=mean_value,
            median_value=median_value,
            standard_deviation=standard_deviation,
            variance=variance,
            skewness=skewness,
            kurtosis=kurtosis,
            sample_count=sample_count,
            confidence_interval_95=confidence_interval_95,
            percentiles=percentiles
        )
    
    def _validate_data(self, data: np.ndarray) -> None:
        """
        Validate input data for statistical analysis.
        
        Args:
            data: Input data to validate
            
        Raises:
            ValueError: If data is invalid
        """
        if len(data) == 0:
            raise ValueError("Data cannot be empty")
        
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("Data must contain numeric values")
        
        if np.any(np.isnan(data)):
            raise ValueError("Data cannot contain NaN values")
        
        if np.any(np.isinf(data)):
            raise ValueError("Data cannot contain infinite values")
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence_level: float = 0.95) -> tuple[float, float]:
        """
        Calculate confidence interval for the mean.
        
        Args:
            data: Input data
            confidence_level: Confidence level (default: 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(data) == 1 or np.all(data == data[0]):
            # For single value or all same values, confidence interval is the value itself
            # Add small epsilon to satisfy validation constraint
            value = float(data[0])
            epsilon = 1e-10
            return (value - epsilon, value + epsilon)
        
        # Calculate standard error
        std_error = np.std(data, ddof=1) / np.sqrt(len(data))
        
        # Get critical value from t-distribution
        degrees_of_freedom = len(data) - 1
        critical_value = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        
        # Calculate confidence interval
        margin_of_error = critical_value * std_error
        mean_value = np.mean(data)
        
        lower_bound = float(mean_value - margin_of_error)
        upper_bound = float(mean_value + margin_of_error)
        
        return (lower_bound, upper_bound)
    
    def _calculate_percentiles(self, data: np.ndarray) -> dict[str, float]:
        """
        Calculate various percentiles of the data.
        
        Args:
            data: Input data
            
        Returns:
            Dictionary with percentile values
        """
        percentile_values = [25, 50, 75, 90, 95, 99]
        percentiles = {}
        
        for p in percentile_values:
            percentiles[str(p)] = float(np.percentile(data, p, method='linear'))
        
        return percentiles 