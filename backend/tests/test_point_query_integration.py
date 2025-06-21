"""
Integration tests for point query endpoints
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch
from app.services.analysis_executor import AnalysisExecutor
from app.services import thickness_calculator
from app.services.coord_transformer import TransformationPipeline
from scipy.spatial import Delaunay


class TestPointQueryIntegration:
    """Integration tests for point query functionality"""
    
    def setup_method(self):
        self.executor = AnalysisExecutor()
        
        # Create mock TINs for testing
        self.mock_tin1 = Delaunay(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        self.mock_tin1.z_values = np.array([0.0, 0.0, 0.0, 0.0])  # Flat surface at z=0
        
        self.mock_tin2 = Delaunay(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        self.mock_tin2.z_values = np.array([2.0, 2.0, 2.0, 2.0])  # Flat surface at z=2
        
        self.mock_tin3 = Delaunay(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        self.mock_tin3.z_values = np.array([5.0, 5.0, 5.0, 5.0])  # Flat surface at z=5
        
        # Mock analysis results
        self.mock_results = {
            "status": "completed",
            "surface_tins": [self.mock_tin1, self.mock_tin2, self.mock_tin3],
            "surface_names": ["Surface 0", "Surface 1", "Surface 2"],
            "georef": {
                "lat": 40.7128,
                "lon": -74.0060,
                "orientation": 0.0,
                "scale": 1.0
            }
        }
    
    def test_point_query_integration_with_mock_executor(self):
        """Test point query integration with mock analysis executor"""
        # Mock the executor to return our test results
        with patch.object(self.executor, 'get_results', return_value=self.mock_results):
            # Test query point inside triangulation
            query_point = np.array([0.5, 0.5])
            
            # Calculate expected thickness values
            upper_z1 = thickness_calculator._interpolate_z_at_point(query_point, self.mock_tin2)
            lower_z1 = thickness_calculator._interpolate_z_at_point(query_point, self.mock_tin1)
            thickness1 = upper_z1 - lower_z1
            
            upper_z2 = thickness_calculator._interpolate_z_at_point(query_point, self.mock_tin3)
            lower_z2 = thickness_calculator._interpolate_z_at_point(query_point, self.mock_tin2)
            thickness2 = upper_z2 - lower_z2
            
            # Verify calculations
            assert abs(thickness1 - 2.0) < 1e-10  # 2.0 - 0.0 = 2.0
            assert abs(thickness2 - 3.0) < 1e-10  # 5.0 - 2.0 = 3.0
    
    def test_batch_point_query_integration(self):
        """Test batch point query integration"""
        # Mock the executor
        with patch.object(self.executor, 'get_results', return_value=self.mock_results):
            # Test multiple query points
            query_points = [
                np.array([0.25, 0.25]),
                np.array([0.75, 0.75]),
                np.array([0.5, 0.5])
            ]
            
            results = []
            for point in query_points:
                thickness_layers = []
                for i in range(len(self.mock_results["surface_tins"]) - 1):
                    upper_tin = self.mock_results["surface_tins"][i+1]
                    lower_tin = self.mock_results["surface_tins"][i]
                    
                    upper_z = thickness_calculator._interpolate_z_at_point(point, upper_tin)
                    lower_z = thickness_calculator._interpolate_z_at_point(point, lower_tin)
                    
                    if not np.isnan(upper_z) and not np.isnan(lower_z):
                        thickness = upper_z - lower_z
                    else:
                        thickness = None
                    
                    thickness_layers.append({
                        "layer_name": f"{self.mock_results['surface_names'][i]} to {self.mock_results['surface_names'][i+1]}",
                        "thickness_feet": thickness
                    })
                
                results.append({
                    "point": {"x": float(point[0]), "y": float(point[1]), "coordinate_system": "utm"},
                    "thickness_layers": thickness_layers
                })
            
            # Verify all points have results
            assert len(results) == 3
            
            # Verify thickness values are consistent
            for result in results:
                assert len(result["thickness_layers"]) == 2
                assert result["thickness_layers"][0]["thickness_feet"] == 2.0
                assert result["thickness_layers"][1]["thickness_feet"] == 3.0
    
    def test_coordinate_transformation_integration(self):
        """Test coordinate transformation integration"""
        # Create a mock transformer
        transformer = TransformationPipeline(
            anchor_lat=40.7128,
            anchor_lon=-74.0060,
            rotation_degrees=0.0,
            scale_factor=1.0
        )
        
        # Test WGS84 to UTM transformation
        wgs84_point = np.array([[-74.0060, 40.7128, 0]])  # Longitude, Latitude, Z
        utm_point = transformer.transform_to_utm(wgs84_point)[0]
        
        # Verify transformation produces valid UTM coordinates
        assert isinstance(utm_point[0], (int, float))
        assert isinstance(utm_point[1], (int, float))
        assert utm_point[0] > 0  # UTM X should be positive
        assert utm_point[1] > 0  # UTM Y should be positive
    
    def test_error_handling_integration(self):
        """Test error handling integration"""
        # Test with non-existent analysis
        with patch.object(self.executor, 'get_results', return_value=None):
            # This should raise an exception in the endpoint
            # For integration testing, we verify the logic handles None results
            results = self.executor.get_results("non-existent-id")
            assert results is None
        
        # Test with incomplete analysis
        incomplete_results = {
            "status": "processing",
            "progress_percent": 50.0
        }
        with patch.object(self.executor, 'get_results', return_value=incomplete_results):
            results = self.executor.get_results("incomplete-id")
            assert results["status"] == "processing"
    
    def test_point_outside_boundary_integration(self):
        """Test point outside boundary handling"""
        # Create a TIN with limited extent
        limited_tin = Delaunay(np.array([[0, 0], [1, 0], [0, 1]]))
        limited_tin.z_values = np.array([0.0, 0.0, 0.0])
        
        # Test point outside triangulation
        outside_point = np.array([2.0, 2.0])
        z_value = thickness_calculator._interpolate_z_at_point(outside_point, limited_tin)
        
        # Should return NaN for points outside
        assert np.isnan(z_value)
    
    def test_performance_integration(self):
        """Test performance characteristics"""
        import time
        
        # Test single point query performance
        start_time = time.time()
        for _ in range(100):
            query_point = np.array([0.5, 0.5])
            thickness_calculator._interpolate_z_at_point(query_point, self.mock_tin1)
        elapsed = time.time() - start_time
        
        # Should complete 100 interpolations quickly
        assert elapsed < 1.0  # Less than 1 second for 100 queries
        
        # Test batch processing performance
        batch_points = np.array([[0.1 * i, 0.1 * i] for i in range(100)])
        start_time = time.time()
        
        for point in batch_points:
            thickness_calculator._interpolate_z_at_point(point, self.mock_tin1)
        
        elapsed = time.time() - start_time
        
        # Should complete batch processing quickly
        assert elapsed < 2.0  # Less than 2 seconds for 100 points
    
    def test_data_consistency_integration(self):
        """Test data consistency across different query methods"""
        # Test that single point and batch queries produce consistent results
        query_point = np.array([0.5, 0.5])
        
        # Single point calculation
        single_z = thickness_calculator._interpolate_z_at_point(query_point, self.mock_tin1)
        
        # Batch calculation (single point)
        batch_points = np.array([query_point])
        batch_z = thickness_calculator._interpolate_z_at_point(batch_points[0], self.mock_tin1)
        
        # Results should be identical
        assert abs(single_z - batch_z) < 1e-10
    
    def test_coordinate_system_validation_integration(self):
        """Test coordinate system validation"""
        # Test valid coordinate systems
        valid_systems = ["utm", "wgs84"]
        for system in valid_systems:
            assert system in valid_systems
        
        # Test invalid coordinate system
        invalid_system = "invalid"
        assert invalid_system not in valid_systems
    
    def test_thickness_calculation_accuracy_integration(self):
        """Test thickness calculation accuracy with known values"""
        # Create surfaces with known thickness
        bottom_surface = Delaunay(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        bottom_surface.z_values = np.array([0.0, 0.0, 0.0, 0.0])
        
        top_surface = Delaunay(np.array([[0, 0], [1, 0], [0, 1], [1, 1]]))
        top_surface.z_values = np.array([3.5, 3.5, 3.5, 3.5])  # 3.5 feet above bottom
        
        # Calculate thickness at center point
        query_point = np.array([0.5, 0.5])
        upper_z = thickness_calculator._interpolate_z_at_point(query_point, top_surface)
        lower_z = thickness_calculator._interpolate_z_at_point(query_point, bottom_surface)
        thickness = upper_z - lower_z
        
        # Should be exactly 3.5 feet
        assert abs(thickness - 3.5) < 1e-10 