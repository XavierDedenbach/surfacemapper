"""
Tests for coordinate transformation functionality
"""
import pytest
import numpy as np
import time
from typing import List, Tuple


class ControlPointTest:
    """Test control point with known WGS84 and UTM coordinates"""
    def __init__(self, name: str, wgs84_lat: float, wgs84_lon: float, 
                 expected_utm_x: float, expected_utm_y: float, utm_zone: str, 
                 tolerance_meters: float = 0.1):
        self.name = name
        self.wgs84_lat = wgs84_lat
        self.wgs84_lon = wgs84_lon
        self.expected_utm_x = expected_utm_x
        self.expected_utm_y = expected_utm_y
        self.utm_zone = utm_zone
        self.tolerance_meters = tolerance_meters


class TestCoordinateTransformer:
    """Test suite for coordinate transformation functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        from app.services.coord_transformer import CoordinateTransformer
        self.transformer = CoordinateTransformer()
        
        # Known control points for accuracy testing
        # Note: These are actual UTM coordinates calculated by pyproj
        self.control_points = [
            ControlPointTest(
                name="Statue of Liberty",
                wgs84_lat=40.6892,
                wgs84_lon=-74.0445,
                expected_utm_x=580735.9,  # Actual UTM X coordinate from pyproj
                expected_utm_y=4504695.2,  # Actual UTM Y coordinate from pyproj
                utm_zone="EPSG:32618"  # UTM Zone 18N
            ),
            ControlPointTest(
                name="Golden Gate Bridge",
                wgs84_lat=37.8199,
                wgs84_lon=-122.4783,
                expected_utm_x=545915.8,  # Actual UTM X coordinate from pyproj
                expected_utm_y=4185961.0,  # Actual UTM Y coordinate from pyproj
                utm_zone="EPSG:32610"  # UTM Zone 10N
            ),
            ControlPointTest(
                name="Mount Everest",
                wgs84_lat=27.9881,
                wgs84_lon=86.9250,
                expected_utm_x=492625.0,  # Actual UTM X coordinate from pyproj
                expected_utm_y=3095886.4,  # Actual UTM Y coordinate from pyproj
                utm_zone="EPSG:32645"  # UTM Zone 45N
            )
        ]
    
    def test_wgs84_to_utm_accuracy(self):
        """Test WGS84 to UTM transformation accuracy with known control points"""
        for control_point in self.control_points:
            # Transform WGS84 to UTM
            utm_coords = self.transformer.transform_wgs84_to_utm(
                control_point.wgs84_lat, 
                control_point.wgs84_lon
            )
            
            # Check accuracy within tolerance
            assert abs(utm_coords[0] - control_point.expected_utm_x) < control_point.tolerance_meters, \
                f"X coordinate accuracy failed for {control_point.name}"
            assert abs(utm_coords[1] - control_point.expected_utm_y) < control_point.tolerance_meters, \
                f"Y coordinate accuracy failed for {control_point.name}"
    
    def test_utm_zone_detection(self):
        """Test automatic UTM zone detection for different longitudes"""
        test_cases = [
            # (longitude, expected_zone_number, expected_hemisphere)
            (-74.0060, 18, "N"),  # New York
            (-122.4783, 10, "N"),  # San Francisco
            (86.9250, 45, "N"),    # Mount Everest
            (151.2093, 56, "S"),   # Sydney (Southern Hemisphere)
            (-0.1276, 30, "N"),    # London
            (139.6917, 54, "N"),   # Tokyo
        ]
        
        for lon, expected_zone, expected_hemisphere in test_cases:
            # Use a test latitude to determine hemisphere
            test_lat = 40.0 if expected_hemisphere == "N" else -40.0
            zone = self.transformer.determine_utm_zone_with_hemisphere(test_lat, lon)
            # Correct EPSG codes: 326xx for Northern, 327xx for Southern
            expected_epsg = f"EPSG:326{expected_zone:02d}" if expected_hemisphere == "N" else f"EPSG:327{expected_zone:02d}"
            
            assert zone == expected_epsg, \
                f"Zone detection failed for longitude {lon}: expected {expected_epsg}, got {zone}"
    
    def test_batch_transformation_performance(self):
        """Test performance with large batches of coordinates"""
        # Generate test coordinates around New York
        base_lat, base_lon = 40.7128, -74.0060
        coords = []
        
        for i in range(1000):
            lat = base_lat + (i * 0.001)  # Small offset
            lon = base_lon + (i * 0.001)
            coords.append((lat, lon))
        
        # Test batch transformation performance
        start_time = time.time()
        results = self.transformer.transform_wgs84_to_utm_batch(coords)
        elapsed = time.time() - start_time
        
        # Performance requirements: must complete in <1 second
        assert elapsed < 1.0, f"Batch transformation took {elapsed:.2f}s, expected <1.0s"
        assert len(results) == 1000, f"Expected 1000 results, got {len(results)}"
        
        # Verify all results are valid coordinates
        for result in results:
            assert isinstance(result, tuple) and len(result) == 2, \
                "Each result should be a tuple of (x, y) coordinates"
            assert isinstance(result[0], (int, float)) and isinstance(result[1], (int, float)), \
                "Coordinates should be numeric"
    
    def test_invalid_coordinates(self):
        """Test handling of invalid coordinate inputs"""
        invalid_cases = [
            (91.0, 0.0),    # Latitude > 90
            (-91.0, 0.0),   # Latitude < -90
            (0.0, 181.0),   # Longitude > 180
            (0.0, -181.0),  # Longitude < -180
            (float('inf'), 0.0),  # Infinite latitude
            (0.0, float('nan')),  # NaN longitude
        ]
        
        for lat, lon in invalid_cases:
            with pytest.raises(ValueError):
                self.transformer.transform_wgs84_to_utm(lat, lon)
    
    def test_edge_case_coordinates(self):
        """Test edge cases and boundary conditions"""
        edge_cases = [
            (0.0, 0.0),     # Prime meridian and equator
            (90.0, 0.0),    # North pole
            (-90.0, 0.0),   # South pole
            (0.0, 180.0),   # International date line
            (0.0, -180.0),  # International date line (negative)
        ]
        
        for lat, lon in edge_cases:
            try:
                result = self.transformer.transform_wgs84_to_utm(lat, lon)
                assert isinstance(result, tuple) and len(result) == 2, \
                    f"Edge case ({lat}, {lon}) should return valid coordinates"
            except ValueError as e:
                # Some edge cases might be expected to fail (e.g., poles)
                assert "pole" in str(e).lower() or "boundary" in str(e).lower(), \
                    f"Unexpected error for edge case ({lat}, {lon}): {e}"
    
    def test_coordinate_precision(self):
        """Test coordinate transformation precision"""
        # Test with high precision coordinates
        lat, lon = 40.7128, -74.0060  # New York
        
        # Transform multiple times and check consistency
        results = []
        for _ in range(10):
            result = self.transformer.transform_wgs84_to_utm(lat, lon)
            results.append(result)
        
        # All results should be identical (within floating point precision)
        first_result = results[0]
        for result in results[1:]:
            assert abs(result[0] - first_result[0]) < 1e-10, \
                "X coordinate should be consistent across transformations"
            assert abs(result[1] - first_result[1]) < 1e-10, \
                "Y coordinate should be consistent across transformations"
    
    def test_utm_zone_boundaries(self):
        """Test UTM zone boundary conditions"""
        # Test zone boundaries (every 6 degrees of longitude)
        # Zone 1: 180°W to 174°W, Zone 2: 174°W to 168°W, etc.
        zone_boundaries = [
            (-180.0, 1),   # Start of zone 1
            (-174.0, 2),   # Start of zone 2 (end of zone 1)
            (-168.0, 3),   # Start of zone 3
            (0.0, 31),     # Prime meridian (zone 31)
            (6.0, 32),     # Start of zone 32
            (174.0, 60),   # Start of zone 60
            (180.0, 61),   # End of zone 60 (should be clamped to 60)
        ]
        
        for lon, expected_zone in zone_boundaries:
            zone = self.transformer.determine_utm_zone(lon)
            # Handle the edge case where 180° should be zone 60, not 61
            if lon == 180.0:
                expected_zone = 60
            
            assert zone == expected_zone, \
                f"Zone boundary test failed for longitude {lon}: expected zone {expected_zone}, got {zone}"
    
    def test_transformation_metadata(self):
        """Test that transformation includes proper metadata"""
        lat, lon = 40.7128, -74.0060
        
        # Test that transformation returns metadata
        result = self.transformer.transform_wgs84_to_utm(lat, lon)
        
        # Should return tuple with coordinates and metadata
        assert isinstance(result, tuple), "Result should be a tuple"
        assert len(result) >= 2, "Result should have at least coordinates"
        
        # If metadata is included, it should contain zone information
        if len(result) > 2:
            metadata = result[2]
            assert 'utm_zone' in metadata, "Metadata should include UTM zone"
            assert 'datum' in metadata, "Metadata should include datum information"
    
    def test_memory_efficiency(self):
        """Test memory efficiency for large coordinate sets"""
        try:
            import psutil
            import os
            
            # Generate large coordinate set
            coords = [(40.0 + i*0.001, -74.0 + i*0.001) for i in range(10000)]
            
            # Measure memory before transformation
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss
            
            # Perform transformation
            results = self.transformer.transform_wgs84_to_utm_batch(coords)
            
            # Measure memory after transformation
            memory_after = process.memory_info().rss
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (<100MB for 10k coordinates)
            assert memory_increase < 100 * 1024 * 1024, \
                f"Memory increase {memory_increase / (1024*1024):.1f}MB exceeds 100MB limit"
            
            assert len(results) == 10000, "Should process all coordinates"
        except ImportError:
            pytest.skip("psutil not available for memory testing")
    
    def test_error_handling_edge_cases(self):
        """Test error handling for various edge cases"""
        # Test with None values
        with pytest.raises(ValueError):
            self.transformer.transform_wgs84_to_utm(None, 0.0)
        
        with pytest.raises(ValueError):
            self.transformer.transform_wgs84_to_utm(0.0, None)
        
        # Test with string values
        with pytest.raises(ValueError):
            self.transformer.transform_wgs84_to_utm("40.0", -74.0)
        
        # Test with empty lists for batch
        with pytest.raises(ValueError):
            self.transformer.transform_wgs84_to_utm_batch([])
        
        # Test with invalid batch input
        with pytest.raises(ValueError):
            self.transformer.transform_wgs84_to_utm_batch([(40.0, -74.0), "invalid"])
    
    def test_coordinate_system_consistency(self):
        """Test consistency across different coordinate systems"""
        # Test that same WGS84 coordinates always produce same UTM coordinates
        lat, lon = 40.7128, -74.0060
        
        # Multiple transformations should be identical
        result1 = self.transformer.transform_wgs84_to_utm(lat, lon)
        result2 = self.transformer.transform_wgs84_to_utm(lat, lon)
        
        assert result1 == result2, "Identical inputs should produce identical outputs"
        
        # Test that zone detection is consistent
        zone1 = self.transformer.determine_utm_zone(lon)
        zone2 = self.transformer.determine_utm_zone(lon)
        
        assert zone1 == zone2, "Zone detection should be consistent"


if __name__ == "__main__":
    pytest.main([__file__]) 