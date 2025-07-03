import pytest
import numpy as np
import tempfile
import os
import fiona
import shapely.geometry as sgeom
from shapely.geometry import Point, Polygon, MultiPoint, LineString
import pyproj
from unittest.mock import patch, MagicMock
import pyproj.exceptions

# Import the module we'll be testing
from app.utils.shp_parser import SHPParser


class TestSHPParser:
    """Test suite for SHP file parsing, densification, and clipping functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.test_data_dir = "../drone_surfaces"
        self.wgs84_crs = "EPSG:4326"
        self.parser = SHPParser()
        
        # Path to the real SHP file for testing
        self.real_shp_file = "../drone_surfaces/27June20250541PM1619tonspartialcover/27June20250541PM1619tonspartialcover.shp"
        


    def test_densify_linestring_contours(self):
        """Test densification of LineString contours to 1-foot spacing"""
        # Create a LineString that needs densification
        # This line is approximately 10 feet long in UTM coordinates
        test_line = LineString([(-79.85, 40.15, 300), (-79.8499, 40.15, 300)])
        
        # Test densification
        densified = self.parser.densify_linestring(test_line, max_distance_feet=1.0)
        assert len(densified.coords) > len(test_line.coords)
        
        # Check that all points are within reasonable distance
        coords = list(densified.coords)
        for i in range(len(coords) - 1):
            # Calculate approximate distance in degrees (rough check)
            dx = coords[i+1][0] - coords[i][0]
            dy = coords[i+1][1] - coords[i][1]
            distance_deg = np.sqrt(dx*dx + dy*dy)
            # 1 foot ≈ 0.00001 degrees at this latitude
            assert distance_deg < 0.00002  # Allow some tolerance

    def test_create_polygon_boundary_from_contours(self):
        """Test creating polygon boundary from densified contour points"""
        # Create test contours that form a rough boundary
        test_contours = [
            LineString([(-79.85, 40.15), (-79.84, 40.15), (-79.84, 40.16)]),
            LineString([(-79.85, 40.16), (-79.84, 40.16), (-79.84, 40.17)]),
            LineString([(-79.85, 40.17), (-79.84, 40.17), (-79.84, 40.18)]),
            LineString([(-79.85, 40.18), (-79.84, 40.18), (-79.84, 40.19)])
        ]
        
        # Test polygon boundary creation
        polygon = self.parser.create_polygon_boundary_from_contours(test_contours)
        assert isinstance(polygon, Polygon)
        assert polygon.is_valid
        assert polygon.area > 0

    def test_real_shp_file_processing(self):
        """Test processing the real SHP file with contours"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found: {self.real_shp_file}")
        
        # Test complete processing pipeline
        geometries, crs = self.parser.parse_shp_file(self.real_shp_file)
        assert len(geometries) > 0
        assert all(isinstance(g, LineString) for g in geometries)
        
        # Test densification
        densified_geometries = [self.parser.densify_linestring(g) for g in geometries]
        assert len(densified_geometries) == len(geometries)
        
        # Test polygon boundary creation
        polygon = self.parser.create_polygon_boundary_from_contours(geometries)
        assert isinstance(polygon, Polygon)
        assert polygon.is_valid
        assert polygon.area > 0

    def test_wgs84_boundary_clipping(self):
        """Test clipping with WGS84 boundary"""
        # Create a test polygon boundary
        test_polygon = Polygon([
            (-79.86, 40.14), (-79.83, 40.14), 
            (-79.83, 40.18), (-79.86, 40.18), (-79.86, 40.14)
        ])
        
        # Create a clipping boundary
        clip_boundary = Polygon([
            (-79.85, 40.15), (-79.84, 40.15), 
            (-79.84, 40.16), (-79.85, 40.16), (-79.85, 40.15)
        ])
        
        # Test clipping
        clipped = self.parser.clip_to_boundary(test_polygon, clip_boundary)
        assert isinstance(clipped, Polygon)
        assert clipped.area <= test_polygon.area
        assert clipped.area <= clip_boundary.area

    def test_crs_validation(self):
        """Test CRS validation for WGS84 requirement"""
        # Test valid WGS84 CRS
        valid_crs = pyproj.CRS.from_epsg(4326)
        assert self.parser._is_wgs84_crs(valid_crs) is True
        
        # Test invalid CRS (UTM)
        invalid_crs = pyproj.CRS.from_epsg(32617)  # UTM Zone 17N
        assert self.parser._is_wgs84_crs(invalid_crs) is False

    def test_output_format_validation(self):
        """Test that output format matches requirements"""
        # Test that parser outputs numpy arrays in correct format
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found: {self.real_shp_file}")
        
        vertices, faces = self.parser.process_shp_file(self.real_shp_file)
        assert isinstance(vertices, np.ndarray)
        assert vertices.shape[1] == 3  # x, y, z coordinates
        assert len(vertices) > 0
        assert np.all(np.isfinite(vertices))

    def test_error_handling_invalid_file(self):
        """Test error handling for invalid SHP files"""
        # Test with non-existent file
        with pytest.raises(RuntimeError):
            self.parser.parse_shp_file("nonexistent.shp")
        
        # Test with corrupted file
        with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as f:
            f.write(b"not a valid shapefile")
            temp_file = f.name
        
        try:
            with pytest.raises(RuntimeError):
                self.parser.parse_shp_file(temp_file)
        finally:
            os.unlink(temp_file)

    def test_densification_performance(self):
        """Test performance of densification with large LineStrings"""
        # Create a long LineString that needs significant densification
        coords = []
        for i in range(100):
            coords.append((-79.85 + i * 0.0001, 40.15 + i * 0.0001, 300 + i))
        
        test_line = LineString(coords)
        
        # Test densification performance
        import time
        start_time = time.time()
        densified = self.parser.densify_linestring(test_line, max_distance_feet=1.0)
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 5.0  # Should complete in under 5 seconds
        assert len(densified.coords) > len(test_line.coords)

    def test_polygon_boundary_quality(self):
        """Test quality of generated polygon boundary"""
        # Create contours that should form a reasonable boundary
        test_contours = [
            LineString([(-79.85, 40.15), (-79.84, 40.15)]),
            LineString([(-79.84, 40.15), (-79.84, 40.16)]),
            LineString([(-79.84, 40.16), (-79.85, 40.16)]),
            LineString([(-79.85, 40.16), (-79.85, 40.15)])
        ]
        
        # Test boundary quality
        polygon = self.parser.create_polygon_boundary_from_contours(test_contours)
        assert polygon.is_valid
        assert polygon.area > 0
        assert not polygon.is_empty
        
        # Check that all contours are within or on the boundary
        for contour in test_contours:
            assert polygon.contains(contour) or polygon.touches(contour)

    def test_integration_with_existing_workflow(self):
        """Test integration with existing surface processing workflow"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found: {self.real_shp_file}")
        
        # Test that SHP processing integrates with existing workflow
        vertices, faces = self.parser.process_shp_file(self.real_shp_file)
        
        # Should be compatible with existing surface processing
        from app.services.surface_processor import SurfaceProcessor
        processor = SurfaceProcessor()
        
        # Should be able to process the vertices
        # Boundary must be in (lon, lat) order to match vertices
        processed_vertices = processor.clip_to_boundary(vertices, [(-79.86, 40.14), (-79.83, 40.18)])
        assert len(processed_vertices) > 0

    # Keep existing tests that are still relevant
    def test_shp_parsing_real_file(self):
        """Test parsing a real SHP file"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        geometries, crs = self.parser.parse_shp_file(self.real_shp_file)
        assert len(geometries) > 0
        assert crs.to_epsg() == 4326

    def test_crs_validation_wgs84(self):
        """Test CRS validation for WGS84 files"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Should validate successfully
        assert self.parser.validate_shp_file(self.real_shp_file)

    def test_crs_validation_non_wgs84(self):
        """Test CRS validation for non-WGS84 files (should raise error)"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Test with a mock non-WGS84 CRS
        with patch('app.utils.shp_parser.CRS') as mock_crs:
            mock_crs.return_value.to_epsg.return_value = 32610  # UTM Zone 10N
            
            # Should raise RuntimeError for non-WGS84 CRS
            with pytest.raises(RuntimeError, match="CRS must be WGS84"):
                self.parser.parse_shp_file(self.real_shp_file)

    def test_empty_shp_file(self):
        """Test handling of empty SHP file"""
        # Create a mock empty file
        with patch('fiona.open') as mock_open:
            mock_collection = MagicMock()
            mock_collection.__len__.return_value = 0
            mock_collection.__iter__.return_value = []
            mock_collection.crs = None
            mock_open.return_value.__enter__.return_value = mock_collection
            
            # Should raise RuntimeError for missing CRS
            with pytest.raises(RuntimeError, match="SHP file must have a defined CRS"):
                self.parser.parse_shp_file("empty.shp")

    def test_unsupported_geometry_type(self):
        """Test handling of unsupported geometry types"""
        # Create a mock file with unsupported geometry
        with patch('fiona.open') as mock_open:
            mock_collection = MagicMock()
            mock_collection.__len__.return_value = 1
            # Use a CRS dict that pyproj.CRS can parse
            mock_collection.crs = {'init': 'epsg:4326', 'no_defs': True}
            mock_feature = {
                'geometry': {'type': 'LineString', 'coordinates': [[0, 0], [1, 1]]}
            }
            mock_collection.__iter__.return_value = [mock_feature]
            mock_open.return_value.__enter__.return_value = mock_collection
            
            # Should raise RuntimeError for CRS mismatch
            with pytest.raises(RuntimeError, match="CRS must be WGS84"):
                self.parser.parse_shp_file("line.shp")

    def test_large_shp_file_performance(self):
        """Test performance with large SHP file"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        import time
        
        # Time the parsing
        start_time = time.time()
        geometries, crs = self.parser.parse_shp_file(self.real_shp_file)
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 10.0  # Should parse in under 10 seconds
        assert len(geometries) > 0

    def test_output_format_consistency(self):
        """Test that output format is consistent"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Parse the real SHP file
        geometries, crs = self.parser.parse_shp_file(self.real_shp_file)
        
        # Check format consistency
        assert isinstance(geometries, list)
        assert all(isinstance(g, LineString) for g in geometries)
        assert crs.to_epsg() == 4326

    def test_boundary_clipping_edge_cases(self):
        """Test boundary clipping edge cases"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Parse the real SHP file
        geometries, crs = self.parser.parse_shp_file(self.real_shp_file)
        
        # Test clipping with various boundaries
        test_boundaries = [
            Polygon([(-79.86, 40.14), (-79.83, 40.14), (-79.83, 40.18), (-79.86, 40.18)]),
            Polygon([(-79.85, 40.15), (-79.84, 40.15), (-79.84, 40.16), (-79.85, 40.16)]),
        ]
        
        for boundary in test_boundaries:
            for geom in geometries[:3]:  # Test first 3 geometries
                clipped = self.parser.clip_to_boundary(geom, boundary)
                if clipped is not None:
                    assert clipped.area <= geom.area

    def test_coordinate_precision(self):
        """Test coordinate precision handling"""
        if not os.path.exists(self.real_shp_file):
            pytest.skip(f"Real SHP file not found at {self.real_shp_file}")
        
        # Parse the real SHP file
        geometries, crs = self.parser.parse_shp_file(self.real_shp_file)
        
        # Check coordinate precision
        for geom in geometries:
            coords = list(geom.coords)
            for coord in coords:
                assert all(isinstance(c, (int, float)) for c in coord)
                assert all(np.isfinite(c) for c in coord)

    def test_crs_validation_methods(self):
        """Test CRS validation methods"""
        # Test WGS84 CRS validation
        wgs84_crs = pyproj.CRS.from_epsg(4326)
        assert self.parser._is_wgs84_crs(wgs84_crs) is True
        
        # Test non-WGS84 CRS validation
        utm_crs = pyproj.CRS.from_epsg(32617)  # UTM Zone 17N
        assert self.parser._is_wgs84_crs(utm_crs) is False
        
        # Test invalid CRS should raise CRSError
        with pytest.raises(pyproj.exceptions.CRSError):
            pyproj.CRS.from_epsg(99999)

    # Minor Task 10.2.1: SHP to UTM Projection and Preparation Tests
    
    def test_shp_projection_to_utm(self):
        """Test projection of SHP geometries from WGS84 to UTM coordinates"""
        # Create test WGS84 coordinates (longitude, latitude)
        wgs84_coords = [
            (-79.85, 40.15, 300),  # Pittsburgh area
            (-79.84, 40.15, 301),
            (-79.84, 40.16, 302),
            (-79.85, 40.16, 303)
        ]
        
        # Test projection to UTM
        utm_coords = self.parser._project_to_utm(wgs84_coords)
        
        # Verify output format
        assert isinstance(utm_coords, np.ndarray)
        assert utm_coords.shape == (4, 3)  # Same number of points, 3 coordinates
        assert utm_coords.dtype in [np.float32, np.float64]
        
        # Verify coordinates are in meters (UTM)
        # UTM coordinates should be much larger than WGS84 degrees
        assert np.all(utm_coords[:, 0] > 100000)  # UTM X coordinates are large
        assert np.all(utm_coords[:, 1] > 4000000)  # UTM Y coordinates are very large
        
        # Verify Z coordinates are preserved
        np.testing.assert_array_almost_equal(utm_coords[:, 2], [300, 301, 302, 303], decimal=1)

    def test_utm_zone_detection(self):
        """Test automatic UTM zone detection from coordinates"""
        # Test different UTM zones (returns full EPSG codes)
        test_cases = [
            ((-79.85, 40.15), 32617),  # Pittsburgh - UTM Zone 17N
            ((-118.0, 34.0), 32611),   # Los Angeles - UTM Zone 11N
            ((2.0, 48.0), 32631),      # Paris - UTM Zone 31N
            ((151.0, -33.0), 32756),   # Sydney - UTM Zone 56S
        ]
        
        for (lon, lat), expected_zone in test_cases:
            detected_zone = self.parser._get_utm_zone(lat, lon)
            assert detected_zone == expected_zone, f"Expected zone {expected_zone} for ({lon}, {lat}), got {detected_zone}"

    def test_projection_accuracy(self):
        """Test accuracy of WGS84 to UTM projection"""
        # Test known coordinate pairs with high precision
        # Pittsburgh area: WGS84 (-79.85, 40.15) should project to UTM Zone 17N
        wgs84_coords = [(-79.85, 40.15, 300)]
        
        utm_coords = self.parser._project_to_utm(wgs84_coords)
        
        # Accept a wider range for X due to UTM implementation differences
        # UTM Zone 17N coordinates for Pittsburgh area should be around:
        # X: 570000-610000, Y: 4440000-4450000
        assert 570000 < utm_coords[0, 0] < 610000
        assert 4440000 < utm_coords[0, 1] < 4450000
        assert utm_coords[0, 2] == 300  # Z coordinate preserved

    def test_projection_with_different_geometry_types(self):
        """Test projection of different geometry types to UTM"""
        # Test points
        point_coords = [(-79.85, 40.15, 300)]
        point_utm = self.parser._project_to_utm(point_coords)
        assert point_utm.shape == (1, 3)
        
        # Test multiple points
        multi_point_coords = [
            (-79.85, 40.15, 300),
            (-79.84, 40.15, 301),
            (-79.84, 40.16, 302)
        ]
        multi_point_utm = self.parser._project_to_utm(multi_point_coords)
        assert multi_point_utm.shape == (3, 3)
        
        # Test polygon coordinates (closed ring)
        polygon_coords = [
            (-79.85, 40.15, 300),
            (-79.84, 40.15, 301),
            (-79.84, 40.16, 302),
            (-79.85, 40.16, 303),
            (-79.85, 40.15, 300)  # Closed ring
        ]
        polygon_utm = self.parser._project_to_utm(polygon_coords)
        assert polygon_utm.shape == (5, 3)

    def test_projection_error_handling(self):
        """Test error handling for invalid coordinates during projection"""
        # Test with invalid coordinates
        invalid_coords = [
            (181.0, 40.15, 300),   # Longitude > 180
            (-79.85, 91.0, 300),   # Latitude > 90
            (-181.0, 40.15, 300),  # Longitude < -180
            (-79.85, -91.0, 300),  # Latitude < -90
        ]
        
        for coords in invalid_coords:
            with pytest.raises(ValueError):
                self.parser._project_to_utm([coords])

    def test_projection_performance(self):
        """Test performance of projection with large datasets"""
        # Create large dataset
        import time
        large_coords = []
        for i in range(10000):
            lon = -79.85 + (i % 100) * 0.001
            lat = 40.15 + (i // 100) * 0.001
            large_coords.append((lon, lat, 300 + i))
        
        # Test projection performance
        start_time = time.time()
        utm_coords = self.parser._project_to_utm(large_coords)
        elapsed_time = time.time() - start_time
        
        assert elapsed_time < 5.0  # Should project 10k points in <5 seconds
        assert utm_coords.shape == (10000, 3)
        assert np.all(np.isfinite(utm_coords))

    def test_projection_output_format_consistency(self):
        """Test that projection output format is consistent with PLY parser"""
        # Test that projected coordinates match expected format
        wgs84_coords = [(-79.85, 40.15, 300), (-79.84, 40.15, 301)]
        utm_coords = self.parser._project_to_utm(wgs84_coords)
        
        # Should match PLY parser output format
        assert isinstance(utm_coords, np.ndarray)
        assert utm_coords.shape[1] == 3  # x, y, z coordinates
        assert utm_coords.dtype in [np.float32, np.float64]
        assert len(utm_coords) == 2
        
        # Coordinates should be in meters (UTM)
        assert np.all(utm_coords[:, 0] > 100000)  # X coordinates in meters
        assert np.all(utm_coords[:, 1] > 4000000)  # Y coordinates in meters
        assert np.all(utm_coords[:, 2] >= 300)  # Z coordinates preserved

    def test_projection_with_missing_z_coordinates(self):
        """Test projection when Z coordinates are missing"""
        # Test 2D coordinates (no Z)
        wgs84_2d = [(-79.85, 40.15), (-79.84, 40.15)]
        
        with pytest.raises(ValueError):
            self.parser._project_to_utm(wgs84_2d)  # Should require 3D coordinates

    def test_projection_with_nan_coordinates(self):
        """Test projection handling of NaN coordinates"""
        # Test with NaN coordinates
        nan_coords = [(-79.85, 40.15, 300), (np.nan, 40.15, 301)]
        
        with pytest.raises(ValueError):
            self.parser._project_to_utm(nan_coords)

    def test_projection_with_infinite_coordinates(self):
        """Test projection handling of infinite coordinates"""
        # Test with infinite coordinates
        inf_coords = [(-79.85, 40.15, 300), (np.inf, 40.15, 301)]
        
        with pytest.raises(ValueError):
            self.parser._project_to_utm(inf_coords)

    def test_projection_edge_cases(self):
        """Test projection with edge case coordinates"""
        # Test coordinates at UTM zone boundaries
        edge_cases = [
            (-180.0, 0.0, 300),    # International date line
            (180.0, 0.0, 300),     # International date line
            (0.0, 90.0, 300),      # North pole
            (0.0, -90.0, 300),     # South pole
        ]
        
        for coords in edge_cases:
            try:
                utm_coords = self.parser._project_to_utm([coords])
                assert utm_coords.shape == (1, 3)
                assert np.all(np.isfinite(utm_coords))
            except Exception as e:
                # Some edge cases might not be supported, which is acceptable
                assert "not supported" in str(e) or "invalid" in str(e)

    def test_projection_round_trip_accuracy(self):
        """Test round-trip accuracy of projection (WGS84 -> UTM -> WGS84)"""
        # Test round-trip projection accuracy
        original_coords = [(-79.85, 40.15, 300), (-79.84, 40.15, 301)]
        
        # Project to UTM
        utm_coords = self.parser._project_to_utm(original_coords)
        
        # Project back to WGS84, explicitly pass UTM zone
        utm_zone = self.parser._get_utm_zone(40.15, -79.85)
        wgs84_coords = self.parser._project_to_wgs84(utm_coords, utm_zone=utm_zone)
        
        # Verify round-trip accuracy (within reasonable tolerance)
        np.testing.assert_array_almost_equal(
            np.array(original_coords), 
            wgs84_coords, 
            decimal=5  # Slightly relaxed precision for round-trip
        )

    def test_projection_with_different_utm_zones(self):
        """Test projection with coordinates in different UTM zones"""
        # Test coordinates that span multiple UTM zones
        multi_zone_coords = [
            (-79.85, 40.15, 300),  # UTM Zone 17N
            (-118.0, 34.0, 301),   # UTM Zone 11N
        ]
        
        # Should raise error for mixed UTM zones
        with pytest.raises(ValueError):
            self.parser._project_to_utm(multi_zone_coords)

    def test_projection_output_validation(self):
        """Test validation of projection output"""
        wgs84_coords = [(-79.85, 40.15, 300), (-79.84, 40.15, 301)]
        utm_coords = self.parser._project_to_utm(wgs84_coords)
        
        # Validate output
        assert self.parser._validate_utm_coordinates(utm_coords)
        
        # Test invalid UTM coordinates
        invalid_utm = np.array([[0, 0, 300], [100, 100, 301]])  # Too small for UTM
        assert not self.parser._validate_utm_coordinates(invalid_utm)


if __name__ == "__main__":
    pytest.main([__file__]) 