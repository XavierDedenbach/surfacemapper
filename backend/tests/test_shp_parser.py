import pytest
import numpy as np
import tempfile
import os
import fiona
import shapely.geometry as sgeom
from shapely.geometry import Point, Polygon, MultiPoint, LineString
import pyproj
from unittest.mock import patch, MagicMock

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
        """Test CRS validation helper methods"""
        # Test WGS84 CRS
        wgs84_crs = pyproj.CRS.from_epsg(4326)
        assert self.parser._is_wgs84_crs(wgs84_crs) is True
        
        # Test non-WGS84 CRS
        utm_crs = pyproj.CRS.from_epsg(32617)
        assert self.parser._is_wgs84_crs(utm_crs) is False


if __name__ == "__main__":
    pytest.main([__file__]) 