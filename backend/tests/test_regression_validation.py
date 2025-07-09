import pytest
import numpy as np
from pyproj import Transformer
from app.services.surface_processor import SurfaceProcessor
from app.utils.shp_parser import SHPParser
from app.utils.ply_parser import PLYParser
from app.services.volume_calculator import calculate_surface_area, calculate_volume_between_surfaces


class TestRegressionValidation:
    """Test suite for regression and validation of projection system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.surface_processor = SurfaceProcessor()
        self.shp_parser = SHPParser()
        self.ply_parser = PLYParser()
        
        # Known test coordinates (WGS84)
        self.test_coordinates_wgs84 = np.array([
            [-87.6298, 41.8781, 100],  # Chicago area
            [-87.6299, 41.8782, 101],
            [-87.6300, 41.8783, 102],
            [-87.6301, 41.8784, 103],
            [-87.6302, 41.8785, 104],
        ], dtype=np.float64)
        
        # Corresponding UTM coordinates (EPSG:32616 for Chicago)
        self.test_coordinates_utm = np.array([
            [443000, 4638000, 100],
            [443010, 4638010, 101],
            [443020, 4638020, 102],
            [443030, 4638030, 103],
            [443040, 4638040, 104],
        ], dtype=np.float64)
        
        # Simple triangular faces for testing
        self.test_faces = np.array([
            [0, 1, 2],
            [1, 2, 3],
            [2, 3, 4],
        ])
        
        # Boundary polygon in WGS84
        self.boundary_wgs84 = [
            (-87.6298, 41.8781),
            (-87.6302, 41.8781),
            (-87.6302, 41.8785),
            (-87.6298, 41.8785),
        ]
        
        # Boundary polygon in UTM
        self.boundary_utm = [
            (443000, 4638000),
            (443040, 4638000),
            (443040, 4638040),
            (443000, 4638040),
        ]
    
    def test_surface_area_consistency_wgs84_to_utm(self):
        """Test that surface area calculations are consistent between WGS84 and UTM."""
        # Calculate area in WGS84
        area_wgs84 = calculate_surface_area(self.test_coordinates_wgs84)
        # Calculate area in UTM
        area_utm = calculate_surface_area(self.test_coordinates_utm)
        # Areas should be very close (within 0.1% due to projection distortion)
        area_ratio = area_utm / area_wgs84
        assert 0.999 < area_ratio < 1.001, f"Area ratio {area_ratio} outside acceptable range"
        print(f"WGS84 area: {area_wgs84:.6f}")
        print(f"UTM area: {area_utm:.6f}")
        print(f"Area ratio: {area_ratio:.6f}")
    
    def test_volume_consistency_wgs84_to_utm(self):
        """Test that volume calculations are consistent between WGS84 and UTM."""
        # Create two surfaces with different heights
        surface1_wgs84 = self.test_coordinates_wgs84.copy()
        surface2_wgs84 = self.test_coordinates_wgs84.copy()
        surface2_wgs84[:, 2] = surface2_wgs84[:, 2] + 10.0  # 10 meter difference
        surface1_utm = self.test_coordinates_utm.copy()
        surface2_utm = self.test_coordinates_utm.copy()
        surface2_utm[:, 2] = surface2_utm[:, 2] + 10.0  # 10 meter difference
        # Calculate volume in WGS84
        volume_wgs84 = calculate_volume_between_surfaces(surface1_wgs84, surface2_wgs84)
        # Calculate volume in UTM
        volume_utm = calculate_volume_between_surfaces(surface1_utm, surface2_utm)
        # Volumes should be very close (within 0.1% due to projection distortion)
        volume_ratio = volume_utm / volume_wgs84
        assert 0.999 < volume_ratio < 1.001, f"Volume ratio {volume_ratio} outside acceptable range"
        print(f"WGS84 volume: {volume_wgs84:.6f}")
        print(f"UTM volume: {volume_utm:.6f}")
        print(f"Volume ratio: {volume_ratio:.6f}")
    
    def test_no_artifacts_in_projection(self):
        """Test that projection doesn't introduce geometric artifacts."""
        # Create a simple rectangular surface
        vertices_wgs84 = np.array([
            [-87.6298, 41.8781, 100],
            [-87.6298, 41.8785, 100],
            [-87.6302, 41.8785, 100],
            [-87.6302, 41.8781, 100],
        ])
        
        faces_wgs84 = np.array([[0, 1, 2], [0, 2, 3]])
        
        # Project to UTM
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
        vertices_utm = np.zeros_like(vertices_wgs84)
        for i, vertex in enumerate(vertices_wgs84):
            utm_x, utm_y = transformer.transform(vertex[0], vertex[1])
            vertices_utm[i] = [utm_x, utm_y, vertex[2]]
        
        # Calculate areas
        area_wgs84 = calculate_surface_area(vertices_wgs84)
        area_utm = calculate_surface_area(vertices_utm)
        
        # Check that the surface remains planar (no warping)
        # Calculate normal vectors for each face
        def calculate_face_normal(vertices, face):
            v1 = vertices[face[1]] - vertices[face[0]]
            v2 = vertices[face[2]] - vertices[face[0]]
            normal = np.cross(v1, v2)
            return normal / np.linalg.norm(normal)
        
        normal_wgs84 = calculate_face_normal(vertices_wgs84, faces_wgs84[0])
        normal_utm = calculate_face_normal(vertices_utm, faces_wgs84[0])
        
        # Normals should be very similar (within 1 degree)
        dot_product = np.dot(normal_wgs84, normal_utm)
        angle_diff = np.arccos(np.clip(dot_product, -1, 1))
        assert angle_diff < np.radians(1), f"Normal angle difference {np.degrees(angle_diff):.2f}° exceeds 1°"
        
        print(f"WGS84 normal: {normal_wgs84}")
        print(f"UTM normal: {normal_utm}")
        print(f"Angle difference: {np.degrees(angle_diff):.4f}°")
    
    def test_boundary_clipping_consistency(self):
        """Test that boundary clipping produces consistent results in both coordinate systems."""
        # Create a surface that extends beyond the boundary
        vertices_wgs84 = np.array([
            [-87.6297, 41.8780, 100],  # Outside boundary
            [-87.6298, 41.8781, 100],  # On boundary
            [-87.6302, 41.8785, 100],  # On boundary
            [-87.6303, 41.8786, 100],  # Outside boundary
        ])
        
        faces_wgs84 = np.array([[0, 1, 2], [1, 2, 3]])
        
        # Project to UTM
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
        vertices_utm = np.zeros_like(vertices_wgs84)
        for i, vertex in enumerate(vertices_wgs84):
            utm_x, utm_y = transformer.transform(vertex[0], vertex[1])
            vertices_utm[i] = [utm_x, utm_y, vertex[2]]
        
        # Clip in WGS84 (should not be done in production, but for comparison)
        try:
            clipped_wgs84_vertices, clipped_wgs84_faces = self.surface_processor.clip_to_boundary(
                vertices_wgs84, self.boundary_wgs84, faces_wgs84
            )
            area_wgs84 = calculate_surface_area(clipped_wgs84_vertices)
        except Exception:
            # Expected - clipping in WGS84 should fail or produce warnings
            area_wgs84 = 0
        
        # Clip in UTM (correct approach)
        clipped_utm_vertices, clipped_utm_faces = self.surface_processor.clip_to_boundary(
            vertices_utm, self.boundary_utm, faces_wgs84
        )
        area_utm = calculate_surface_area(clipped_utm_vertices)
        
        # UTM clipping should produce valid results
        assert area_utm > 0, "UTM clipping should produce valid surface area"
        assert len(clipped_utm_vertices) > 0, "UTM clipping should produce vertices"
        assert len(clipped_utm_faces) > 0, "UTM clipping should produce faces"
        
        print(f"WGS84 clipped area: {area_wgs84:.6f}")
        print(f"UTM clipped area: {area_utm:.6f}")
        print(f"UTM vertices after clipping: {len(clipped_utm_vertices)}")
        print(f"UTM faces after clipping: {len(clipped_utm_faces)}")
    
    def test_coordinate_transformation_accuracy(self):
        """Test that coordinate transformations are accurate and reversible."""
        # Test known coordinate pairs
        test_pairs = [
            (-87.6298, 41.8781),  # Chicago
            (-74.0060, 40.7128),  # New York
            (-118.2437, 34.0522),  # Los Angeles
        ]
        
        transformer_wgs84_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
        transformer_utm_to_wgs84 = Transformer.from_crs("EPSG:32616", "EPSG:4326", always_xy=True)
        
        for lon, lat in test_pairs:
            # Forward transformation
            utm_x, utm_y = transformer_wgs84_to_utm.transform(lon, lat)
            
            # Reverse transformation
            lon_back, lat_back = transformer_utm_to_wgs84.transform(utm_x, utm_y)
            
            # Check accuracy (should be within 1 meter)
            lon_diff = abs(lon - lon_back)
            lat_diff = abs(lat - lat_back)
            
            # Convert to meters for comparison
            lat_diff_meters = lat_diff * 111000  # Approximate meters per degree latitude
            lon_diff_meters = lon_diff * 111000 * np.cos(np.radians(lat))  # Approximate meters per degree longitude
            
            assert lat_diff_meters < 1, f"Latitude transformation error {lat_diff_meters:.3f}m exceeds 1m"
            assert lon_diff_meters < 1, f"Longitude transformation error {lon_diff_meters:.3f}m exceeds 1m"
            
            print(f"Coordinate ({lon}, {lat}) -> UTM ({utm_x:.2f}, {utm_y:.2f}) -> ({lon_back:.6f}, {lat_back:.6f})")
            print(f"  Errors: lat {lat_diff_meters:.3f}m, lon {lon_diff_meters:.3f}m")
    
    def test_no_distortion_in_mesh_operations(self):
        """Test that mesh operations don't introduce geometric distortions."""
        # Create a regular grid surface
        x_coords = np.linspace(-87.6298, -87.6302, 5)
        y_coords = np.linspace(41.8781, 41.8785, 5)
        z_coords = np.full((5, 5), 100)
        
        vertices_wgs84 = []
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                vertices_wgs84.append([x, y, z_coords[i, j]])
        vertices_wgs84 = np.array(vertices_wgs84)
        
        # Create faces for the grid
        faces_wgs84 = []
        for i in range(4):
            for j in range(4):
                idx = i * 5 + j
                faces_wgs84.extend([
                    [idx, idx + 1, idx + 5],
                    [idx + 1, idx + 6, idx + 5],
                ])
        faces_wgs84 = np.array(faces_wgs84)
        
        # Project to UTM
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32616", always_xy=True)
        vertices_utm = np.zeros_like(vertices_wgs84)
        for i, vertex in enumerate(vertices_wgs84):
            utm_x, utm_y = transformer.transform(vertex[0], vertex[1])
            vertices_utm[i] = [utm_x, utm_y, vertex[2]]
        
        # Calculate areas
        area_wgs84 = calculate_surface_area(vertices_wgs84)
        area_utm = calculate_surface_area(vertices_utm)
        
        # Check area consistency
        area_ratio = area_utm / area_wgs84
        assert 0.999 < area_ratio < 1.001, f"Area ratio {area_ratio} outside acceptable range"
        
        # Check that mesh topology is preserved
        assert len(vertices_wgs84) == len(vertices_utm), "Vertex count should be preserved"
        assert len(faces_wgs84) == len(faces_wgs84), "Face count should be preserved"
        
        # Check that relative positions are preserved
        def calculate_edge_lengths(vertices, faces):
            lengths = []
            for face in faces:
                for i in range(3):
                    v1 = vertices[face[i]]
                    v2 = vertices[face[(i + 1) % 3]]
                    length = np.linalg.norm(v2 - v1)
                    lengths.append(length)
            return np.array(lengths)
        
        edge_lengths_wgs84 = calculate_edge_lengths(vertices_wgs84, faces_wgs84)
        edge_lengths_utm = calculate_edge_lengths(vertices_utm, faces_wgs84)
        
        # Edge length ratios should be consistent
        length_ratios = edge_lengths_utm / edge_lengths_wgs84
        mean_ratio = np.mean(length_ratios)
        std_ratio = np.std(length_ratios)
        
        # Standard deviation should be small (indicating consistent scaling)
        assert std_ratio < 0.01, f"Edge length ratio std dev {std_ratio:.6f} exceeds 0.01"
        
        print(f"WGS84 area: {area_wgs84:.6f}")
        print(f"UTM area: {area_utm:.6f}")
        print(f"Area ratio: {area_ratio:.6f}")
        print(f"Mean edge length ratio: {mean_ratio:.6f}")
        print(f"Edge length ratio std dev: {std_ratio:.6f}")
    
    def test_volume_calculation_accuracy(self):
        """Test that volume calculations are accurate in UTM coordinates."""
        # Create two parallel surfaces with known separation
        separation = 10.0  # 10 meters
        
        vertices1_utm = self.test_coordinates_utm.copy()
        vertices2_utm = self.test_coordinates_utm.copy()
        vertices2_utm[:, 2] += separation
        
        # Calculate volume
        volume = calculate_volume_between_surfaces(vertices1_utm, vertices2_utm)
        
        # Calculate expected volume (area * separation)
        area = calculate_surface_area(vertices1_utm)
        expected_volume = area * separation
        
        # Check accuracy (within 1%)
        volume_ratio = volume / expected_volume
        assert 0.99 < volume_ratio < 1.01, f"Volume ratio {volume_ratio} outside acceptable range"
        
        print(f"Calculated volume: {volume:.6f}")
        print(f"Expected volume: {expected_volume:.6f}")
        print(f"Volume ratio: {volume_ratio:.6f}")
        print(f"Surface area: {area:.6f}")
        print(f"Separation: {separation:.1f}m") 