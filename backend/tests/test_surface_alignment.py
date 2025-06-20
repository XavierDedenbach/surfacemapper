import numpy as np
import pytest

class TestSurfaceAlignment:
    def setup_method(self):
        from app.services.coord_transformer import SurfaceAlignment
        self.aligner = SurfaceAlignment()

    def create_test_surface(self, n=100, offset=(0, 0, 0), noise=0.0):
        # Create a flat grid surface in XY, Z=0, with optional offset and noise
        x, y = np.meshgrid(np.linspace(0, 10, int(np.sqrt(n))), np.linspace(0, 10, int(np.sqrt(n))))
        z = np.zeros_like(x)
        points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        points += np.array(offset)
        if noise > 0:
            points += np.random.normal(0, noise, points.shape)
        return points

    def apply_similarity_transform(self, points, rotation_deg, scale, offset):
        # Serial approach: translation -> rotation -> scale
        theta = np.radians(rotation_deg)
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        # Step 1: Apply translation
        pts_xy = points[:, :2] + offset[:2]
        # Step 2: Apply rotation (around translated centroid)
        translated_centroid = np.mean(pts_xy, axis=0)
        pts_xy = (pts_xy - translated_centroid) @ rot.T + translated_centroid
        # Step 3: Apply scale (around translated centroid)
        pts_xy = (pts_xy - translated_centroid) * scale + translated_centroid
        pts_z = points[:, 2] + offset[2]
        return np.column_stack([pts_xy, pts_z])

    def test_alignment_with_known_transform(self):
        # Known transform: rotation 30 deg, scale 1.5, offset [10, 20, 0]
        surface1 = self.create_test_surface(100)
        theta = np.radians(30)
        scale = 1.5
        rot = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0, 0, 1]
        ])
        offset = np.array([10, 20, 0])
        surface2 = (surface1 @ rot.T) * scale + offset
        # Align
        estimated = self.aligner.align_surfaces(surface1, surface2, method='icp')
        print('Estimated:', estimated)
        print('True scale:', scale, 'True rotation:', 30, 'True offset:', offset)
        centroid = np.mean(surface1[:, :2], axis=0)
        print('Centroid:', centroid)
        aligned = self.apply_similarity_transform(surface1, estimated['rotation'], estimated['scale'], estimated['offset'])
        mean_error = np.mean(np.linalg.norm(aligned - surface2, axis=1))
        print('Mean error:', mean_error)
        assert mean_error < 0.2

    def test_alignment_with_different_reference_points(self):
        # Surfaces with different reference points
        surface1 = self.create_test_surface(100)
        surface2 = surface1 + np.array([5, -3, 2])
        estimated = self.aligner.align_surfaces(surface1, surface2, method='point')
        np.testing.assert_allclose(estimated['offset'], [5, -3, 2], atol=1e-6)
        assert abs(estimated['rotation']) < 1e-6
        assert abs(estimated['scale'] - 1.0) < 1e-6

    def test_alignment_with_noisy_data(self):
        # Add noise to surface2
        surface1 = self.create_test_surface(100)
        surface2 = surface1 + np.array([2, 2, 0]) + np.random.normal(0, 0.05, surface1.shape)
        estimated = self.aligner.align_surfaces(surface1, surface2, method='icp')
        aligned = self.apply_similarity_transform(surface1, estimated['rotation'], estimated['scale'], estimated['offset'])
        mean_error = np.mean(np.linalg.norm(aligned - surface2, axis=1))
        assert mean_error < 0.3

    def test_alignment_validation_metrics(self):
        # Check that alignment returns quality metrics
        surface1 = self.create_test_surface(100)
        surface2 = surface1 + np.array([1, 1, 0])
        result = self.aligner.align_surfaces(surface1, surface2, method='icp', return_metrics=True)
        assert 'rmse' in result
        assert result['rmse'] < 0.5
        assert 'inlier_ratio' in result
        assert 0.9 <= result['inlier_ratio'] <= 1.0

    def test_alignment_with_outliers(self):
        # Add outliers to surface2
        surface1 = self.create_test_surface(100)
        surface2 = surface1.copy()
        surface2[:10] += 100  # 10 outliers
        estimated = self.aligner.align_surfaces(surface1, surface2, method='icp', reject_outliers=True)
        # Should still align the inlier part
        assert abs(estimated['rotation']) < 1.0
        assert abs(estimated['scale'] - 1.0) < 0.05
        assert np.linalg.norm(estimated['offset']) < 2.0 