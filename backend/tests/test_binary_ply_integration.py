import os
import time
import tempfile
import pytest
from fastapi.testclient import TestClient
from app.main import app

class TestBinaryPLYIntegration:
    def setup_method(self):
        self.client = TestClient(app)
        self.tv_test_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/test_files/tv_test.ply"))

    def test_full_workflow_with_real_binary_ply(self):
        if not os.path.exists(self.tv_test_path):
            pytest.skip("tv_test.ply not available for testing")
        # 1. Upload binary PLY file
        with open(self.tv_test_path, "rb") as f:
            response = self.client.post("/api/surfaces/upload", files={"file": ("tv_test.ply", f, "application/octet-stream")})
        assert response.status_code == 200
        upload_data = response.json()
        surface_id = upload_data.get("surface_id") or upload_data.get("file_id")
        assert surface_id
        # 2. Start analysis with binary PLY
        analysis_request = {
            "surface_ids": [surface_id],
            "analysis_type": "volume",
            "generate_base_surface": True,
            "georeference_params": [
                {
                    "wgs84_lat": 35.05,
                    "wgs84_lon": -118.15,
                    "orientation_degrees": 0.0,
                    "scaling_factor": 1.0
                }
            ],
            "analysis_boundary": {
                "wgs84_coordinates": [[35.0, -118.2], [35.1, -118.1]]
            },
            "params": {
                "base_surface_offset": 3.0
            }
        }
        response = self.client.post("/api/analysis/start", json=analysis_request)
        assert response.status_code == 202
        analysis_id = response.json()["analysis_id"]
        # 3. Monitor analysis progress
        max_wait = 120
        start_time = time.time()
        while time.time() - start_time < max_wait:
            status_response = self.client.get(f"/api/analysis/{analysis_id}/status")
            status_data = status_response.json()
            if status_data["status"] == "completed":
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Analysis failed: {status_data.get('error_message', 'Unknown error')}")
            time.sleep(2)
        # 4. Verify results
        results_response = self.client.get(f"/api/analysis/{analysis_id}/results")
        assert results_response.status_code == 200
        results_data = results_response.json()
        assert "volume_results" in results_data
        assert len(results_data["volume_results"]) > 0
        assert "thickness_results" in results_data
        assert len(results_data["thickness_results"]) > 0
        volume_result = results_data["volume_results"][0]
        assert volume_result["volume_cubic_yards"] > 0
        assert "Surface 0 to 1" in volume_result["layer_name"]

    def test_error_recovery_with_corrupted_binary_ply(self):
        # Create corrupted binary PLY file
        corrupted_content = b"""ply\nformat binary_little_endian 1.0\nelement vertex 1000\nproperty float x\nproperty float y\nproperty float z\nend_header\n"""
        corrupted_content += b"\x00" * 100  # Incomplete data
        with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
            f.write(corrupted_content)
            temp_file = f.name
        try:
            with open(temp_file, "rb") as f:
                response = self.client.post("/api/surfaces/upload", files={"file": ("corrupted.ply", f, "application/octet-stream")})
            assert response.status_code == 400
            error_data = response.json()
            assert "invalid" in error_data["detail"].lower() or "corrupted" in error_data["detail"].lower() or "error" in error_data["detail"].lower()
        finally:
            os.unlink(temp_file)
        # System should still process valid files after error
        if not os.path.exists(self.tv_test_path):
            pytest.skip("tv_test.ply not available for testing")
        with open(self.tv_test_path, "rb") as f:
            response = self.client.post("/api/surfaces/upload", files={"file": ("tv_test.ply", f, "application/octet-stream")})
        assert response.status_code == 200
        upload_data = response.json()
        surface_id = upload_data.get("surface_id") or upload_data.get("file_id")
        assert surface_id
        analysis_request = {
            "surface_ids": [surface_id],
            "analysis_type": "volume",
            "generate_base_surface": True,
            "georeference_params": [
                {
                    "wgs84_lat": 35.05,
                    "wgs84_lon": -118.15,
                    "orientation_degrees": 0.0,
                    "scaling_factor": 1.0
                }
            ],
            "analysis_boundary": {
                "wgs84_coordinates": [[35.0, -118.2], [35.1, -118.1]]
            },
            "params": {
                "base_surface_offset": 3.0
            }
        }
        response = self.client.post("/api/analysis/start", json=analysis_request)
        assert response.status_code == 202 