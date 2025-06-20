"""
Synchronous unit tests for file validation and upload response logic.
"""
import pytest
from app.utils.file_validator import validate_file_extension, validate_file_size, validate_ply_format
from app.models.data_models import SurfaceUploadResponse, ProcessingStatus

class TestFileValidation:
    def test_valid_ply_extension(self):
        assert validate_file_extension("test.ply")
        assert validate_file_extension("TEST.PLY")
        assert not validate_file_extension("test.txt")
        assert not validate_file_extension(".ply")
        assert not validate_file_extension("")

    def test_file_size(self):
        assert validate_file_size(1024)
        assert not validate_file_size(0)
        assert not validate_file_size(3 * 1024 * 1024 * 1024)  # 3GB

    def test_valid_ascii_ply_format(self):
        ply = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n0.0 0.0 0.0\n"
        assert validate_ply_format(ply)

    def test_invalid_ply_format(self):
        not_ply = b"not a ply file"
        assert not validate_ply_format(not_ply)
        missing_header = b"ply\nformat ascii 1.0\n"
        assert not validate_ply_format(missing_header)

    def test_valid_binary_ply_format(self):
        ply = b"ply\nformat binary_little_endian 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n"
        assert validate_ply_format(ply)

class TestSurfaceUploadResponse:
    def test_response_success(self):
        resp = SurfaceUploadResponse(message="ok", filename="a.ply", status=ProcessingStatus.PENDING)
        assert resp.message == "ok"
        assert resp.filename == "a.ply"
        assert resp.status == ProcessingStatus.PENDING

    def test_response_fields(self):
        resp = SurfaceUploadResponse(message="done", filename="b.ply", status=ProcessingStatus.COMPLETED)
        assert hasattr(resp, "message")
        assert hasattr(resp, "filename")
        assert hasattr(resp, "status") 