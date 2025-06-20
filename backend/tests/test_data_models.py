"""
Tests for Pydantic data models used in the Surface Volume and Layer Thickness Analysis Tool
"""
import pytest
from pydantic import ValidationError
from app.models.data_models import (
    SurfaceUploadResponse,
    ProcessingStatus,
    GeoreferenceParams,
    AnalysisBoundary,
    TonnageInput,
    ProcessingRequest,
    ProcessingResponse,
    VolumeResult,
    ThicknessResult,
    CompactionResult,
    AnalysisResults
)


class TestSurfaceUploadResponse:
    """Test cases for SurfaceUploadResponse model"""

    def test_surface_upload_response_validation(self):
        """Test valid SurfaceUploadResponse creation"""
        response = SurfaceUploadResponse(
            message="File uploaded successfully",
            filename="test_surface.ply",
            status=ProcessingStatus.COMPLETED
        )
        
        assert response.message == "File uploaded successfully"
        assert response.filename == "test_surface.ply"
        assert response.status == ProcessingStatus.COMPLETED

    def test_surface_upload_response_invalid_status(self):
        """Test invalid status enum value"""
        with pytest.raises(ValidationError) as exc_info:
            SurfaceUploadResponse(
                message="File uploaded successfully",
                filename="test_surface.ply",
                status="invalid_status"
            )
        
        assert "Input should be" in str(exc_info.value)

    def test_surface_upload_response_missing_fields(self):
        """Test missing required fields"""
        with pytest.raises(ValidationError) as exc_info:
            SurfaceUploadResponse(
                message="File uploaded successfully"
                # Missing filename and status
            )
        
        assert "Field required" in str(exc_info.value)

    def test_surface_upload_response_serialization(self):
        """Test model serialization to dict"""
        response = SurfaceUploadResponse(
            message="File uploaded successfully",
            filename="test_surface.ply",
            status=ProcessingStatus.COMPLETED
        )
        
        response_dict = response.model_dump()
        assert response_dict["message"] == "File uploaded successfully"
        assert response_dict["filename"] == "test_surface.ply"
        assert response_dict["status"] == "completed"

    def test_surface_upload_response_deserialization(self):
        """Test model deserialization from dict"""
        data = {
            "message": "File uploaded successfully",
            "filename": "test_surface.ply",
            "status": "completed"
        }
        
        response = SurfaceUploadResponse.model_validate(data)
        assert response.message == "File uploaded successfully"
        assert response.filename == "test_surface.ply"
        assert response.status == ProcessingStatus.COMPLETED


class TestGeoreferenceParams:
    """Test cases for GeoreferenceParams model"""

    def test_georeference_params_validation(self):
        """Test valid GeoreferenceParams creation"""
        params = GeoreferenceParams(
            wgs84_lat=40.7128,
            wgs84_lon=-74.0060,
            orientation_degrees=45.0,
            scaling_factor=1.5
        )
        
        assert params.wgs84_lat == 40.7128
        assert params.wgs84_lon == -74.0060
        assert params.orientation_degrees == 45.0
        assert params.scaling_factor == 1.5

    def test_georeference_params_invalid_latitude(self):
        """Test invalid latitude values"""
        # Test latitude > 90
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=91.0,
                wgs84_lon=-74.0060,
                orientation_degrees=45.0,
                scaling_factor=1.5
            )
        assert "Input should be less than or equal to 90" in str(exc_info.value)

        # Test latitude < -90
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=-91.0,
                wgs84_lon=-74.0060,
                orientation_degrees=45.0,
                scaling_factor=1.5
            )
        assert "Input should be greater than or equal to -90" in str(exc_info.value)

    def test_georeference_params_invalid_longitude(self):
        """Test invalid longitude values"""
        # Test longitude > 180
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=40.7128,
                wgs84_lon=181.0,
                orientation_degrees=45.0,
                scaling_factor=1.5
            )
        assert "Input should be less than or equal to 180" in str(exc_info.value)

        # Test longitude < -180
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=40.7128,
                wgs84_lon=-181.0,
                orientation_degrees=45.0,
                scaling_factor=1.5
            )
        assert "Input should be greater than or equal to -180" in str(exc_info.value)

    def test_georeference_params_invalid_orientation(self):
        """Test invalid orientation values"""
        # Test negative orientation
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=40.7128,
                wgs84_lon=-74.0060,
                orientation_degrees=-10.0,
                scaling_factor=1.5
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)

        # Test orientation > 360
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=40.7128,
                wgs84_lon=-74.0060,
                orientation_degrees=370.0,
                scaling_factor=1.5
            )
        assert "Input should be less than or equal to 360" in str(exc_info.value)

    def test_georeference_params_invalid_scaling_factor(self):
        """Test invalid scaling factor values"""
        # Test zero scaling factor
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=40.7128,
                wgs84_lon=-74.0060,
                orientation_degrees=45.0,
                scaling_factor=0.0
            )
        assert "Input should be greater than 0" in str(exc_info.value)

        # Test negative scaling factor
        with pytest.raises(ValidationError) as exc_info:
            GeoreferenceParams(
                wgs84_lat=40.7128,
                wgs84_lon=-74.0060,
                orientation_degrees=45.0,
                scaling_factor=-1.0
            )
        assert "Input should be greater than 0" in str(exc_info.value)


class TestAnalysisBoundary:
    """Test cases for AnalysisBoundary model"""

    def test_analysis_boundary_validation(self):
        """Test valid AnalysisBoundary creation"""
        boundary = AnalysisBoundary(
            wgs84_coordinates=[
                (40.7128, -74.0060),
                (40.7128, -73.9960),
                (40.7028, -73.9960),
                (40.7028, -74.0060)
            ]
        )
        
        assert len(boundary.wgs84_coordinates) == 4
        assert boundary.wgs84_coordinates[0] == (40.7128, -74.0060)

    def test_analysis_boundary_invalid_coordinates_count(self):
        """Test invalid number of coordinates"""
        # Test with 3 coordinates (should be 4)
        with pytest.raises(ValidationError) as exc_info:
            AnalysisBoundary(
                wgs84_coordinates=[
                    (40.7128, -74.0060),
                    (40.7128, -73.9960),
                    (40.7028, -73.9960)
                ]
            )
        assert "List should have at least 4 items" in str(exc_info.value)

        # Test with 5 coordinates (should be 4)
        with pytest.raises(ValidationError) as exc_info:
            AnalysisBoundary(
                wgs84_coordinates=[
                    (40.7128, -74.0060),
                    (40.7128, -73.9960),
                    (40.7028, -73.9960),
                    (40.7028, -74.0060),
                    (40.7128, -74.0060)
                ]
            )
        assert "List should have at most 4 items" in str(exc_info.value)

    def test_analysis_boundary_invalid_coordinate_format(self):
        """Test invalid coordinate format"""
        # Test with wrong tuple size
        with pytest.raises(ValidationError) as exc_info:
            AnalysisBoundary(
                wgs84_coordinates=[
                    (40.7128, -74.0060, 100.0),  # 3 values instead of 2
                    (40.7128, -73.9960),
                    (40.7028, -73.9960),
                    (40.7028, -74.0060)
                ]
            )
        assert "Tuple should have at most 2 items" in str(exc_info.value)


class TestTonnageInput:
    """Test cases for TonnageInput model"""

    def test_tonnage_input_validation(self):
        """Test valid TonnageInput creation"""
        tonnage = TonnageInput(
            layer_index=0,
            tonnage=150.5
        )
        
        assert tonnage.layer_index == 0
        assert tonnage.tonnage == 150.5

    def test_tonnage_input_invalid_layer_index(self):
        """Test invalid layer index"""
        # Test negative layer index
        with pytest.raises(ValidationError) as exc_info:
            TonnageInput(
                layer_index=-1,
                tonnage=150.5
            )
        assert "Input should be greater than or equal to 0" in str(exc_info.value)

    def test_tonnage_input_invalid_tonnage(self):
        """Test invalid tonnage values"""
        # Test negative tonnage
        with pytest.raises(ValidationError) as exc_info:
            TonnageInput(
                layer_index=0,
                tonnage=-10.0
            )
        assert "Input should be greater than 0" in str(exc_info.value)

        # Test zero tonnage
        with pytest.raises(ValidationError) as exc_info:
            TonnageInput(
                layer_index=0,
                tonnage=0.0
            )
        assert "Input should be greater than 0" in str(exc_info.value)


class TestProcessingRequest:
    """Test cases for ProcessingRequest model"""

    def test_processing_request_validation(self):
        """Test valid ProcessingRequest creation"""
        request = ProcessingRequest(
            surface_files=["surface1.ply", "surface2.ply"],
            georeference_params=[
                GeoreferenceParams(
                    wgs84_lat=40.7128,
                    wgs84_lon=-74.0060,
                    orientation_degrees=45.0,
                    scaling_factor=1.5
                ),
                GeoreferenceParams(
                    wgs84_lat=40.7128,
                    wgs84_lon=-74.0060,
                    orientation_degrees=45.0,
                    scaling_factor=1.5
                )
            ],
            analysis_boundary=AnalysisBoundary(
                wgs84_coordinates=[
                    (40.7128, -74.0060),
                    (40.7128, -73.9960),
                    (40.7028, -73.9960),
                    (40.7028, -74.0060)
                ]
            ),
            tonnage_inputs=[
                TonnageInput(layer_index=0, tonnage=150.5)
            ],
            generate_base_surface=True,
            base_surface_offset=10.0
        )
        
        assert len(request.surface_files) == 2
        assert len(request.georeference_params) == 2
        assert request.generate_base_surface is True
        assert request.base_surface_offset == 10.0

    def test_processing_request_mismatched_files_and_params(self):
        """Test mismatch between surface files and georeference params"""
        with pytest.raises(ValidationError) as exc_info:
            ProcessingRequest(
                surface_files=["surface1.ply", "surface2.ply"],
                georeference_params=[
                    GeoreferenceParams(
                        wgs84_lat=40.7128,
                        wgs84_lon=-74.0060,
                        orientation_degrees=45.0,
                        scaling_factor=1.5
                    )
                    # Missing second georeference params
                ],
                analysis_boundary=AnalysisBoundary(
                    wgs84_coordinates=[
                        (40.7128, -74.0060),
                        (40.7128, -73.9960),
                        (40.7028, -73.9960),
                        (40.7028, -74.0060)
                    ]
                )
            )
        assert "Number of georeference parameters (1) must match number of surface files (2)" in str(exc_info.value)


class TestProcessingResponse:
    """Test cases for ProcessingResponse model"""

    def test_processing_response_validation(self):
        """Test valid ProcessingResponse creation"""
        response = ProcessingResponse(
            message="Processing started",
            status=ProcessingStatus.PROCESSING,
            job_id="job-12345"
        )
        
        assert response.message == "Processing started"
        assert response.status == ProcessingStatus.PROCESSING
        assert response.job_id == "job-12345"

    def test_processing_response_invalid_status(self):
        """Test invalid status enum value"""
        with pytest.raises(ValidationError) as exc_info:
            ProcessingResponse(
                message="Processing started",
                status="invalid_status",
                job_id="job-12345"
            )
        assert "Input should be" in str(exc_info.value)


class TestVolumeResult:
    """Test cases for VolumeResult model"""

    def test_volume_result_validation(self):
        """Test valid VolumeResult creation"""
        result = VolumeResult(
            layer_designation="Surface 0 to Surface 1",
            volume_cubic_yards=150.5,
            confidence_interval=(145.0, 155.0),
            uncertainty=2.5
        )
        
        assert result.layer_designation == "Surface 0 to Surface 1"
        assert result.volume_cubic_yards == 150.5
        assert result.confidence_interval == (145.0, 155.0)
        assert result.uncertainty == 2.5

    def test_volume_result_invalid_confidence_interval(self):
        """Test invalid confidence interval format"""
        with pytest.raises(ValidationError) as exc_info:
            VolumeResult(
                layer_designation="Surface 0 to Surface 1",
                volume_cubic_yards=150.5,
                confidence_interval=(155.0, 145.0),  # Lower bound > upper bound
                uncertainty=2.5
            )
        assert "Confidence interval lower bound must be less than upper bound" in str(exc_info.value)


class TestThicknessResult:
    """Test cases for ThicknessResult model"""

    def test_thickness_result_validation(self):
        """Test valid ThicknessResult creation"""
        result = ThicknessResult(
            layer_designation="Surface 0 to Surface 1",
            average_thickness_feet=2.5,
            min_thickness_feet=1.0,
            max_thickness_feet=4.0,
            confidence_interval=(2.3, 2.7)
        )
        
        assert result.layer_designation == "Surface 0 to Surface 1"
        assert result.average_thickness_feet == 2.5
        assert result.min_thickness_feet == 1.0
        assert result.max_thickness_feet == 4.0
        assert result.confidence_interval == (2.3, 2.7)

    def test_thickness_result_invalid_thickness_values(self):
        """Test invalid thickness values"""
        with pytest.raises(ValidationError) as exc_info:
            ThicknessResult(
                layer_designation="Surface 0 to Surface 1",
                average_thickness_feet=2.5,
                min_thickness_feet=4.0,  # Min > average
                max_thickness_feet=1.0,  # Max < average
                confidence_interval=(2.3, 2.7)
            )
        assert "Minimum thickness cannot be greater than maximum thickness" in str(exc_info.value)


class TestCompactionResult:
    """Test cases for CompactionResult model"""

    def test_compaction_result_validation(self):
        """Test valid CompactionResult creation"""
        result = CompactionResult(
            layer_designation="Surface 0 to Surface 1",
            compaction_rate_lbs_per_cubic_yard=1800.0,
            tonnage_used=150.5
        )
        
        assert result.layer_designation == "Surface 0 to Surface 1"
        assert result.compaction_rate_lbs_per_cubic_yard == 1800.0
        assert result.tonnage_used == 150.5

    def test_compaction_result_optional_fields(self):
        """Test CompactionResult with optional fields as None"""
        result = CompactionResult(
            layer_designation="Surface 0 to Surface 1",
            compaction_rate_lbs_per_cubic_yard=None,
            tonnage_used=None
        )
        
        assert result.layer_designation == "Surface 0 to Surface 1"
        assert result.compaction_rate_lbs_per_cubic_yard is None
        assert result.tonnage_used is None


class TestAnalysisResults:
    """Test cases for AnalysisResults model"""

    def test_analysis_results_validation(self):
        """Test valid AnalysisResults creation"""
        results = AnalysisResults(
            volume_results=[
                VolumeResult(
                    layer_designation="Surface 0 to Surface 1",
                    volume_cubic_yards=150.5,
                    confidence_interval=(145.0, 155.0),
                    uncertainty=2.5
                )
            ],
            thickness_results=[
                ThicknessResult(
                    layer_designation="Surface 0 to Surface 1",
                    average_thickness_feet=2.5,
                    min_thickness_feet=1.0,
                    max_thickness_feet=4.0,
                    confidence_interval=(2.3, 2.7)
                )
            ],
            compaction_results=[
                CompactionResult(
                    layer_designation="Surface 0 to Surface 1",
                    compaction_rate_lbs_per_cubic_yard=1800.0,
                    tonnage_used=150.5
                )
            ],
            processing_metadata={
                "processing_time_seconds": 120.5,
                "algorithm_version": "1.0.0",
                "validation_passed": True
            }
        )
        
        assert len(results.volume_results) == 1
        assert len(results.thickness_results) == 1
        assert len(results.compaction_results) == 1
        assert results.processing_metadata["processing_time_seconds"] == 120.5 