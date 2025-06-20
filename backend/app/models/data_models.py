"""
Pydantic data models for API requests and responses
"""
from pydantic import BaseModel, Field, field_validator, model_validator, ValidationError
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import uuid

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SurfaceUploadResponse(BaseModel):
    """Response model for surface upload"""
    message: str
    filename: str
    status: ProcessingStatus

class GeoreferenceParams(BaseModel):
    """Georeferencing parameters for a surface"""
    wgs84_lat: float = Field(..., ge=-90, le=90, description="WGS84 latitude of reference vertex")
    wgs84_lon: float = Field(..., ge=-180, le=180, description="WGS84 longitude of reference vertex")
    orientation_degrees: float = Field(..., ge=0, le=360, description="Orientation angle in degrees clockwise from North")
    scaling_factor: float = Field(..., gt=0, description="Scaling factor to apply to coordinates")

class AnalysisBoundary(BaseModel):
    """Analysis boundary definition"""
    wgs84_coordinates: List[Tuple[float, float]] = Field(..., min_length=4, max_length=4, description="Four WGS84 lat/lon coordinates defining rectangular boundary")

class TonnageInput(BaseModel):
    """Tonnage input for compaction rate calculation"""
    layer_index: int = Field(..., ge=0, description="Index of the layer (0-based)")
    tonnage: float = Field(..., gt=0, description="Tonnage in imperial tons")

class ProcessingRequest(BaseModel):
    """Request model for surface processing"""
    surface_files: List[str] = Field(..., min_length=1, max_length=4, description="List of uploaded surface file paths")
    georeference_params: List[GeoreferenceParams] = Field(..., description="Georeferencing parameters for each surface")
    analysis_boundary: AnalysisBoundary = Field(..., description="Analysis boundary definition")
    tonnage_inputs: Optional[List[TonnageInput]] = Field(None, description="Optional tonnage inputs for compaction calculation")
    generate_base_surface: bool = Field(False, description="Whether to generate a base surface")
    base_surface_offset: Optional[float] = Field(None, gt=0, description="Vertical offset for base surface in feet")

    @field_validator('georeference_params')
    @classmethod
    def validate_georeference_params_match_files(cls, v, info):
        """Validate that georeference params match the number of surface files"""
        if 'surface_files' in info.data:
            if len(v) != len(info.data['surface_files']):
                raise ValueError(f"Number of georeference parameters ({len(v)}) must match number of surface files ({len(info.data['surface_files'])})")
        return v

class ProcessingResponse(BaseModel):
    """Response model for processing request"""
    message: str
    status: ProcessingStatus
    job_id: str

class VolumeResult(BaseModel):
    """Volume calculation result"""
    layer_designation: str = Field(..., description="Layer designation (e.g., 'Surface 0 to Surface 1')")
    volume_cubic_yards: float = Field(..., ge=0, description="Volume in cubic yards")
    confidence_interval: Tuple[float, float] = Field(..., description="Confidence interval for volume calculation")
    uncertainty: float = Field(..., ge=0, description="Uncertainty estimate")

    @field_validator('confidence_interval')
    @classmethod
    def validate_confidence_interval(cls, v):
        """Validate confidence interval has exactly 2 values"""
        if len(v) != 2:
            raise ValueError("Confidence interval must have exactly 2 values")
        if v[0] > v[1]:
            raise ValueError("Confidence interval lower bound must be less than upper bound")
        return v

class ThicknessResult(BaseModel):
    """Thickness calculation result"""
    layer_designation: str = Field(..., description="Layer designation")
    average_thickness_feet: float = Field(..., ge=0, description="Average thickness in feet")
    min_thickness_feet: float = Field(..., ge=0, description="Minimum thickness in feet")
    max_thickness_feet: float = Field(..., ge=0, description="Maximum thickness in feet")
    confidence_interval: Tuple[float, float] = Field(..., description="Confidence interval for thickness")

    @field_validator('confidence_interval')
    @classmethod
    def validate_confidence_interval(cls, v):
        """Validate confidence interval has exactly 2 values"""
        if len(v) != 2:
            raise ValueError("Confidence interval must have exactly 2 values")
        if v[0] > v[1]:
            raise ValueError("Confidence interval lower bound must be less than upper bound")
        return v

    @model_validator(mode='after')
    def validate_thickness_consistency(self):
        """Validate thickness values are consistent"""
        if self.min_thickness_feet > self.max_thickness_feet:
            raise ValueError("Minimum thickness cannot be greater than maximum thickness")
        if self.average_thickness_feet < self.min_thickness_feet or self.average_thickness_feet > self.max_thickness_feet:
            raise ValueError("Average thickness must be between minimum and maximum thickness")
        return self

class CompactionResult(BaseModel):
    """Compaction rate calculation result"""
    layer_designation: str = Field(..., description="Layer designation")
    compaction_rate_lbs_per_cubic_yard: Optional[float] = Field(None, ge=0, description="Compaction rate in lbs/cubic yard")
    tonnage_used: Optional[float] = Field(None, ge=0, description="Tonnage used in calculation")

class AnalysisResults(BaseModel):
    """Complete analysis results"""
    volume_results: List[VolumeResult]
    thickness_results: List[ThicknessResult]
    compaction_results: List[CompactionResult]
    processing_metadata: dict = Field(..., description="Processing metadata and performance metrics") 