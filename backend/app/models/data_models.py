"""
Pydantic data models for API requests and responses
"""
from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from enum import Enum

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
    wgs84_lat: float = Field(..., description="WGS84 latitude of reference vertex")
    wgs84_lon: float = Field(..., description="WGS84 longitude of reference vertex")
    orientation_degrees: float = Field(..., description="Orientation angle in degrees clockwise from North")
    scaling_factor: float = Field(..., description="Scaling factor to apply to coordinates")

class AnalysisBoundary(BaseModel):
    """Analysis boundary definition"""
    wgs84_coordinates: List[Tuple[float, float]] = Field(..., description="Four WGS84 lat/lon coordinates defining rectangular boundary")

class TonnageInput(BaseModel):
    """Tonnage input for compaction rate calculation"""
    layer_index: int = Field(..., description="Index of the layer (0-based)")
    tonnage: float = Field(..., description="Tonnage in imperial tons")

class ProcessingRequest(BaseModel):
    """Request model for surface processing"""
    surface_files: List[str] = Field(..., description="List of uploaded surface file paths")
    georeference_params: List[GeoreferenceParams] = Field(..., description="Georeferencing parameters for each surface")
    analysis_boundary: AnalysisBoundary = Field(..., description="Analysis boundary definition")
    tonnage_inputs: Optional[List[TonnageInput]] = Field(None, description="Optional tonnage inputs for compaction calculation")
    generate_base_surface: bool = Field(False, description="Whether to generate a base surface")
    base_surface_offset: Optional[float] = Field(None, description="Vertical offset for base surface in feet")

class ProcessingResponse(BaseModel):
    """Response model for processing request"""
    message: str
    status: ProcessingStatus
    job_id: str

class VolumeResult(BaseModel):
    """Volume calculation result"""
    layer_designation: str = Field(..., description="Layer designation (e.g., 'Surface 0 to Surface 1')")
    volume_cubic_yards: float = Field(..., description="Volume in cubic yards")
    confidence_interval: Tuple[float, float] = Field(..., description="Confidence interval for volume calculation")
    uncertainty: float = Field(..., description="Uncertainty estimate")

class ThicknessResult(BaseModel):
    """Thickness calculation result"""
    layer_designation: str = Field(..., description="Layer designation")
    average_thickness_feet: float = Field(..., description="Average thickness in feet")
    min_thickness_feet: float = Field(..., description="Minimum thickness in feet")
    max_thickness_feet: float = Field(..., description="Maximum thickness in feet")
    confidence_interval: Tuple[float, float] = Field(..., description="Confidence interval for thickness")

class CompactionResult(BaseModel):
    """Compaction rate calculation result"""
    layer_designation: str = Field(..., description="Layer designation")
    compaction_rate_lbs_per_cubic_yard: Optional[float] = Field(None, description="Compaction rate in lbs/cubic yard")
    tonnage_used: Optional[float] = Field(None, description="Tonnage used in calculation")

class AnalysisResults(BaseModel):
    """Complete analysis results"""
    volume_results: List[VolumeResult]
    thickness_results: List[ThicknessResult]
    compaction_results: List[CompactionResult]
    processing_metadata: dict = Field(..., description="Processing metadata and performance metrics") 