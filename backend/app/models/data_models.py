"""
Pydantic data models for API requests and responses
"""
from pydantic import BaseModel, Field, field_validator
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import uuid

# --- New Models for Frontend Alignment ---

class SurfaceInfo(BaseModel):
    id: str
    name: str

class AnalysisBoundary(BaseModel):
    min_x: float
    max_x: float
    min_y: float
    max_y: float

class TonnagePerLayer(BaseModel):
    layer_index: int
    tonnage: float

class GeoreferenceParams(BaseModel):
    wgs84_lat: float = Field(..., ge=-90, le=90, description="WGS84 latitude")
    wgs84_lon: float = Field(..., ge=-180, le=180, description="WGS84 longitude")
    orientation_degrees: float = Field(..., ge=0, le=360, description="Orientation in degrees")
    scaling_factor: float = Field(..., gt=0, description="Scaling factor")

class TonnageInput(BaseModel):
    layer_index: int = Field(..., ge=0, description="Layer index")
    tonnage: float = Field(..., gt=0, description="Tonnage value")

class CoordinateSystem(BaseModel):
    name: str = Field(..., description="Coordinate system name")
    epsg_code: int = Field(..., description="EPSG code")
    bounds: Optional[Tuple[float, float, float, float]] = Field(None, description="System bounds")

class ProcessingParameters(BaseModel):
    method: str = Field(..., description="Processing method")
    grid_resolution: float = Field(..., gt=0, description="Grid resolution")
    confidence_level: float = Field(..., ge=0.5, le=0.99, description="Confidence level")

class SurfaceConfiguration(BaseModel):
    quality_threshold: float = Field(..., ge=0, le=1, description="Quality threshold")
    export_format: str = Field(..., description="Export format")

class QualityMetrics(BaseModel):
    coverage_percentage: float = Field(..., ge=0, le=100, description="Coverage percentage")
    noise_level: float = Field(..., ge=0, le=1, description="Noise level")

class DetailedAnalysisReport(BaseModel):
    analysis_id: str = Field(..., description="Analysis ID")
    duration_seconds: float = Field(..., ge=0, description="Analysis duration")
    surface_count: int = Field(..., gt=0, description="Number of surfaces")

class ProcessingRequest(BaseModel):
    surface_ids: List[str]
    analysis_type: str
    generate_base_surface: bool
    georeference_params: List[Dict[str, Any]]
    analysis_boundary: Dict[str, Any]
    params: Dict[str, Any]

    @field_validator('analysis_type')
    @classmethod
    def validate_analysis_type(cls, v):
        if v not in ['compaction', 'volume', 'thickness']:
            raise ValueError("Analysis type must be 'compaction', 'volume', or 'thickness'")
        return v

# --- Existing Models ---

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class SurfaceUploadResponse(BaseModel):
    """Response model for surface upload"""
    message: str
    filename: str
    surface_id: str = Field(..., description="Unique identifier for the uploaded surface")
    status: ProcessingStatus
    vertices: Optional[List[Tuple[float, float, float]]] = Field(None, description="List of vertex coordinates")
    faces: Optional[List[List[int]]] = Field(None, description="List of face indices")

class ProcessingResponse(BaseModel):
    """Response model for processing request"""
    message: str
    status: str
    job_id: str

class VolumeResult(BaseModel):
    """Volume calculation result"""
    layer_designation: str = Field(..., description="Layer designation (e.g., 'Surface 0 to Surface 1')")
    volume_cubic_yards: float = Field(..., ge=0, description="Volume in cubic yards")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="Confidence interval for volume calculation")
    uncertainty: Optional[float] = Field(None, ge=0, description="Uncertainty estimate")

class ThicknessResult(BaseModel):
    """Thickness calculation result"""
    layer_designation: str
    average_thickness_feet: float
    min_thickness_feet: float
    max_thickness_feet: float
    std_dev_thickness_feet: float
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="Confidence interval for thickness")

class CompactionResult(BaseModel):
    """Compaction rate calculation result"""
    layer_designation: str = Field(..., description="Layer designation")
    compaction_rate_lbs_per_cubic_yard: Optional[float] = Field(None, ge=0, description="Compaction rate in lbs/cubic yard")
    tonnage_used: Optional[float] = Field(None, ge=0, description="Tonnage used in calculation")

class AnalysisResults(BaseModel):
    """Complete analysis results"""
    analysis_id: str
    status: str
    results: Optional[dict] = None
    volume_results: Optional[List[VolumeResult]] = None
    thickness_results: Optional[List[ThicknessResult]] = None
    compaction_results: Optional[List[CompactionResult]] = None
    processing_metadata: Optional[dict] = Field(None, description="Processing metadata and performance metrics")


class StatisticalAnalysis(BaseModel):
    """Statistical analysis results"""
    mean_value: float = Field(..., description="Mean value")
    median_value: float = Field(..., description="Median value")
    standard_deviation: float = Field(..., ge=0, description="Standard deviation")
    variance: float = Field(..., ge=0, description="Variance")
    skewness: float = Field(..., description="Skewness")
    kurtosis: float = Field(..., description="Kurtosis")
    sample_count: int = Field(..., gt=0, description="Number of samples")
    confidence_interval_95: Tuple[float, float] = Field(..., description="95% confidence interval")
    percentiles: Dict[str, float] = Field(..., description="Percentile values")
