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
    wgs84_coordinates: List[Tuple[float, float]] = Field(..., description="WGS84 coordinate pairs")
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None

    @field_validator('wgs84_coordinates')
    @classmethod
    def validate_coordinates(cls, v):
        if len(v) < 4:
            raise ValueError("List should have at least 4 items")
        for coord in v:
            if len(coord) != 2:
                raise ValueError("Tuple should have at most 2 items")
        return v

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
    epsg_code: int = Field(..., gt=0, description="EPSG code")
    bounds: Tuple[float, float, float, float] = Field(..., description="System bounds (min_lat, max_lat, min_lon, max_lon)")

    @field_validator('bounds')
    @classmethod
    def validate_bounds(cls, v):
        min_lat, max_lat, min_lon, max_lon = v
        if min_lat >= max_lat:
            raise ValueError("min_lat must be less than max_lat")
        if min_lon >= max_lon:
            raise ValueError("min_lon must be less than max_lon")
        return v

class ProcessingParameters(BaseModel):
    method: str = Field(..., description="Processing method")
    triangulation_method: str = Field(..., description="Triangulation method")
    grid_resolution: float = Field(..., gt=0, description="Grid resolution")
    confidence_level: float = Field(..., ge=0.5, le=0.99, description="Confidence level")
    convergence_tolerance: float = Field(1e-6, gt=0, description="Convergence tolerance")

    @field_validator('triangulation_method')
    @classmethod
    def validate_triangulation_method(cls, v):
        valid_methods = ['delaunay', 'alpha_shape', 'convex_hull']
        if v not in valid_methods:
            raise ValueError(f"Triangulation method must be one of {valid_methods}")
        return v

class SurfaceConfiguration(BaseModel):
    coordinate_system: CoordinateSystem
    quality_threshold: float = Field(..., ge=0, le=1, description="Quality threshold")
    export_format: str = Field(..., description="Export format")
    min_point_density: float = Field(..., ge=0, description="Minimum point density")

    @field_validator('min_point_density')
    @classmethod
    def validate_point_density(cls, v):
        if v < 0:
            raise ValueError("Quality threshold min_point_density must be greater than or equal to 0")
        return v

    @field_validator('export_format')
    @classmethod
    def validate_export_format(cls, v):
        valid_formats = ['csv', 'json', 'excel', 'geojson']
        if v not in valid_formats:
            raise ValueError(f"Export format must be one of {valid_formats}")
        return v

class QualityMetrics(BaseModel):
    point_density: float = Field(..., ge=0, description="Point density")
    spatial_coverage: float = Field(..., ge=0, le=1, description="Spatial coverage")
    coverage_percentage: float = Field(..., ge=0, le=100, description="Coverage percentage")
    noise_level: float = Field(..., ge=0, le=1, description="Noise level")
    data_quality_score: float = Field(..., ge=0, le=1, description="Data quality score")
    acceptable_accuracy: bool = Field(..., description="Whether accuracy is acceptable")

class DetailedAnalysisReport(BaseModel):
    analysis_id: str = Field(..., description="Analysis ID")
    duration_seconds: float = Field(..., ge=0, description="Analysis duration")
    surface_count: int = Field(..., gt=0, description="Number of surfaces")
    quality_metrics: QualityMetrics
    processing_metadata: Dict[str, Any] = Field(default_factory=dict, description="Processing metadata")

    @field_validator('duration_seconds')
    @classmethod
    def validate_duration(cls, v):
        if v < 0:
            raise ValueError("Input should be greater than or equal to 0")
        return v

    @field_validator('surface_count')
    @classmethod
    def validate_surface_count(cls, v):
        if v <= 0:
            raise ValueError("Input should be greater than 0")
        return v

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

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        valid_statuses = ['pending', 'processing', 'completed', 'failed']
        if v not in valid_statuses:
            raise ValueError(f"Status must be one of {valid_statuses}")
        return v

class VolumeResult(BaseModel):
    """Volume calculation result"""
    layer_designation: str = Field(..., description="Layer designation (e.g., 'Surface 0 to Surface 1')")
    volume_cubic_yards: float = Field(..., ge=0, description="Volume in cubic yards")
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="Confidence interval for volume calculation")
    uncertainty: Optional[float] = Field(None, ge=0, description="Uncertainty estimate")

    @field_validator('confidence_interval')
    @classmethod
    def validate_confidence_interval(cls, v):
        if v is not None:
            lower, upper = v
            if lower >= upper:
                raise ValueError("Lower bound must be less than upper bound")
        return v

class ThicknessResult(BaseModel):
    """Thickness calculation result"""
    layer_designation: str
    average_thickness_feet: float
    min_thickness_feet: float
    max_thickness_feet: float
    std_dev_thickness_feet: float
    confidence_interval: Optional[Tuple[float, float]] = Field(None, description="Confidence interval for thickness")

    @field_validator('min_thickness_feet', 'max_thickness_feet')
    @classmethod
    def validate_thickness_values(cls, v, info):
        if 'min_thickness_feet' in info.data and 'max_thickness_feet' in info.data:
            min_val = info.data['min_thickness_feet']
            max_val = info.data['max_thickness_feet']
            if min_val > max_val:
                raise ValueError("Minimum thickness cannot be greater than maximum thickness")
        return v

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

    @field_validator('confidence_interval_95')
    @classmethod
    def validate_confidence_interval(cls, v):
        lower, upper = v
        if lower >= upper:
            raise ValueError("Lower bound must be less than upper bound")
        return v
