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
    wgs84_coordinates: List[Tuple[float, float]] = Field(..., min_length=4, max_length=4, description="WGS84 coordinate pairs")
    min_x: Optional[float] = None
    max_x: Optional[float] = None
    min_y: Optional[float] = None
    max_y: Optional[float] = None

    @field_validator('wgs84_coordinates')
    @classmethod
    def validate_coordinates(cls, v):
        if len(v) != 4:
            raise ValueError("List should have at least 4 items")
        for coord in v:
            if len(coord) != 2:
                raise ValueError("Each coordinate must be a tuple of 2 values")
            lat, lon = coord
            if not (-90 <= lat <= 90):
                raise ValueError("Latitude must be between -90 and 90")
            if not (-180 <= lon <= 180):
                raise ValueError("Longitude must be between -180 and 180")
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
    description: str = Field(..., description="Coordinate system description")
    units: str = Field(..., description="Coordinate system units")
    bounds: Dict[str, float] = Field(..., description="System bounds")

    @field_validator('bounds')
    @classmethod
    def validate_bounds(cls, v):
        if 'min_lat' in v and 'max_lat' in v:
            if v['min_lat'] >= v['max_lat']:
                raise ValueError("min_lat must be less than max_lat")
        if 'min_lon' in v and 'max_lon' in v:
            if v['min_lon'] >= v['max_lon']:
                raise ValueError("min_lon must be less than max_lon")
        return v

class ProcessingParameters(BaseModel):
    triangulation_method: str = Field(..., description="Triangulation method")
    interpolation_method: str = Field(..., description="Interpolation method")
    grid_resolution: float = Field(..., gt=0, description="Grid resolution")
    smoothing_factor: float = Field(..., ge=0, le=1, description="Smoothing factor")
    outlier_threshold: float = Field(..., gt=0, description="Outlier threshold")
    confidence_level: float = Field(..., ge=0.5, le=0.99, description="Confidence level")
    max_iterations: int = Field(..., gt=0, description="Maximum iterations")
    convergence_tolerance: float = Field(1e-6, gt=0, description="Convergence tolerance")

    @field_validator('triangulation_method')
    @classmethod
    def validate_triangulation_method(cls, v):
        valid_methods = ['delaunay', 'alpha_shape', 'convex_hull']
        if v not in valid_methods:
            raise ValueError(f"Triangulation method must be one of {valid_methods}")
        return v

    @field_validator('interpolation_method')
    @classmethod
    def validate_interpolation_method(cls, v):
        valid_methods = ['linear', 'cubic', 'nearest']
        if v not in valid_methods:
            raise ValueError(f"Interpolation method must be one of {valid_methods}")
        return v

class SurfaceConfiguration(BaseModel):
    coordinate_system: CoordinateSystem
    processing_params: ProcessingParameters
    quality_thresholds: Dict[str, float] = Field(..., description="Quality thresholds")
    export_settings: Dict[str, Any] = Field(..., description="Export settings")

    @field_validator('quality_thresholds')
    @classmethod
    def validate_quality_thresholds(cls, v):
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Quality threshold {key} must be greater than or equal to 0")
        return v

    @field_validator('export_settings')
    @classmethod
    def validate_export_settings(cls, v):
        if 'format' in v:
            valid_formats = ['ply', 'csv', 'json', 'excel', 'geojson']
            if v['format'] not in valid_formats:
                raise ValueError(f"Export format must be one of {valid_formats}")
        return v

class QualityMetrics(BaseModel):
    point_density: float = Field(..., ge=0, description="Point density")
    surface_coverage: float = Field(..., ge=0, le=1, description="Surface coverage")
    data_completeness: float = Field(..., ge=0, le=1, description="Data completeness")
    noise_level: float = Field(..., ge=0, le=1, description="Noise level")
    accuracy_estimate: float = Field(..., ge=0, description="Accuracy estimate")
    precision_estimate: float = Field(..., ge=0, description="Precision estimate")
    reliability_score: float = Field(..., ge=0, le=1, description="Reliability score")
    quality_flags: Dict[str, bool] = Field(..., description="Quality flags")

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
            raise ValueError("Confidence interval lower bound must be less than upper bound")
        return v

class DetailedAnalysisReport(BaseModel):
    analysis_id: str = Field(..., description="Analysis ID")
    timestamp: str = Field(..., description="Analysis timestamp")
    processing_duration_seconds: float = Field(..., ge=0, description="Processing duration")
    input_surfaces_count: int = Field(..., gt=0, description="Number of input surfaces")
    analysis_boundary_area_sq_meters: float = Field(..., ge=0, description="Boundary area")
    statistical_analysis: StatisticalAnalysis
    quality_metrics: QualityMetrics
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    recommendations: List[str] = Field(default_factory=list, description="Recommendations")

    @field_validator('processing_duration_seconds')
    @classmethod
    def validate_duration(cls, v):
        if v < 0:
            raise ValueError("Input should be greater than or equal to 0")
        return v

    @field_validator('input_surfaces_count')
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
    surface_id: Optional[str] = Field(None, description="Unique identifier for the uploaded surface")
    status: ProcessingStatus
    vertices: Optional[List[Tuple[float, float, float]]] = Field(None, description="List of vertex coordinates")
    faces: Optional[List[List[int]]] = Field(None, description="List of face indices")

class ProcessingResponse(BaseModel):
    """Response model for processing request"""
    message: str
    status: ProcessingStatus
    job_id: str

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
                raise ValueError("Confidence interval lower bound must be less than upper bound")
        return v

class ThicknessResult(BaseModel):
    """Thickness calculation result"""
    layer_designation: str
    average_thickness_feet: float
    min_thickness_feet: float
    max_thickness_feet: float
    std_dev_thickness_feet: Optional[float] = None
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
    analysis_id: str = Field(..., description="Analysis ID")
    status: str = Field(..., description="Analysis status")
    results: Optional[dict] = None
    volume_results: Optional[List[VolumeResult]] = None
    thickness_results: Optional[List[ThicknessResult]] = None
    compaction_results: Optional[List[CompactionResult]] = None
    processing_metadata: Optional[dict] = Field(None, description="Processing metadata and performance metrics")
