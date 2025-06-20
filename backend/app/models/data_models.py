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


class CoordinateSystem(BaseModel):
    """Coordinate system definition"""
    name: str = Field(..., description="Name of the coordinate system")
    epsg_code: int = Field(..., gt=0, description="EPSG code for the coordinate system")
    description: str = Field(..., description="Description of the coordinate system")
    units: str = Field(..., description="Units of the coordinate system")
    bounds: Dict[str, float] = Field(..., description="Geographic bounds of the coordinate system")

    @field_validator('bounds')
    @classmethod
    def validate_bounds(cls, v):
        """Validate coordinate bounds are logical"""
        required_keys = ['min_lat', 'max_lat', 'min_lon', 'max_lon']
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Missing required bound key: {key}")
        
        if v['min_lat'] >= v['max_lat']:
            raise ValueError("min_lat must be less than max_lat")
        if v['min_lon'] >= v['max_lon']:
            raise ValueError("min_lon must be less than max_lon")
        
        return v


class ProcessingParameters(BaseModel):
    """Processing parameters for surface analysis"""
    triangulation_method: str = Field(..., description="Triangulation method to use")
    interpolation_method: str = Field(..., description="Interpolation method to use")
    grid_resolution: float = Field(..., gt=0, description="Grid resolution in units")
    smoothing_factor: float = Field(..., ge=0, le=1, description="Smoothing factor (0-1)")
    outlier_threshold: float = Field(..., gt=0, description="Outlier detection threshold")
    confidence_level: float = Field(..., gt=0, le=1, description="Confidence level (0-1)")
    max_iterations: int = Field(..., gt=0, description="Maximum iterations for convergence")
    convergence_tolerance: float = Field(..., gt=0, description="Convergence tolerance")

    @field_validator('triangulation_method')
    @classmethod
    def validate_triangulation_method(cls, v):
        """Validate triangulation method"""
        valid_methods = ['delaunay', 'convex_hull', 'alpha_shape']
        if v not in valid_methods:
            raise ValueError(f"Triangulation method must be one of: {valid_methods}")
        return v

    @field_validator('interpolation_method')
    @classmethod
    def validate_interpolation_method(cls, v):
        """Validate interpolation method"""
        valid_methods = ['linear', 'cubic', 'nearest']
        if v not in valid_methods:
            raise ValueError(f"Interpolation method must be one of: {valid_methods}")
        return v


class SurfaceConfiguration(BaseModel):
    """Complete surface configuration"""
    coordinate_system: CoordinateSystem = Field(..., description="Coordinate system configuration")
    processing_params: ProcessingParameters = Field(..., description="Processing parameters")
    quality_thresholds: Dict[str, float] = Field(..., description="Quality control thresholds")
    export_settings: Dict[str, Any] = Field(..., description="Export configuration")

    @field_validator('quality_thresholds')
    @classmethod
    def validate_quality_thresholds(cls, v):
        """Validate quality thresholds are non-negative"""
        for key, value in v.items():
            if value < 0:
                raise ValueError(f"Quality threshold {key} must be greater than or equal to 0")
        return v

    @field_validator('export_settings')
    @classmethod
    def validate_export_settings(cls, v):
        """Validate export settings"""
        if 'format' not in v:
            raise ValueError("Export settings must include 'format'")
        
        valid_formats = ['ply', 'obj', 'stl', 'xyz']
        if v['format'] not in valid_formats:
            raise ValueError(f"Export format must be one of: {valid_formats}")
        
        return v


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
        """Validate confidence interval has exactly 2 values and lower < upper"""
        if len(v) != 2:
            raise ValueError("Confidence interval must have exactly 2 values")
        if v[0] >= v[1]:
            raise ValueError("Confidence interval lower bound must be less than upper bound")
        return v


class QualityMetrics(BaseModel):
    """Quality metrics for analysis results"""
    point_density: float = Field(..., ge=0, description="Point density per unit area")
    surface_coverage: float = Field(..., ge=0, le=1, description="Surface coverage ratio (0-1)")
    data_completeness: float = Field(..., ge=0, le=1, description="Data completeness ratio (0-1)")
    noise_level: float = Field(..., ge=0, description="Noise level estimate")
    accuracy_estimate: float = Field(..., ge=0, description="Accuracy estimate")
    precision_estimate: float = Field(..., ge=0, description="Precision estimate")
    reliability_score: float = Field(..., ge=0, le=1, description="Reliability score (0-1)")
    quality_flags: Dict[str, bool] = Field(..., description="Quality assessment flags")


class DetailedAnalysisReport(BaseModel):
    """Detailed analysis report with comprehensive results"""
    analysis_id: str = Field(..., description="Unique analysis identifier")
    timestamp: str = Field(..., description="Analysis timestamp")
    processing_duration_seconds: float = Field(..., ge=0, description="Processing duration in seconds")
    input_surfaces_count: int = Field(..., gt=0, description="Number of input surfaces")
    analysis_boundary_area_sq_meters: float = Field(..., ge=0, description="Analysis boundary area")
    statistical_analysis: StatisticalAnalysis = Field(..., description="Statistical analysis results")
    quality_metrics: QualityMetrics = Field(..., description="Quality metrics")
    warnings: List[str] = Field(default_factory=list, description="Analysis warnings")
    recommendations: List[str] = Field(default_factory=list, description="Analysis recommendations") 