// TypeScript interfaces for API communication
// These interfaces define the structure of data exchanged between frontend and backend

export interface SurfaceUploadResponse {
  message: string;
  filename: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
}

export interface GeoreferenceParams {
  wgs84_lat: number;
  wgs84_lon: number;
  orientation_degrees: number;
  scaling_factor: number;
}

export interface AnalysisBoundary {
  wgs84_coordinates: [number, number][];
}

export interface TonnageInput {
  layer_index: number;
  tonnage: number;
}

export interface ProcessingRequest {
  surface_files: string[];
  georeference_params: GeoreferenceParams[];
  analysis_boundary: AnalysisBoundary;
  tonnage_inputs?: TonnageInput[];
  generate_base_surface: boolean;
  base_surface_offset?: number;
}

export interface ProcessingResponse {
  message: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  job_id: string;
}

export interface VolumeResult {
  layer_designation: string;
  volume_cubic_yards: number;
  confidence_interval: [number, number];
  uncertainty: number;
}

export interface ThicknessResult {
  layer_designation: string;
  average_thickness_feet: number;
  min_thickness_feet: number;
  max_thickness_feet: number;
  confidence_interval: [number, number];
}

export interface CompactionResult {
  layer_designation: string;
  compaction_rate_lbs_per_cubic_yard?: number;
  tonnage_used?: number;
}

export interface AnalysisResults {
  volume_results: VolumeResult[];
  thickness_results: ThicknessResult[];
  compaction_results: CompactionResult[];
  processing_metadata: Record<string, any>;
}

export interface CoordinateSystem {
  name: string;
  epsg_code: number;
  description: string;
  units: string;
  bounds: {
    min_lat: number;
    max_lat: number;
    min_lon: number;
    max_lon: number;
  };
}

export interface ProcessingParameters {
  triangulation_method: 'delaunay' | 'convex_hull' | 'alpha_shape';
  interpolation_method: 'linear' | 'cubic' | 'nearest';
  grid_resolution: number;
  smoothing_factor: number;
  outlier_threshold: number;
  confidence_level: number;
  max_iterations: number;
  convergence_tolerance: number;
}

export interface SurfaceConfiguration {
  coordinate_system: CoordinateSystem;
  processing_params: ProcessingParameters;
  quality_thresholds: Record<string, number>;
  export_settings: Record<string, any>;
}

export interface StatisticalAnalysis {
  mean_value: number;
  median_value: number;
  standard_deviation: number;
  variance: number;
  skewness: number;
  kurtosis: number;
  sample_count: number;
  confidence_interval_95: [number, number];
  percentiles: Record<string, number>;
}

export interface QualityMetrics {
  point_density: number;
  surface_coverage: number;
  data_completeness: number;
  noise_level: number;
  accuracy_estimate: number;
  precision_estimate: number;
  reliability_score: number;
  quality_flags: Record<string, boolean>;
}

export interface DetailedAnalysisReport {
  analysis_id: string;
  timestamp: string;
  processing_duration_seconds: number;
  input_surfaces_count: number;
  analysis_boundary_area_sq_meters: number;
  statistical_analysis: StatisticalAnalysis;
  quality_metrics: QualityMetrics;
  warnings: string[];
  recommendations: string[];
} 