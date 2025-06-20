// TypeScript compilation tests for API interfaces
// These tests validate that the TypeScript interfaces compile correctly and enforce proper types

import {
  SurfaceUploadResponse,
  GeoreferenceParams,
  AnalysisBoundary,
  TonnageInput,
  ProcessingRequest,
  ProcessingResponse,
  VolumeResult,
  ThicknessResult,
  CompactionResult,
  AnalysisResults,
  CoordinateSystem,
  ProcessingParameters,
  SurfaceConfiguration,
  StatisticalAnalysis,
  QualityMetrics,
  DetailedAnalysisReport
} from './api';

// Test 1: SurfaceUploadResponse interface validation
function testSurfaceUploadResponse(): SurfaceUploadResponse {
  return {
    message: "File uploaded successfully",
    filename: "surface1.ply",
    status: "completed" // Valid status
  };
}

// Test 2: GeoreferenceParams interface validation
function testGeoreferenceParams(): GeoreferenceParams {
  return {
    wgs84_lat: 40.7128,
    wgs84_lon: -74.0060,
    orientation_degrees: 45.0,
    scaling_factor: 1.5
  };
}

// Test 3: AnalysisBoundary interface validation
function testAnalysisBoundary(): AnalysisBoundary {
  return {
    wgs84_coordinates: [
      [40.7128, -74.0060],
      [40.7128, -73.9960],
      [40.7028, -73.9960],
      [40.7028, -74.0060]
    ]
  };
}

// Test 4: TonnageInput interface validation
function testTonnageInput(): TonnageInput {
  return {
    layer_index: 0,
    tonnage: 150.5
  };
}

// Test 5: ProcessingRequest interface validation
function testProcessingRequest(): ProcessingRequest {
  return {
    surface_files: ["surface1.ply", "surface2.ply"],
    georeference_params: [
      {
        wgs84_lat: 40.7128,
        wgs84_lon: -74.0060,
        orientation_degrees: 45.0,
        scaling_factor: 1.5
      },
      {
        wgs84_lat: 40.7128,
        wgs84_lon: -74.0060,
        orientation_degrees: 45.0,
        scaling_factor: 1.5
      }
    ],
    analysis_boundary: {
      wgs84_coordinates: [
        [40.7128, -74.0060],
        [40.7128, -73.9960],
        [40.7028, -73.9960],
        [40.7028, -74.0060]
      ]
    },
    tonnage_inputs: [
      {
        layer_index: 0,
        tonnage: 150.5
      }
    ],
    generate_base_surface: true,
    base_surface_offset: 10.0
  };
}

// Test 6: ProcessingResponse interface validation
function testProcessingResponse(): ProcessingResponse {
  return {
    message: "Processing started",
    status: "processing",
    job_id: "job-12345"
  };
}

// Test 7: VolumeResult interface validation
function testVolumeResult(): VolumeResult {
  return {
    layer_designation: "Surface 0 to Surface 1",
    volume_cubic_yards: 150.5,
    confidence_interval: [145.0, 155.0],
    uncertainty: 2.5
  };
}

// Test 8: ThicknessResult interface validation
function testThicknessResult(): ThicknessResult {
  return {
    layer_designation: "Surface 0 to Surface 1",
    average_thickness_feet: 2.5,
    min_thickness_feet: 1.0,
    max_thickness_feet: 4.0,
    confidence_interval: [2.3, 2.7]
  };
}

// Test 9: CompactionResult interface validation
function testCompactionResult(): CompactionResult {
  return {
    layer_designation: "Surface 0 to Surface 1",
    compaction_rate_lbs_per_cubic_yard: 1800.0,
    tonnage_used: 150.5
  };
}

// Test 10: AnalysisResults interface validation
function testAnalysisResults(): AnalysisResults {
  return {
    volume_results: [
      {
        layer_designation: "Surface 0 to Surface 1",
        volume_cubic_yards: 150.5,
        confidence_interval: [145.0, 155.0],
        uncertainty: 2.5
      }
    ],
    thickness_results: [
      {
        layer_designation: "Surface 0 to Surface 1",
        average_thickness_feet: 2.5,
        min_thickness_feet: 1.0,
        max_thickness_feet: 4.0,
        confidence_interval: [2.3, 2.7]
      }
    ],
    compaction_results: [
      {
        layer_designation: "Surface 0 to Surface 1",
        compaction_rate_lbs_per_cubic_yard: 1800.0,
        tonnage_used: 150.5
      }
    ],
    processing_metadata: {
      processing_time_seconds: 120.5,
      algorithm_version: "1.0.0",
      validation_passed: true
    }
  };
}

// Test 11: CoordinateSystem interface validation
function testCoordinateSystem(): CoordinateSystem {
  return {
    name: "UTM Zone 10N",
    epsg_code: 26910,
    description: "UTM Zone 10N (NAD83)",
    units: "meters",
    bounds: {
      min_lat: 32.0,
      max_lat: 84.0,
      min_lon: -126.0,
      max_lon: -120.0
    }
  };
}

// Test 12: ProcessingParameters interface validation
function testProcessingParameters(): ProcessingParameters {
  return {
    triangulation_method: "delaunay",
    interpolation_method: "linear",
    grid_resolution: 1.0,
    smoothing_factor: 0.1,
    outlier_threshold: 3.0,
    confidence_level: 0.95,
    max_iterations: 1000,
    convergence_tolerance: 1e-6
  };
}

// Test 13: SurfaceConfiguration interface validation
function testSurfaceConfiguration(): SurfaceConfiguration {
  return {
    coordinate_system: {
      name: "UTM Zone 10N",
      epsg_code: 26910,
      description: "UTM Zone 10N (NAD83)",
      units: "meters",
      bounds: {
        min_lat: 32.0,
        max_lat: 84.0,
        min_lon: -126.0,
        max_lon: -120.0
      }
    },
    processing_params: {
      triangulation_method: "delaunay",
      interpolation_method: "linear",
      grid_resolution: 1.0,
      smoothing_factor: 0.1,
      outlier_threshold: 3.0,
      confidence_level: 0.95,
      max_iterations: 1000,
      convergence_tolerance: 1e-6
    },
    quality_thresholds: {
      min_point_density: 10.0,
      max_gap_size: 5.0,
      surface_roughness_threshold: 0.5
    },
    export_settings: {
      format: "ply",
      include_normals: true,
      compression: false
    }
  };
}

// Test 14: StatisticalAnalysis interface validation
function testStatisticalAnalysis(): StatisticalAnalysis {
  return {
    mean_value: 25.5,
    median_value: 24.8,
    standard_deviation: 3.2,
    variance: 10.24,
    skewness: 0.15,
    kurtosis: 2.8,
    sample_count: 1500,
    confidence_interval_95: [22.3, 28.7],
    percentiles: {
      p10: 20.1,
      p25: 22.5,
      p75: 28.2,
      p90: 30.8
    }
  };
}

// Test 15: QualityMetrics interface validation
function testQualityMetrics(): QualityMetrics {
  return {
    point_density: 125.5,
    surface_coverage: 0.95,
    data_completeness: 0.98,
    noise_level: 0.02,
    accuracy_estimate: 0.15,
    precision_estimate: 0.08,
    reliability_score: 0.92,
    quality_flags: {
      high_density: true,
      good_coverage: true,
      low_noise: true,
      acceptable_accuracy: true
    }
  };
}

// Test 16: DetailedAnalysisReport interface validation
function testDetailedAnalysisReport(): DetailedAnalysisReport {
  return {
    analysis_id: "analysis_20241219_001",
    timestamp: "2024-12-19T10:30:00Z",
    processing_duration_seconds: 245.7,
    input_surfaces_count: 2,
    analysis_boundary_area_sq_meters: 15000.0,
    statistical_analysis: {
      mean_value: 25.5,
      median_value: 24.8,
      standard_deviation: 3.2,
      variance: 10.24,
      skewness: 0.15,
      kurtosis: 2.8,
      sample_count: 1500,
      confidence_interval_95: [22.3, 28.7],
      percentiles: {
        p10: 20.1,
        p25: 22.5,
        p75: 28.2,
        p90: 30.8
      }
    },
    quality_metrics: {
      point_density: 125.5,
      surface_coverage: 0.95,
      data_completeness: 0.98,
      noise_level: 0.02,
      accuracy_estimate: 0.15,
      precision_estimate: 0.08,
      reliability_score: 0.92,
      quality_flags: {
        high_density: true,
        good_coverage: true,
        low_noise: true,
        acceptable_accuracy: true
      }
    },
    warnings: [
      "Gap detected in northwest corner",
      "Low point density in southern region"
    ],
    recommendations: [
      "Consider additional survey points in low-density areas",
      "Verify coordinate system accuracy"
    ]
  };
}

// Export all test functions to ensure they are used
export {
  testSurfaceUploadResponse,
  testGeoreferenceParams,
  testAnalysisBoundary,
  testTonnageInput,
  testProcessingRequest,
  testProcessingResponse,
  testVolumeResult,
  testThicknessResult,
  testCompactionResult,
  testAnalysisResults,
  testCoordinateSystem,
  testProcessingParameters,
  testSurfaceConfiguration,
  testStatisticalAnalysis,
  testQualityMetrics,
  testDetailedAnalysisReport
}; 