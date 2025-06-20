/**
 * API client for communicating with the Surface Mapper backend
 * Uses axios for better error handling and request/response interceptors
 */

import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8081';

// Create axios instance with default configuration
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 second timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor for logging and authentication
apiClient.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error);
    if (error.response) {
      // Server responded with error status
      const errorMessage = error.response.data?.detail || 
                          error.response.data?.message || 
                          `HTTP ${error.response.status}: ${error.response.statusText}`;
      throw new Error(errorMessage);
    } else if (error.request) {
      // Network error
      throw new Error('Network error: Unable to connect to backend server');
    } else {
      // Other error
      throw new Error(error.message || 'Unknown error occurred');
    }
  }
);

/**
 * Upload a surface file
 */
export const uploadSurface = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/surfaces/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Process surfaces for analysis
 */
export const processSurfaces = async (processingRequest) => {
  const response = await apiClient.post('/surfaces/process', processingRequest);
  return response.data;
};

/**
 * Get processing status
 */
export const getProcessingStatus = async (jobId) => {
  const response = await apiClient.get(`/surfaces/status/${jobId}`);
  return response.data;
};

/**
 * Get analysis results
 */
export const getAnalysisResults = async (jobId) => {
  const response = await apiClient.get(`/surfaces/results/${jobId}`);
  return response.data;
};

/**
 * Get point analysis data for interactive 3D visualization
 */
export const getPointAnalysis = async (x, y, coordinateSystem = 'utm') => {
  const response = await apiClient.post('/surfaces/point-analysis', {
    x,
    y,
    coordinate_system: coordinateSystem,
  });
  return response.data;
};

/**
 * Get surface visualization data (mesh data for Three.js)
 */
export const getSurfaceVisualization = async (surfaceIds, levelOfDetail = 'medium') => {
  const response = await apiClient.post('/surfaces/visualization', {
    surface_ids: surfaceIds,
    level_of_detail: levelOfDetail,
  });
  return response.data;
};

/**
 * Get 3D mesh data for specific surface
 */
export const getSurfaceMesh = async (surfaceId, levelOfDetail = 'medium') => {
  const response = await apiClient.get(`/surfaces/${surfaceId}/mesh`, {
    params: { level_of_detail: levelOfDetail },
  });
  return response.data;
};

/**
 * Export analysis results
 */
export const exportResults = async (jobId, format = 'json') => {
  const response = await apiClient.get(`/surfaces/export/${jobId}`, {
    headers: {
      'Accept': format === 'csv' ? 'text/csv' : 'application/json',
    },
    responseType: format === 'csv' ? 'text' : 'json',
  });
  return response.data;
};

/**
 * Get system health status
 */
export const getHealthStatus = async () => {
  const response = await apiClient.get('/health');
  return response.data;
};

/**
 * Get API documentation
 */
export const getApiDocs = async () => {
  const response = await apiClient.get('/docs');
  return response.data;
};

/**
 * Validate PLY file
 */
export const validatePlyFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/surfaces/validate', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Get supported coordinate systems
 */
export const getCoordinateSystems = async () => {
  const response = await apiClient.get('/coordinate-systems');
  return response.data;
};

/**
 * Transform coordinates
 */
export const transformCoordinates = async (coordinates, fromSystem, toSystem) => {
  const response = await apiClient.post('/coordinate-transform', {
    coordinates,
    from_system: fromSystem,
    to_system: toSystem,
  });
  return response.data;
};

/**
 * Get processing configuration
 */
export const getProcessingConfig = async () => {
  const response = await apiClient.get('/config/processing');
  return response.data;
};

/**
 * Update processing configuration
 */
export const updateProcessingConfig = async (config) => {
  const response = await apiClient.put('/config/processing', config);
  return response.data;
};

/**
 * Get processing history
 */
export const getProcessingHistory = async (limit = 10, offset = 0) => {
  const response = await apiClient.get('/surfaces/history', {
    params: { limit, offset },
  });
  return response.data;
};

/**
 * Delete processing job
 */
export const deleteProcessingJob = async (jobId) => {
  const response = await apiClient.delete(`/surfaces/job/${jobId}`);
  return response.data;
};

/**
 * Retry processing job
 */
export const retryProcessingJob = async (jobId) => {
  const response = await apiClient.post(`/surfaces/job/${jobId}/retry`);
  return response.data;
};

/**
 * Get processing statistics
 */
export const getProcessingStats = async () => {
  const response = await apiClient.get('/surfaces/stats');
  return response.data;
};

/**
 * Batch upload surfaces
 */
export const batchUploadSurfaces = async (files) => {
  const formData = new FormData();
  files.forEach((file, index) => {
    formData.append(`files`, file);
  });

  const response = await apiClient.post('/surfaces/batch-upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Get surface metadata
 */
export const getSurfaceMetadata = async (surfaceId) => {
  const response = await apiClient.get(`/surfaces/${surfaceId}/metadata`);
  return response.data;
};

/**
 * Update surface metadata
 */
export const updateSurfaceMetadata = async (surfaceId, metadata) => {
  const response = await apiClient.put(`/surfaces/${surfaceId}/metadata`, metadata);
  return response.data;
};

/**
 * Delete surface
 */
export const deleteSurface = async (surfaceId) => {
  const response = await apiClient.delete(`/surfaces/${surfaceId}`);
  return response.data;
};

/**
 * Get analysis boundary validation
 */
export const validateAnalysisBoundary = async (boundary) => {
  const response = await apiClient.post('/surfaces/validate-boundary', boundary);
  return response.data;
};

/**
 * Get surface overlap analysis
 */
export const getSurfaceOverlap = async (surfaceIds) => {
  const response = await apiClient.post('/surfaces/overlap-analysis', {
    surface_ids: surfaceIds,
  });
  return response.data;
};

/**
 * Get volume calculation preview
 */
export const getVolumePreview = async (surfaceIds, boundary) => {
  const response = await apiClient.post('/surfaces/volume-preview', {
    surface_ids: surfaceIds,
    boundary,
  });
  return response.data;
};

export default apiClient; 