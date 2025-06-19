/**
 * API client for communicating with the Surface Mapper backend
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';

/**
 * Make HTTP request to backend API
 */
const apiRequest = async (endpoint, options = {}) => {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
  };

  const config = {
    ...defaultOptions,
    ...options,
    headers: {
      ...defaultOptions.headers,
      ...options.headers,
    },
  };

  try {
    const response = await fetch(url, config);
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `HTTP ${response.status}: ${response.statusText}`);
    }

    return await response.json();
  } catch (error) {
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
      throw new Error('Network error: Unable to connect to backend server');
    }
    throw error;
  }
};

/**
 * Upload a surface file
 */
export const uploadSurface = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiRequest('/surfaces/upload', {
    method: 'POST',
    headers: {}, // Let browser set Content-Type for FormData
    body: formData,
  });

  return response;
};

/**
 * Process surfaces for analysis
 */
export const processSurfaces = async (processingRequest) => {
  return await apiRequest('/surfaces/process', {
    method: 'POST',
    body: JSON.stringify(processingRequest),
  });
};

/**
 * Get processing status
 */
export const getProcessingStatus = async (jobId) => {
  return await apiRequest(`/surfaces/status/${jobId}`, {
    method: 'GET',
  });
};

/**
 * Get point analysis data
 */
export const getPointAnalysis = async (x, y) => {
  return await apiRequest('/surfaces/point-analysis', {
    method: 'POST',
    body: JSON.stringify({ x, y }),
  });
};

/**
 * Get surface visualization data
 */
export const getSurfaceVisualization = async (surfaceIds) => {
  return await apiRequest('/surfaces/visualization', {
    method: 'POST',
    body: JSON.stringify({ surface_ids: surfaceIds }),
  });
};

/**
 * Export analysis results
 */
export const exportResults = async (jobId, format = 'json') => {
  return await apiRequest(`/surfaces/export/${jobId}`, {
    method: 'GET',
    headers: {
      'Accept': format === 'csv' ? 'text/csv' : 'application/json',
    },
  });
};

/**
 * Get system health status
 */
export const getHealthStatus = async () => {
  return await apiRequest('/health', {
    method: 'GET',
  });
};

/**
 * Get API documentation
 */
export const getApiDocs = async () => {
  return await apiRequest('/docs', {
    method: 'GET',
  });
};

/**
 * Validate PLY file
 */
export const validatePlyFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  return await apiRequest('/surfaces/validate', {
    method: 'POST',
    headers: {}, // Let browser set Content-Type for FormData
    body: formData,
  });
};

/**
 * Get supported coordinate systems
 */
export const getCoordinateSystems = async () => {
  return await apiRequest('/coordinate-systems', {
    method: 'GET',
  });
};

/**
 * Transform coordinates
 */
export const transformCoordinates = async (coordinates, fromSystem, toSystem) => {
  return await apiRequest('/coordinate-transform', {
    method: 'POST',
    body: JSON.stringify({
      coordinates,
      from_system: fromSystem,
      to_system: toSystem,
    }),
  });
};

/**
 * Get processing configuration
 */
export const getProcessingConfig = async () => {
  return await apiRequest('/config/processing', {
    method: 'GET',
  });
};

/**
 * Update processing configuration
 */
export const updateProcessingConfig = async (config) => {
  return await apiRequest('/config/processing', {
    method: 'PUT',
    body: JSON.stringify(config),
  });
};

/**
 * Get processing history
 */
export const getProcessingHistory = async (limit = 10, offset = 0) => {
  return await apiRequest(`/surfaces/history?limit=${limit}&offset=${offset}`, {
    method: 'GET',
  });
};

/**
 * Delete processing job
 */
export const deleteProcessingJob = async (jobId) => {
  return await apiRequest(`/surfaces/job/${jobId}`, {
    method: 'DELETE',
  });
};

/**
 * Retry failed processing job
 */
export const retryProcessingJob = async (jobId) => {
  return await apiRequest(`/surfaces/job/${jobId}/retry`, {
    method: 'POST',
  });
};

/**
 * Get processing statistics
 */
export const getProcessingStats = async () => {
  return await apiRequest('/surfaces/stats', {
    method: 'GET',
  });
};

/**
 * Batch upload surfaces
 */
export const batchUploadSurfaces = async (files) => {
  const formData = new FormData();
  files.forEach((file, index) => {
    formData.append(`files`, file);
  });

  return await apiRequest('/surfaces/batch-upload', {
    method: 'POST',
    headers: {}, // Let browser set Content-Type for FormData
    body: formData,
  });
};

/**
 * Get surface metadata
 */
export const getSurfaceMetadata = async (surfaceId) => {
  return await apiRequest(`/surfaces/${surfaceId}/metadata`, {
    method: 'GET',
  });
};

/**
 * Update surface metadata
 */
export const updateSurfaceMetadata = async (surfaceId, metadata) => {
  return await apiRequest(`/surfaces/${surfaceId}/metadata`, {
    method: 'PUT',
    body: JSON.stringify(metadata),
  });
};

/**
 * Delete surface
 */
export const deleteSurface = async (surfaceId) => {
  return await apiRequest(`/surfaces/${surfaceId}`, {
    method: 'DELETE',
  });
};

export default {
  uploadSurface,
  processSurfaces,
  getProcessingStatus,
  getPointAnalysis,
  getSurfaceVisualization,
  exportResults,
  getHealthStatus,
  getApiDocs,
  validatePlyFile,
  getCoordinateSystems,
  transformCoordinates,
  getProcessingConfig,
  updateProcessingConfig,
  getProcessingHistory,
  deleteProcessingJob,
  retryProcessingJob,
  getProcessingStats,
  batchUploadSurfaces,
  getSurfaceMetadata,
  updateSurfaceMetadata,
  deleteSurface,
}; 