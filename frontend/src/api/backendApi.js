/**
 * API client for communicating with the Surface Mapper backend
 * Uses axios for better error handling and request/response interceptors
 */

import axios from 'axios';

const api = axios.create({
  baseURL: '/',
  headers: {
    'Content-Type': 'application/json',
  },
});

const backendApi = {
  uploadSurface: async (file) => {
    const formData = new FormData();
    formData.append('file', file, file.name);

    try {
      const response = await api.post('/api/surfaces/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error uploading file:', error);
      if (error.response) {
        const errorData = error.response.data;
        const errorMessage = errorData.detail || JSON.stringify(errorData);
        throw new Error(errorMessage);
      } else if (error.request) {
        throw new Error('No response from server during file upload.');
      } else {
        throw new Error(error.message);
      }
    }
  },

  startAnalysis: async (formData) => {
    try {
      const response = await api.post('/api/analysis/start', formData);
      return response.data;
    } catch (error) {
      console.error('Error starting analysis:', error);
      if (error.response) {
        const errorData = error.response.data;
        const errorMessage = errorData.detail || JSON.stringify(errorData);
        throw new Error(errorMessage);
      } else if (error.request) {
        throw new Error('No response from server. The backend might be down or unresponsive.');
      } else {
        throw new Error(error.message || 'An unknown error occurred while setting up the request.');
      }
    }
  },
};

// Request interceptor for logging and authentication
api.interceptors.request.use(
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
api.interceptors.response.use(
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
 * Process surfaces for analysis
 */
export const processSurfaces = async (processingRequest) => {
  const response = await api.post('/api/analysis/start', processingRequest);
  return response.data;
};

/**
 * Get processing status
 */
export const getProcessingStatus = async (jobId) => {
  const response = await api.get(`/api/analysis/${jobId}/status`);
  return response.data;
};

/**
 * Get analysis results
 */
export const getAnalysisResults = async (jobId) => {
  const response = await api.get(`/api/analysis/${jobId}/results`);
  return response.data;
};

/**
 * Get point analysis data for interactive 3D visualization
 */
export const getPointAnalysis = async (x, y, coordinateSystem = 'utm') => {
  const response = await api.post(`/api/analysis/point_query`, {
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
  const response = await api.post('/api/analysis/3d-visualization', {
    surface_ids: surfaceIds,
    level_of_detail: levelOfDetail,
  });
  return response.data;
};

/**
 * Get 3D mesh data for specific surface
 */
export const getSurfaceMesh = async (analysisId, surfaceIndex, levelOfDetail = 'medium') => {
  const response = await api.get(`/api/analysis/${analysisId}/surface/${surfaceIndex}/mesh`, {
    params: { level_of_detail: levelOfDetail },
  });
  return response.data;
};

/**
 * Export analysis results
 */
export const exportResults = async (analysisId, format = 'json') => {
  const response = await api.get(`/api/analysis/${analysisId}/export`, {
    params: { format },
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
  const response = await api.get('/health');
  return response.data;
};

/**
 * Get API documentation
 */
export const getApiDocs = async () => {
  const response = await api.get('/docs');
  return response.data;
};

/**
 * Validate PLY file
 */
export const validatePlyFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post('/api/surfaces/validate', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

/**
 * Get supported coordinate systems
 */
export const getSupportedCoordinateSystems = async () => {
  const response = await api.get('/api/surfaces/coordinate-systems');
  return response.data;
};

/**
 * Transform coordinates
 */
export const transformCoordinates = async (coordinates, fromSystem, toSystem) => {
  const response = await api.post('/surfaces/coordinate-transform', {
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
  const response = await api.get('/config/processing');
  return response.data;
};

/**
 * Update processing configuration
 */
export const updateProcessingConfig = async (config) => {
  const response = await api.put('/config/processing', config);
  return response.data;
};

/**
 * Get processing history
 */
export const getProcessingHistory = async (limit = 10, offset = 0) => {
  const response = await api.get('/surfaces/history', {
    params: { limit, offset },
  });
  return response.data;
};

/**
 * Delete processing job
 */
export const deleteProcessingJob = async (jobId) => {
  const response = await api.delete(`/surfaces/job/${jobId}`);
  return response.data;
};

/**
 * Retry processing job
 */
export const retryProcessingJob = async (jobId) => {
  const response = await api.post(`/surfaces/job/${jobId}/retry`);
  return response.data;
};

/**
 * Get processing statistics
 */
export const getProcessingStats = async () => {
  const response = await api.get('/surfaces/stats');
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

  const response = await api.post('/surfaces/batch-upload', formData, {
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
  const response = await api.get(`/surfaces/${surfaceId}/metadata`);
  return response.data;
};

/**
 * Update surface metadata
 */
export const updateSurfaceMetadata = async (surfaceId, metadata) => {
  const response = await api.put(`/surfaces/${surfaceId}/metadata`, metadata);
  return response.data;
};

/**
 * Delete surface
 */
export const deleteSurface = async (surfaceId) => {
  const response = await api.delete(`/surfaces/${surfaceId}`);
  return response.data;
};

/**
 * Get analysis boundary validation
 */
export const validateAnalysisBoundary = async (boundary) => {
  const response = await api.post('/surfaces/validate-boundary', boundary);
  return response.data;
};

/**
 * Get surface overlap analysis
 */
export const getSurfaceOverlap = async (surfaceIds) => {
  const response = await api.post('/surfaces/overlap-analysis', {
    surface_ids: surfaceIds,
  });
  return response.data;
};

/**
 * Get volume calculation preview
 */
export const getVolumePreview = async (surfaceIds, boundary) => {
  const response = await api.post('/surfaces/volume-preview', {
    surface_ids: surfaceIds,
    boundary,
  });
  return response.data;
};

export default backendApi; 