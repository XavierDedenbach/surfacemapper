/**
 * API client for communicating with the Surface Mapper backend
 * Uses axios for better error handling and request/response interceptors
 */

import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8081',
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
      let errorMessage;
      if (error.response) {
        const errorData = error.response.data;
        if (typeof errorData === 'string') {
          errorMessage = errorData;
        } else if (errorData && typeof errorData.detail === 'string') {
          errorMessage = errorData.detail;
        } else {
          errorMessage = JSON.stringify(errorData);
        }
      } else if (error.request) {
        errorMessage = 'No response from server. The backend might be down or unresponsive.';
      } else {
        errorMessage = error.message || 'An unknown error occurred while setting up the request.';
      }
      throw new Error(errorMessage);
    }
  },

  getAnalysisResults: async (analysisId) => {
    try {
      const response = await api.get(`/api/analysis/${analysisId}/results`);
      return response.data;
    } catch (error) {
      if (error.response && error.response.status === 202) {
        return error.response.data;
      }
      console.error('Error fetching analysis results:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  /**
   * Get processing status
   */
  getProcessingStatus: async (jobId) => {
    const response = await api.get(`/api/analysis/${jobId}/status`);
    return response.data;
  },

  /**
   * Get point analysis data for interactive 3D visualization
   */
  getPointAnalysis: async (x, y, coordinateSystem = 'utm') => {
    const response = await api.post(`/api/analysis/point_query`, {
      x,
      y,
      coordinate_system: coordinateSystem,
    });
    return response.data;
  },

  /**
   * Get surface visualization data (mesh data for Three.js)
   */
  getSurfaceVisualization: async (surfaceIds, levelOfDetail = 'medium') => {
    const response = await api.post('/api/analysis/3d-visualization', {
      surface_ids: surfaceIds,
      level_of_detail: levelOfDetail,
    });
    return response.data;
  },

  /**
   * Get 3D mesh data for specific surface
   */
  getSurfaceMesh: async (analysisId, surfaceIndex, levelOfDetail = 'medium') => {
    const response = await api.get(`/api/analysis/${analysisId}/surface/${surfaceIndex}/mesh`, {
      params: { level_of_detail: levelOfDetail },
    });
    return response.data;
  },

  /**
   * Export analysis results
   */
  exportResults: async (analysisId, format = 'json') => {
    const response = await api.get(`/api/analysis/${analysisId}/export`, {
      params: { format },
      headers: {
        'Accept': format === 'csv' ? 'text/csv' : 'application/json',
      },
      responseType: format === 'csv' ? 'text' : 'json',
    });
    return response.data;
  },

  /**
   * Get system health status
   */
  getHealthStatus: async () => {
    const response = await api.get('/health');
    return response.data;
  },

  /**
   * Get API documentation
   */
  getApiDocs: async () => {
    const response = await api.get('/docs');
    return response.data;
  },

  /**
   * Validate PLY file
   */
  validatePlyFile: async (file) => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post('/api/surfaces/validate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Get supported coordinate systems
   */
  getSupportedCoordinateSystems: async () => {
    const response = await api.get('/api/surfaces/coordinate-systems');
    return response.data;
  },

  /**
   * Transform coordinates
   */
  transformCoordinates: async (coordinates, fromSystem, toSystem) => {
    const response = await api.post('/surfaces/coordinate-transform', {
      coordinates,
      from_system: fromSystem,
      to_system: toSystem,
    });
    return response.data;
  },

  /**
   * Get processing configuration
   */
  getProcessingConfig: async () => {
    const response = await api.get('/config/processing');
    return response.data;
  },

  /**
   * Update processing configuration
   */
  updateProcessingConfig: async (config) => {
    const response = await api.put('/config/processing', config);
    return response.data;
  },

  /**
   * Get processing history
   */
  getProcessingHistory: async (limit = 10, offset = 0) => {
    const response = await api.get('/surfaces/history', {
      params: { limit, offset },
    });
    return response.data;
  },

  /**
   * Delete processing job
   */
  deleteProcessingJob: async (jobId) => {
    const response = await api.delete(`/surfaces/job/${jobId}`);
    return response.data;
  },

  /**
   * Retry processing job
   */
  retryProcessingJob: async (jobId) => {
    const response = await api.post(`/surfaces/job/${jobId}/retry`);
    return response.data;
  },

  /**
   * Get processing stats
   */
  getProcessingStats: async () => {
    const response = await api.get('/surfaces/stats');
    return response.data;
  },

  /**
   * Batch upload surfaces
   */
  batchUploadSurfaces: async (files) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file, file.name);
    });

    const response = await api.post('/surfaces/batch-upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  /**
   * Get surface metadata
   */
  getSurfaceMetadata: async (surfaceId) => {
    const response = await api.get(`/surfaces/${surfaceId}/metadata`);
    return response.data;
  },

  /**
   * Update surface metadata
   */
  updateSurfaceMetadata: async (surfaceId, metadata) => {
    const response = await api.put(`/surfaces/${surfaceId}/metadata`, metadata);
    return response.data;
  },

  /**
   * Delete a surface
   */
  deleteSurface: async (surfaceId) => {
    const response = await api.delete(`/surfaces/${surfaceId}`);
    return response.data;
  },

  /**
   * Validate analysis boundary
   */
  validateAnalysisBoundary: async (boundary) => {
    const response = await api.post('/surfaces/validate-boundary', boundary);
    return response.data;
  },

  /**
   * Analyze surface overlap
   */
  getSurfaceOverlap: async (surfaceIds) => {
    const response = await api.post('/surfaces/overlap-analysis', {
      surface_ids: surfaceIds,
    });
    return response.data;
  },

  /**
   * Get volume calculation preview
   */
  getVolumePreview: async (surfaceIds, boundary) => {
    const response = await api.post('/surfaces/volume-preview', {
      surface_ids: surfaceIds,
      boundary: boundary,
    });
    return response.data;
  },

  getAnalysisStatus: async (analysisId) => {
    const response = await api.get(`/api/analysis/${analysisId}/status`);
    return response.data;
  },

  /**
   * Query thickness at a point for a specific analysis
   */
  queryPointAnalysis: async (analysisId, x, y, coordinateSystem = 'utm') => {
    const response = await api.post(`/api/analysis/${analysisId}/point_query`, {
      x,
      y,
      coordinate_system: coordinateSystem,
    });
    return response.data;
  },

  /**
   * Point query method for interactive thickness analysis
   */
  pointQuery: async (analysisId, queryParams) => {
    const response = await api.post(`/api/analysis/${analysisId}/point_query`, queryParams);
    return response.data;
  },

  /**
   * Download thickness grid CSV for a completed analysis
   */
  downloadThicknessGridCSV: async (analysisId, spacing = 1.0) => {
    try {
      const response = await api.get(`/api/analysis/${analysisId}/thickness_grid_csv`, {
        params: { spacing },
        responseType: 'blob',
      });
      
      // Create a download link
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `thickness_grid_${analysisId}.csv`);
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(url);
      
      return { success: true };
    } catch (error) {
      console.error('Error downloading thickness grid CSV:', error.response ? error.response.data : error.message);
      throw error;
    }
  },

  uploadShpFiles: async (files) => {
    const formData = new FormData();
    files.forEach((file) => {
      formData.append('files', file, file.name);
    });
    try {
      const response = await api.post('/api/surfaces/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Error uploading SHP files:', error);
      if (error.response) {
        const errorData = error.response.data;
        const errorMessage = errorData.detail || JSON.stringify(errorData);
        throw new Error(errorMessage);
      } else if (error.request) {
        throw new Error('No response from server during SHP file upload.');
      } else {
        throw new Error(error.message);
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
    let errorMessage = 'An unknown error occurred';

    if (error.response) {
      const errorData = error.response.data;
      if (typeof errorData === 'string') {
        errorMessage = errorData;
      } else if (errorData && typeof errorData.detail === 'string') {
        errorMessage = errorData.detail;
      } else if (errorData && typeof errorData.message === 'string') {
        errorMessage = errorData.message;
      } else {
        errorMessage = `HTTP ${error.response.status}: ${error.response.statusText}`;
      }
    } else if (error.request) {
      errorMessage = 'Network error: Unable to connect to backend server';
    } else {
      errorMessage = error.message || 'Unknown error occurred during request setup';
    }
    
    // Use Promise.reject to pass the error along the promise chain
    return Promise.reject(new Error(errorMessage));
  }
);

export default backendApi;