import { useState, useCallback } from 'react';
import { uploadSurface, processSurfaces, getProcessingStatus } from '../api/backendApi';

/**
 * Custom hook for managing surface data and processing state
 */
const useSurfaceData = () => {
  const [surfaces, setSurfaces] = useState([]);
  const [processingStatus, setProcessingStatus] = useState('idle');
  const [processingJobId, setProcessingJobId] = useState(null);
  const [analysisResults, setAnalysisResults] = useState({
    volumeResults: [],
    thicknessResults: [],
    compactionResults: []
  });
  const [error, setError] = useState(null);

  // Upload surface files
  const uploadSurfaces = useCallback(async (files) => {
    setProcessingStatus('uploading');
    setError(null);

    try {
      const uploadPromises = files.map(file => uploadSurface(file));
      const uploadResults = await Promise.all(uploadPromises);
      
      setSurfaces(prev => [...prev, ...uploadResults]);
      setProcessingStatus('uploaded');
      
      return uploadResults;
    } catch (err) {
      setError(`Upload failed: ${err.message}`);
      setProcessingStatus('error');
      throw err;
    }
  }, []);

  // Poll processing status
  const pollProcessingStatus = useCallback(async (jobId) => {
    const pollInterval = setInterval(async () => {
      try {
        const status = await getProcessingStatus(jobId);
        
        if (status.status === 'completed') {
          setProcessingStatus('completed');
          setAnalysisResults(status.results);
          clearInterval(pollInterval);
        } else if (status.status === 'failed') {
          setProcessingStatus('error');
          setError(`Processing failed: ${status.error || 'Unknown error'}`);
          clearInterval(pollInterval);
        }
        // Continue polling if status is 'processing' or 'pending'
      } catch (err) {
        setProcessingStatus('error');
        setError(`Status check failed: ${err.message}`);
        clearInterval(pollInterval);
      }
    }, 2000); // Poll every 2 seconds

    // Cleanup interval after 10 minutes
    setTimeout(() => {
      clearInterval(pollInterval);
      if (processingStatus === 'processing') {
        setProcessingStatus('timeout');
        setError('Processing timeout - please check job status manually');
      }
    }, 600000);
  }, [processingStatus]);

  // Process surfaces for analysis
  const processSurfaceData = useCallback(async (processingRequest) => {
    setProcessingStatus('processing');
    setError(null);

    try {
      const response = await processSurfaces(processingRequest);
      setProcessingJobId(response.job_id);
      
      // Start polling for status
      pollProcessingStatus(response.job_id);
      
      return response;
    } catch (err) {
      setError(`Processing failed: ${err.message}`);
      setProcessingStatus('error');
      throw err;
    }
  }, [pollProcessingStatus]);

  // Clear surfaces
  const clearSurfaces = useCallback(() => {
    setSurfaces([]);
    setAnalysisResults({
      volumeResults: [],
      thicknessResults: [],
      compactionResults: []
    });
    setProcessingStatus('idle');
    setProcessingJobId(null);
    setError(null);
  }, []);

  // Remove specific surface
  const removeSurface = useCallback((surfaceId) => {
    setSurfaces(prev => prev.filter(surface => surface.id !== surfaceId));
  }, []);

  // Update surface metadata
  const updateSurfaceMetadata = useCallback((surfaceId, metadata) => {
    setSurfaces(prev => prev.map(surface => 
      surface.id === surfaceId 
        ? { ...surface, metadata: { ...surface.metadata, ...metadata } }
        : surface
    ));
  }, []);

  // Get surface by ID
  const getSurfaceById = useCallback((surfaceId) => {
    return surfaces.find(surface => surface.id === surfaceId);
  }, [surfaces]);

  // Get processing progress
  const getProcessingProgress = useCallback(() => {
    switch (processingStatus) {
      case 'idle':
        return 0;
      case 'uploading':
        return 10;
      case 'uploaded':
        return 20;
      case 'processing':
        return 50;
      case 'completed':
        return 100;
      case 'error':
      case 'timeout':
        return 0;
      default:
        return 0;
    }
  }, [processingStatus]);

  // Validate processing request
  const validateProcessingRequest = useCallback((request) => {
    const errors = [];

    // Check if surfaces are uploaded
    if (!request.surface_files || request.surface_files.length < 2) {
      errors.push('At least 2 surfaces are required for analysis');
    }

    // Check georeference parameters
    if (!request.georeference_params || request.georeference_params.length !== request.surface_files.length) {
      errors.push('Georeference parameters must be provided for all surfaces');
    }

    // Validate georeference parameters
    request.georeference_params?.forEach((params, index) => {
      if (!params.wgs84_lat || !params.wgs84_lon) {
        errors.push(`Surface ${index + 1}: WGS84 coordinates are required`);
      }
      if (params.wgs84_lat < -90 || params.wgs84_lat > 90) {
        errors.push(`Surface ${index + 1}: Invalid latitude value`);
      }
      if (params.wgs84_lon < -180 || params.wgs84_lon > 180) {
        errors.push(`Surface ${index + 1}: Invalid longitude value`);
      }
    });

    // Check analysis boundary
    if (!request.analysis_boundary || !request.analysis_boundary.wgs84_coordinates) {
      errors.push('Analysis boundary coordinates are required');
    }

    if (request.analysis_boundary?.wgs84_coordinates?.length !== 4) {
      errors.push('Analysis boundary must have exactly 4 coordinate points');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }, []);

  return {
    // State
    surfaces,
    processingStatus,
    processingJobId,
    analysisResults,
    error,
    
    // Actions
    uploadSurfaces,
    processSurfaceData,
    clearSurfaces,
    removeSurface,
    updateSurfaceMetadata,
    getSurfaceById,
    
    // Utilities
    getProcessingProgress,
    validateProcessingRequest
  };
};

export default useSurfaceData; 