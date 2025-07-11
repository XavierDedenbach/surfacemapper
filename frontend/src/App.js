import React, { useState, useEffect } from 'react';
import './App.css';
import InputForms from './components/InputForms';
import ThreeDViewer from './components/ThreeDViewer';
import DataTable from './components/DataTable';
// import PointAnalysisPopup from './components/PointAnalysisPopup';
import backendApi from './api/backendApi'; // Adjusted path if necessary

function App() {
  const [wizardStep, setWizardStep] = useState(0); // 0: setup, 1: loading, 2: results
  const [error, setError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [tonnageParams, setTonnageParams] = useState([]);
  const [elapsedTime, setElapsedTime] = useState(0);
  const [csvDownloadLoading, setCsvDownloadLoading] = useState(false);

  // State for the InputForms component
  const [surfaces, setSurfaces] = useState([]);
  const [numLayers, setNumLayers] = useState(1);
  const [tonnages, setTonnages] = useState({});
  const [generateBaseline, setGenerateBaseline] = useState(false);
  const [baseSurfaceOffset, setBaseSurfaceOffset] = useState(0);

  // New state for PRD requirements
  const [georeferenceParams, setGeoreferenceParams] = useState([{}]);
  const [analysisBoundary, setAnalysisBoundary] = useState({
    coordinates: [
      { lat: '', lon: '' },
      { lat: '', lon: '' },
      { lat: '', lon: '' },
      { lat: '', lon: '' },
    ],
  });

  // Point query state
  const [hoverInfo, setHoverInfo] = useState(null); // { x, y, thickness, lat, lon }
  const [pointQueryLoading, setPointQueryLoading] = useState(false);
  const [lastQueryPoint, setLastQueryPoint] = useState(null); // { x, y } for distance threshold

  useEffect(() => {
    let timer;
    if (wizardStep === 1) {
      setElapsedTime(0);
      timer = setInterval(() => {
        setElapsedTime(prevTime => prevTime + 1);
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [wizardStep]);

  const handleWizardComplete = async () => {
    setWizardStep(1); // Show loading/processing view
    setError(null);
    setAnalysisResult(null);

    const surface_ids = surfaces.map(sf => sf && sf.surface_id).filter(id => id);
    
    let requiredSurfaceCount = numLayers;
    if (generateBaseline) {
      requiredSurfaceCount = 1;
    }

    if (surface_ids.length !== requiredSurfaceCount) {
      setError(new Error(`Expected ${requiredSurfaceCount} surface(s), but uploaded ${surface_ids.length}. Please upload all required files.`));
      setWizardStep(2);
      return;
    }

    const formData = {
      surface_ids: surface_ids,
      analysis_type: 'compaction',
      generate_base_surface: generateBaseline,
      georeference_params: georeferenceParams,
      analysis_boundary: {
        wgs84_coordinates: analysisBoundary.coordinates.map(c => [parseFloat(c.lat), parseFloat(c.lon)])
      },
      params: {
        tonnage_per_layer: Object.keys(tonnages).map(key => ({
          layer_index: parseInt(key, 10),
          tonnage: tonnages[key]
        })),
        base_surface_offset: baseSurfaceOffset,
      },
    };

    setTonnageParams(formData.params.tonnage_per_layer);

    try {
      const initialResponse = await backendApi.startAnalysis(formData);
      const { analysis_id } = initialResponse;

      if (!analysis_id) {
        throw new Error("Did not receive an analysis ID from the server.");
      }

      // Poll for results
      const pollForResults = async () => {
        try {
          const result = await backendApi.getAnalysisResults(analysis_id);
          
          if (!result) {
            throw new Error("Received an empty response from the server while checking status.");
          }

          // The backend now sends status directly in the response body for 202
          const status = result.status || (result.analysis_metadata && result.analysis_metadata.status);

          if (status === 'completed') {
            console.log("Analysis reported as 'completed'. Full result from backend:", result);

            const isVolumeAnalysis = surfaces.length > 1 || generateBaseline;
            const resultsAreReady = !isVolumeAnalysis || (result.volume_results && result.volume_results.length > 0);
            
            console.log(`Is Volume Analysis? ${isVolumeAnalysis}`);
            console.log(`Are Results Ready? ${resultsAreReady}`);
            
            // Debug volume results specifically
            console.log("Volume results:", result.volume_results);
            console.log("Thickness results:", result.thickness_results);
            console.log("Analysis summary:", result.analysis_summary);
            console.log("All result keys:", Object.keys(result));

            if (resultsAreReady) {
              console.log("Results are ready. Setting state and displaying results page.");
              setAnalysisResult(result);
              setWizardStep(2);
            } else {
              console.log("Results are NOT ready. Polling again in 3 seconds.");
              // It's marked 'completed' but key results are missing. Poll again.
              setTimeout(pollForResults, 3000);
            }
          } else if (status === 'failed') {
            const errorMsg = result.error_message || (result.analysis_metadata && result.analysis_metadata.error_message) || 'Analysis failed on the server.';
            throw new Error(errorMsg);
          } else { // Catches 'processing', 'pending', 'running', 'completed_caching'
            setTimeout(pollForResults, 3000);
          }
        } catch (err) {
          // Check if the error is from a 202 response, which indicates polling should continue
          if (err.response && err.response.status === 202) {
              setTimeout(pollForResults, 3000);
          } else {
              setError(err);
              setWizardStep(2);
          }
        }
      };

      setTimeout(pollForResults, 1000);

    } catch (err) {
      setError(err);
      setWizardStep(2);
    }
  };
  
  const handleReturnToSetup = () => {
      setWizardStep(0);
      setError(null);
      setAnalysisResult(null);
      // Do not reset form state, so user can fix input
  }

  const getErrorMessage = (e) => {
    if (typeof e === 'string') return e;
    if (e && typeof e.message === 'string') return e.message;
    if (typeof e === 'object' && e !== null) return JSON.stringify(e);
    return 'An unknown error occurred.';
  };

  // Handler for CSV download
  const handleDownloadCSV = async () => {
    console.log('CSV Download button clicked');
    console.log('Analysis result:', analysisResult);
    console.log('Analysis result keys:', Object.keys(analysisResult || {}));
    
    // Use only analysis_id from analysisResult
    const analysisId = analysisResult?.analysis_id;
    
    console.log('Found analysis ID:', analysisId);
    
    if (!analysisResult || !analysisId) {
      console.error('No analysis ID available for CSV download');
      console.log('Available fields in analysisResult:', Object.keys(analysisResult || {}));
      return;
    }
    
    setCsvDownloadLoading(true);
    try {
      console.log('Calling downloadThicknessGridCSV with ID:', analysisId);
      await backendApi.downloadThicknessGridCSV(analysisId, 1.0);
      console.log('CSV download triggered (check browser for download or block)');
    } catch (error) {
      console.error('CSV download failed or was blocked by the browser:', error);
    } finally {
      setCsvDownloadLoading(false);
    }
  };

  // Handler for point hover/thickness query
  const handlePointHover = async (x, y, z, mouseX, mouseY) => {
    // Use only analysis_id from analysisResult
    const analysisId = analysisResult?.analysis_id;
    if (!analysisId) return;
    
    // Check if mouse has moved more than 1 foot from last query point
    const distanceThreshold = 1.0; // 1 foot in UTM coordinates
    const currentPoint = { x, y };
    
    if (lastQueryPoint) {
      const distance = Math.sqrt(
        Math.pow(currentPoint.x - lastQueryPoint.x, 2) + 
        Math.pow(currentPoint.y - lastQueryPoint.y, 2)
      );
      
      if (distance < distanceThreshold) {
        // Mouse hasn't moved enough, keep existing hover info
        return;
      }
    }
    
    setPointQueryLoading(true);
    setHoverInfo(null);
    try {
      const response = await backendApi.pointQuery(analysisId, { x, y, coordinate_system: 'utm' });
      
      // Show all layer thicknesses, not just the first one
      const thicknessLayers = response?.thickness_layers || [];
      let totalThickness = 0;
      let layerCount = 0;
      
      // Calculate total thickness from all layers
      thicknessLayers.forEach(layer => {
        if (layer.thickness_feet !== null && layer.thickness_feet !== undefined) {
          totalThickness += layer.thickness_feet;
          layerCount++;
        }
      });
      
      // Use average thickness if multiple layers, or single layer thickness
      const averageThickness = layerCount > 0 ? totalThickness / layerCount : 
                              (thicknessLayers[0]?.thickness_feet || null);
      
      const newHoverInfo = {
        x: response?.query_point?.x,
        y: response?.query_point?.y,
        thickness: averageThickness,
        lat: response?.query_point?.lat,
        lon: response?.query_point?.lon,
        layerCount: layerCount,
        totalThickness: totalThickness
      };
      
      setHoverInfo(newHoverInfo);
      setLastQueryPoint(currentPoint);
    } catch (error) {
      console.error('Point query error:', error);
      setHoverInfo(null);
    } finally {
      setPointQueryLoading(false);
    }
  };

  // Handler for mouse leave
  const handleMouseLeave = () => {
    setHoverInfo(null);
    setPointQueryLoading(false);
    setLastQueryPoint(null); // Reset last query point when mouse leaves
  };

  const renderContent = () => {
    switch (wizardStep) {
      case 1:
        const formatTime = (seconds) => {
            const mins = Math.floor(seconds / 60);
            const secs = seconds % 60;
            return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
        };
        return (
          <div className="flex flex-col items-center justify-center h-screen">
            <h2 className="text-2xl font-bold mb-4">Processing...</h2>
            <p>Your analysis is being run.</p>
            <p className="text-lg font-mono my-2">{formatTime(elapsedTime)}</p>
            <p className="text-sm text-gray-500">Please note: This process can take up to 30 minutes for large surface files.</p>
          </div>
        );
      case 2:
        if (error) {
          return (
            <div className="flex flex-col items-center justify-center h-screen">
              <h2 className="text-2xl font-bold text-red-500 mb-4">Error</h2>
              <p className="text-lg text-gray-700 mb-8">Processing failed: {getErrorMessage(error)}</p>
              <button onClick={handleReturnToSetup} className="bg-indigo-600 text-white py-2 px-4 rounded-md">
                Return to Setup
              </button>
            </div>
          );
        }
        return (
          <div className="flex flex-col h-screen">
            <div className="flex-grow">
                <ThreeDViewer 
                  analysisResult={analysisResult} 
                  onBack={handleReturnToSetup}
                  onPointHover={handlePointHover}
                  onMouseLeave={handleMouseLeave}
                />
            </div>
            <div className="flex-none h-1/4 bg-white p-4 overflow-auto">
                <DataTable analysisResult={analysisResult} tonnages={tonnageParams} />
            </div>
          </div>
        );
      case 0:
      default:
        return (
          <InputForms
            onWizardComplete={handleWizardComplete}
            onBack={handleReturnToSetup}
            surfaces={surfaces}
            setSurfaces={setSurfaces}
            numLayers={numLayers}
            setNumLayers={setNumLayers}
            tonnages={tonnages}
            setTonnages={setTonnages}
            generateBaseline={generateBaseline}
            setGenerateBaseline={setGenerateBaseline}
            baseSurfaceOffset={baseSurfaceOffset}
            setBaseSurfaceOffset={setBaseSurfaceOffset}
            georeferenceParams={georeferenceParams}
            setGeoreferenceParams={setGeoreferenceParams}
            analysisBoundary={analysisBoundary}
            setAnalysisBoundary={setAnalysisBoundary}
          />
        );
    }
  };

  return (
    <div className="App bg-gray-50 min-h-screen">
      <header className="bg-white shadow-sm">
        <div className="container mx-auto px-4 py-4">
            <h1 className="text-3xl font-bold text-gray-800">Surface Mapper</h1>
        </div>
      </header>
      <main className="p-4">
        {renderContent()}
      </main>

      {/* Static Thickness Info Table - Bottom Center */}
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50 bg-white rounded-lg shadow-lg px-6 py-3 border border-gray-300" style={{ minWidth: 340 }}>
        <div className="font-semibold mb-1 text-center">Thickness at Point</div>
        <table className="w-full text-sm">
          <tbody>
            <tr>
              <td className="pr-2 text-gray-600">X (ft)</td>
              <td>{pointQueryLoading ? '...' : (hoverInfo?.x?.toFixed(2) ?? '--')}</td>
              <td className="pr-2 text-gray-600">Y (ft)</td>
              <td>{pointQueryLoading ? '...' : (hoverInfo?.y?.toFixed(2) ?? '--')}</td>
            </tr>
            <tr>
              <td className="pr-2 text-gray-600">Thickness (ft)</td>
              <td colSpan={3}>{pointQueryLoading ? '...' : (hoverInfo?.thickness?.toFixed(3) ?? '--')}</td>
            </tr>
            <tr>
              <td className="pr-2 text-gray-600">Latitude (WGS84)</td>
              <td>{pointQueryLoading ? '...' : (hoverInfo?.lat?.toFixed(6) ?? '--')}</td>
              <td className="pr-2 text-gray-600">Longitude (WGS84)</td>
              <td>{pointQueryLoading ? '...' : (hoverInfo?.lon?.toFixed(6) ?? '--')}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* CSV Download Button - Fixed position in bottom right */}
      {analysisResult && (
        <div className="fixed bottom-4 right-4 z-50">
          <button
            onClick={handleDownloadCSV}
            disabled={csvDownloadLoading}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white px-4 py-2 rounded-lg shadow-lg transition-colors duration-200 flex items-center space-x-2"
          >
            {csvDownloadLoading ? (
              <>
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Downloading...</span>
              </>
            ) : (
              <>
                <svg className="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
                <span>Download Full Thickness CSV</span>
              </>
            )}
          </button>
        </div>
      )}
    </div>
  );
}

export default App;
