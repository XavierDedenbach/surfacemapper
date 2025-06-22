import React, { useState, useEffect } from 'react';
import './App.css';
import InputForms from './components/InputForms';
import ThreeDViewer from './components/ThreeDViewer';
import DataTable from './components/DataTable';
import backendApi from './api/backendApi'; // Adjusted path if necessary

function App() {
  const [wizardStep, setWizardStep] = useState(0); // 0: setup, 1: loading, 2: results
  const [error, setError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [tonnageParams, setTonnageParams] = useState([]);
  const [elapsedTime, setElapsedTime] = useState(0);

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
                <ThreeDViewer analysisResult={analysisResult} onBack={handleReturnToSetup} />
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
    </div>
  );
}

export default App;
