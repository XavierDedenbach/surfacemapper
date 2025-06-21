import React, { useState } from 'react';
import './App.css';
import InputForms from './components/InputForms';
import ThreeDViewer from './components/ThreeDViewer';
import backendApi from './api/backendApi'; // Adjusted path if necessary

function App() {
  const [wizardStep, setWizardStep] = useState(0); // 0: setup, 1: loading, 2: results
  const [error, setError] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);

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

    try {
      const result = await backendApi.startAnalysis(formData);
      setAnalysisResult(result);
      setWizardStep(2);
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

  const renderContent = () => {
    switch (wizardStep) {
      case 1:
        return (
          <div className="flex flex-col items-center justify-center h-screen">
            <h2 className="text-2xl font-bold mb-4">Processing...</h2>
            <p>Your analysis is being run.</p>
          </div>
        );
      case 2:
        if (error) {
          return (
            <div className="flex flex-col items-center justify-center h-screen">
              <h2 className="text-2xl font-bold text-red-500 mb-4">Error</h2>
              <p className="text-lg text-gray-700 mb-8">Processing failed: {error.message || 'An unknown error occurred.'}</p>
              <button onClick={handleReturnToSetup} className="bg-indigo-600 text-white py-2 px-4 rounded-md">
                Return to Setup
              </button>
            </div>
          );
        }
        return (
          <div>
            <ThreeDViewer analysisResult={analysisResult} onBack={handleReturnToSetup} />
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
