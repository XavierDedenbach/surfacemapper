import React, { useState } from 'react';
import './App.css';
import InputForms from './components/InputForms';
import ThreeDViewer from './components/ThreeDViewer';
import DataTable from './components/DataTable';
import useSurfaceData from './hooks/useSurfaceData';

function App() {
  const [currentView, setCurrentView] = useState('wizard'); // 'wizard', 'analysis', 'results'
  const [analysisData, setAnalysisData] = useState(null);
  const [selectedSurfaces, setSelectedSurfaces] = useState([]);
  
  const {
    surfaces,
    analysisResults,
    processingStatus,
    error,
    uploadSurfaces,
    processSurfaceData,
    clearSurfaces
  } = useSurfaceData();

  // Extract results from analysisResults
  const { volumeResults = [], thicknessResults = [], compactionResults = [] } = analysisResults;

  const handleWizardComplete = (data) => {
    setAnalysisData(data);
    setCurrentView('analysis');
  };

  const handleAnalysisComplete = (results) => {
    setCurrentView('results');
  };

  const handleSurfaceSelection = (surfaceId) => {
    setSelectedSurfaces(prev => 
      prev.includes(surfaceId) 
        ? prev.filter(id => id !== surfaceId)
        : [...prev, surfaceId]
    );
  };

  const handlePointHover = (pointData) => {
    // Handle point analysis hover events
    console.log('Point hover:', pointData);
  };

  const renderWizardView = () => (
    <div className="wizard-container">
      <header className="app-header">
        <h1>Surface Volume and Layer Thickness Analysis Tool</h1>
        <p>Upload PLY files and configure analysis parameters</p>
      </header>
      
      <InputForms
        onSurfaceUpload={uploadSurfaces}
        onGeoreferenceSubmit={(params) => console.log('Georeference params:', params)}
        onBoundarySubmit={(boundary) => console.log('Analysis boundary:', boundary)}
        onTonnageSubmit={(tonnage) => console.log('Tonnage data:', tonnage)}
        onComplete={handleWizardComplete}
      />
    </div>
  );

  const renderAnalysisView = () => (
    <div className="analysis-container">
      <header className="app-header">
        <h1>3D Surface Analysis</h1>
        <div className="view-controls">
          <button 
            onClick={() => setCurrentView('wizard')}
            className="btn-secondary"
          >
            Back to Setup
          </button>
          <button 
            onClick={() => setCurrentView('results')}
            className="btn-primary"
          >
            View Results
          </button>
        </div>
      </header>

      <div className="analysis-content">
        <div className="viewer-section">
          <ThreeDViewer
            surfaces={surfaces}
            analysisBoundary={analysisData?.boundary}
            onPointHover={handlePointHover}
            selectedSurfaces={selectedSurfaces}
          />
        </div>
        
        <div className="data-section">
          <DataTable
            volumeResults={volumeResults}
            thicknessResults={thicknessResults}
            compactionResults={compactionResults}
            onSurfaceSelection={handleSurfaceSelection}
            selectedSurfaces={selectedSurfaces}
          />
        </div>
      </div>
    </div>
  );

  const renderResultsView = () => (
    <div className="results-container">
      <header className="app-header">
        <h1>Analysis Results</h1>
        <div className="view-controls">
          <button 
            onClick={() => setCurrentView('analysis')}
            className="btn-secondary"
          >
            Back to Analysis
          </button>
          <button 
            onClick={() => {
              clearSurfaces();
              setCurrentView('wizard');
            }}
            className="btn-secondary"
          >
            New Analysis
          </button>
        </div>
      </header>

      <div className="results-content">
        <DataTable
          volumeResults={volumeResults}
          thicknessResults={thicknessResults}
          compactionResults={compactionResults}
          onSurfaceSelection={handleSurfaceSelection}
          selectedSurfaces={selectedSurfaces}
        />
      </div>
    </div>
  );

  const renderLoading = () => (
    <div className="loading-container">
      <div className="loading-spinner"></div>
      <p>Processing analysis...</p>
    </div>
  );

  const renderError = () => (
    <div className="error-container">
      <h2>Error</h2>
      <p>{error}</p>
      <button onClick={() => setCurrentView('wizard')}>
        Return to Setup
      </button>
    </div>
  );

  // Show loading state when processing
  if (processingStatus === 'uploading' || processingStatus === 'processing') {
    return renderLoading();
  }

  if (error) {
    return renderError();
  }

  return (
    <div className="App">
      {currentView === 'wizard' && renderWizardView()}
      {currentView === 'analysis' && renderAnalysisView()}
      {currentView === 'results' && renderResultsView()}
    </div>
  );
}

export default App;
