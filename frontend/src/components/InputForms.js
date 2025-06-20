import React, { useState, useEffect } from 'react';

/**
 * InputForms component for surface upload and georeferencing
 * Handles the wizard-based workflow for surface data input
 */
const InputForms = ({ 
  onSurfaceUpload, 
  onGeoreferenceSubmit, 
  onBoundarySubmit, 
  onTonnageSubmit, 
  onComplete 
}) => {
  // Validate props
  const callbacks = {
    onSurfaceUpload: onSurfaceUpload || (() => {}),
    onGeoreferenceSubmit: onGeoreferenceSubmit || (() => {}),
    onBoundarySubmit: onBoundarySubmit || (() => {}),
    onTonnageSubmit: onTonnageSubmit || (() => {}),
    onComplete: onComplete || (() => {})
  };

  const [currentStep, setCurrentStep] = useState(1);
  const [surfaceFiles, setSurfaceFiles] = useState([]);
  const [georeferenceParams, setGeoreferenceParams] = useState([]);
  const [analysisBoundary, setAnalysisBoundary] = useState({
    coordinates: [
      { lat: '', lon: '' },
      { lat: '', lon: '' },
      { lat: '', lon: '' },
      { lat: '', lon: '' }
    ]
  });
  const [tonnageInputs, setTonnageInputs] = useState([]);

  // Validate step boundaries
  const nextStep = () => {
    if (currentStep < 5) {
      setCurrentStep(Math.min(currentStep + 1, 5));
    }
  };

  const prevStep = () => {
    if (currentStep > 1) {
      setCurrentStep(Math.max(currentStep - 1, 1));
    }
  };

  // Handle file upload with validation
  const handleFileUpload = (event) => {
    const files = Array.from(event.target.files || []);
    // Validate file types
    const validFiles = files.filter(file => 
      file && file.name && file.name.toLowerCase().endsWith('.ply')
    );
    setSurfaceFiles(validFiles);
  };

  // Handle georeference changes with validation
  const handleGeoreferenceChange = (index, field, value) => {
    const updatedParams = [...georeferenceParams];
    if (!updatedParams[index]) {
      updatedParams[index] = {
        wgs84_lat: '',
        wgs84_lon: '',
        orientation_degrees: '',
        scaling_factor: '1.0'
      };
    }
    updatedParams[index][field] = value;
    setGeoreferenceParams(updatedParams);
  };

  // Handle boundary changes with validation
  const handleBoundaryChange = (index, field, value) => {
    const updatedBoundary = { ...analysisBoundary };
    if (updatedBoundary.coordinates[index]) {
      updatedBoundary.coordinates[index][field] = value;
      setAnalysisBoundary(updatedBoundary);
    }
  };

  // Handle tonnage changes with validation
  const handleTonnageChange = (index, value) => {
    const updatedTonnage = [...tonnageInputs];
    const tonnageValue = parseFloat(value) || 0;
    updatedTonnage[index] = { layer_index: index, tonnage: tonnageValue };
    setTonnageInputs(updatedTonnage);
  };

  // Validate surface count change
  const handleSurfaceCountChange = (count) => {
    const validCount = Math.max(1, Math.min(4, parseInt(count) || 1));
    setSurfaceFiles(new Array(validCount).fill(null));
    setGeoreferenceParams(new Array(validCount).fill({}));
  };

  const renderStep1 = () => (
    <div className="step-container">
      <h3>Step 1: Project Setup</h3>
      <div className="form-group">
        <label>Number of Surfaces:</label>
        <select 
          value={surfaceFiles.length} 
          onChange={(e) => handleSurfaceCountChange(e.target.value)}
        >
          <option value={1}>1 Surface</option>
          <option value={2}>2 Surfaces</option>
          <option value={3}>3 Surfaces</option>
          <option value={4}>4 Surfaces</option>
        </select>
      </div>
      <button onClick={nextStep} disabled={surfaceFiles.length === 0}>
        Next: Surface Upload
      </button>
    </div>
  );

  const renderStep2 = () => (
    <div className="step-container">
      <h3>Step 2: Surface Upload</h3>
      <div className="form-group">
        <label>Upload PLY Files:</label>
        <input 
          type="file" 
          multiple 
          accept=".ply"
          onChange={handleFileUpload}
        />
        <div className="file-list">
          {surfaceFiles.map((file, index) => (
            <div key={index} className="file-item">
              {file ? file.name : `Surface ${index + 1} - No file selected`}
            </div>
          ))}
        </div>
      </div>
      <div className="step-buttons">
        <button onClick={prevStep}>Previous</button>
        <button onClick={nextStep} disabled={surfaceFiles.length === 0}>
          Next: Georeferencing
        </button>
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className="step-container">
      <h3>Step 3: Georeferencing</h3>
      {surfaceFiles.map((file, index) => (
        <div key={index} className="surface-georeference">
          <h4>Surface {index + 1}: {file?.name}</h4>
          <div className="form-row">
            <div className="form-group">
              <label>WGS84 Latitude:</label>
              <input
                type="number"
                step="any"
                value={georeferenceParams[index]?.wgs84_lat || ''}
                onChange={(e) => handleGeoreferenceChange(index, 'wgs84_lat', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>WGS84 Longitude:</label>
              <input
                type="number"
                step="any"
                value={georeferenceParams[index]?.wgs84_lon || ''}
                onChange={(e) => handleGeoreferenceChange(index, 'wgs84_lon', e.target.value)}
              />
            </div>
          </div>
          <div className="form-row">
            <div className="form-group">
              <label>Orientation (degrees):</label>
              <input
                type="number"
                step="any"
                value={georeferenceParams[index]?.orientation_degrees || ''}
                onChange={(e) => handleGeoreferenceChange(index, 'orientation_degrees', e.target.value)}
              />
            </div>
            <div className="form-group">
              <label>Scaling Factor:</label>
              <input
                type="number"
                step="any"
                value={georeferenceParams[index]?.scaling_factor || '1.0'}
                onChange={(e) => handleGeoreferenceChange(index, 'scaling_factor', e.target.value)}
              />
            </div>
          </div>
        </div>
      ))}
      <div className="step-buttons">
        <button onClick={prevStep}>Previous</button>
        <button onClick={nextStep}>
          Next: Analysis Boundary
        </button>
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="step-container">
      <h3>Step 4: Analysis Boundary</h3>
      <p>Define rectangular analysis boundary using WGS84 coordinates:</p>
      <div className="boundary-coordinates">
        {analysisBoundary.coordinates.map((coord, index) => (
          <div key={index} className="coordinate-pair">
            <h5>Point {index + 1}:</h5>
            <div className="form-row">
              <div className="form-group">
                <label>Latitude:</label>
                <input
                  type="number"
                  step="any"
                  value={coord.lat}
                  onChange={(e) => handleBoundaryChange(index, 'lat', e.target.value)}
                />
              </div>
              <div className="form-group">
                <label>Longitude:</label>
                <input
                  type="number"
                  step="any"
                  value={coord.lon}
                  onChange={(e) => handleBoundaryChange(index, 'lon', e.target.value)}
                />
              </div>
            </div>
          </div>
        ))}
      </div>
      <div className="step-buttons">
        <button onClick={prevStep}>Previous</button>
        <button onClick={nextStep}>
          Next: Material Input
        </button>
      </div>
    </div>
  );

  const renderStep5 = () => (
    <div className="step-container">
      <h3>Step 5: Material Input</h3>
      <p>Enter tonnage for each layer (optional):</p>
      {Array.from({ length: Math.max(0, surfaceFiles.length - 1) }, (_, index) => (
        <div key={index} className="tonnage-input">
          <label>Layer {index + 1} Tonnage (tons):</label>
          <input
            type="number"
            step="any"
            value={tonnageInputs[index]?.tonnage || ''}
            onChange={(e) => handleTonnageChange(index, e.target.value)}
            placeholder="Optional"
          />
        </div>
      ))}
      <div className="step-buttons">
        <button onClick={prevStep}>Previous</button>
        <button onClick={() => {
          // Submit all data for processing
          const processingData = {
            surfaceFiles,
            georeferenceParams,
            analysisBoundary,
            tonnageInputs: tonnageInputs.filter(t => t && t.tonnage > 0)
          };
          
          // Call the onComplete callback with the processing data
          if (onComplete) {
            onComplete(processingData);
          }
        }}>
          Start Analysis
        </button>
      </div>
    </div>
  );

  const renderCurrentStep = () => {
    switch (currentStep) {
      case 1: return renderStep1();
      case 2: return renderStep2();
      case 3: return renderStep3();
      case 4: return renderStep4();
      case 5: return renderStep5();
      default: return renderStep1();
    }
  };

  return (
    <div className="input-forms">
      <div className="progress-indicator">
        {[1, 2, 3, 4, 5].map(step => (
          <div 
            key={step} 
            className={`progress-step ${currentStep >= step ? 'active' : ''}`}
          >
            Step {step}
          </div>
        ))}
      </div>
      {renderCurrentStep()}
    </div>
  );
};

export default InputForms; 