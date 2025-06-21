import React from 'react';
import backendApi from '../api/backendApi';

/**
 * InputForms component for surface upload and georeferencing
 * This is a "controlled component" that receives all its state and handlers from a parent.
 */
const InputForms = ({
  onWizardComplete,
  onBack,
  surfaces,
  setSurfaces,
  numLayers,
  setNumLayers,
  tonnages,
  setTonnages,
  generateBaseline,
  setGenerateBaseline,
  baseSurfaceOffset,
  setBaseSurfaceOffset,
  georeferenceParams,
  setGeoreferenceParams,
  analysisBoundary,
  setAnalysisBoundary,
}) => {

  const handleConfigUpload = async (file) => {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
        try {
            const config = JSON.parse(event.target.result);
            
            if (config.layer_0) {
                if (config.layer_0.georeference) {
                    const newGeorefParams = [...georeferenceParams];
                    newGeorefParams[0] = { ...newGeorefParams[0], ...config.layer_0.georeference };
                    setGeoreferenceParams(newGeorefParams);
                }
                if (config.layer_0.tonnage != null) {
                    setTonnages({ ...tonnages, '0': config.layer_0.tonnage });
                }
            }

            if (config.analysis_boundary && config.analysis_boundary.coordinates) {
                setAnalysisBoundary({ coordinates: config.analysis_boundary.coordinates });
            }

        } catch (error) {
            alert('Failed to parse configuration file: ' + error.message);
        }
    };
    reader.readAsText(file);
  };

  const handleFileUpload = async (file, index) => {
    if (!file) return;
    try {
      const response = await backendApi.uploadSurface(file);
      const newSurfaces = [...surfaces];
      newSurfaces[index] = { ...response, file };
      setSurfaces(newSurfaces);
    } catch (error) {
      alert(`Upload failed: ${error.message}`);
    }
  };

  const handleGeoreferenceChange = (index, field, value) => {
    const updatedParams = [...georeferenceParams];
    if (!updatedParams[index]) {
      updatedParams[index] = {};
    }
    updatedParams[index][field] = value ? parseFloat(value) : null;
    setGeoreferenceParams(updatedParams);
  };

  const handleBoundaryChange = (index, field, value) => {
    const newCoordinates = [...analysisBoundary.coordinates];
    const parsedValue = value ? parseFloat(value) : null;
    newCoordinates[index] = { ...newCoordinates[index], [field]: parsedValue };
    setAnalysisBoundary({ coordinates: newCoordinates });
  };

  const canProceed = () => {
    const requiredSurfaces = generateBaseline ? 1 : numLayers;
    const uploadedCount = surfaces.filter(s => s).length;

    if (uploadedCount < requiredSurfaces) {
        return false;
    }
    
    if (generateBaseline) {
        if (baseSurfaceOffset <= 0) {
            return false;
        }
    }
    
    // Check georeferencing
    for (let i = 0; i < requiredSurfaces; i++) {
        const params = georeferenceParams[i];
        if (!params || 
            params.wgs84_lat == null || 
            params.wgs84_lon == null || 
            params.orientation_degrees == null || 
            params.scaling_factor == null) {
            return false;
        }
    }

    // Check boundary
    for (let i = 0; i < 4; i++) {
        const coord = analysisBoundary.coordinates[i];
        if (!coord || coord.lat == null || coord.lon == null) {
            return false;
        }
    }

    return true;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canProceed()) {
      alert("Please fill all required fields, including georeferencing and boundary coordinates for all surfaces.");
      return;
    }
    onWizardComplete();
  };

  return (
    <div className="container mx-auto p-8 bg-white shadow-lg rounded-lg">
      <h1 className="text-2xl font-bold mb-6">Analysis Setup</h1>
      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6">
          {/* Left Column: Layers and Surfaces */}
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold text-gray-800 border-b pb-2">Project Setup</h2>
              <div className="mt-4">
                  <label htmlFor="config-file" className="block text-sm font-medium text-gray-700">
                    Upload Configuration (.json)
                  </label>
                  <input type="file" id="config-file" accept=".json" onChange={(e) => handleConfigUpload(e.target.files[0])} className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"/>
              </div>
            </div>
            <div className="mb-4">
              <label htmlFor="numLayers" className="block text-sm font-medium text-gray-700">
                Number of Surfaces (Layers)
              </label>
              <select
                id="numLayers"
                value={numLayers}
                onChange={(e) => {
                  const newNumLayers = parseInt(e.target.value, 10);
                  setNumLayers(newNumLayers);
                  const newSurfaces = new Array(newNumLayers).fill(null);
                  surfaces.forEach((surface, index) => {
                    if (index < newNumLayers) newSurfaces[index] = surface;
                  });
                  setSurfaces(newSurfaces);
                  setGeoreferenceParams(new Array(newNumLayers).fill({}));
                }}
                disabled={generateBaseline}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              >
                {[1, 2, 3, 4, 5].map((n) => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>

            <div className="flex items-center">
              <input
                id="generateBaseline"
                type="checkbox"
                checked={generateBaseline}
                onChange={(e) => {
                  setGenerateBaseline(e.target.checked);
                  if (e.target.checked) {
                    setNumLayers(1);
                    const newSurfaces = new Array(1).fill(null);
                    if (surfaces[0]) newSurfaces[0] = surfaces[0];
                    setSurfaces(newSurfaces);
                    setGeoreferenceParams(new Array(1).fill({}));
                  }
                }}
                className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
              />
              <label htmlFor="generateBaseline" className="ml-2 block text-sm text-gray-900">
                Generate Baseline Surface
              </label>
            </div>
            
            {generateBaseline && (
                <div className="mb-4">
                    <label htmlFor="baseSurfaceOffset" className="block text-sm font-medium text-gray-700">
                        Base Surface Offset (feet)
                    </label>
                    <input
                        type="number"
                        id="baseSurfaceOffset"
                        value={baseSurfaceOffset}
                        onChange={(e) => setBaseSurfaceOffset(parseFloat(e.target.value) || 0)}
                        className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                    />
                </div>
            )}

            {Array.from({ length: numLayers }, (_, i) => (
              <div key={i} className="mt-6 border-t border-gray-200 pt-6">
                <h3 className="text-lg font-medium leading-6 text-gray-900">
                  {generateBaseline ? 'Reference Surface' : `Layer ${i}`}
                </h3>
                {/* Surface Upload */}
                <div className="mt-4">
                  <label htmlFor={`surface-file-${i}`} className="block text-sm font-medium text-gray-700">
                    Surface File (.ply)
                  </label>
                  <input type="file" id={`surface-file-${i}`} accept=".ply" onChange={(e) => handleFileUpload(e.target.files[0], i)} className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"/>
                  {surfaces[i] && <span className="text-sm text-green-600">Uploaded: {surfaces[i].file.name}</span>}
                </div>
                
                {!generateBaseline && (
                    <div className="mt-4">
                        <label htmlFor={`tonnage-${i}`} className="block text-sm font-medium text-gray-700">Tonnage</label>
                        <input
                            type="number"
                            id={`tonnage-${i}`}
                            value={tonnages[i] || ''}
                            onChange={(e) => setTonnages({ ...tonnages, [i]: parseFloat(e.target.value) || 0 })}
                            className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"
                        />
                    </div>
                )}

                {/* Georeferencing */}
                <div className="mt-4 grid grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700">WGS84 Lat</label>
                        <input type="number" step="any" value={georeferenceParams[i]?.wgs84_lat || ''} onChange={(e) => handleGeoreferenceChange(i, 'wgs84_lat', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700">WGS84 Lon</label>
                        <input type="number" step="any" value={georeferenceParams[i]?.wgs84_lon || ''} onChange={(e) => handleGeoreferenceChange(i, 'wgs84_lon', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Orientation (°)</label>
                        <input type="number" step="any" value={georeferenceParams[i]?.orientation_degrees || ''} onChange={(e) => handleGeoreferenceChange(i, 'orientation_degrees', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700">Scale Factor</label>
                        <input type="number" step="any" value={georeferenceParams[i]?.scaling_factor || '1.0'} onChange={(e) => handleGeoreferenceChange(i, 'scaling_factor', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                    </div>
                </div>
              </div>
            ))}
          </div>

          {/* Right Column: Config and Boundary */}
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold text-gray-800 border-b pb-2 mt-6">Analysis Boundary</h2>
              <p className="text-sm text-gray-600 mt-1">Define the four corners of the analysis area in WGS84 coordinates.</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
                {analysisBoundary.coordinates.map((coord, index) => (
                    <div key={index} className="border-t border-gray-200 pt-4">
                        <h4 className="text-md font-medium text-gray-800">Point {index + 1}</h4>
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Latitude</label>
                            <input type="number" step="any" value={coord.lat} onChange={(e) => handleBoundaryChange(index, 'lat', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                        </div>
                        <div>
                            <label className="block text-sm font-medium text-gray-700">Longitude</label>
                            <input type="number" step="any" value={coord.lon} onChange={(e) => handleBoundaryChange(index, 'lon', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                        </div>
                    </div>
                ))}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="mt-8 pt-5 border-t border-gray-200">
          <div className="flex justify-end">
            <button type="button" onClick={onBack} className="bg-white py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500">
              Back
            </button>
            <button type="submit" disabled={!canProceed()} className="ml-3 inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 disabled:bg-gray-400">
              Run Analysis
            </button>
          </div>
        </div>
      </form>
    </div>
  );
};

export default InputForms; 