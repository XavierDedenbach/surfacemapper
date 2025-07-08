import React from 'react';
import backendApi from '../api/backendApi';

/**
 * InputForms component for surface upload and georeferencing
 * This is a "controlled component" that receives all its state and handlers from a parent.
 */

// A single layer's input form
const LayerForm = ({ 
  index,
  title, 
  handleFileUpload,
  handleShpFileUpload,
  handleGeoreferenceChange,
  handleTonnageChange,
  surface,
  georef,
  tonnage
}) => (
  <div className="mt-6 border-t border-gray-200 pt-6">
    <h3 className="text-lg font-medium leading-6 text-gray-900">{title}</h3>
    
    {/* Surface Upload */}
    <div className="mt-4">
      <label htmlFor={`surface-file-${index}`} className="block text-sm font-medium text-gray-700">Surface File (.ply or .shp)</label>
      <input type="file" id={`surface-file-${index}`} accept=".ply,.shp,.shx,.dbf,.prj" multiple onChange={(e) => {
        const files = Array.from(e.target.files);
        const hasShp = files.some(f => f.name.toLowerCase().endsWith('.shp'));
        if (hasShp) {
          handleShpFileUpload(files, index);
        } else {
          handleFileUpload(files[0], index);
        }
      }} className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"/>
      {surface && surface.files && <span className="text-sm text-green-600">Uploaded: {surface.files.map(f => f.name).join(', ')}</span>}
      {surface && surface.file && <span className="text-sm text-green-600">Uploaded: {surface.file.name}</span>}
    </div>
    
    {/* Tonnage input */}
    <div className="mt-4">
      <label htmlFor={`tonnage-${index}`} className="block text-sm font-medium text-gray-700">Tonnage</label>
      <input type="number" id={`tonnage-${index}`} value={tonnage || ''} onChange={(e) => handleTonnageChange(index, e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" />
    </div>

    {/* Georeferencing */}
    <div className="mt-4 grid grid-cols-2 gap-4">
      <div>
        <label className="block text-sm font-medium text-gray-700">WGS84 Lat</label>
        <input type="number" step="any" value={georef?.wgs84_lat || ''} onChange={(e) => handleGeoreferenceChange(index, 'wgs84_lat', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700">WGS84 Lon</label>
        <input type="number" step="any" value={georef?.wgs84_lon || ''} onChange={(e) => handleGeoreferenceChange(index, 'wgs84_lon', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700">Orientation (°)</label>
        <input type="number" step="any" value={georef?.orientation || ''} onChange={(e) => handleGeoreferenceChange(index, 'orientation', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" />
      </div>
      <div>
        <label className="block text-sm font-medium text-gray-700">Scale Factor</label>
        <input type="number" step="any" value={georef?.scaling_factor || ''} onChange={(e) => handleGeoreferenceChange(index, 'scaling_factor', e.target.value)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" />
      </div>
    </div>
  </div>
);

const InputForms = ({
  onWizardComplete,
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
        const newGeorefParams = [];
        const newTonnages = {};

        if (config.analysis_boundary?.coordinates) {
          setAnalysisBoundary({ coordinates: config.analysis_boundary.coordinates });
        }
        
        for (let i = 0; i < 5; i++) { // Assume max 5 layers for config parsing
            const layerConfig = config[`layer_${i}`];
            if (layerConfig) {
                if (layerConfig.georeference) {
                    newGeorefParams[i] = { ...newGeorefParams[i], ...layerConfig.georeference };
                }
                if (layerConfig.tonnage != null) {
                    newTonnages[i] = layerConfig.tonnage;
                }
            }
        }
        
        if (config.orientation != null) {
            if (!newGeorefParams[0]) newGeorefParams[0] = {};
            newGeorefParams[0].orientation = config.orientation;
        }
        
        setGeoreferenceParams(newGeorefParams);
        setTonnages(newTonnages);

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

  const handleShpFileUpload = async (files, index) => {
    if (!files || files.length === 0) return;
    try {
      const response = await backendApi.uploadShpFiles(files);
      const newSurfaces = [...surfaces];
      newSurfaces[index] = { ...response, files };
      setSurfaces(newSurfaces);
    } catch (error) {
      alert(`Upload failed: ${error.message}`);
    }
  };

  const handleGeoreferenceChange = (index, field, value) => {
    const updatedParams = [...georeferenceParams];
    if (!updatedParams[index]) updatedParams[index] = {};
    updatedParams[index][field] = value !== '' ? parseFloat(value) : null;
    setGeoreferenceParams(updatedParams);
  };

  const handleTonnageChange = (index, value) => {
    const newTonnages = { ...tonnages };
    newTonnages[index] = value !== '' ? parseFloat(value) : 0;
    setTonnages(newTonnages);
  };

  const handleBoundaryChange = (index, field, value) => {
    const newCoordinates = [...analysisBoundary.coordinates];
    newCoordinates[index] = { ...newCoordinates[index], [field]: value !== '' ? parseFloat(value) : null };
    setAnalysisBoundary({ coordinates: newCoordinates });
  };
  
  const canProceed = () => {
    const uploadedCount = surfaces.filter(s => s).length;
    if (uploadedCount < numLayers) {
      return false;
    }
    if (!generateBaseline && numLayers < 2) {
      return false;
    }
    return true;
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!canProceed()) {
      let alertMsg = "Please upload all required surface files.";
      if (!generateBaseline && numLayers < 2) {
        alertMsg = "Analysis requires at least two surfaces if you are not generating a baseline.";
      }
      alert(alertMsg);
      return;
    }
    onWizardComplete();
  };

  return (
    <div className="container mx-auto p-8 bg-white shadow-lg rounded-lg">
      <h1 className="text-2xl font-bold mb-6">Analysis Setup</h1>
      <form onSubmit={handleSubmit}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-6">
          {/* Left Column: Project and Layer Setup */}
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold text-gray-800 border-b pb-2">Project Setup</h2>
              <div className="mt-4">
                <label htmlFor="config-file" className="block text-sm font-medium text-gray-700">Upload Configuration (.json)</label>
                <input type="file" id="config-file" accept=".json" onChange={(e) => handleConfigUpload(e.target.files[0])} className="mt-1 block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-violet-50 file:text-violet-700 hover:file:bg-violet-100"/>
              </div>
            </div>
            
            <div className="mb-4">
              <label htmlFor="numLayers" className="block text-sm font-medium text-gray-700">Number of Surfaces/Layers</label>
              <select id="numLayers" value={numLayers} onChange={(e) => setNumLayers(parseInt(e.target.value, 10))} className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md">
                {[1, 2, 3, 4, 5].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>

            <div className="flex items-center">
              <input id="generateBaseline" type="checkbox" checked={generateBaseline} onChange={(e) => setGenerateBaseline(e.target.checked)} className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"/>
              <label htmlFor="generateBaseline" className="ml-2 block text-sm text-gray-900">Generate Baseline Surface</label>
            </div>
            
            {generateBaseline && (
              <div className="mb-4">
                <label htmlFor="baseSurfaceOffset" className="block text-sm font-medium text-gray-700">Base Surface Offset (feet)</label>
                <input type="number" id="baseSurfaceOffset" value={baseSurfaceOffset} onChange={(e) => setBaseSurfaceOffset(parseFloat(e.target.value) || 0)} className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md" />
              </div>
            )}
            
            {Array.from({ length: numLayers }, (_, i) => (
              <LayerForm
                key={i}
                index={i}
                title={`Layer ${i}`}
                handleFileUpload={handleFileUpload}
                handleShpFileUpload={handleShpFileUpload}
                handleGeoreferenceChange={handleGeoreferenceChange}
                handleTonnageChange={handleTonnageChange}
                surface={surfaces[i]}
                georef={georeferenceParams[i]}
                tonnage={tonnages[i]}
              />
            ))}
          </div>

          {/* Right Column: Analysis Boundary */}
          <div className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold text-gray-800 border-b pb-2">Analysis Boundary</h2>
                <p className="text-sm text-gray-600 mt-2">Define the four corners of the analysis area in WGS84 coordinates.</p>
            </div>
            <div className="grid grid-cols-2 gap-x-8 gap-y-4">
                {analysisBoundary.coordinates.map((coord, i) => (
                    <div key={i}>
                        <h4 className="font-semibold">Point {i + 1}</h4>
                        <div className="mt-2">
                            <label className="block text-sm font-medium">Latitude</label>
                            <input type="number" step="any" value={coord.lat || ''} onChange={(e) => handleBoundaryChange(i, 'lat', e.target.value)} className="mt-1 w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                        </div>
                        <div className="mt-2">
                            <label className="block text-sm font-medium">Longitude</label>
                            <input type="number" step="any" value={coord.lon || ''} onChange={(e) => handleBoundaryChange(i, 'lon', e.target.value)} className="mt-1 w-full shadow-sm sm:text-sm border-gray-300 rounded-md"/>
                        </div>
                    </div>
                ))}
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="flex justify-end mt-8 pt-6 border-t">
            <button type="submit" disabled={!canProceed()} className="bg-indigo-600 text-white py-2 px-6 rounded-md shadow-sm hover:bg-indigo-700 disabled:bg-gray-400 disabled:cursor-not-allowed">
                Run Analysis
            </button>
        </div>
      </form>
    </div>
  );
};

export default InputForms; 