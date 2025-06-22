import React, { useState } from 'react';

/**
 * DataTable component for displaying analysis results
 * Shows volume, thickness, and compaction rate data in organized tables
 */
const DataTable = ({ analysisResult, tonnages }) => {
  // Validate props and extract data from the main analysisResult object
  const volumeData = analysisResult?.volume_results || [];
  const thicknessData = analysisResult?.thickness_results || [];
  const compactionData = analysisResult?.compaction_results || [];

  const [activeTab, setActiveTab] = useState('summary');
  const [sortField, setSortField] = useState('layer_name');
  const [sortDirection, setSortDirection] = useState('asc');

  // Handle sort with validation
  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  // Sort data with validation
  const sortData = (dataArray) => {
    if (!Array.isArray(dataArray) || dataArray.length === 0) return dataArray;

    return [...dataArray].sort((a, b) => {
      if (!a || !b) return 0;
      
      let aValue = a[sortField];
      let bValue = b[sortField];

      // Handle numeric values
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
      }

      // Handle string values
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortDirection === 'asc' 
          ? aValue.localeCompare(bValue) 
          : bValue.localeCompare(aValue);
      }

      return 0;
    });
  };

  // Format number with validation
  const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || value === '--') {
      return '--';
    }
    return typeof value === 'number' ? value.toFixed(decimals) : value;
  };

  const getTonnageForLayer = (layerName) => {
    const tonnageInfo = tonnages.find(t => t.layer_name === layerName);
    return tonnageInfo ? tonnageInfo.tonnage : 'N/A';
  };

  const renderVolumeTable = () => (
    <div className="table-container">
      <h3>Volume Analysis Results</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('layer_name')}>Layer Name</th>
            <th onClick={() => handleSort('volume_cubic_yards')}>Volume (cubic yards)</th>
          </tr>
        </thead>
        <tbody>
          {sortData(volumeData).map((result, index) => (
            <tr 
              key={index}
            >
              <td>{result.layer_name}</td>
              <td>{formatNumber(result.volume_cubic_yards)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const renderThicknessTable = () => (
    <div className="table-container">
      <h3>Thickness Analysis Results</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('layer_name')}>Layer Name</th>
            <th onClick={() => handleSort('average_thickness_feet')}>Average (feet)</th>
            <th onClick={() => handleSort('min_thickness_feet')}>Minimum (feet)</th>
            <th onClick={() => handleSort('max_thickness_feet')}>Maximum (feet)</th>
          </tr>
        </thead>
        <tbody>
          {sortData(thicknessData).map((layer, index) => (
            <tr 
              key={index}
            >
              <td>{layer.layer_name}</td>
              <td>{formatNumber(layer.average_thickness_feet, 3)}</td>
              <td>{formatNumber(layer.min_thickness_feet, 3)}</td>
              <td>{formatNumber(layer.max_thickness_feet, 3)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const renderCompactionTable = () => (
    <div className="table-container">
      <h3>Compaction Rate Analysis</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('layer_name')}>Layer Name</th>
            <th onClick={() => handleSort('compaction_rate_lbs_per_cubic_yard')}>Compaction Rate (lbs/cubic yard)</th>
          </tr>
        </thead>
        <tbody>
          {sortData(compactionData).map((result, index) => (
            <tr 
              key={index}
            >
              <td>{result.layer_name}</td>
              <td>{formatNumber(result.compaction_rate_lbs_per_cubic_yard)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const renderSummary = () => {
    const summaryData = analysisResult?.analysis_summary || [];

    return (
      <div className="table-container">
        <h3>Analysis Summary</h3>
        <table className="data-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('layer_name')}>Layer Name</th>
              <th onClick={() => handleSort('volume_cubic_yards')}>Volume (Cubic Yards)</th>
              <th onClick={() => handleSort('avg_thickness_feet')}>Avg. Thickness (ft)</th>
              <th onClick={() => handleSort('compaction_rate_lbs_per_cubic_yard')}>Compaction (lbs/cu.yd)</th>
            </tr>
          </thead>
          <tbody>
            {sortData(summaryData).map((item, index) => (
              <tr key={index}>
                <td>{item.layer_name}</td>
                <td>{formatNumber(item.volume_cubic_yards)}</td>
                <td>{formatNumber(item.avg_thickness_feet, 3)}</td>
                <td>{formatNumber(item.compaction_rate_lbs_per_cubic_yard)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );
  };

  const renderTabs = () => (
    <div className="tabs">
      <button 
        className={`tab ${activeTab === 'summary' ? 'active' : ''}`}
        onClick={() => setActiveTab('summary')}
      >
        Summary
      </button>
      <button 
        className={`tab ${activeTab === 'volume' ? 'active' : ''}`}
        onClick={() => setActiveTab('volume')}
      >
        Volume
      </button>
      <button 
        className={`tab ${activeTab === 'thickness' ? 'active' : ''}`}
        onClick={() => setActiveTab('thickness')}
      >
        Thickness
      </button>
      <button 
        className={`tab ${activeTab === 'compaction' ? 'active' : ''}`}
        onClick={() => setActiveTab('compaction')}
      >
        Compaction
      </button>
    </div>
  );

  const renderActiveTab = () => {
    switch (activeTab) {
      case 'volume':
        return renderVolumeTable();
      case 'thickness':
        return renderThicknessTable();
      case 'compaction':
        return renderCompactionTable();
      case 'summary':
      default:
        return renderSummary();
    }
  };

  return (
    <div className="data-table-container">
      {renderTabs()}
      <div className="tab-content">
        {renderActiveTab()}
      </div>
    </div>
  );
};

export default DataTable; 