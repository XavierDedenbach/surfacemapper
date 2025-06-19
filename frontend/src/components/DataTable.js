import React, { useState } from 'react';

/**
 * DataTable component for displaying analysis results
 * Shows volume, thickness, and compaction rate data in organized tables
 */
const DataTable = ({ 
  volumeResults, 
  thicknessResults, 
  compactionResults, 
  onSurfaceSelection,
  selectedSurfaces 
}) => {
  const [activeTab, setActiveTab] = useState('volume');
  const [sortField, setSortField] = useState('layer_designation');
  const [sortDirection, setSortDirection] = useState('asc');

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const sortData = (data) => {
    if (!data || data.length === 0) return data;

    return [...data].sort((a, b) => {
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

  const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined || value === '--') {
      return '--';
    }
    return typeof value === 'number' ? value.toFixed(decimals) : value;
  };

  const renderVolumeTable = () => (
    <div className="table-container">
      <h3>Volume Analysis Results</h3>
      <table className="data-table">
        <thead>
          <tr>
            <th onClick={() => handleSort('layer_designation')}>
              Layer Designation
              {sortField === 'layer_designation' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th onClick={() => handleSort('volume_cubic_yards')}>
              Volume (cubic yards)
              {sortField === 'volume_cubic_yards' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th>Confidence Interval</th>
            <th onClick={() => handleSort('uncertainty')}>
              Uncertainty (%)
              {sortField === 'uncertainty' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortData(volumeResults || []).map((result, index) => (
            <tr 
              key={index}
              className={selectedSurfaces?.includes(result.layer_designation) ? 'selected' : ''}
              onClick={() => onSurfaceSelection?.(result.layer_designation)}
            >
              <td>{result.layer_designation}</td>
              <td>{formatNumber(result.volume_cubic_yards)}</td>
              <td>
                {formatNumber(result.confidence_interval[0])} - {formatNumber(result.confidence_interval[1])}
              </td>
              <td>{formatNumber(result.uncertainty)}</td>
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
            <th onClick={() => handleSort('layer_designation')}>
              Layer Designation
              {sortField === 'layer_designation' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th onClick={() => handleSort('average_thickness_feet')}>
              Average (feet)
              {sortField === 'average_thickness_feet' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th onClick={() => handleSort('min_thickness_feet')}>
              Minimum (feet)
              {sortField === 'min_thickness_feet' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th onClick={() => handleSort('max_thickness_feet')}>
              Maximum (feet)
              {sortField === 'max_thickness_feet' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th>Confidence Interval</th>
          </tr>
        </thead>
        <tbody>
          {sortData(thicknessResults || []).map((result, index) => (
            <tr 
              key={index}
              className={selectedSurfaces?.includes(result.layer_designation) ? 'selected' : ''}
              onClick={() => onSurfaceSelection?.(result.layer_designation)}
            >
              <td>{result.layer_designation}</td>
              <td>{formatNumber(result.average_thickness_feet)}</td>
              <td>{formatNumber(result.min_thickness_feet)}</td>
              <td>{formatNumber(result.max_thickness_feet)}</td>
              <td>
                {formatNumber(result.confidence_interval[0])} - {formatNumber(result.confidence_interval[1])}
              </td>
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
            <th onClick={() => handleSort('layer_designation')}>
              Layer Designation
              {sortField === 'layer_designation' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th onClick={() => handleSort('compaction_rate_lbs_per_cubic_yard')}>
              Compaction Rate (lbs/cubic yard)
              {sortField === 'compaction_rate_lbs_per_cubic_yard' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
            <th onClick={() => handleSort('tonnage_used')}>
              Tonnage Used (tons)
              {sortField === 'tonnage_used' && (
                <span className={`sort-arrow ${sortDirection}`}>▼</span>
              )}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortData(compactionResults || []).map((result, index) => (
            <tr 
              key={index}
              className={selectedSurfaces?.includes(result.layer_designation) ? 'selected' : ''}
              onClick={() => onSurfaceSelection?.(result.layer_designation)}
            >
              <td>{result.layer_designation}</td>
              <td>
                {result.compaction_rate_lbs_per_cubic_yard 
                  ? formatNumber(result.compaction_rate_lbs_per_cubic_yard) 
                  : '--'
                }
              </td>
              <td>
                {result.tonnage_used 
                  ? formatNumber(result.tonnage_used) 
                  : '--'
                }
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const renderSummary = () => {
    const totalVolume = volumeResults?.reduce((sum, result) => sum + (result.volume_cubic_yards || 0), 0) || 0;
    const avgThickness = thicknessResults?.reduce((sum, result) => sum + (result.average_thickness_feet || 0), 0) / (thicknessResults?.length || 1) || 0;
    const validCompactionRates = compactionResults?.filter(result => result.compaction_rate_lbs_per_cubic_yard) || [];
    const avgCompactionRate = validCompactionRates.length > 0 
      ? validCompactionRates.reduce((sum, result) => sum + result.compaction_rate_lbs_per_cubic_yard, 0) / validCompactionRates.length 
      : 0;

    return (
      <div className="summary-container">
        <h3>Analysis Summary</h3>
        <div className="summary-grid">
          <div className="summary-item">
            <label>Total Volume:</label>
            <span>{formatNumber(totalVolume)} cubic yards</span>
          </div>
          <div className="summary-item">
            <label>Average Thickness:</label>
            <span>{formatNumber(avgThickness)} feet</span>
          </div>
          <div className="summary-item">
            <label>Average Compaction Rate:</label>
            <span>{formatNumber(avgCompactionRate)} lbs/cubic yard</span>
          </div>
          <div className="summary-item">
            <label>Layers Analyzed:</label>
            <span>{volumeResults?.length || 0}</span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="data-table-container">
      <div className="table-tabs">
        <button 
          className={`tab-button ${activeTab === 'summary' ? 'active' : ''}`}
          onClick={() => setActiveTab('summary')}
        >
          Summary
        </button>
        <button 
          className={`tab-button ${activeTab === 'volume' ? 'active' : ''}`}
          onClick={() => setActiveTab('volume')}
        >
          Volume Analysis
        </button>
        <button 
          className={`tab-button ${activeTab === 'thickness' ? 'active' : ''}`}
          onClick={() => setActiveTab('thickness')}
        >
          Thickness Analysis
        </button>
        <button 
          className={`tab-button ${activeTab === 'compaction' ? 'active' : ''}`}
          onClick={() => setActiveTab('compaction')}
        >
          Compaction Analysis
        </button>
      </div>

      <div className="table-content">
        {activeTab === 'summary' && renderSummary()}
        {activeTab === 'volume' && renderVolumeTable()}
        {activeTab === 'thickness' && renderThicknessTable()}
        {activeTab === 'compaction' && renderCompactionTable()}
      </div>

      <div className="table-actions">
        <button onClick={() => {
          // TODO: Implement export functionality
          console.log('Export data');
        }}>
          Export Results
        </button>
        <button onClick={() => {
          // TODO: Implement print functionality
          window.print();
        }}>
          Print Report
        </button>
      </div>
    </div>
  );
};

export default DataTable; 