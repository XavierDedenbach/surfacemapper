import React from 'react';

const PointAnalysisPopup = ({ isVisible, position, thicknessData, loading, onClose }) => {
  if (!isVisible || !position) return null;

  const style = {
    position: 'absolute',
    left: position.x + 12, // offset from cursor
    top: position.y + 12,
    zIndex: 1000,
    minWidth: 180,
    background: 'rgba(30,30,30,0.97)',
    color: '#fff',
    borderRadius: 8,
    boxShadow: '0 2px 12px rgba(0,0,0,0.25)',
    padding: '12px 16px',
    pointerEvents: 'none',
    fontSize: 14,
    maxWidth: 320,
    border: '1px solid #333',
  };

  return (
    <div style={style}>
      {loading ? (
        <div style={{ textAlign: 'center', padding: '8px 0' }}>
          <span className="loader" style={{ display: 'inline-block', width: 18, height: 18, border: '2px solid #fff', borderTop: '2px solid #888', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
          <span style={{ marginLeft: 8 }}>Loading...</span>
        </div>
      ) : thicknessData && thicknessData.error ? (
        <div style={{ color: '#ffb3b3' }}>{thicknessData.error}</div>
      ) : thicknessData && Array.isArray(thicknessData.thickness_layers) ? (
        <>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>
            Thickness at Point
          </div>
          <div style={{ fontSize: 12, color: '#ccc', marginBottom: 8 }}>
            <div>UTM: X: {thicknessData.query_point?.x?.toFixed(2) || 'N/A'}, Y: {thicknessData.query_point?.y?.toFixed(2) || 'N/A'}</div>
            <div>WGS84: Lat: {thicknessData.query_point?.lat?.toFixed(6) || 'N/A'}, Lon: {thicknessData.query_point?.lon?.toFixed(6) || 'N/A'}</div>
          </div>
          <table style={{ width: '100%', fontSize: 12 }}>
            <thead>
              <tr style={{ borderBottom: '1px solid #444' }}>
                <th style={{ textAlign: 'left', padding: '2px 4px' }}>Layer</th>
                <th style={{ textAlign: 'right', padding: '2px 4px' }}>Thickness (ft)</th>
              </tr>
            </thead>
            <tbody>
              {thicknessData.thickness_layers.map((layer, index) => (
                <tr key={index} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: '2px 4px' }}>{layer.layer_designation}</td>
                  <td style={{ textAlign: 'right', padding: '2px 4px' }}>
                    {layer.thickness_feet?.toFixed(3) || 'N/A'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <div style={{ color: '#ccc' }}>No thickness data available</div>
      )}
    </div>
  );
};

export default PointAnalysisPopup;

// Add a simple CSS spinner animation
const styleSheet = document.createElement('style');
styleSheet.type = 'text/css';
styleSheet.innerText = `@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }`;
document.head.appendChild(styleSheet); 