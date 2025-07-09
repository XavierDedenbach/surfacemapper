import React from 'react';
import { render, screen } from '@testing-library/react';
import ThreeDViewer from '../ThreeDViewer';

// Utility to validate UTM coordinates (to be added to ThreeDViewer)
function validateUTMCoordinates(vertices) {
  if (!vertices || vertices.length === 0) return false;
  const xs = vertices.map(v => v[0]);
  const ys = vertices.map(v => v[1]);
  return xs.every(x => Math.abs(x) > 180) && ys.every(y => Math.abs(y) > 90);
}

describe('ThreeDViewer UTM coordinate validation', () => {
  beforeEach(() => {
    jest.spyOn(console, 'warn').mockImplementation(() => {});
  });
  afterEach(() => {
    jest.restoreAllMocks();
  });

  it('renders with UTM coordinates and does not warn', () => {
    const utmSurface = {
      vertices: [ [500000, 4100000, 0], [500100, 4100000, 0], [500100, 4100100, 0], [500000, 4100100, 0] ],
      faces: []
    };
    render(<ThreeDViewer analysisResult={{ surfaces: [utmSurface] }} />);
    expect(console.warn).not.toHaveBeenCalled();
  });

  it('renders with WGS84 coordinates and logs a warning', () => {
    const wgsSurface = {
      vertices: [ [-120.0, 35.0, 0], [-120.0, 35.001, 0], [-119.999, 35.001, 0], [-119.999, 35.0, 0] ],
      faces: []
    };
    render(<ThreeDViewer analysisResult={{ surfaces: [wgsSurface] }} />);
    expect(console.warn).toHaveBeenCalledWith(expect.stringContaining('WGS84'));
  });

  it('validateUTMCoordinates returns true for UTM, false for WGS84', () => {
    const utm = [ [500000, 4100000, 0], [500100, 4100000, 0] ];
    const wgs = [ [-120.0, 35.0, 0], [-119.9, 35.1, 0] ];
    expect(validateUTMCoordinates(utm)).toBe(true);
    expect(validateUTMCoordinates(wgs)).toBe(false);
  });

  it('handles invalid/empty/missing vertices gracefully', () => {
    render(<ThreeDViewer analysisResult={{ surfaces: [{ vertices: [] }] }} />);
    render(<ThreeDViewer analysisResult={{ surfaces: [{}] }} />);
    render(<ThreeDViewer analysisResult={{ surfaces: [] }} />);
    expect(console.warn).not.toHaveBeenCalled();
  });
}); 