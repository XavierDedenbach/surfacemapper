/**
 * Geometry utility functions for surface analysis
 */

/**
 * Calculate distance between two 3D points
 */
export const calculateDistance = (point1, point2) => {
  const dx = point2.x - point1.x;
  const dy = point2.y - point1.y;
  const dz = point2.z - point1.z;
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
};

/**
 * Calculate distance between two 2D points
 */
export const calculateDistance2D = (point1, point2) => {
  const dx = point2.x - point1.x;
  const dy = point2.y - point1.y;
  return Math.sqrt(dx * dx + dy * dy);
};

/**
 * Calculate area of a triangle given three vertices
 */
export const calculateTriangleArea = (vertex1, vertex2, vertex3) => {
  const v1 = { x: vertex2.x - vertex1.x, y: vertex2.y - vertex1.y };
  const v2 = { x: vertex3.x - vertex1.x, y: vertex3.y - vertex1.y };
  return Math.abs(v1.x * v2.y - v1.y * v2.x) / 2;
};

/**
 * Calculate volume of a triangular prism
 */
export const calculatePrismVolume = (baseVertices, height) => {
  const baseArea = calculateTriangleArea(...baseVertices);
  return baseArea * height;
};

/**
 * Check if a point is inside a polygon using ray casting algorithm
 */
export const isPointInPolygon = (point, polygon) => {
  let inside = false;
  const x = point.x;
  const y = point.y;

  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const xi = polygon[i].x;
    const yi = polygon[i].y;
    const xj = polygon[j].x;
    const yj = polygon[j].y;

    if (((yi > y) !== (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
};

/**
 * Calculate bounding box for a set of points
 */
export const calculateBoundingBox = (points) => {
  if (!points || points.length === 0) {
    return null;
  }

  let minX = points[0].x;
  let maxX = points[0].x;
  let minY = points[0].y;
  let maxY = points[0].y;
  let minZ = points[0].z;
  let maxZ = points[0].z;

  for (const point of points) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
    minZ = Math.min(minZ, point.z);
    maxZ = Math.max(maxZ, point.z);
  }

  return {
    min: { x: minX, y: minY, z: minZ },
    max: { x: maxX, y: maxY, z: maxZ },
    center: {
      x: (minX + maxX) / 2,
      y: (minY + maxY) / 2,
      z: (minZ + maxZ) / 2
    },
    size: {
      x: maxX - minX,
      y: maxY - minY,
      z: maxZ - minZ
    }
  };
};

/**
 * Convert degrees to radians
 */
export const degreesToRadians = (degrees) => {
  return degrees * (Math.PI / 180);
};

/**
 * Convert radians to degrees
 */
export const radiansToDegrees = (radians) => {
  return radians * (180 / Math.PI);
};

/**
 * Rotate a point around the origin by given angle
 */
export const rotatePoint = (point, angleRadians) => {
  const cos = Math.cos(angleRadians);
  const sin = Math.sin(angleRadians);
  
  return {
    x: point.x * cos - point.y * sin,
    y: point.x * sin + point.y * cos,
    z: point.z
  };
};

/**
 * Scale a point by given factors
 */
export const scalePoint = (point, scaleX, scaleY, scaleZ = 1) => {
  return {
    x: point.x * scaleX,
    y: point.y * scaleY,
    z: point.z * scaleZ
  };
};

/**
 * Transform a point using rotation and scaling
 */
export const transformPoint = (point, rotationRadians, scaleX, scaleY, scaleZ = 1) => {
  const rotated = rotatePoint(point, rotationRadians);
  return scalePoint(rotated, scaleX, scaleY, scaleZ);
};

/**
 * Calculate surface normal for a triangle
 */
export const calculateTriangleNormal = (vertex1, vertex2, vertex3) => {
  const v1 = {
    x: vertex2.x - vertex1.x,
    y: vertex2.y - vertex1.y,
    z: vertex2.z - vertex1.z
  };
  
  const v2 = {
    x: vertex3.x - vertex1.x,
    y: vertex3.y - vertex1.y,
    z: vertex3.z - vertex1.z
  };

  const normal = {
    x: v1.y * v2.z - v1.z * v2.y,
    y: v1.z * v2.x - v1.x * v2.z,
    z: v1.x * v2.y - v1.y * v2.x
  };

  const length = Math.sqrt(normal.x * normal.x + normal.y * normal.y + normal.z * normal.z);
  
  if (length > 0) {
    normal.x /= length;
    normal.y /= length;
    normal.z /= length;
  }

  return normal;
};

/**
 * Calculate centroid of a triangle
 */
export const calculateTriangleCentroid = (vertex1, vertex2, vertex3) => {
  return {
    x: (vertex1.x + vertex2.x + vertex3.x) / 3,
    y: (vertex1.y + vertex2.y + vertex3.y) / 3,
    z: (vertex1.z + vertex2.z + vertex3.z) / 3
  };
};

/**
 * Interpolate Z value at given X,Y coordinates using bilinear interpolation
 */
export const interpolateZ = (x, y, vertices, faces) => {
  // Find the triangle containing the point
  for (const face of faces) {
    const v1 = vertices[face[0]];
    const v2 = vertices[face[1]];
    const v3 = vertices[face[2]];

    if (isPointInTriangle({ x, y }, v1, v2, v3)) {
      // Calculate barycentric coordinates
      const barycentric = calculateBarycentricCoordinates({ x, y }, v1, v2, v3);
      
      // Interpolate Z value
      return barycentric.a * v1.z + barycentric.b * v2.z + barycentric.c * v3.z;
    }
  }

  return null; // Point not found in any triangle
};

/**
 * Check if a point is inside a triangle
 */
export const isPointInTriangle = (point, vertex1, vertex2, vertex3) => {
  const barycentric = calculateBarycentricCoordinates(point, vertex1, vertex2, vertex3);
  return barycentric.a >= 0 && barycentric.b >= 0 && barycentric.c >= 0 &&
         Math.abs(barycentric.a + barycentric.b + barycentric.c - 1) < 1e-10;
};

/**
 * Calculate barycentric coordinates of a point relative to a triangle
 */
export const calculateBarycentricCoordinates = (point, vertex1, vertex2, vertex3) => {
  const v0 = { x: vertex2.x - vertex1.x, y: vertex2.y - vertex1.y };
  const v1 = { x: vertex3.x - vertex1.x, y: vertex3.y - vertex1.y };
  const v2 = { x: point.x - vertex1.x, y: point.y - vertex1.y };

  const d00 = v0.x * v0.x + v0.y * v0.y;
  const d01 = v0.x * v1.x + v0.y * v1.y;
  const d11 = v1.x * v1.x + v1.y * v1.y;
  const d20 = v2.x * v0.x + v2.y * v0.y;
  const d21 = v2.x * v1.x + v2.y * v1.y;

  const denom = d00 * d11 - d01 * d01;
  
  if (Math.abs(denom) < 1e-10) {
    return { a: 0, b: 0, c: 0 };
  }

  const b = (d11 * d20 - d01 * d21) / denom;
  const c = (d00 * d21 - d01 * d20) / denom;
  const a = 1.0 - b - c;

  return { a, b, c };
};

/**
 * Convert feet to cubic yards
 */
export const feetToCubicYards = (cubicFeet) => {
  return cubicFeet / 27; // 1 cubic yard = 27 cubic feet
};

/**
 * Convert cubic yards to cubic feet
 */
export const cubicYardsToFeet = (cubicYards) => {
  return cubicYards * 27;
};

/**
 * Convert tons to pounds
 */
export const tonsToPounds = (tons) => {
  return tons * 2000; // 1 US short ton = 2000 pounds
};

/**
 * Calculate compaction rate in lbs/cubic yard
 */
export const calculateCompactionRate = (tons, cubicYards) => {
  if (cubicYards <= 0) return null;
  const pounds = tonsToPounds(tons);
  return pounds / cubicYards;
};

/**
 * Calculate volume between two surfaces using triangulated irregular network
 */
export const calculateVolumeBetweenSurfaces = (lowerSurface, upperSurface, boundary) => {
  // This is a simplified implementation
  // In practice, this would use more sophisticated algorithms
  
  let totalVolume = 0;
  
  // For each triangle in the lower surface
  for (const face of lowerSurface.faces) {
    const v1 = lowerSurface.vertices[face[0]];
    const v2 = lowerSurface.vertices[face[1]];
    const v3 = lowerSurface.vertices[face[2]];
    
    // Get corresponding points on upper surface
    const u1 = interpolateZ(v1.x, v1.y, upperSurface.vertices, upperSurface.faces) || v1.z;
    const u2 = interpolateZ(v2.x, v2.y, upperSurface.vertices, upperSurface.faces) || v2.z;
    const u3 = interpolateZ(v3.x, v3.y, upperSurface.vertices, upperSurface.faces) || v3.z;
    
    // Calculate average height difference
    const avgHeight = ((u1 + u2 + u3) / 3) - ((v1.z + v2.z + v3.z) / 3);
    
    // Calculate triangle area
    const area = calculateTriangleArea(v1, v2, v3);
    
    // Add to total volume
    totalVolume += area * avgHeight;
  }
  
  return Math.abs(totalVolume);
};

/**
 * Validate coordinate bounds
 */
export const validateCoordinates = (lat, lon) => {
  return lat >= -90 && lat <= 90 && lon >= -180 && lon <= 180;
};

/**
 * Calculate distance between two WGS84 coordinates in meters
 */
export const calculateWGS84Distance = (lat1, lon1, lat2, lon2) => {
  const R = 6371000; // Earth's radius in meters
  const dLat = degreesToRadians(lat2 - lat1);
  const dLon = degreesToRadians(lon2 - lon1);
  
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(degreesToRadians(lat1)) * Math.cos(degreesToRadians(lat2)) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  
  return R * c;
};

export default {
  calculateDistance,
  calculateDistance2D,
  calculateTriangleArea,
  calculatePrismVolume,
  isPointInPolygon,
  calculateBoundingBox,
  degreesToRadians,
  radiansToDegrees,
  rotatePoint,
  scalePoint,
  transformPoint,
  calculateTriangleNormal,
  calculateTriangleCentroid,
  interpolateZ,
  isPointInTriangle,
  calculateBarycentricCoordinates,
  feetToCubicYards,
  cubicYardsToFeet,
  tonsToPounds,
  calculateCompactionRate,
  calculateVolumeBetweenSurfaces,
  validateCoordinates,
  calculateWGS84Distance
}; 