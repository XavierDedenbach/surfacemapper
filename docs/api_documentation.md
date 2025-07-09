# Surface Mapper API Documentation

## Overview

The Surface Mapper API provides endpoints for surface data processing, volume calculations, and analysis. This document describes all available endpoints, request/response formats, and usage examples.

## Coordinate System Requirements

All API endpoints that process surface data require coordinates to be in UTM (meters). The system will automatically project WGS84 coordinates to UTM for SHP files and boundaries, but PLY files must be in UTM. All outputs are in UTM (meters). Calculations assume UTM coordinates.

- SHP files: Projected from WGS84 to UTM automatically (mesh operations and boundary are projected together; project together)
- PLY files: Must be in UTM coordinates
- Boundaries: Projected from WGS84 to UTM automatically
- All returned coordinates, areas, and volumes are in UTM (meters)

**Note:** All relevant endpoints (upload, process, results, visualization) require and return UTM coordinates. All mesh operations are performed in UTM coordinates only. SHP workflows project mesh and boundary together before mesh operations (project together).

## Base URL

```
http://localhost:8081
```

## Authentication

Currently, the API does not require authentication. All endpoints are publicly accessible.

## Content Types

- **Request**: `application/json` for JSON data, `multipart/form-data` for file uploads
- **Response**: `application/json`

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Successful operation
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `422 Unprocessable Entity`: Validation errors
- `500 Internal Server Error`: Server error

Error responses include a `detail` field with error information:

```json
{
  "detail": "Error description"
}
```

## Endpoints

### Health Check

#### GET /health

Check the health status of the API.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-12-19T21:00:00Z",
  "version": "1.0.0"
}
```

### Surface Management

#### POST /surfaces/upload

Upload a surface file (PLY or SHP format).

**Supported File Formats:**
- **PLY files**: ASCII or binary PLY format with vertex coordinates (x, y, z)
- **SHP files**: ESRI Shapefile format with LineString contours in WGS84 coordinate system

**SHP File Requirements:**
- Must be in WGS84 coordinate system (EPSG:4326)
- Should contain LineString geometries for contour lines
- Will be automatically densified to 1-foot spacing and converted to polygon boundary
- Output will be projected to UTM coordinates for analysis

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload with key `file`

**Response:**
```json
{
  "surface_id": "uuid-string",
  "filename": "surface.ply",
  "file_size": 1024000,
  "upload_time": "2024-12-19T21:00:00Z",
  "status": "uploaded"
}
```

#### POST /surfaces/validate

Validate a PLY or SHP file before processing.

**Request:**
- Content-Type: `multipart/form-data`
- Body: File upload with key `file`

**Response:**
```json
{
  "is_valid": true,
  "vertex_count": 10000,
  "face_count": 20000,
  "file_size": 1024000,
  "warnings": [],
  "errors": []
}
```

#### POST /surfaces/process

Process surfaces for volume and thickness analysis.

**Request:**
```json
{
  "surface_files": [
    {
      "surface_id": "uuid-string",
      "filename": "surface1.ply"
    },
    {
      "surface_id": "uuid-string",
      "filename": "surface2.ply"
    }
  ],
  "georeference_params": [
    {
      "wgs84_lat": 37.7749,
      "wgs84_lon": -122.4194,
      "orientation_degrees": 0.0,
      "scaling_factor": 1.0
    },
    {
      "wgs84_lat": 37.7749,
      "wgs84_lon": -122.4194,
      "orientation_degrees": 0.0,
      "scaling_factor": 1.0
    }
  ],
  "analysis_boundary": {
    "wgs84_coordinates": [
      {"lat": 37.7749, "lon": -122.4194},
      {"lat": 37.7849, "lon": -122.4194},
      {"lat": 37.7849, "lon": -122.4094},
      {"lat": 37.7749, "lon": -122.4094}
    ]
  },
  "tonnage_inputs": [
    {
      "layer_index": 0,
      "tonnage": 100.5
    }
  ],
  "processing_params": {
    "volume_calculation": {
      "primary_algorithm": "pyvista_delaunay",
      "validation_tolerance": 1.0
    }
  }
}
```

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "estimated_completion": "2024-12-19T21:30:00Z",
  "message": "Processing started successfully"
}
```

#### GET /surfaces/status/{job_id}

Get the status of a processing job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "completed",
  "progress": 100,
  "start_time": "2024-12-19T21:00:00Z",
  "completion_time": "2024-12-19T21:25:00Z",
  "results": {
    "volume_results": [
      {
        "layer_designation": "Surface 0 to Surface 1",
        "volume_cubic_yards": 1250.5,
        "confidence_interval": [1240.2, 1260.8],
        "uncertainty": 0.8
      }
    ],
    "thickness_results": [
      {
        "layer_designation": "Surface 0 to Surface 1",
        "average_thickness_feet": 2.5,
        "min_thickness_feet": 0.1,
        "max_thickness_feet": 8.2,
        "confidence_interval": [2.4, 2.6]
      }
    ],
    "compaction_results": [
      {
        "layer_designation": "Surface 0 to Surface 1",
        "compaction_rate_lbs_per_cubic_yard": 2160.0,
        "tonnage_used": 100.5
      }
    ]
  }
}
```

#### POST /surfaces/point-analysis

Get point analysis data for specific coordinates.

**Request:**
```json
{
  "x": 100.5,
  "y": 200.3
}
```

**Response:**
```json
{
  "coordinates": {
    "x": 100.5,
    "y": 200.3
  },
  "surface_elevations": [
    {
      "surface_index": 0,
      "elevation_feet": 150.2
    },
    {
      "surface_index": 1,
      "elevation_feet": 152.7
    }
  ],
  "layer_thicknesses": [
    {
      "layer_index": 0,
      "thickness_feet": 2.5
    }
  ]
}
```

#### POST /surfaces/visualization

Get surface data for 3D visualization.

**Request:**
```json
{
  "surface_ids": ["uuid-string", "uuid-string"]
}
```

**Response:**
```json
{
  "surfaces": [
    {
      "surface_id": "uuid-string",
      "vertices": [[x1, y1, z1], [x2, y2, z2], ...],
      "faces": [[v1, v2, v3], [v4, v5, v6], ...],
      "color": "#4285f4",
      "metadata": {
        "filename": "surface1.ply",
        "upload_time": "2024-12-19T21:00:00Z"
      }
    }
  ],
  "boundary": {
    "min": {"x": 0, "y": 0, "z": 100},
    "max": {"x": 500, "y": 300, "z": 200}
  }
}
```

#### GET /surfaces/export/{job_id}

Export analysis results.

**Query Parameters:**
- `format`: Export format (`json`, `csv`, `xlsx`, `pdf`)

**Response:**
File download with appropriate content type.

#### GET /surfaces/history

Get processing history.

**Query Parameters:**
- `limit`: Number of records to return (default: 10)
- `offset`: Number of records to skip (default: 0)

**Response:**
```json
{
  "jobs": [
    {
      "job_id": "uuid-string",
      "status": "completed",
      "created_at": "2024-12-19T21:00:00Z",
      "completed_at": "2024-12-19T21:25:00Z",
      "surface_count": 2
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

#### DELETE /surfaces/job/{job_id}

Delete a processing job.

**Response:**
```json
{
  "message": "Job deleted successfully"
}
```

#### POST /surfaces/job/{job_id}/retry

Retry a failed processing job.

**Response:**
```json
{
  "job_id": "uuid-string",
  "status": "processing",
  "message": "Job restarted successfully"
}
```

### SHP File Processing Workflow

The API supports processing ESRI Shapefile (.shp) format files containing LineString contours for surface analysis. SHP files are automatically processed to create compatible surface data for volume and thickness calculations.

#### SHP File Processing Steps

1. **File Validation**: SHP files are validated for:
   - WGS84 coordinate system (EPSG:4326)
   - Valid LineString geometries
   - Proper file structure and CRS definition

2. **Contour Densification**: LineString contours are densified to ensure points are no more than 1 foot apart for accurate analysis.

3. **Polygon Boundary Creation**: Densified contours are converted to a polygon boundary using:
   - Line merging and boundary extraction
   - Convex hull fallback for complex geometries
   - Area validation to ensure reasonable boundary size

4. **Coordinate Projection**: The polygon boundary is projected from WGS84 to UTM coordinates for analysis:
   - Automatic UTM zone detection based on coordinates
   - Projection to UTM meters for accurate distance calculations
   - Output format compatible with PLY file processing

#### SHP File Requirements

**Format Requirements:**
- ESRI Shapefile format (.shp, .shx, .dbf, .prj files)
- WGS84 coordinate system (EPSG:4326)
- LineString geometries representing contour lines

**Content Requirements:**
- Contour lines should represent elevation boundaries
- Lines should be properly closed or form continuous boundaries
- Coordinate precision should be sufficient for analysis (typically 6+ decimal places)

**Processing Output:**
- Vertices in UTM coordinates (meters)
- Polygon boundary ready for volume calculation
- Compatible with all downstream analysis operations

#### SHP vs PLY File Processing

| Feature | PLY Files | SHP Files |
|---------|-----------|-----------|
| Input Format | 3D point cloud/mesh | 2D contour lines |
| Coordinate System | Any (specified in file) | WGS84 (EPSG:4326) |
| Processing | Direct vertex extraction | Densification + boundary creation |
| Output Format | UTM coordinates | UTM coordinates |
| Analysis Compatibility | Full | Full |
| File Size | Typically larger | Typically smaller |

Both file types produce identical output formats and are fully compatible with all analysis operations.

### Coordinate Systems

#### GET /coordinate-systems

Get available coordinate systems.

**Response:**
```json
{
  "coordinate_systems": {
    "WGS84": {
      "name": "WGS84",
      "description": "World Geodetic System 1984",
      "epsg_code": "EPSG:4326",
      "type": "geographic",
      "units": "degrees"
    }
  }
}
```

#### POST /coordinate-transform

Transform coordinates between coordinate systems.

**Request:**
```json
{
  "coordinates": [
    {"lat": 37.7749, "lon": -122.4194},
    {"lat": 37.7849, "lon": -122.4094}
  ],
  "from_system": "WGS84",
  "to_system": "UTM_NAD83_Zone_10N"
}
```

**Response:**
```json
{
  "transformed_coordinates": [
    {"x": 549304.5, "y": 4181845.2},
    {"x": 550304.5, "y": 4182845.2}
  ],
  "transformation_info": {
    "from_system": "WGS84",
    "to_system": "UTM_NAD83_Zone_10N",
    "accuracy_meters": 0.1
  }
}
```

### Configuration

#### GET /config/processing

Get current processing configuration.

**Response:**
```json
{
  "volume_calculation": {
    "primary_algorithm": "pyvista_delaunay",
    "validation_tolerance": 1.0
  },
  "surface_processing": {
    "max_points": 50000000,
    "decimation_factor": 1.0
  }
}
```

#### PUT /config/processing

Update processing configuration.

**Request:**
```json
{
  "volume_calculation": {
    "primary_algorithm": "pyvista_delaunay",
    "validation_tolerance": 2.0
  }
}
```

**Response:**
```json
{
  "message": "Configuration updated successfully"
}
```

### Statistics

#### GET /surfaces/stats

Get processing statistics.

**Response:**
```json
{
  "total_jobs": 150,
  "completed_jobs": 142,
  "failed_jobs": 8,
  "average_processing_time_minutes": 15.5,
  "total_volume_calculated_cubic_yards": 25000.5,
  "most_common_coordinate_system": "UTM_NAD83_Zone_10N"
}
```

### API Documentation

#### GET /docs

Get interactive API documentation (Swagger UI).

#### GET /openapi.json

Get OpenAPI specification.

## Rate Limiting

Currently, no rate limiting is implemented. However, large file uploads and processing operations may take significant time.

## File Size Limits

- Maximum file size: 2GB per PLY file
- Maximum total project size: 8GB
- Recommended file size: < 500MB for optimal performance

## Performance Considerations

- Processing time scales with point count
- Large datasets (>10M points) may take 10-20 minutes
- Memory usage scales with dataset size
- Consider decimation for very large point clouds

## Examples

### Complete Workflow Example

```bash
# 1. Upload surface files
curl -X POST -F "file=@surface1.ply" http://localhost:8081/surfaces/upload
curl -X POST -F "file=@surface2.ply" http://localhost:8081/surfaces/upload

# 2. Process surfaces
curl -X POST -H "Content-Type: application/json" \
  -d @processing_request.json \
  http://localhost:8081/surfaces/process

# 3. Check status
curl http://localhost:8081/surfaces/status/{job_id}

# 4. Export results
curl http://localhost:8081/surfaces/export/{job_id}?format=csv
```

### Processing Request Example

```json
{
  "surface_files": [
    {"surface_id": "uuid1", "filename": "surface1.ply"},
    {"surface_id": "uuid2", "filename": "surface2.ply"}
  ],
  "georeference_params": [
    {
      "wgs84_lat": 37.7749,
      "wgs84_lon": -122.4194,
      "orientation_degrees": 0.0,
      "scaling_factor": 1.0
    },
    {
      "wgs84_lat": 37.7749,
      "wgs84_lon": -122.4194,
      "orientation_degrees": 0.0,
      "scaling_factor": 1.0
    }
  ],
  "analysis_boundary": {
    "wgs84_coordinates": [
      {"lat": 37.7749, "lon": -122.4194},
      {"lat": 37.7849, "lon": -122.4194},
      {"lat": 37.7849, "lon": -122.4094},
      {"lat": 37.7749, "lon": -122.4094}
    ]
  },
  "tonnage_inputs": [
    {"layer_index": 0, "tonnage": 100.5}
  ]
}
```

## Support

For API support and questions, please refer to the project documentation or create an issue in the project repository. 