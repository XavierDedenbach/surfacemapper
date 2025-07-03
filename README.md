# Surface Mapper

A comprehensive tool for measuring volume and thickness between surface profiles, supporting both PLY and SHP file formats.

## Features

- **Multi-format Support**: Process PLY files (3D point clouds/meshes) and SHP files (2D contour lines)
- **Volume Calculation**: Accurate volume calculations between surface layers
- **Thickness Analysis**: Detailed thickness distribution analysis
- **Coordinate Transformation**: Automatic WGS84 to UTM projection for SHP files
- **Real-time Analysis**: Interactive point queries and batch processing
- **3D Visualization**: Web-based 3D surface visualization
- **Quality Control**: Comprehensive data validation and quality metrics

## Supported File Formats

### PLY Files
- ASCII or binary PLY format
- 3D point clouds and mesh data
- Any coordinate system (specified in file)
- Direct vertex extraction for analysis

### SHP Files
- ESRI Shapefile format (.shp, .shx, .dbf, .prj)
- WGS84 coordinate system (EPSG:4326)
- LineString geometries representing contour lines
- Automatic densification and polygon boundary creation

## Quick Start

### Using Docker (Recommended)

```bash
# Build and start services
docker-compose down && docker-compose up -d --build --force-recreate

# View backend logs
docker-compose logs --tail "100" backend
```

### API Access

- **Backend API**: http://localhost:8081
- **API Documentation**: http://localhost:8081/docs
- **Frontend**: http://localhost:3000

## Example Configuration

### WGS84 Boundary Coordinates
```json
{
  "wgs84_coordinates": [
    [37.773530, -122.421130],  // Southwest corner
    [37.773530, -122.417670],  // Southeast corner  
    [37.776270, -122.417670],  // Northeast corner
    [37.776270, -122.421130]   // Northwest corner
  ]
}
```

### Center Point
Latitude: 37.774900, Longitude: -122.419400

## Documentation

- [API Documentation](docs/api_documentation.md) - Complete API reference
- [Algorithm Specifications](docs/algorithm_specifications.md) - Technical implementation details
- [Task Tracking](requirements/task_tracking.md) - Development progress and status

## Development

The project follows a comprehensive development methodology with:
- Test-driven development (TDD)
- Comprehensive unit and integration testing
- Modular architecture with clear separation of concerns
- Production-ready error handling and validation