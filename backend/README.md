# Surface Mapper Backend

This is the Python FastAPI backend for the Surface Volume and Layer Thickness Analysis Tool.

## Architecture

The backend follows a modular architecture with the following structure:

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI app entry point
│   ├── routes/                 # API endpoints
│   │   └── surfaces.py         # Surface upload and processing endpoints
│   ├── services/               # Core business logic
│   │   ├── surface_processor.py # PLY parsing, clipping, base surface
│   │   ├── volume_calculator.py # Volume and thickness calculations
│   │   └── coord_transformer.py # Coordinate transformations
│   ├── models/                 # Data models (Pydantic)
│   │   └── data_models.py      # Request/response data structures
│   └── utils/                  # Helper functions
│       └── ply_parser.py       # PLY file parsing logic
├── tests/                      # Unit and integration tests
│   ├── test_services.py
│   └── test_routes.py
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Features

- **PLY File Processing**: Parse and validate .ply surface files
- **Coordinate Transformation**: Transform local coordinates to geo-referenced systems using pyproj
- **Volume Calculation**: Calculate volume differences between surfaces using PyVista (Note: PyVista provides native 3D Delaunay triangulation and advanced mesh operations via VTK backend)
- **Thickness Analysis**: Compute layer thickness statistics
- **Compaction Rate**: Calculate compaction rates from tonnage inputs
- **RESTful API**: FastAPI-based endpoints for all operations

## Dependencies

Key dependencies include:
- `fastapi`: Web framework
- `uvicorn`: ASGI server
- `plyfile`: PLY file parsing
- `pyvista`: 3D geometry processing (replaces Trimesh/Open3D; provides advanced capabilities via VTK backend)
- `numpy`: Numerical computations
- `pyproj`: Coordinate transformations
- `pydantic`: Data validation
- `pytest`: Testing framework

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the development server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## API Endpoints

### Health Check
- `GET /health` - Service health status

### Surface Operations
- `POST /surfaces/upload` - Upload .ply surface file
- `POST /surfaces/process` - Process surfaces for analysis
- `GET /surfaces/status/{job_id}` - Get processing status

## Development

### Adding New Endpoints
1. Create route handlers in `app/routes/`
2. Add corresponding data models in `app/models/`
3. Implement business logic in `app/services/`
4. Add tests in `tests/`

### Testing
- Unit tests for services in `tests/test_services.py`
- Integration tests for routes in `tests/test_routes.py`
- Run with `pytest tests/`

## Performance Considerations

- PLY files up to 50M points supported
- Memory usage limited to 8GB for large datasets
- Processing benchmarks:
  - 1M points: <30 seconds
  - 10M points: <5 minutes
  - 50M points: <20 minutes

## Error Handling

- Comprehensive validation using Pydantic models
- Detailed error messages for debugging
- Graceful handling of malformed PLY files
- Coordinate transformation accuracy validation

## Logging

- Structured logging for all operations
- Performance metrics tracking
- Error logging with context
- Processing logs for algorithm validation 