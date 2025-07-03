# Surface Mapper Backend

## Getting Started

### 1. Prerequisites
- **Python**: 3.9+ (ideally matches your dev environment)
- **Node.js**: 16+ (for frontend, if you want the UI)
- **Docker** (optional, for containerized deployment)
- **.ply file**: Your surface mesh file

### 2. Backend Setup

#### A. Using Python (Local Dev)
1. **Install dependencies**  
   From the `backend/` directory:
   ```bash
   pip install -r requirements.txt
   ```
2. **Start the backend server**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8081
   ```
   - The API will be available at: [http://localhost:8081](http://localhost:8081)
   - Docs: [http://localhost:8081/docs](http://localhost:8081/docs)
   - Health check: [http://localhost:8081/health](http://localhost:8081/health)

#### B. Using Docker
1. **Build and run the backend container**
   ```bash
   docker build -f ../Dockerfile.backend -t surfacemapper-backend ..
   docker run -p 8081:8081 surfacemapper-backend
   ```

### 3. Frontend Setup (Optional, for Web UI)

#### A. Using Node.js
1. From the `frontend/` directory:
   ```bash
   npm install
   npm run build
   npm start
   ```
   - The UI will be available at: [http://localhost:3000](http://localhost:3000)

#### B. Using Docker
1. **Build and run the frontend container**
   ```bash
   docker build -f ../Dockerfile.frontend -t surfacemapper-frontend ..
   docker run -p 3000:80 surfacemapper-frontend
   ```

### 4. Uploading Your First Surface File

The backend supports both PLY and SHP file formats:

#### PLY Files
- 3D point clouds and mesh data
- ASCII or binary format
- Any coordinate system

#### SHP Files
- ESRI Shapefile format (.shp, .shx, .dbf, .prj)
- WGS84 coordinate system (EPSG:4326)
- LineString geometries representing contour lines
- Automatic densification and polygon boundary creation

#### A. Using the API (cURL)
```bash
# Upload PLY file
curl -X POST "http://localhost:8081/api/v1/surfaces/upload" \
  -F "file=@/path/to/your/surface.ply"

# Upload SHP file
curl -X POST "http://localhost:8081/api/v1/surfaces/upload" \
  -F "file=@/path/to/your/contours.shp"
```
- You should get a JSON response confirming upload.

#### B. Using the Web UI
1. Open [http://localhost:3000](http://localhost:3000)
2. Use the upload form to select and submit your surface file (PLY or SHP).

### 5. Processing the Surface
- After upload, use the API or UI to trigger processing.
- Example API call (adjust parameters as needed):
  ```bash
  curl -X POST "http://localhost:8081/api/v1/surfaces/process" \
    -H "Content-Type: application/json" \
    -d '{
      "surface_files": ["surface.ply"],
      "georeference_params": [{"wgs84_lat": 40.0, "wgs84_lon": -120.0, "orientation_degrees": 0.0, "scaling_factor": 1.0}],
      "analysis_boundary": {"wgs84_coordinates": [[40.0, -120.0], [40.0, -119.0], [41.0, -119.0], [41.0, -120.0]]}
    }'
  ```

### 6. Checking Status & Results
- **Status:**  
  ```bash
  curl "http://localhost:8081/api/v1/surfaces/status/<job_id>"
  ```
- **Results:**  
  Use the `/api/analysis/` endpoints as documented in [http://localhost:8081/docs](http://localhost:8081/docs).

### 7. Troubleshooting
- Check logs in your terminal for errors.
- Use the `/health` endpoint to verify the backend is running.
- For CORS issues, ensure both frontend and backend are running on the correct ports.

---

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

- **Multi-format File Processing**: Parse and validate PLY files (3D point clouds/meshes) and SHP files (2D contour lines)
- **SHP File Processing**: Automatic densification, polygon boundary creation, and WGS84 to UTM projection
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
- `fiona`: SHP file parsing
- `shapely`: Geometric operations for SHP processing
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
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8081
   ```

3. Run tests:
   ```bash
   pytest tests/
   ```

## API Endpoints

### Health Check
- `GET /health` - Service health status

### Surface Operations
- `POST /surfaces/upload` - Upload PLY or SHP surface file
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

## Running the Application

### Development Mode
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8081
``` 