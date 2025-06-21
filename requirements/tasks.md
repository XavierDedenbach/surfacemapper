# Task Breakdown Document: Surface Volume and Layer Thickness Analysis Tool

## Development Methodology
**Test-Driven Development (TDD)**: All tasks follow the pattern: Write Tests → Implement Code → Refactor. Each task specifies exact implementation requirements, code length expectations, and testing criteria suitable for intern-level developers.

## Phase 1: Foundation & Core Infrastructure (Weeks 1-4)

### Major Task 1.0: Project Setup & Development Environment

#### Subtask 1.1: Core Project Structure Setup

##### Minor Task 1.1.1 (Test First): Define docker-compose.yml
**Action**: Create docker-compose.yml that defines two services: backend and frontend
**What to do**: Create a YAML file defining backend service (Python) and frontend service (Node.js) with port mappings and volume mounts
**Implementation Level**: Basic Docker Compose configuration with service definitions
**Code Estimate**: ~30 lines of YAML configuration
**How to Test**: Run `docker-compose config` to validate syntax
**Acceptance Criteria**: File is syntactically correct, defines both services with proper port mappings

##### Minor Task 1.1.2 (Test First): Create Dockerfile.backend
**Action**: Create Dockerfile for Python FastAPI backend
**What to do**: Write Dockerfile using python:3.9-slim base, install requirements, copy code, expose port
**Implementation Level**: Multi-stage Docker build with dependency installation
**Code Estimate**: ~20 lines of Dockerfile commands
**How to Test**: Run `docker build -f Dockerfile.backend -t backend-image .` and verify success
**Acceptance Criteria**: Dockerfile builds without errors, creates working Python environment

##### Minor Task 1.1.3 (Test First): Create Dockerfile.frontend
**Action**: Create Dockerfile for React frontend with production build
**What to do**: Write Dockerfile using node:16-alpine, install deps, build React app, serve with nginx
**Implementation Level**: Multi-stage build (Node build + nginx serve)
**Code Estimate**: ~25 lines of Dockerfile commands
**How to Test**: Run `docker build -f Dockerfile.frontend -t frontend-image .` and verify success
**Acceptance Criteria**: Dockerfile builds without errors, creates production-ready static files

##### Minor Task 1.1.4 (Test First): Verify Docker Compose Integration
**Action**: Test complete Docker Compose orchestration
**What to do**: Update docker-compose.yml to reference new Dockerfiles and verify both services start
**Implementation Level**: Service orchestration with health checks
**Code Estimate**: ~10 lines of docker-compose modifications
**How to Test**: Run `docker-compose up --build` and verify both services start without errors
**Acceptance Criteria**: Both services start, logs show "started" messages, no container exits

##### Minor Task 1.1.5: Initialize Python Backend Directory Structure
**Action**: Create complete backend directory structure per PRD
**What to do**: Create backend/, backend/app/, backend/app/routes/, backend/app/services/, backend/app/models/, backend/app/utils/, backend/tests/ with __init__.py files
**Implementation Level**: Standard Python package structure
**Code Estimate**: ~15 __init__.py files and directory creation
**How to Test**: Verify all directories exist and Python can import packages
**Acceptance Criteria**: All directories created, Python imports work without errors

##### Minor Task 1.1.6: Initialize React Frontend Directory Structure
**Action**: Create React application using Create React App
**What to do**: Run `npx create-react-app frontend` and organize according to PRD structure
**Implementation Level**: Standard React project initialization
**Code Estimate**: ~50 generated files from CRA
**How to Test**: Run `npm start` in frontend directory and verify React app loads
**Acceptance Criteria**: React development server starts, default page renders in browser

##### Minor Task 1.1.7: Create Additional Project Directories
**Action**: Create data/, config/, docs/ directories per PRD specification
**What to do**: Create additional directories with placeholder README.md files
**Implementation Level**: Basic directory structure with documentation placeholders
**Code Estimate**: ~5 directories, ~3 placeholder README files
**How to Test**: Verify directories exist and contain placeholder files
**Acceptance Criteria**: All directories created, README files present

##### Minor Task 1.1.8: Initialize Git Repository
**Action**: Set up version control with appropriate .gitignore
**What to do**: Run `git init`, create .gitignore for Python/Node.js/Docker, make initial commit
**Implementation Level**: Standard Git initialization with comprehensive .gitignore
**Code Estimate**: ~50 lines in .gitignore file
**How to Test**: Run `git status` and verify tracked/untracked files are appropriate
**Acceptance Criteria**: Git repository initialized, .gitignore excludes build artifacts

#### Subtask 1.2: Dependency Installation and Configuration

##### Minor Task 1.2.1: Create backend/requirements.txt
**Action**: Define all Python dependencies for the backend
**What to do**: List fastapi, uvicorn, plyfile, pyvista, numpy, scipy, pyproj, pydantic with specific versions (Note: PyVista provides advanced 3D mesh processing with VTK backend, including native 3D Delaunay triangulation)
**Implementation Level**: Pinned dependency versions for reproducible builds
**Code Estimate**: ~15 lines of package specifications
**How to Test**: Run `pip install -r backend/requirements.txt` and verify no errors
**Acceptance Criteria**: All dependencies install successfully, no version conflicts

##### Minor Task 1.2.2: Install Frontend Dependencies
**Action**: Add Three.js and related packages to React project
**What to do**: Run `npm install three @react-three/fiber @react-three/drei` in frontend directory
**Implementation Level**: Standard npm package installation
**Code Estimate**: Package.json will have ~5 new dependencies
**How to Test**: Run `npm list` and verify packages are installed
**Acceptance Criteria**: All packages install without warnings, no dependency conflicts

##### Minor Task 1.2.3: Configure Tailwind CSS
**Action**: Set up Tailwind CSS for React styling
**What to do**: Install Tailwind, create tailwind.config.js, update CSS imports
**Implementation Level**: Standard Tailwind installation with React integration
**Code Estimate**: ~20 lines in config files, ~5 lines of CSS imports
**How to Test**: Add Tailwind class to component and verify styling applies
**Acceptance Criteria**: Tailwind classes work in React components

##### Minor Task 1.2.4: Create Basic FastAPI Application
**Action**: Create minimal FastAPI app structure
**What to do**: Write backend/app/main.py with basic FastAPI instance and health check endpoint
**Implementation Level**: Minimal FastAPI setup with CORS configuration
**Code Estimate**: ~25 lines of Python code
**How to Test**: Run `uvicorn app.main:app --reload` and access health endpoint
**Acceptance Criteria**: FastAPI starts, health endpoint returns 200 OK

##### Minor Task 1.2.5: Create Basic React Component Structure
**Action**: Set up basic React component organization
**What to do**: Create src/components/, src/api/, src/utils/ directories with placeholder components
**Implementation Level**: Basic React component organization
**Code Estimate**: ~5 placeholder component files
**How to Test**: Import components in App.js and verify no errors
**Acceptance Criteria**: Components can be imported and rendered without errors

## Phase 2: Backend Core Infrastructure (Weeks 5-8)

### Major Task 2.0: Data Models and Type Definitions

#### Subtask 2.1: Pydantic Model Definitions

##### Minor Task 2.1.1 (Test First): Write Tests for File Upload Models
**Task**: Create comprehensive tests for file upload data structures
**What to do**: Create `backend/tests/test_data_models.py` with tests for file upload request/response models
**Implementation Level**: 
- Test FileUploadResponse model validation
- Test invalid data rejection
- Test serialization/deserialization
**Code Estimate**: ~40 lines of test code
**How to Test**:
```python
def test_file_upload_response_validation():
    response = FileUploadResponse(
        success=True,
        file_id="test-123",
        filename="test.ply",
        size_bytes=1024
    )
    assert response.success is True
    assert response.file_id == "test-123"

def test_file_upload_response_invalid_data():
    with pytest.raises(ValidationError):
        FileUploadResponse(success="invalid_boolean")
```
**Acceptance Criteria**: All model validation tests pass, invalid data properly rejected

##### Minor Task 2.1.2 (Implementation): Create File Upload Models
**Task**: Implement Pydantic models for file upload operations
**What to do**: Create file upload request/response models in `backend/app/models/data_models.py`
**Implementation Level**:
- FileUploadResponse with success, file_id, filename, size_bytes fields
- Input validation for file types and sizes
- Proper field types and constraints
**Code Estimate**: ~50 lines
**How to Test**: Use tests from 2.1.1 - all tests must pass
**Acceptance Criteria**: Models validate correctly, serialization works

##### Minor Task 2.1.3 (Test First): Write Tests for Surface Configuration Models
**Task**: Create tests for surface transformation parameter models
**What to do**: Test SurfaceConfig model with anchor coordinates, rotation, scaling
**Implementation Level**:
- Test valid coordinate ranges (-90 to 90 lat, -180 to 180 lon)
- Test rotation angle validation (0-360 degrees)
- Test scaling factor validation (positive values)
**Code Estimate**: ~60 lines of test code
**How to Test**:
```python
def test_surface_config_validation():
    config = SurfaceConfig(
        file_id="test-123",
        anchor_lat=40.7128,
        anchor_lon=-74.0060,
        rotation_degrees=45.0,
        scale_factor=1.5
    )
    assert config.anchor_lat == 40.7128
    assert config.rotation_degrees == 45.0

def test_surface_config_invalid_coordinates():
    with pytest.raises(ValidationError):
        SurfaceConfig(anchor_lat=91.0, anchor_lon=0.0)  # Invalid latitude
```
**Acceptance Criteria**: All validation tests pass, edge cases handled properly

##### Minor Task 2.1.4 (Implementation): Create Surface Configuration Models
**Task**: Implement models for surface transformation parameters
**What to do**: Create SurfaceConfig, AnalysisConfig models with proper validation
**Implementation Level**:
- SurfaceConfig with coordinate validation
- AnalysisConfig for boundary definition
- Field validators for numeric ranges
**Code Estimate**: ~80 lines
**How to Test**: Use tests from 2.1.3 - all validation tests must pass
**Acceptance Criteria**: All models validate correctly, proper error messages

##### Minor Task 2.1.5 (Test First): Write Tests for Analysis Result Models
**Task**: Create tests for volume and thickness calculation result models
**What to do**: Test VolumeResult, ThicknessResult, AnalysisResponse models
**Implementation Level**:
- Test volume calculation results with units
- Test thickness statistics (min, max, average)
- Test compaction rate calculations
**Code Estimate**: ~50 lines of test code
**How to Test**:
```python
def test_volume_result_model():
    result = VolumeResult(
        layer_name="Surface 0 to 1",
        volume_cubic_yards=150.5,
        compaction_rate_lbs_per_cubic_yard=1800.0
    )
    assert result.volume_cubic_yards == 150.5
    assert result.compaction_rate_lbs_per_cubic_yard == 1800.0
```
**Acceptance Criteria**: Result models accurately represent calculation outputs

##### Minor Task 2.1.6 (Implementation): Create Analysis Result Models
**Task**: Implement models for analysis calculation results
**What to do**: Create VolumeResult, ThicknessResult, AnalysisResponse models
**Implementation Level**:
- VolumeResult with volume and compaction rate
- ThicknessResult with statistics
- AnalysisResponse combining all results
**Code Estimate**: ~100 lines
**How to Test**: Use tests from 2.1.5 - all result model tests must pass
**Acceptance Criteria**: Models accurately represent all calculation outputs

#### Subtask 2.2: Frontend Type Definitions

##### Minor Task 2.2.1 (Test First): Write Tests for TypeScript Interfaces
**Task**: Create tests for frontend TypeScript interface definitions
**What to do**: Create `frontend/src/types/__tests__/interfaces.test.ts` with type validation tests
**Implementation Level**:
- Test interface compatibility with backend models
- Test type safety for API responses
- Test component prop type validation
**Code Estimate**: ~40 lines of test code
**How to Test**:
```typescript
describe('TypeScript Interfaces', () => {
  it('should validate FileUploadResponse interface', () => {
    const response: FileUploadResponse = {
      success: true,
      file_id: "test-123",
      filename: "test.ply",
      size_bytes: 1024
    };
    expect(response.success).toBe(true);
  });
});
```
**Acceptance Criteria**: All TypeScript interfaces compile without errors

##### Minor Task 2.2.2 (Implementation): Create TypeScript Interface Definitions
**Task**: Implement TypeScript interfaces matching backend Pydantic models
**What to do**: Create `frontend/src/types/interfaces.ts` with all data model interfaces
**Implementation Level**:
- Interfaces matching all backend Pydantic models
- API response type definitions
- Component prop type definitions
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 2.2.1 - all TypeScript compilation tests must pass
**Acceptance Criteria**: All interfaces compile, match backend models exactly

### Major Task 3.0: PLY File Processing Foundation

#### Subtask 3.1: PLY File Upload API

##### Minor Task 3.1.1 (Test First): Write Upload Endpoint Tests
**Task**: Create comprehensive tests for PLY file upload endpoint
**What to do**: Create `backend/tests/test_upload_api.py` with upload validation tests
**Implementation Level**:
- Test successful PLY file upload
- Test invalid file type rejection
- Test file size validation
- Test multiple file handling
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_upload_valid_ply_file():
    client = TestClient(app)
    with open("test_data/sample.ply", "rb") as f:
        response = client.post("/api/upload", files={"file": f})
    assert response.status_code == 200
    assert "file_id" in response.json()

def test_upload_invalid_file_type():
    client = TestClient(app)
    with open("test_data/invalid.txt", "rb") as f:
        response = client.post("/api/upload", files={"file": f})
    assert response.status_code == 400
    assert "invalid file type" in response.json()["detail"]
```
**Acceptance Criteria**: All upload scenarios tested, proper error handling

##### Minor Task 3.1.2 (Implementation): Create Upload Endpoint
**Task**: Implement FastAPI endpoint for PLY file uploads
**What to do**: Create `/api/upload` endpoint in `backend/app/routes/surfaces.py`
**Implementation Level**:
- File type validation (.ply extension)
- File size limits (2GB max)
- Unique file ID generation
- Temporary file storage
**Code Estimate**: ~60 lines
**How to Test**: Use tests from 3.1.1 - all upload tests must pass
**Acceptance Criteria**: Endpoint handles all test scenarios correctly

##### Minor Task 3.1.3 (Test First): Write File Validation Tests
**Task**: Create tests for comprehensive file validation logic
**What to do**: Test PLY format validation, corruption detection, size limits
**Implementation Level**:
- Test ASCII and binary PLY format detection
- Test corrupted file handling
- Test oversized file rejection
- Test empty file handling
**Code Estimate**: ~60 lines of test code
**How to Test**:
```python
def test_ply_format_validation():
    valid_ply_content = b"ply\nformat ascii 1.0\nelement vertex 3\n"
    assert validate_ply_format(valid_ply_content) is True
    
    invalid_content = b"not a ply file"
    assert validate_ply_format(invalid_content) is False
```
**Acceptance Criteria**: All file validation edge cases handled correctly

##### Minor Task 3.1.4 (Implementation): Create File Validation Logic
**Task**: Implement comprehensive PLY file validation
**What to do**: Create validation functions in `backend/app/utils/file_validator.py`
**Implementation Level**:
- PLY header validation
- File size checking
- Format detection (ASCII/binary)
- Corruption detection
**Code Estimate**: ~80 lines
**How to Test**: Use tests from 3.1.3 - all validation tests must pass
**Acceptance Criteria**: Robust validation catches all invalid files

#### Subtask 3.2: PLY File Parsing and Processing

##### Minor Task 3.2.1 (Test First): Write PLY Parser Tests
**Task**: Create tests for PLY file parsing into PyVista structures (was: Trimesh/Open3D)
**What to do**: Create `backend/tests/test_ply_parser.py` with parsing accuracy tests
**Implementation Level**:
- Test ASCII PLY parsing
- Test binary PLY parsing
- Test vertex data extraction accuracy
- Test face data handling (if present)
**Code Estimate**: ~100 lines of test code
**How to Test**:
```python
def test_parse_ascii_ply():
    ply_content = create_test_ply_ascii(vertices=[(0,0,0), (1,0,0), (0,1,0)])
    point_cloud = parse_ply_to_point_cloud(ply_content)
    points = np.asarray(point_cloud.points)
    assert len(points) == 3
    np.testing.assert_array_almost_equal(points[0], [0, 0, 0])
```
**Acceptance Criteria**: Parser accurately extracts all vertex data

##### Minor Task 3.2.2 (Implementation): Create PLY Parser
**Task**: Implement PLY file parser using plyfile and PyVista (was: Trimesh/Open3D)
**What to do**: Create parser in `backend/app/utils/ply_parser.py`
**Implementation Level**:
- Use plyfile for reading PLY format
- Convert to PyVista PointCloud/Mesh (Note: PyVista provides advanced point cloud to mesh conversion via VTK backend)
- Handle both ASCII and binary formats
- Extract vertex coordinates and attributes
**Code Estimate**: ~100 lines
**How to Test**: Use tests from 3.2.1 - all parsing tests must pass
**Acceptance Criteria**: Parser handles all PLY format variations correctly

##### Minor Task 3.2.3 (Test First): Write Point Cloud Processing Tests
**Task**: Create tests for point cloud manipulation and optimization
**What to do**: Test point cloud filtering, downsampling, and optimization
**Implementation Level**:
- Test point cloud filtering by bounds
- Test downsampling for performance
- Test outlier removal
- Test coordinate system consistency
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_point_cloud_filtering():
    points = np.array([[0,0,0], [1,1,1], [5,5,5], [10,10,10]])
    pc = create_point_cloud_from_array(points)
    filtered = filter_point_cloud_by_bounds(pc, min_bound=[0,0,0], max_bound=[2,2,2])
    assert len(np.asarray(filtered.points)) == 2
```
**Acceptance Criteria**: Point cloud operations maintain data integrity

##### Minor Task 3.2.4 (Implementation): Create Point Cloud Processing
**Task**: Implement point cloud manipulation functions
**What to do**: Create processing functions in `backend/app/services/point_cloud_processor.py`
**Implementation Level**:
- Bounding box filtering
- Statistical outlier removal
- Downsampling for large datasets
- Coordinate transformation application
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 3.2.3 - all processing tests must pass
**Acceptance Criteria**: All point cloud operations work efficiently and accurately

##### Minor Task 3.2.5: Validate Mesh Simplification and Point Cloud Meshing Quality with PyVista
**Task**: Test and document the quality of mesh simplification and point cloud to mesh conversion using PyVista. PyVista provides advanced capabilities via VTK backend.
**What to do**: Run test cases for mesh simplification and point cloud meshing. Document PyVista's advanced capabilities and compare with Open3D results if possible.
**Acceptance Criteria**: Quality and capabilities of PyVista mesh operations are documented and validated against project requirements.

#### Subtask 3.3: Memory Management and Performance

##### Minor Task 3.3.1 (Test First): Write Memory Usage Tests
**Task**: Create tests for memory efficiency with large PLY files
**What to do**: Test memory usage patterns with increasing file sizes
**Implementation Level**:
- Test memory usage with 1M, 10M, 50M point files
- Test memory cleanup after processing
- Test streaming for large files
- Test memory leak detection
**Code Estimate**: ~60 lines of test code
**How to Test**:
```python
def test_memory_usage_large_file():
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss
    
    large_pc = generate_large_point_cloud(10_000_000)  # 10M points
    processed_pc = process_point_cloud(large_pc)
    
    peak_memory = process.memory_info().rss
    memory_increase = (peak_memory - initial_memory) / 1024 / 1024  # MB
    
    # Should not exceed 2x the theoretical minimum
    assert memory_increase < 10_000_000 * 3 * 8 * 2 / 1024 / 1024  # 2x(points * 3 coords * 8 bytes)
```
**Acceptance Criteria**: Memory usage stays within specified limits from NFR-P1.3

##### Minor Task 3.3.2 (Implementation): Implement Memory Optimization
**Task**: Optimize memory usage for large file processing
**What to do**: Implement memory-efficient processing in point cloud operations
**Implementation Level**:
- Streaming processing for large files
- Memory pool management
- Efficient NumPy array operations
- Garbage collection optimization
**Code Estimate**: ~80 lines
**How to Test**: Use tests from 3.3.1 - all memory tests must pass
**Acceptance Criteria**: Memory usage meets NFR-P1.3 requirements

### Major Task 4.0: Coordinate System Transformation

#### Subtask 4.1: PyProj Integration and Core Transformations

##### Minor Task 4.1.1 (Test First): Write WGS84 to UTM Transformation Tests
**Task**: Create comprehensive tests for coordinate transformation accuracy
**What to do**: Create `backend/tests/test_coordinate_transformation.py` with known control points
**Implementation Level**:
- Test at least 10 known WGS84/UTM coordinate pairs with <0.1m tolerance
- Test edge cases: poles, date line, UTM zone boundaries
- Test batch transformation of 1000+ points for performance
**Code Estimate**: ~120 lines of test code
**How to Test**:
```python
def test_wgs84_to_utm_accuracy():
    # Test known control point: Statue of Liberty
    wgs84_lat, wgs84_lon = 40.6892, -74.0445
    expected_utm_x, expected_utm_y = 583960.0, 4507523.0
    result = transform_wgs84_to_utm(wgs84_lat, wgs84_lon)
    assert abs(result.x - expected_utm_x) < 0.1
    assert abs(result.y - expected_utm_y) < 0.1

def test_utm_zone_detection():
    # Test automatic UTM zone detection
    lat, lon = 40.7128, -74.0060  # New York
    zone = get_utm_zone(lat, lon)
    assert zone == 18  # UTM Zone 18N
    
def test_batch_transformation_performance():
    # Test performance with large batches
    coords = [(40.0 + i*0.001, -74.0 + i*0.001) for i in range(1000)]
    start_time = time.time()
    results = transform_wgs84_to_utm_batch(coords)
    elapsed = time.time() - start_time
    assert elapsed < 1.0  # Must complete in <1 second
    assert len(results) == 1000
```
**Acceptance Criteria**: All coordinate transformation tests pass with <0.1m accuracy

##### Minor Task 4.1.2 (Implementation): Implement WGS84 to UTM Transformation
**Task**: Create coordinate transformation service using PyProj
**What to do**: Create transformation functions in `backend/app/services/coord_transformer.py`
**Implementation Level**:
- Use PyProj for WGS84 to UTM transformations
- Implement automatic UTM zone detection
- Support batch transformations for performance
- Handle edge cases and validation
**Code Estimate**: ~150 lines
**How to Test**: Use tests from 4.1.1 - all transformation accuracy tests must pass
**Acceptance Criteria**: Transformations accurate to <0.1m, performance meets requirements

##### Minor Task 4.1.3 (Test First): Write Rotation and Scaling Tests
**Task**: Create tests for rotation matrix and scaling operations
**What to do**: Test 3D rotation and scaling transformations with known results
**Implementation Level**:
- Test rotation angles: 0°, 45°, 90°, 180°, 270°, 359°
- Test scaling factors: 0.1, 0.5, 1.0, 2.0, 10.0
- Test combined transformations (rotation + scaling + translation)
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_rotation_transformation():
    # Test 90-degree rotation
    points = np.array([[1, 0, 0], [0, 1, 0]])
    rotated = apply_rotation_z(points, 90)  # 90 degrees
    expected = np.array([[0, 1, 0], [-1, 0, 0]])
    np.testing.assert_allclose(rotated[:, :2], expected[:, :2], atol=1e-10)

def test_scaling_transformation():
    points = np.array([[1, 1, 1], [2, 2, 2]])
    scaled = apply_scaling(points, 2.0)
    expected = np.array([[2, 2, 2], [4, 4, 4]])
    np.testing.assert_allclose(scaled, expected, atol=1e-10)
```
**Acceptance Criteria**: Rotation tests accurate to 1e-10 precision, scaling maintains proportions

##### Minor Task 4.1.4 (Implementation): Implement Rotation and Scaling
**Task**: Create 3D rotation and scaling transformation functions
**What to do**: Implement transformation matrices using NumPy
**Implementation Level**:
- 3D rotation matrices around Z-axis
- Uniform and non-uniform scaling
- Efficient matrix operations for large point sets
- Transformation composition and order
**Code Estimate**: ~100 lines
**How to Test**: Use tests from 4.1.3 - all rotation and scaling tests must pass
**Acceptance Criteria**: All transformations maintain numerical precision

##### Minor Task 4.1.5 (Test First): Write Transformation Pipeline Tests
**Task**: Create tests for complete coordinate transformation pipeline
**What to do**: Test end-to-end transformation from PLY local coordinates to UTM
**Implementation Level**:
- Test pipeline with multiple PLY files (different scales/orientations)
- Test inverse transformation accuracy
- Test pipeline composition order
- Test transformation metadata tracking
**Code Estimate**: ~100 lines of test code
**How to Test**:
```python
def test_transformation_pipeline():
    # Test complete pipeline: local PLY -> scaled -> rotated -> translated -> UTM
    local_points = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    pipeline = TransformationPipeline(
        anchor_lat=40.7128, anchor_lon=-74.0060,
        rotation_degrees=45.0, scale_factor=2.0
    )
    
    utm_points = pipeline.transform_to_utm(local_points)
    recovered_points = pipeline.inverse_transform(utm_points)
    
    np.testing.assert_allclose(local_points, recovered_points, atol=1e-6)
    
def test_pipeline_consistency():
    # Test that different transformation orders give same result
    points = generate_test_points(100)
    pipeline1 = create_pipeline_method1(params)
    pipeline2 = create_pipeline_method2(params)
    
    result1 = pipeline1.transform(points)
    result2 = pipeline2.transform(points)
    
    np.testing.assert_allclose(result1, result2, atol=1e-8)
```
**Acceptance Criteria**: Pipeline maintains round-trip accuracy within 1e-6

##### Minor Task 4.1.6 (Implementation): Create Transformation Pipeline
**Task**: Implement complete coordinate transformation system
**What to do**: Create unified pipeline combining all transformations
**Implementation Level**:
- Pipeline class managing transformation sequence
- Transformation parameter validation
- Inverse transformation capability
- Performance optimization for large datasets
**Code Estimate**: ~200 lines
**How to Test**: Use tests from 4.1.5 - all pipeline tests must pass
**Acceptance Criteria**: Pipeline maintains accuracy, supports all transformation requirements

#### Subtask 4.2: Surface Registration and Alignment

##### Minor Task 4.2.1 (Test First): Write Surface Alignment Tests
**Task**: Create tests for multi-surface alignment accuracy
**What to do**: Test alignment of multiple surfaces using reference points
**Implementation Level**:
- Test alignment with known control points
- Test alignment accuracy with different reference points
- Test alignment with noisy data
- Test alignment validation metrics
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_surface_alignment_accuracy():
    # Create two surfaces with known transformation
    surface1 = create_test_surface(100, 100)
    transform_params = TransformParams(rotation=30, scale=1.5, offset=[10, 20])
    surface2 = apply_known_transform(surface1, transform_params)
    
    # Test alignment recovery
    estimated_params = align_surfaces(surface1, surface2, method='icp')
    
    assert abs(estimated_params.rotation - 30) < 0.1
    assert abs(estimated_params.scale - 1.5) < 0.01
    np.testing.assert_allclose(estimated_params.offset, [10, 20], atol=0.1)
```
**Acceptance Criteria**: Surface alignment accurate within survey tolerances

##### Minor Task 4.2.2 (Implementation): Implement Surface Alignment
**Task**: Create surface alignment and registration algorithms
**What to do**: Implement alignment functions in coordinate transformation module
**Implementation Level**:
- Reference point-based alignment
- ICP (Iterative Closest Point) refinement
- Alignment quality assessment
- Robust alignment with outlier rejection
**Code Estimate**: ~150 lines
**How to Test**: Use tests from 4.2.1 - all alignment tests must pass
**Acceptance Criteria**: Alignment achieves required accuracy for multi-surface analysis

### Major Task 5.0: Volume Calculation Engine

#### Subtask 5.1: Delaunay Triangulation and TIN Creation

##### Minor Task 5.1.1 (Test First): Write Delaunay Triangulation Tests
**Task**: Create comprehensive tests for triangulation correctness and performance
**What to do**: Create `backend/tests/test_triangulation.py` with synthetic and real-world test cases
**Implementation Level**:
- Test triangulation of simple geometries (square, triangle, circle)
- Test edge cases: collinear points, duplicate points, single points
- Test large datasets (10k, 100k, 1M points) for performance
- Test triangulation quality metrics
**Code Estimate**: ~150 lines of test code
**How to Test**:
```python
def test_delaunay_triangulation_square():
    # Test triangulation of unit square
    points = np.array([[0,0], [1,0], [1,1], [0,1]])
    triangulation = create_delaunay_triangulation(points)
    assert len(triangulation.simplices) == 2  # Square should have 2 triangles
    assert validate_triangulation_quality(triangulation) > 0.5  # Quality metric
    
def test_triangulation_performance():
    points = generate_random_points_2d(100000)
    start_time = time.time()
    triangulation = create_delaunay_triangulation(points)
    elapsed = time.time() - start_time
    assert elapsed < 30.0  # Must complete in <30 seconds
    assert len(triangulation.simplices) > 0

def test_triangulation_edge_cases():
    # Test collinear points
    collinear_points = np.array([[0,0], [1,0], [2,0]])
    triangulation = create_delaunay_triangulation(collinear_points)
    assert triangulation.simplices.size == 0  # No valid triangles from collinear points
    
    # Test duplicate points
    duplicate_points = np.array([[0,0], [1,0], [1,1], [0,0]])
    triangulation = create_delaunay_triangulation(duplicate_points)
    unique_points = len(np.unique(triangulation.points.view(np.void), axis=0))
    assert unique_points == 3  # Duplicates removed
```
**Acceptance Criteria**: Triangulation produces valid non-overlapping triangles, performance meets 100k points in <30 seconds

##### Minor Task 5.1.2 (Implementation): Implement Delaunay Triangulation
**Task**: Create triangulation service using SciPy spatial
**What to do**: Write triangulation functions in `backend/app/services/triangulation.py`
**Implementation Level**:
- Use scipy.spatial.Delaunay for triangulation
- Handle edge cases (collinear, duplicate points)
- Optimize for large point sets
- Quality validation and metrics
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 5.1.1 - all triangulation tests must pass
**Acceptance Criteria**: Triangulation handles all test cases, meets performance requirements

##### Minor Task 5.1.3 (Test First): Write TIN Interpolation Tests
**Task**: Create tests for Z-value interpolation from triangulated surfaces
**What to do**: Test interpolation accuracy for various surface types
**Implementation Level**:
- Test interpolation on flat planes (analytical solution)
- Test interpolation on curved surfaces
- Test edge cases (points outside convex hull)
- Test interpolation performance with large TINs
**Code Estimate**: ~100 lines of test code
**How to Test**:
```python
def test_interpolation_flat_plane():
    # Create flat plane at z=5
    points_2d = np.array([[0,0], [1,0], [1,1], [0,1]])
    z_values = np.array([5, 5, 5, 5])
    points_3d = np.column_stack([points_2d, z_values])
    
    tin = create_tin_from_points(points_3d)
    
    # Test interpolation at interior point
    query_point = np.array([0.5, 0.5])
    interpolated_z = interpolate_z_from_tin(tin, query_point)
    assert abs(interpolated_z - 5.0) < 1e-10

def test_interpolation_sloped_plane():
    # Create sloped plane: z = x + y
    points_2d = np.array([[0,0], [1,0], [1,1], [0,1]])
    z_values = np.array([0, 1, 2, 1])  # z = x + y
    points_3d = np.column_stack([points_2d, z_values])
    
    tin = create_tin_from_points(points_3d)
    
    # Test interpolation accuracy
    query_point = np.array([0.5, 0.5])
    expected_z = 0.5 + 0.5  # 1.0
    interpolated_z = interpolate_z_from_tin(tin, query_point)
    assert abs(interpolated_z - expected_z) < 1e-6

def test_interpolation_outside_hull():
    points_3d = np.array([[0,0,0], [1,0,1], [0,1,1]])
    tin = create_tin_from_points(points_3d)
    
    # Query point outside convex hull
    query_point = np.array([2.0, 2.0])
    interpolated_z = interpolate_z_from_tin(tin, query_point)
    assert np.isnan(interpolated_z)  # Should return NaN for points outside
```
**Acceptance Criteria**: Interpolation accurate for analytical cases, handles edge cases correctly

##### Minor Task 5.1.4 (Implementation): Implement TIN Interpolation
**Task**: Create Z-value interpolation algorithms for triangulated surfaces
**What to do**: Implement interpolation functions using barycentric coordinates
**Implementation Level**:
- Barycentric coordinate interpolation within triangles
- Efficient point location in triangulation
- Handling points outside convex hull
- Vectorized operations for batch interpolation
**Code Estimate**: ~180 lines
**How to Test**: Use tests from 5.1.3 - all interpolation tests must pass
**Acceptance Criteria**: Interpolation meets accuracy requirements, handles all edge cases

#### Subtask 5.2: Volume Calculation Algorithms

##### Minor Task 5.2.1 (Test First): Write Volume Calculation Tests for Simple Geometries
**Task**: Create tests using known geometric shapes with analytical solutions
**What to do**: Test volume calculations against pyramids, prisms, and other simple shapes
**Implementation Level**:
- Test pyramid volume: V = (1/3) * base_area * height
- Test rectangular prism volume: V = length * width * height
- Test truncated pyramid (frustum) volume
- Test volume calculation accuracy within ±1% tolerance
**Code Estimate**: ~120 lines of test code
**How to Test**:
```python
def test_pyramid_volume_calculation():
    # Create pyramid with square base 10x10, height 15
    base_points = create_square_grid(10, 10, z=0)
    apex_point = np.array([[5, 5, 15]])
    bottom_surface = base_points
    top_surface = np.full((len(base_points), 3), [5, 5, 15])  # All points at apex
    
    calculated_volume = calculate_volume_between_surfaces(bottom_surface, top_surface)
    expected_volume = (10 * 10 * 15) / 3  # 500 cubic units
    
    relative_error = abs(calculated_volume - expected_volume) / expected_volume
    assert relative_error < 0.01  # Within 1% accuracy

def test_rectangular_prism_volume():
    # Create two parallel rectangular surfaces
    bottom = create_rectangular_grid(10, 20, z=0)  # 10x20 base
    top = create_rectangular_grid(10, 20, z=5)     # 5 units higher
    
    calculated_volume = calculate_volume_between_surfaces(bottom, top)
    expected_volume = 10 * 20 * 5  # 1000 cubic units
    
    relative_error = abs(calculated_volume - expected_volume) / expected_volume
    assert relative_error < 0.005  # Within 0.5% for simple shapes

def test_irregular_surface_volume():
    # Create surfaces with known volume difference
    bottom = create_sine_wave_surface(amplitude=2, wavelength=10)
    top = bottom + 3.0  # Uniform 3-unit thickness
    
    calculated_volume = calculate_volume_between_surfaces(bottom, top)
    expected_volume = calculate_surface_area(bottom) * 3.0
    
    relative_error = abs(calculated_volume - expected_volume) / expected_volume
    assert relative_error < 0.02  # Within 2% for irregular shapes
```
**Acceptance Criteria**: Volume calculations accurate within ±1% for geometric primitives

##### Minor Task 5.2.2 (Implementation): Implement Primary Volume Calculation (PyVista)
**Task**: Create volume calculation using PyVista mesh operations (was: Trimesh/Open3D)
**What to do**: Implement volume calculation using PyVista mesh volume and convex hull methods. (Note: PyVista provides native 3D Delaunay triangulation and advanced volume calculation via VTK backend.)
- Convert point clouds to meshes using PyVista
**Implementation Level**:
- Convert point clouds to meshes using PyVista
- Calculate volume differences using mesh operations
- Handle surface interpolation and mesh quality
- Optimize for large datasets
**Code Estimate**: ~200 lines
**How to Test**: Use tests from 5.2.1 - all geometric volume tests must pass
**Acceptance Criteria**: Volume calculations meet accuracy requirements for all test geometries

##### Minor Task 5.2.3 (Test First): Write Secondary Volume Calculation Tests
**Task**: Create tests for alternative volume calculation method (cross-validation)
**What to do**: Test prism-based volume calculation for validation
**Implementation Level**:
- Test volume calculation using vertical prisms
- Test cross-validation between primary and secondary methods
- Test method agreement within specified tolerances
- Test performance comparison between methods
**Code Estimate**: ~100 lines of test code
**How to Test**:
```python
def test_prism_method_volume_calculation():
    # Create simple test surfaces
    bottom_surface = create_planar_surface(10, 10, z=0)
    top_surface = create_planar_surface(10, 10, z=5)
    
    # Calculate using prism method
    prism_volume = calculate_volume_prism_method(bottom_surface, top_surface)
    expected_volume = 10 * 10 * 5  # 500 cubic units
    
    relative_error = abs(prism_volume - expected_volume) / expected_volume
    assert relative_error < 0.01

def test_volume_method_cross_validation():
    # Generate moderately complex surface
    bottom = create_random_surface(50, 50, roughness=0.5)
    top = bottom + generate_varying_thickness(50, 50, mean=3, std=0.5)
    
    # Calculate using both methods
    pyvista_volume = calculate_volume_pyvista_method(bottom, top)
    prism_volume = calculate_volume_prism_method(bottom, top)
    
    # Methods should agree within tolerance
    relative_difference = abs(pyvista_volume - prism_volume) / pyvista_volume
    assert relative_difference < 0.005  # Methods agree within 0.5%
```
**Acceptance Criteria**: Secondary method provides validation within 0.5% of primary method

##### Minor Task 5.2.4 (Implementation): Implement Secondary Volume Calculation
**Task**: Create alternative volume calculation for cross-validation
**What to do**: Implement prism-based volume calculation method
**Implementation Level**:
- Triangulate lower surface into triangular prisms
- Calculate volume of each prism to upper surface
- Sum all prism volumes for total difference
- Optimize calculation for large triangulations
**Code Estimate**: ~180 lines
**How to Test**: Use tests from 5.2.3 - all cross-validation tests must pass
**Acceptance Criteria**: Secondary method validates primary calculations within tolerance

##### Minor Task 5.2.5 (Test First): Write Unit Conversion Tests
**Task**: Create tests for volume unit conversions (cubic feet to cubic yards)
**What to do**: Test accurate conversion between measurement units
**Implementation Level**:
- Test cubic feet to cubic yards conversion
- Test conversion accuracy with various input ranges
- Test conversion with edge cases (zero, very large numbers)
- Test round-trip conversion accuracy
**Code Estimate**: ~40 lines of test code
**How to Test**:
```python
def test_cubic_feet_to_yards_conversion():
    # Test known conversion: 27 cubic feet = 1 cubic yard
    volume_cf = 27.0
    volume_cy = convert_cubic_feet_to_yards(volume_cf)
    assert abs(volume_cy - 1.0) < 1e-10
    
    # Test various values
    test_values_cf = [1.0, 27.0, 54.0, 100.0, 1000.0]
    expected_cy = [v / 27.0 for v in test_values_cf]
    
    for cf, expected in zip(test_values_cf, expected_cy):
        converted = convert_cubic_feet_to_yards(cf)
        assert abs(converted - expected) < 1e-10

def test_conversion_edge_cases():
    # Test zero
    assert convert_cubic_feet_to_yards(0.0) == 0.0
    
    # Test very large numbers
    large_value = 1e9
    converted = convert_cubic_feet_to_yards(large_value)
    expected = large_value / 27.0
    assert abs(converted - expected) / expected < 1e-10
```
**Acceptance Criteria**: All unit conversions accurate to machine precision

##### Minor Task 5.2.6 (Implementation): Implement Unit Conversions
**Task**: Create unit conversion functions for volume calculations
**What to do**: Implement conversion functions with proper validation
**Implementation Level**:
- Cubic feet to cubic yards conversion
- Input validation for negative values
- Precision preservation for large numbers
- Documentation of conversion factors
**Code Estimate**: ~30 lines
**How to Test**: Use tests from 5.2.5 - all conversion tests must pass
**Acceptance Criteria**: Conversions maintain precision, handle all edge cases

#### Subtask 5.3: Thickness Calculation Engine

##### Minor Task 5.3.1 (Test First): Write Point-to-Surface Distance Tests
**Task**: Create tests for vertical distance calculations between points and surfaces
**What to do**: Create `backend/tests/test_thickness_calculation.py` with distance scenarios
**Implementation Level**:
- Test distance from point to flat plane (analytical solution)
- Test distance to sloped plane with known geometry
- Test interpolation accuracy for points between grid vertices
- Test performance with large point sets
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_point_to_flat_plane_distance():
    # Create flat plane at z=10
    plane_points = create_planar_surface(10, 10, z=10)
    plane_tin = create_tin_from_points(plane_points)
    
    # Test point 5 units above plane center
    test_point = np.array([5, 5, 15])
    distance = calculate_point_to_surface_distance(test_point, plane_tin)
    assert abs(distance - 5.0) < 1e-10

def test_point_to_sloped_plane_distance():
    # Create sloped plane: z = 0.5*x + 0.3*y + 10
    x, y = np.meshgrid(np.linspace(0, 10, 11), np.linspace(0, 10, 11))
    z = 0.5*x + 0.3*y + 10
    plane_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    plane_tin = create_tin_from_points(plane_points)
    
    # Test point above known location
    test_x, test_y = 5.0, 5.0
    expected_surface_z = 0.5*test_x + 0.3*test_y + 10  # 12.5
    test_point = np.array([test_x, test_y, 15.0])
    distance = calculate_point_to_surface_distance(test_point, plane_tin)
    expected_distance = 15.0 - expected_surface_z  # 2.5
    assert abs(distance - expected_distance) < 1e-6

def test_batch_distance_calculation():
    surface_tin = create_test_surface_tin()
    test_points = generate_test_points_above_surface(1000)
    
    start_time = time.time()
    distances = calculate_batch_point_to_surface_distances(test_points, surface_tin)
    elapsed = time.time() - start_time
    
    assert len(distances) == 1000
    assert elapsed < 1.0  # Must complete 1000 points in <1 second
    assert np.all(distances >= 0)  # All distances should be positive
```
**Acceptance Criteria**: Distance calculations accurate to 1e-10 for analytical cases

##### Minor Task 5.3.2 (Implementation): Implement Point-to-Surface Distance
**Task**: Create distance calculation service for thickness analysis
**What to do**: Implement distance functions in thickness calculation module
**Implementation Level**:
- Vertical distance from point to TIN surface
- Efficient interpolation using TIN structure
- Batch processing for multiple points
- Handling points outside surface boundaries
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 5.3.1 - all distance calculation tests must pass
**Acceptance Criteria**: Distance calculations meet accuracy and performance requirements

##### Minor Task 5.3.3 (Test First): Write Thickness Sampling Strategy Tests
**Task**: Create tests for systematic thickness sampling across analysis area
**What to do**: Test sampling point generation and distribution quality
**Implementation Level**:
- Test uniform grid sampling
- Test adaptive sampling based on surface complexity
- Test sampling density control
- Test sampling within irregular boundaries
**Code Estimate**: ~60 lines of test code
**How to Test**:
```python
def test_uniform_grid_sampling():
    # Test regular grid generation
    boundary = [(0, 0), (10, 0), (10, 10), (0, 10)]
    sample_spacing = 1.0
    
    sample_points = generate_uniform_sample_points(boundary, sample_spacing)
    
    # Should generate 11x11 = 121 points
    assert len(sample_points) == 121
    
    # Check point spacing
    x_coords = np.unique(sample_points[:, 0])
    y_coords = np.unique(sample_points[:, 1])
    assert len(x_coords) == 11
    assert len(y_coords) == 11
    
    # Check spacing accuracy
    x_spacing = np.diff(x_coords)
    assert np.allclose(x_spacing, sample_spacing, atol=1e-10)

def test_adaptive_sampling():
    # Test adaptive sampling based on surface roughness
    rough_surface = create_rough_test_surface()
    smooth_surface = create_smooth_test_surface()
    
    rough_samples = generate_adaptive_sample_points(rough_surface, max_spacing=2.0)
    smooth_samples = generate_adaptive_sample_points(smooth_surface, max_spacing=2.0)
    
    # Rough surface should generate more sample points
    assert len(rough_samples) > len(smooth_samples)
```
**Acceptance Criteria**: Sampling generates appropriate point distributions for analysis

##### Minor Task 5.3.4 (Implementation): Implement Thickness Sampling
**Task**: Create sampling strategies for thickness calculation
**What to do**: Implement sampling functions for systematic thickness analysis
**Implementation Level**:
- Regular grid sampling with user-defined spacing
- Adaptive sampling based on surface complexity
- Boundary-aware sampling
- Sample point optimization for accuracy vs performance
**Code Estimate**: ~100 lines
**How to Test**: Use tests from 5.3.3 - all sampling tests must pass
**Acceptance Criteria**: Sampling provides adequate coverage for thickness analysis

##### Minor Task 5.3.5 (Test First): Write Statistical Thickness Analysis Tests
**Task**: Create tests for thickness statistics (min, max, average) calculations
**What to do**: Test statistical aggregation accuracy with known distributions
**Implementation Level**:
- Test uniform thickness distributions
- Test linearly varying thickness
- Test random thickness distributions with known statistics
- Test handling of NaN values from interpolation
**Code Estimate**: ~70 lines of test code
**How to Test**:
```python
def test_uniform_thickness_statistics():
    # Create surfaces with uniform 3-unit thickness
    bottom = create_planar_surface(10, 10, z=0)
    top = create_planar_surface(10, 10, z=3)
    
    thickness_values = calculate_thickness_at_sample_points(bottom, top, sample_spacing=1.0)
    stats = calculate_thickness_statistics(thickness_values)
    
    assert abs(stats.min - 3.0) < 1e-10
    assert abs(stats.max - 3.0) < 1e-10
    assert abs(stats.average - 3.0) < 1e-10
    assert abs(stats.std_dev - 0.0) < 1e-10

def test_varying_thickness_statistics():
    # Create surfaces with known thickness variation
    bottom = create_planar_surface(10, 10, z=0)
    top = create_linearly_varying_surface(10, 10, z_min=2, z_max=6)  # 2-6 unit thickness
    
    thickness_values = calculate_thickness_at_sample_points(bottom, top, sample_spacing=0.5)
    stats = calculate_thickness_statistics(thickness_values)
    
    assert abs(stats.min - 2.0) < 0.1
    assert abs(stats.max - 6.0) < 0.1
    assert abs(stats.average - 4.0) < 0.1  # Average should be (2+6)/2 = 4

def test_thickness_with_nan_values():
    # Test handling of NaN values (points outside surface)
    thickness_values = np.array([1.0, 2.0, np.nan, 3.0, np.nan, 4.0])
    stats = calculate_thickness_statistics(thickness_values)
    
    # Should ignore NaN values
    assert abs(stats.min - 1.0) < 1e-10
    assert abs(stats.max - 4.0) < 1e-10
    assert abs(stats.average - 2.5) < 1e-10  # (1+2+3+4)/4 = 2.5
    assert stats.valid_count == 4
```
**Acceptance Criteria**: Statistical calculations mathematically correct for all test distributions

##### Minor Task 5.3.6 (Implementation): Implement Statistical Thickness Analysis
**Task**: Create comprehensive thickness statistics calculator
**What to do**: Implement statistical functions for thickness distribution analysis
**Implementation Level**:
- Calculate min, max, mean, median, standard deviation
- Handle NaN values from interpolation outside boundaries
- Calculate percentiles and distribution statistics
- Generate thickness distribution histograms
**Code Estimate**: ~80 lines
**How to Test**: Use tests from 5.3.5 - all statistical calculation tests must pass
**Acceptance Criteria**: Statistics mathematically correct, properly handle edge cases

#### Subtask 5.4: Compaction Rate Calculations

##### Minor Task 5.4.1 (Test First): Write Compaction Rate Formula Tests
**Task**: Create tests for compaction rate calculation accuracy
**What to do**: Test compaction rate formula with known tonnage and volume values
**Implementation Level**:
- Test basic formula: (tonnage * 2000 lbs/ton) / volume_cubic_yards
- Test edge cases: zero volume, negative volume
- Test unit consistency and conversion accuracy
- Test precision with various input ranges
**Code Estimate**: ~50 lines of test code
**How to Test**:
```python
def test_compaction_rate_calculation():
    # Test known values
    tonnage = 100.0  # tons
    volume_cubic_yards = 50.0  # cubic yards
    
    compaction_rate = calculate_compaction_rate(tonnage, volume_cubic_yards)
    expected_rate = (100.0 * 2000) / 50.0  # 4000 lbs/cubic yard
    
    assert abs(compaction_rate - expected_rate) < 1e-10

def test_compaction_rate_edge_cases():
    # Test zero volume (should raise error or return special value)
    with pytest.raises(ValueError):
        calculate_compaction_rate(100.0, 0.0)
    
    # Test negative volume (should raise error)
    with pytest.raises(ValueError):
        calculate_compaction_rate(100.0, -10.0)
    
    # Test zero tonnage (should return 0)
    rate = calculate_compaction_rate(0.0, 50.0)
    assert rate == 0.0

def test_compaction_rate_precision():
    # Test with various input ranges
    test_cases = [
        (1.5, 0.8),    # Small values
        (1000, 500),   # Large values
        (0.001, 0.0005)  # Very small values
    ]
    
    for tonnage, volume in test_cases:
        rate = calculate_compaction_rate(tonnage, volume)
        expected = (tonnage * 2000) / volume
        relative_error = abs(rate - expected) / expected
        assert relative_error < 1e-12  # High precision required
```
**Acceptance Criteria**: Compaction rate calculations accurate to machine precision

##### Minor Task 5.4.2 (Implementation): Implement Compaction Rate Calculation
**Task**: Create compaction rate calculation functions
**What to do**: Implement formula with proper validation and error handling
**Implementation Level**:
- Basic compaction rate formula implementation
- Input validation for physical constraints
- Error handling for edge cases
- Support for optional tonnage (return None or "--")
**Code Estimate**: ~40 lines
**How to Test**: Use tests from 5.4.1 - all compaction rate tests must pass
**Acceptance Criteria**: All calculations accurate, proper error handling

##### Minor Task 5.4.3 (Test First): Write Multi-Layer Compaction Analysis Tests
**Task**: Create tests for compaction analysis across multiple surface layers
**What to do**: Test compaction rate calculation for sequential layers
**Implementation Level**:
- Test compaction rates for surface pairs (0-1, 1-2, 2-3)
- Test handling of missing tonnage data
- Test compaction rate trends and validation
- Test result formatting and display
**Code Estimate**: ~60 lines of test code
**How to Test**:
```python
def test_multi_layer_compaction_analysis():
    # Test 3 surfaces with known volumes and tonnages
    volumes = [150.0, 200.0, 175.0]  # cubic yards between layers
    tonnages = [300.0, 400.0, None]  # tons (last layer no tonnage data)
    
    compaction_rates = calculate_multi_layer_compaction_rates(volumes, tonnages)
    
    # Layer 0-1: (300 * 2000) / 150 = 4000 lbs/cy
    assert abs(compaction_rates[0] - 4000.0) < 1e-10
    
    # Layer 1-2: (400 * 2000) / 200 = 4000 lbs/cy
    assert abs(compaction_rates[1] - 4000.0) < 1e-10
    
    # Layer 2-3: No tonnage data, should be None or "--"
    assert compaction_rates[2] is None or compaction_rates[2] == "--"

def test_compaction_rate_validation():
    # Test realistic compaction rate ranges
    typical_rates = [1500, 2000, 2500, 3000]  # lbs/cy
    for rate in typical_rates:
        assert validate_compaction_rate(rate) is True
    
    # Test unrealistic rates (should generate warnings)
    unrealistic_rates = [100, 10000]  # Too low or too high
    for rate in unrealistic_rates:
        warnings = validate_compaction_rate(rate)
        assert len(warnings) > 0
```
**Acceptance Criteria**: Multi-layer analysis handles all scenarios correctly

##### Minor Task 5.4.4 (Implementation): Implement Multi-Layer Compaction Analysis
**Task**: Create compaction analysis for multiple surface layers
**What to do**: Implement functions for analyzing compaction across all layers
**Implementation Level**:
- Sequential layer compaction rate calculation
- Missing data handling (optional tonnage)
- Compaction rate validation and quality checks
- Result formatting for display
**Code Estimate**: ~70 lines
**How to Test**: Use tests from 5.4.3 - all multi-layer tests must pass
**Acceptance Criteria**: Complete analysis handles all layer combinations

## Phase 3: API Development & Frontend Integration (Weeks 9-12)

### Major Task 6.0: Backend API Implementation

#### Subtask 6.1: Core API Endpoints

##### Minor Task 6.1.1 (Test First): Write File Upload API Tests
**Task**: Create comprehensive tests for file upload endpoints
**What to do**: Create `backend/tests/test_upload_api.py` with upload validation tests
**Implementation Level**:
- Test valid PLY file uploads (ASCII and binary)
- Test invalid file rejection (wrong format, corrupted, oversized)
- Test concurrent upload handling
- Test upload progress tracking
- Test file cleanup and resource management
**Code Estimate**: ~120 lines of test code
**How to Test**:
```python
def test_upload_valid_ply_file():
    client = TestClient(app)
    
    # Create test PLY file
    ply_content = create_test_ply_content(1000)  # 1000 vertices
    
    with NamedTemporaryFile(suffix=".ply", delete=False) as tmp_file:
        tmp_file.write(ply_content)
        tmp_file.flush()
        
        with open(tmp_file.name, "rb") as f:
            response = client.post("/api/upload", files={"file": ("test.ply", f, "application/octet-stream")})
    
    assert response.status_code == 200
    data = response.json()
    assert "file_id" in data
    assert data["success"] is True
    assert data["filename"] == "test.ply"

def test_upload_invalid_file_type():
    client = TestClient(app)
    
    # Create invalid file
    invalid_content = b"This is not a PLY file"
    
    with NamedTemporaryFile(suffix=".txt", delete=False) as tmp_file:
        tmp_file.write(invalid_content)
        tmp_file.flush()
        
        with open(tmp_file.name, "rb") as f:
            response = client.post("/api/upload", files={"file": ("test.txt", f, "text/plain")})
    
    assert response.status_code == 400
    assert "invalid file type" in response.json()["detail"].lower()

def test_upload_oversized_file():
    client = TestClient(app)
    
    # Create file larger than 2GB limit (mock large file)
    with patch('app.routes.surfaces.get_file_size') as mock_size:
        mock_size.return_value = 3 * 1024 * 1024 * 1024  # 3GB
        
        with NamedTemporaryFile(suffix=".ply") as tmp_file:
            tmp_file.write(b"ply\nformat ascii 1.0\nelement vertex 1\n")
            tmp_file.flush()
            
            with open(tmp_file.name, "rb") as f:
                response = client.post("/api/upload", files={"file": ("large.ply", f, "application/octet-stream")})
    
    assert response.status_code == 413  # Payload Too Large
```
**Acceptance Criteria**: All upload scenarios tested, proper error handling validated

##### Minor Task 6.1.2 (Implementation): Create File Upload Endpoint
**Task**: Implement FastAPI endpoint for PLY file uploads
**What to do**: Create upload handling in `backend/app/routes/surfaces.py`
**Implementation Level**:
- File validation (type, size, format)
- Secure file storage with unique IDs
- Progress tracking capability
- Comprehensive error handling
**Code Estimate**: ~100 lines
**How to Test**: Use tests from 6.1.1 - all upload tests must pass
**Acceptance Criteria**: Upload endpoint handles all test scenarios securely

##### Minor Task 6.1.3 (Test First): Write Analysis Configuration API Tests
**Task**: Create tests for analysis parameter validation and configuration
**What to do**: Test analysis setup endpoint with various parameter combinations
**Implementation Level**:
- Test valid analysis configurations
- Test parameter validation (coordinates, transformations)
- Test boundary definition validation
- Test surface combination validation
**Code Estimate**: ~100 lines of test code
**How to Test**:
```python
def test_valid_analysis_configuration():
    client = TestClient(app)
    
    config = {
        "surfaces": [
            {
                "file_id": "test-file-1",
                                "anchor_lat": 40.7128,
                "anchor_lon": -74.0060,
                "rotation_degrees": 45.0,
                "scale_factor": 1.0
            },
            {
                "file_id": "test-file-2",
                "anchor_lat": 40.7130,
                "anchor_lon": -74.0062,
                "rotation_degrees": 45.0,
                "scale_factor": 1.0
            }
        ],
        "boundary": {
            "corners": [
                [40.7120, -74.0070],
                [40.7140, -74.0070],
                [40.7140, -74.0050],
                [40.7120, -74.0050]
            ]
        },
        "tonnage_per_layer": [1000.0],
        "base_surface_offset": None
    }
    
    response = client.post("/api/analysis/configure", json=config)
    assert response.status_code == 200
    data = response.json()
    assert "analysis_id" in data
    assert data["success"] is True

def test_invalid_coordinate_validation():
    client = TestClient(app)
    
    config = {
        "surfaces": [{
            "file_id": "test-file-1",
            "anchor_lat": 91.0,  # Invalid latitude > 90
            "anchor_lon": -74.0060,
            "rotation_degrees": 45.0,
            "scale_factor": 1.0
        }],
        "boundary": {"corners": [[40.7, -74.0], [40.8, -74.0], [40.8, -73.9], [40.7, -73.9]]}
    }
    
    response = client.post("/api/analysis/configure", json=config)
    assert response.status_code == 422  # Validation Error
    assert "latitude" in response.json()["detail"][0]["msg"].lower()
```
**Acceptance Criteria**: Configuration validation catches all invalid parameter combinations

##### Minor Task 6.1.4 (Implementation): Create Analysis Configuration Endpoint
**Task**: Implement analysis configuration and validation endpoint
**What to do**: Create configuration endpoint with comprehensive validation
**Implementation Level**:
- Parameter validation using Pydantic models
- Coordinate system validation
- Surface combination validation
- Configuration storage and retrieval
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 6.1.3 - all configuration tests must pass
**Acceptance Criteria**: Configuration endpoint validates all parameters correctly

##### Minor Task 6.1.5 (Test First): Write Analysis Execution API Tests
**Task**: Create tests for analysis execution and progress tracking
**What to do**: Test analysis execution with background processing
**Implementation Level**:
- Test analysis execution initiation
- Test progress tracking and status updates
- Test result generation and retrieval
- Test analysis cancellation
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_analysis_execution():
    client = TestClient(app)
    
    # Assume analysis was configured with ID "test-analysis-123"
    analysis_id = "test-analysis-123"
    
    response = client.post(f"/api/analysis/{analysis_id}/execute")
    assert response.status_code == 202  # Accepted for processing
    data = response.json()
    assert data["status"] == "started"
    assert "estimated_duration" in data

def test_analysis_progress_tracking():
    client = TestClient(app)
    analysis_id = "test-analysis-123"
    
    # Start analysis
    client.post(f"/api/analysis/{analysis_id}/execute")
    
    # Check progress
    response = client.get(f"/api/analysis/{analysis_id}/status")
    assert response.status_code == 200
    data = response.json()
    assert "progress_percent" in data
    assert "current_step" in data
    assert data["progress_percent"] >= 0
    assert data["progress_percent"] <= 100

def test_analysis_cancellation():
    client = TestClient(app)
    analysis_id = "test-analysis-123"
    
    # Start analysis
    client.post(f"/api/analysis/{analysis_id}/execute")
    
    # Cancel analysis
    response = client.post(f"/api/analysis/{analysis_id}/cancel")
    assert response.status_code == 200
    
    # Check status
    status_response = client.get(f"/api/analysis/{analysis_id}/status")
    assert status_response.json()["status"] == "cancelled"
```
**Acceptance Criteria**: Analysis execution provides proper progress tracking and control

##### Minor Task 6.1.6 (Implementation): Create Analysis Execution Endpoint
**Task**: Implement analysis execution with background processing
**What to do**: Create execution endpoint with progress tracking
**Implementation Level**:
- Background task management for analysis
- Progress tracking and status updates
- Result caching and retrieval
- Analysis cancellation capability
**Code Estimate**: ~150 lines
**How to Test**: Use tests from 6.1.5 - all execution tests must pass
**Acceptance Criteria**: Analysis execution meets all functional requirements

##### Minor Task 6.1.7 (Test First): Write Results Retrieval API Tests
**Task**: Create tests for analysis results retrieval endpoints
**What to do**: Test result formatting and delivery
**Implementation Level**:
- Test complete results retrieval
- Test partial results (volume only, thickness only)
- Test result formatting and units
- Test result caching behavior
**Code Estimate**: ~70 lines of test code
**How to Test**:
```python
def test_complete_results_retrieval():
    client = TestClient(app)
    analysis_id = "completed-analysis-123"
    
    response = client.get(f"/api/analysis/{analysis_id}/results")
    assert response.status_code == 200
    data = response.json()
    
    # Check required result components
    assert "volume_results" in data
    assert "thickness_results" in data
    assert "compaction_rates" in data
    assert "analysis_metadata" in data
    
    # Validate volume results structure
    volume_results = data["volume_results"]
    assert len(volume_results) > 0
    for layer_result in volume_results:
        assert "layer_name" in layer_result
        assert "volume_cubic_yards" in layer_result
        assert isinstance(layer_result["volume_cubic_yards"], (int, float))

def test_results_not_ready():
    client = TestClient(app)
    analysis_id = "running-analysis-123"
    
    response = client.get(f"/api/analysis/{analysis_id}/results")
    assert response.status_code == 202  # Accepted, not ready yet
    data = response.json()
    assert data["status"] == "processing"
    assert "estimated_completion" in data
```
**Acceptance Criteria**: Results retrieval handles all analysis states correctly

##### Minor Task 6.1.8 (Implementation): Create Results Retrieval Endpoint
**Task**: Implement results retrieval and formatting endpoints
**What to do**: Create results endpoint with proper formatting
**Implementation Level**:
- Complete results formatting
- Unit conversion and validation
- Result caching for performance
- Error handling for incomplete analysis
**Code Estimate**: ~100 lines
**How to Test**: Use tests from 6.1.7 - all results retrieval tests must pass
**Acceptance Criteria**: Results delivered in correct format with proper units

#### Subtask 6.2: Interactive Analysis API

##### Minor Task 6.2.1 (Test First): Write Point Query API Tests
**Task**: Create tests for real-time point-based thickness queries
**What to do**: Test interactive point analysis endpoint
**Implementation Level**:
- Test point coordinate to thickness conversion
- Test query performance (<100ms response time)
- Test spatial interpolation accuracy
- Test batch query handling
**Code Estimate**: ~90 lines of test code
**How to Test**:
```python
def test_point_thickness_query():
    client = TestClient(app)
    analysis_id = "completed-analysis-123"
    
    query = {
        "x": 583960.0,  # UTM coordinates
        "y": 4507523.0,
        "coordinate_system": "utm"
    }
    
    response = client.post(f"/api/analysis/{analysis_id}/point_query", json=query)
    assert response.status_code == 200
    data = response.json()
    
    assert "thickness_layers" in data
    assert isinstance(data["thickness_layers"], list)
    
    for layer in data["thickness_layers"]:
        assert "layer_name" in layer
        assert "thickness_feet" in layer
        assert isinstance(layer["thickness_feet"], (int, float))

def test_point_query_performance():
    client = TestClient(app)
    analysis_id = "completed-analysis-123"
    
    query = {"x": 583960.0, "y": 4507523.0, "coordinate_system": "utm"}
    
    start_time = time.time()
    response = client.post(f"/api/analysis/{analysis_id}/point_query", json=query)
    elapsed = time.time() - start_time
    
    assert response.status_code == 200
    assert elapsed < 0.1  # Must respond in <100ms

def test_batch_point_queries():
    client = TestClient(app)
    analysis_id = "completed-analysis-123"
    
    queries = [
        {"x": 583960.0 + i*10, "y": 4507523.0 + i*10, "coordinate_system": "utm"}
        for i in range(100)
    ]
    
    batch_query = {"points": queries}
    
    start_time = time.time()
    response = client.post(f"/api/analysis/{analysis_id}/batch_point_query", json=batch_query)
    elapsed = time.time() - start_time
    
    assert response.status_code == 200
    assert elapsed < 2.0  # 100 points in <2 seconds
    assert len(response.json()["results"]) == 100
```
**Acceptance Criteria**: Point queries meet performance requirements and accuracy standards

##### Minor Task 6.2.2 (Implementation): Create Point Query Endpoint
**Task**: Implement real-time point-based thickness analysis
**What to do**: Create interactive query endpoint
**Implementation Level**:
- Efficient point-to-surface interpolation
- Coordinate system handling
- Response caching for performance
- Batch query optimization
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 6.2.1 - all point query tests must pass
**Acceptance Criteria**: Point queries respond within 100ms with accurate results

##### Minor Task 6.2.3 (Test First): Write 3D Visualization Data API Tests
**Task**: Create tests for 3D mesh data delivery endpoints
**What to do**: Test mesh data preparation and delivery for frontend
**Implementation Level**:
- Test mesh simplification for different detail levels
- Test data format consistency
- Test large mesh handling and streaming
- Test mesh quality validation
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_3d_mesh_data_retrieval():
    client = TestClient(app)
    analysis_id = "completed-analysis-123"
    surface_id = 0  # First surface
    
    response = client.get(f"/api/analysis/{analysis_id}/surface/{surface_id}/mesh")
    assert response.status_code == 200
    data = response.json()
    
    assert "vertices" in data
    assert "faces" in data
    assert "bounds" in data
    
    # Validate data structure
    vertices = data["vertices"]
    faces = data["faces"]
    assert len(vertices) > 0
    assert len(faces) > 0
    assert all(len(vertex) == 3 for vertex in vertices)  # x, y, z
    assert all(len(face) == 3 for face in faces)  # triangle indices

def test_mesh_simplification_levels():
    client = TestClient(app)
    analysis_id = "completed-analysis-123"
    surface_id = 0
    
    # Test different levels of detail
    for lod in ["high", "medium", "low"]:
        response = client.get(
            f"/api/analysis/{analysis_id}/surface/{surface_id}/mesh",
            params={"level_of_detail": lod}
        )
        assert response.status_code == 200
        mesh_data = response.json()
        
        # Lower LOD should have fewer vertices
        if lod == "low":
            low_vertex_count = len(mesh_data["vertices"])
        elif lod == "high":
            high_vertex_count = len(mesh_data["vertices"])
    
    assert low_vertex_count < high_vertex_count
```
**Acceptance Criteria**: 3D mesh data optimized for frontend rendering performance

##### Minor Task 6.2.4 (Implementation): Create 3D Visualization Data Endpoint
**Task**: Implement mesh data preparation for 3D visualization
**What to do**: Create optimized mesh delivery endpoint
**Implementation Level**:
- Mesh simplification algorithms
- Level-of-detail generation
- Efficient data serialization
- Memory-conscious mesh processing
**Code Estimate**: ~140 lines
**How to Test**: Use tests from 6.2.3 - all 3D data tests must pass
**Acceptance Criteria**: Mesh data optimized for real-time 3D rendering

### Major Task 7.0: Frontend Core Development

#### Subtask 7.1: Wizard Interface Implementation

##### Minor Task 7.1.1 (Test First): Write Wizard Navigation Tests
**Task**: Create tests for step-by-step wizard navigation
**What to do**: Create `frontend/src/components/__tests__/WizardInterface.test.js` with navigation tests
**Implementation Level**:
- Test wizard step progression and validation
- Test form data persistence across steps
- Test error handling and validation feedback
- Test backward navigation without data loss
**Code Estimate**: ~100 lines of test code
**How to Test**:
```javascript
import { render, fireEvent, screen, waitFor } from '@testing-library/react';
import WizardInterface from '../WizardInterface';

describe('WizardInterface', () => {
  test('progresses through all steps correctly', async () => {
    render(<WizardInterface />);
    
    // Step 1: Project Setup
    expect(screen.getByText('Project Setup')).toBeInTheDocument();
    
    // Fill required fields
    fireEvent.change(screen.getByLabelText('Number of Surfaces'), {
      target: { value: '2' }
    });
    
    // Add boundary coordinates
    const latInputs = screen.getAllByLabelText(/Latitude/);
    const lonInputs = screen.getAllByLabelText(/Longitude/);
    
    latInputs.forEach((input, index) => {
      fireEvent.change(input, { target: { value: `40.${7120 + index}` } });
    });
    
    lonInputs.forEach((input, index) => {
      fireEvent.change(input, { target: { value: `-74.${60 + index}` } });
    });
    
    // Proceed to next step
    fireEvent.click(screen.getByText('Next'));
    
    await waitFor(() => {
      expect(screen.getByText('Surface Upload')).toBeInTheDocument();
    });
  });

  test('validates required fields before proceeding', () => {
    render(<WizardInterface />);
    
    // Try to proceed without filling required fields
    fireEvent.click(screen.getByText('Next'));
    
    // Should show validation errors
    expect(screen.getByText(/Number of surfaces is required/)).toBeInTheDocument();
  });

  test('preserves data when navigating backwards', async () => {
    render(<WizardInterface />);
    
    // Fill data in step 1
    fireEvent.change(screen.getByLabelText('Number of Surfaces'), {
      target: { value: '3' }
    });
    
    // Navigate forward then back
    fireEvent.click(screen.getByText('Next'));
    await waitFor(() => screen.getByText('Surface Upload'));
    
    fireEvent.click(screen.getByText('Back'));
    await waitFor(() => screen.getByText('Project Setup'));
    
    // Data should be preserved
    expect(screen.getByDisplayValue('3')).toBeInTheDocument();
  });
});
```
**Acceptance Criteria**: Wizard navigation passes all test scenarios

##### Minor Task 7.1.2 (Implementation): Create Wizard Interface Component
**Task**: Implement 5-step wizard workflow component
**What to do**: Create `frontend/src/components/WizardInterface.js` with complete workflow
**Implementation Level**:
- 5-step wizard as specified in PRD
- Form validation with real-time feedback
- Progress indicators and step navigation
- Data persistence using React Context
**Code Estimate**: ~300 lines
**How to Test**: Use tests from 7.1.1 - all wizard tests must pass
**Acceptance Criteria**: Wizard guides users through complete analysis workflow

##### Minor Task 7.1.3 (Test First): Write Step Components Tests
**Task**: Create tests for individual wizard step components
**What to do**: Test each step component independently
**Implementation Level**:
- Test ProjectSetup component validation
- Test SurfaceUpload component file handling
- Test Georeferencing component parameter entry
- Test MaterialInput component tonnage entry
- Test AnalysisReview component result display
**Code Estimate**: ~150 lines of test code
**How to Test**:
```javascript
describe('ProjectSetup Component', () => {
  test('validates surface count selection', () => {
    render(<ProjectSetup onDataChange={jest.fn()} />);
    
    const surfaceCountSelect = screen.getByLabelText('Number of Surfaces');
    
    // Test valid selections
    fireEvent.change(surfaceCountSelect, { target: { value: '2' } });
    expect(screen.queryByText(/invalid surface count/)).not.toBeInTheDocument();
    
    // Test invalid selection
    fireEvent.change(surfaceCountSelect, { target: { value: '5' } });
    expect(screen.getByText(/surface count must be between 2 and 4/)).toBeInTheDocument();
  });

  test('validates boundary coordinates', () => {
    render(<ProjectSetup onDataChange={jest.fn()} />);
    
    const latInput = screen.getByLabelText('Corner 1 Latitude');
    
    // Test valid latitude
    fireEvent.change(latInput, { target: { value: '40.7128' } });
    expect(screen.queryByText(/invalid latitude/)).not.toBeInTheDocument();
    
    // Test invalid latitude
    fireEvent.change(latInput, { target: { value: '91.0' } });
    expect(screen.getByText(/latitude must be between -90 and 90/)).toBeInTheDocument();
  });
});
```
**Acceptance Criteria**: All step components validate inputs correctly

##### Minor Task 7.1.4 (Implementation): Create Individual Step Components
**Task**: Implement each wizard step as separate component
**What to do**: Create all 5 step components with proper validation
**Implementation Level**:
- ProjectSetup: boundary and surface count input
- SurfaceUpload: file selection and validation
- Georeferencing: transformation parameters
- MaterialInput: tonnage entry
- AnalysisReview: results display and export
**Code Estimate**: ~400 lines total across all components
**How to Test**: Use tests from 7.1.3 - all step component tests must pass
**Acceptance Criteria**: Each step component meets functional requirements

#### Subtask 7.2: File Upload Interface

##### Minor Task 7.2.1 (Test First): Write File Upload Component Tests
**Task**: Create tests for file upload interface functionality
**What to do**: Test drag-and-drop, progress tracking, and validation
**Implementation Level**:
- Test file selection and upload initiation
- Test upload progress display and cancellation
- Test file validation feedback
- Test multiple file management
**Code Estimate**: ~120 lines of test code
**How to Test**:
```javascript
import { render, fireEvent, screen, waitFor } from '@testing-library/react';
import FileUploadComponent from '../FileUploadComponent';

describe('FileUploadComponent', () => {
  test('handles file selection and upload', async () => {
    const mockOnUpload = jest.fn();
    render(<FileUploadComponent onUpload={mockOnUpload} />);
    
    const fileInput = screen.getByLabelText(/choose file/i);
    const file = new File(['test ply content'], 'test.ply', { type: 'application/octet-stream' });
    
    fireEvent.change(fileInput, { target: { files: [file] } });
    
    await waitFor(() => {
      expect(screen.getByText('test.ply')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByText('Upload'));
    
    await waitFor(() => {
      expect(mockOnUpload).toHaveBeenCalledWith(expect.objectContaining({
        name: 'test.ply'
      }));
    });
  });

  test('shows upload progress', async () => {
    // Mock fetch to simulate upload progress
    global.fetch = jest.fn(() => 
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve({ file_id: 'test-123', success: true })
      })
    );

    render(<FileUploadComponent onUpload={jest.fn()} />);
    
    const file = new File(['content'], 'test.ply', { type: 'application/octet-stream' });
    const fileInput = screen.getByLabelText(/choose file/i);
    
    fireEvent.change(fileInput, { target: { files: [file] } });
    fireEvent.click(screen.getByText('Upload'));
    
    // Should show progress indicator
    expect(screen.getByText(/uploading/i)).toBeInTheDocument();
    expect(screen.getByRole('progressbar')).toBeInTheDocument();
    
    await waitFor(() => {
      expect(screen.getByText(/upload complete/i)).toBeInTheDocument();
    });
  });

  test('validates file type', () => {
    render(<FileUploadComponent onUpload={jest.fn()} />);
    
    const fileInput = screen.getByLabelText(/choose file/i);
    const invalidFile = new File(['content'], 'test.txt', { type: 'text/plain' });
    
    fireEvent.change(fileInput, { target: { files: [invalidFile] } });
    
    expect(screen.getByText(/invalid file type/i)).toBeInTheDocument();
    expect(screen.getByText(/only .ply files/i)).toBeInTheDocument();
  });
});
```
**Acceptance Criteria**: File upload component handles all scenarios correctly

##### Minor Task 7.2.2 (Implementation): Create File Upload Component
**Task**: Implement drag-and-drop file upload with progress tracking
**What to do**: Create comprehensive file upload component
**Implementation Level**:
- Drag-and-drop interface
- Upload progress tracking
- File validation and error display
- Multiple file management
**Code Estimate**: ~250 lines
**How to Test**: Use tests from 7.2.1 - all file upload tests must pass
**Acceptance Criteria**: Upload component provides excellent user experience

##### Minor Task 7.2.3 (Test First): Write File Management Tests
**Task**: Create tests for managing multiple uploaded files
**What to do**: Test file list management, removal, and replacement
**Implementation Level**:
- Test file list display
- Test file removal functionality
- Test file replacement
- Test file information display
**Code Estimate**: ~80 lines of test code
**How to Test**:
```javascript
describe('File Management', () => {
  test('displays uploaded files in list', async () => {
    render(<FileUploadComponent onUpload={jest.fn()} />);
    
    // Upload multiple files
    const files = [
      new File(['content1'], 'surface1.ply', { type: 'application/octet-stream' }),
      new File(['content2'], 'surface2.ply', { type: 'application/octet-stream' })
    ];
    
    const fileInput = screen.getByLabelText(/choose file/i);
    fireEvent.change(fileInput, { target: { files } });
    
    await waitFor(() => {
      expect(screen.getByText('surface1.ply')).toBeInTheDocument();
      expect(screen.getByText('surface2.ply')).toBeInTheDocument();
    });
  });

  test('removes files from list', async () => {
    render(<FileUploadComponent onUpload={jest.fn()} />);
    
    const file = new File(['content'], 'test.ply', { type: 'application/octet-stream' });
    const fileInput = screen.getByLabelText(/choose file/i);
    
    fireEvent.change(fileInput, { target: { files: [file] } });
    
    await waitFor(() => {
      expect(screen.getByText('test.ply')).toBeInTheDocument();
    });
    
    fireEvent.click(screen.getByLabelText(/remove test.ply/i));
    
    expect(screen.queryByText('test.ply')).not.toBeInTheDocument();
  });
});
```
**Acceptance Criteria**: File management provides intuitive control over uploaded files

##### Minor Task 7.2.4 (Implementation): Create File Management Interface
**Task**: Implement file list management functionality
**What to do**: Create interface for managing multiple files
**Implementation Level**:
- File list display with metadata
- File removal and replacement
- File status indicators
- File validation feedback
**Code Estimate**: ~150 lines
**How to Test**: Use tests from 7.2.3 - all file management tests must pass
**Acceptance Criteria**: File management interface is intuitive and functional

#### Subtask 7.3: API Integration Layer

##### Minor Task 7.3.1 (Test First): Write API Client Tests
**Task**: Create tests for frontend API communication layer
**What to do**: Test all API calls with proper error handling
**Implementation Level**:
- Test successful API calls
- Test network error handling
- Test server error handling
- Test response data validation
**Code Estimate**: ~100 lines of test code
**How to Test**:
```javascript
import { uploadFile, configureAnalysis, executeAnalysis } from '../api/backendApi';

describe('Backend API Client', () => {
  beforeEach(() => {
    global.fetch = jest.fn();
  });

  test('uploads file successfully', async () => {
    const mockResponse = {
      success: true,
      file_id: 'test-123',
      filename: 'test.ply'
    };

    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve(mockResponse)
    });

    const file = new File(['content'], 'test.ply');
    const result = await uploadFile(file);

    expect(result).toEqual(mockResponse);
    expect(global.fetch).toHaveBeenCalledWith('/api/upload', expect.objectContaining({
      method: 'POST',
      body: expect.any(FormData)
    }));
  });

  test('handles upload errors', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 400,
      json: () => Promise.resolve({ detail: 'Invalid file type' })
    });

    const file = new File(['content'], 'test.txt');
    
    await expect(uploadFile(file)).rejects.toThrow('Invalid file type');
  });

  test('configures analysis with validation', async () => {
    const config = {
      surfaces: [{ file_id: 'test-123', anchor_lat: 40.7, anchor_lon: -74.0 }],
      boundary: { corners: [[40.7, -74.0], [40.8, -74.0], [40.8, -73.9], [40.7, -73.9]] }
    };

    global.fetch.mockResolvedValue({
      ok: true,
      json: () => Promise.resolve({ analysis_id: 'analysis-456', success: true })
    });

    const result = await configureAnalysis(config);
    expect(result.analysis_id).toBe('analysis-456');
  });
});
```
**Acceptance Criteria**: API client handles all communication scenarios robustly

##### Minor Task 7.3.2 (Implementation): Create API Client Layer
**Task**: Implement comprehensive API communication layer
**What to do**: Create `frontend/src/api/backendApi.js` with all API functions
**Implementation Level**:
- File upload API calls
- Analysis configuration API calls
- Analysis execution and status API calls
- Results retrieval API calls
- Point query API calls
**Code Estimate**: ~200 lines
**How to Test**: Use tests from 7.3.1 - all API client tests must pass
**Acceptance Criteria**: API client provides reliable communication with backend

##### Minor Task 7.3.3 (Test First): Write Error Handling Tests
**Task**: Create tests for comprehensive error handling
**What to do**: Test error propagation and user feedback
**Implementation Level**:
- Test network connectivity errors
- Test server validation errors
- Test timeout handling
- Test error message display
**Code Estimate**: ~70 lines of test code
**How to Test**:
```javascript
describe('Error Handling', () => {
  test('handles network timeout', async () => {
    global.fetch.mockImplementation(() => 
      new Promise((resolve, reject) => {
        setTimeout(() => reject(new Error('Network timeout')), 100);
      })
    );

    await expect(uploadFile(new File([''], 'test.ply'))).rejects.toThrow('Network timeout');
  });

  test('displays user-friendly error messages', async () => {
    global.fetch.mockResolvedValue({
      ok: false,
      status: 422,
      json: () => Promise.resolve({
        detail: [{ msg: 'Latitude must be between -90 and 90', loc: ['anchor_lat'] }]
      })
    });

    render(<AnalysisConfigForm />);
    
    fireEvent.change(screen.getByLabelText('Anchor Latitude'), { target: { value: '91' } });
    fireEvent.click(screen.getByText('Submit'));

    await waitFor(() => {
      expect(screen.getByText(/latitude must be between -90 and 90/i)).toBeInTheDocument();
    });
  });
});
```
**Acceptance Criteria**: Error handling provides clear guidance to users

##### Minor Task 7.3.4 (Implementation): Implement Error Handling System
**Task**: Create comprehensive error handling and user feedback
**What to do**: Implement error handling throughout the application
**Implementation Level**:
- Global error boundary component
- API error parsing and display
- User-friendly error messages
- Error recovery mechanisms
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 7.3.3 - all error handling tests must pass
**Acceptance Criteria**: Error handling enhances user experience and provides clear guidance

## Success Metrics and Validation

### Development Progress Tracking
- **Total Tasks**: 127 discrete tasks across 6 phases
- **Test Coverage Target**: >95% for all backend modules, >90% for frontend components
- **Performance Validation**: All NFR-P1.3 benchmarks met in automated testing
- **Integration Testing**: Complete workflow validation from file upload to results

### Quality Assurance Metrics
- **Unit Test Execution**: All 500+ test cases pass without errors
- **Algorithm Accuracy**: Volume calculations within ±1% of analytical solutions
- **API Response Times**: <100ms for interactive queries, <30s for 1M point processing
- **Memory Usage**: <8GB for 50M point datasets as specified in NFR-P1.3

### User Experience Validation
- **Wizard Completion Rate**: >95% of users complete full workflow
- **Error Recovery**: Users can recover from all error scenarios
- **Performance Perception**: UI remains responsive during all operations
- **Documentation Effectiveness**: Users complete tasks using documentation alone

This comprehensive task breakdown provides intern-level specificity for implementing the complete Surface Volume and Layer Thickness Analysis Tool, ensuring every requirement is translated into concrete, testable development tasks.

## Future Iterations: Advanced Features and Improvements

### Major Task 8.0: Robust Surface Alignment with Advanced Outlier Rejection

#### Subtask 8.1: Enhanced Outlier Detection and Rejection

##### Minor Task 8.1.1 (Test First): Write Robust Outlier Rejection Tests
**Task**: Create comprehensive tests for advanced outlier rejection methods
**What to do**: Test RANSAC-based outlier rejection, iterative trimming, and statistical outlier detection
**Implementation Level**:
- Test RANSAC with varying outlier percentages (10%, 20%, 30%)
- Test iterative outlier rejection with convergence criteria
- Test statistical methods (MAD, IQR-based outlier detection)
- Test performance with large datasets containing outliers
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_ransac_outlier_rejection():
    # Create surface with 20% outliers
    surface1 = create_test_surface(1000)
    surface2 = surface1.copy()
    outlier_indices = np.random.choice(1000, 200, replace=False)
    surface2[outlier_indices] += np.random.normal(0, 50, (200, 3))
    
    result = align_surfaces_ransac(surface1, surface2, max_iterations=100)
    assert result['inlier_ratio'] > 0.8
    assert abs(result['rotation']) < 1.0
    assert abs(result['scale'] - 1.0) < 0.01

def test_iterative_outlier_trimming():
    # Test iterative outlier removal
    surface1 = create_test_surface(500)
    surface2 = surface1.copy()
    surface2[:50] += 100  # 10% outliers
    
    result = align_surfaces_iterative(surface1, surface2, trim_percentile=90)
    assert result['final_inlier_count'] > 400
    assert result['convergence_iterations'] < 10
```
**Acceptance Criteria**: All outlier rejection methods handle 30%+ outliers while maintaining alignment accuracy

##### Minor Task 8.1.2 (Implementation): Implement RANSAC-Based Surface Alignment
**Task**: Create robust surface alignment using RANSAC for outlier rejection
**What to do**: Implement RANSAC algorithm for surface alignment with configurable parameters
**Implementation Level**:
- RANSAC with random sampling of point correspondences
- Consensus set evaluation using alignment quality metrics
- Iterative refinement of transformation parameters
- Automatic parameter tuning based on outlier percentage
**Code Estimate**: ~150 lines
**How to Test**: Use tests from 8.1.1 - all RANSAC tests must pass
**Acceptance Criteria**: RANSAC alignment handles 50% outliers with >95% success rate

##### Minor Task 8.1.3 (Test First): Write Statistical Outlier Detection Tests
**Task**: Create tests for statistical outlier detection methods
**What to do**: Test MAD (Median Absolute Deviation) and IQR-based outlier detection
**Implementation Level**:
- Test MAD-based outlier detection with configurable thresholds
- Test IQR-based outlier detection for different distributions
- Test adaptive threshold selection based on data characteristics
- Test performance comparison between methods
**Code Estimate**: ~60 lines of test code
**How to Test**:
```python
def test_mad_outlier_detection():
    data = np.random.normal(0, 1, 1000)
    data[100:120] += 10  # Add outliers
    outliers = detect_outliers_mad(data, threshold=3.0)
    assert 100 <= len(outliers) <= 120
    assert all(100 <= i <= 120 for i in outliers)

def test_adaptive_threshold_selection():
    # Test automatic threshold selection
    data = generate_mixed_data(clean_ratio=0.8, noise_level=0.1)
    threshold = select_adaptive_threshold(data)
    outliers = detect_outliers_mad(data, threshold=threshold)
    assert 0.15 <= len(outliers)/len(data) <= 0.25
```
**Acceptance Criteria**: Statistical methods correctly identify outliers with <5% false positive rate

##### Minor Task 8.1.4 (Implementation): Implement Statistical Outlier Detection
**Task**: Create statistical outlier detection algorithms
**What to do**: Implement MAD and IQR-based outlier detection with adaptive thresholds
**Implementation Level**:
- MAD-based outlier detection with robust statistics
- IQR-based outlier detection for skewed distributions
- Adaptive threshold selection using data characteristics
- Performance optimization for large datasets
**Code Estimate**: ~100 lines
**How to Test**: Use tests from 8.1.3 - all statistical outlier tests must pass
**Acceptance Criteria**: Statistical methods provide reliable outlier detection for various data distributions

#### Subtask 8.2: Performance Optimization for Large Datasets

##### Minor Task 8.2.1 (Test First): Write Large Dataset Alignment Tests
**Task**: Create performance tests for surface alignment with large datasets
**What to do**: Test alignment performance with 1M+ point surfaces containing outliers
**Implementation Level**:
- Test alignment time with 1M, 5M, 10M point surfaces
- Test memory usage during alignment process
- Test parallel processing for large datasets
- Test accuracy vs. performance trade-offs
**Code Estimate**: ~70 lines of test code
**How to Test**:
```python
def test_large_dataset_alignment_performance():
    surface1 = create_large_surface(1000000)
    surface2 = surface1.copy()
    surface2[:100000] += np.random.normal(0, 10, (100000, 3))  # 10% outliers
    
    start_time = time.time()
    result = align_surfaces_parallel(surface1, surface2, n_jobs=4)
    elapsed = time.time() - start_time
    
    assert elapsed < 60.0  # Must complete in <60 seconds
    assert result['inlier_ratio'] > 0.85
    assert result['rmse'] < 0.5
```
**Acceptance Criteria**: Large dataset alignment completes within performance requirements

##### Minor Task 8.2.2 (Implementation): Implement Parallel Surface Alignment
**Task**: Create parallel processing for surface alignment algorithms
**What to do**: Implement parallel RANSAC and outlier detection for large datasets
**Implementation Level**:
- Parallel RANSAC with multiple worker processes
- Chunked processing for memory-efficient large dataset handling
- Load balancing for optimal parallel performance
- Progress tracking and cancellation support
**Code Estimate**: ~120 lines
**How to Test**: Use tests from 8.2.1 - all performance tests must pass
**Acceptance Criteria**: Parallel alignment provides 2-4x speedup on multi-core systems

#### Subtask 8.3: Quality Assessment and Validation

##### Minor Task 8.3.1 (Test First): Write Alignment Quality Assessment Tests
**Task**: Create comprehensive tests for alignment quality assessment
**What to do**: Test quality metrics, confidence intervals, and validation methods
**Implementation Level**:
- Test alignment quality metrics (RMSE, inlier ratio, confidence)
- Test cross-validation between different alignment methods
- Test quality assessment for edge cases and degenerate data
- Test automated quality reporting and recommendations
**Code Estimate**: ~80 lines of test code
**How to Test**:
```python
def test_alignment_quality_assessment():
    surface1 = create_test_surface(1000)
    surface2 = surface1.copy()
    surface2[:100] += np.random.normal(0, 5, (100, 3))  # 10% noise
    
    quality = assess_alignment_quality(surface1, surface2)
    assert quality['rmse'] < 1.0
    assert quality['inlier_ratio'] > 0.9
    assert quality['confidence'] > 0.95
    assert quality['recommendation'] in ['good', 'acceptable', 'poor']

def test_cross_validation_alignment():
    # Test agreement between different alignment methods
    surface1 = create_test_surface(500)
    surface2 = surface1.copy()
    surface2[:50] += 20  # 10% outliers
    
    result1 = align_surfaces_ransac(surface1, surface2)
    result2 = align_surfaces_iterative(surface1, surface2)
    
    # Methods should agree within tolerance
    assert abs(result1['rotation'] - result2['rotation']) < 1.0
    assert abs(result1['scale'] - result2['scale']) < 0.01
```
**Acceptance Criteria**: Quality assessment provides reliable metrics and actionable recommendations

##### Minor Task 8.3.2 (Implementation): Implement Alignment Quality Assessment
**Task**: Create comprehensive quality assessment system for surface alignment
**What to do**: Implement quality metrics, confidence estimation, and validation methods
**Implementation Level**:
- Multiple quality metrics (RMSE, inlier ratio, geometric consistency)
- Confidence interval estimation using bootstrap methods
- Cross-validation between alignment algorithms
- Automated quality reporting with recommendations
**Code Estimate**: ~150 lines
**How to Test**: Use tests from 8.3.1 - all quality assessment tests must pass
**Acceptance Criteria**: Quality assessment system provides actionable insights for alignment quality

### Success Metrics for Future Iterations

#### Robustness Improvements
- **Outlier Tolerance**: Handle 50% outliers with >95% success rate
- **Data Quality**: Maintain alignment accuracy with noisy, incomplete, or corrupted data
- **Edge Cases**: Robust handling of degenerate geometries and extreme transformations

#### Performance Enhancements
- **Large Dataset Support**: 10M+ point surfaces with <2 minute alignment time
- **Parallel Processing**: 2-4x speedup on multi-core systems
- **Memory Efficiency**: <16GB memory usage for largest supported datasets

#### Quality Assurance
- **Automated Validation**: Cross-validation between multiple alignment methods
- **Quality Metrics**: Comprehensive quality assessment with confidence intervals
- **User Guidance**: Automated recommendations for alignment quality improvement

This future iteration plan addresses the current limitation in outlier rejection while providing a roadmap for advanced surface alignment capabilities suitable for production environments with challenging data quality conditions.

## Volume Calculation Method Recommendation

For production use, the mesh-based (PyVista/triangle) volume calculation method is the standard and should be used for all critical and irregular/rough surfaces. The prism method is only suitable for quick estimates or regular grid/planar surfaces, and may diverge significantly for rough or non-uniform surfaces. Strict cross-validation is not required for irregular surfaces; use mesh-based results for all reporting and analysis.
