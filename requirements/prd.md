# Product Requirements Document: Surface Volume and Layer Thickness Analysis Tool

## 1. Introduction
This document outlines the requirements for a software tool designed to analyze changes in surface topology over time, specifically focusing on the volume and thickness of material added between successive layers. The primary use case involves processing .ply files extracted from drone-based surface mapping, where each file represents a distinct capture of a surface at different stages of material addition. The tool will provide 3D visualization, quantitative analysis of volume and thickness, and compaction rate calculations.

## 2. Purpose
The purpose of this software is to enable users (e.g., construction managers, surveyors, civil engineers) to accurately quantify material additions, understand compaction, and visualize topographical changes on a surface over a period. This will support better decision-making in earthworks, mining, and other industries where precise volume and layer thickness measurements are critical.

## 3. Scope
This document covers the functional and non-functional requirements for the initial version of the Surface Volume and Layer Thickness Analysis Tool. It includes data ingestion, boundary definition, core calculations, 3D visualization, interactive analysis, and tabular data output. The application will be delivered as a containerized solution runnable on a local machine.

## 4. User Roles
**Primary User**: An individual who operates drones for mapping, processes .ply files, and requires analytical insights into surface changes. They are familiar with geospatial concepts and require precise measurements.

## 5. Architectural Decisions & File Structure
This application will utilize a Python backend for all heavy-duty data processing, calculations, and 3D model preparation, exposed via a FastAPI web framework. The frontend will be built with React for an interactive user interface and Three.js for browser-based 3D rendering. The entire application will be deployed within a Docker container.

### 5.1. Proposed File Structure
```
├── docker-compose.yml              # Defines multi-service Docker application
├── Dockerfile.backend              # Dockerfile for Python backend
├── Dockerfile.frontend             # Dockerfile for React frontend
├── data/                           # Local data storage and cache
│   ├── temp/                       # Temporary processing files
│   └── exports/                    # Generated reports and exports
├── config/                         # Configuration management
│   ├── coordinate_systems.json     # Predefined coordinate system templates
│   └── processing_defaults.yaml    # Default algorithm parameters
├── docs/                          # Technical documentation
│   ├── api_documentation.md
│   └── algorithm_specifications.md
├── backend/
│   ├── app/                        # Main FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app entry point, defines endpoints
│   │   ├── routes/                 # API endpoints
│   │   │   └── surfaces.py         # Endpoints for surface upload, processing, etc.
│   │   ├── services/               # Core business logic, computations
│   │   │   ├── surface_processor.py # Handles PLY parsing, clipping, base surface
│   │   │   ├── volume_calculator.py # Logic for volume and thickness
│   │   │   └── coord_transformer.py # Handles pyproj transformations
│   │   ├── models/                 # Data models (Pydantic for API)
│   │   │   └── data_models.py      # Request/response data structures
│   │   └── utils/                  # Helper functions
│   │       └── ply_parser.py       # Specific PLY file parsing logic (using plyfile)
│   ├── tests/                      # Backend unit and integration tests
│   │   ├── test_services.py
│   │   └── test_routes.py
│   ├── requirements.txt            # Python dependencies
│   └── README.md                   # Backend specific README
├── frontend/
│   ├── public/                     # Static assets
│   ├── src/
│   │   ├── App.js                  # Main React component
│   │   ├── index.js                # React entry point
│   │   ├── components/             # Reusable UI components
│   │   │   ├── InputForms.js       # User input forms
│   │   │   ├── ThreeDViewer.js     # Three.js integration
│   │   │   └── DataTable.js        # Tabular results display
│   │   ├── hooks/                  # Custom React hooks
│   │   ├── api/                    # Frontend API client (interacts with backend)
│   │   │   └── backendApi.js
│   │   ├── styles/                 # Tailwind CSS configuration
│   │   └── utils/                  # Frontend specific utilities
│   ├── package.json                # Node.js dependencies
│   ├── tailwind.config.js          # Tailwind CSS config
│   └── README.md                   # Frontend specific README
├── prd.md                          # Product Requirements Document
├── task.md                         # Task Breakdown Document
└── README.md                       # Overall project README
```

## 6. Functional Requirements

### 6.1. Surface Data Ingestion
**FR-S1.1**: The system shall allow users to upload or specify the file paths for 1 to 4 .ply surface files.

**FR-S1.2**: The backend shall validate that the provided files are valid .ply format and contain sufficient data (vertices, faces if applicable).

**FR-S1.3**: The system shall require at least two surfaces for analysis. If only one .ply file is provided, the user must select the option to generate a "base level" surface (FR-S2.1) to fulfill this requirement. If neither a second .ply file nor the base level option is selected, the system shall prompt the user and prevent further processing.

### 6.2. Base Surface Generation
**FR-S2.1**: If fewer than the desired number of ply files are provided (e.g., only one ply file, but 2 surfaces are selected for analysis), the system shall provide an option to generate a "base level" flat surface.

**FR-S2.2**: The base level surface shall be defined by a user-specified positive vertical offset (in feet) from the lowest Z-coordinate (elevation) of the first provided .ply surface within the defined analysis boundary.

**FR-S2.3**: The generated base surface shall cover the entire horizontal extent (X, Y) of the defined analysis boundary.

### 6.3. Input Data Details & Preprocessing
**FR-ID3.1**: For each .ply file input by the user, the system shall require the user to provide:
- The WGS84 Latitude and Longitude coordinates of a known vertex within the .ply file's local coordinate system.
- The orientation (angle in degrees, clockwise from North) of the local X-axis relative to true North.
- A scaling factor (decimal, e.g., 1.0 for no scaling) to be applied to the .ply file's coordinates.

**FR-ID3.2**: The system shall use pyproj in the backend to transform the .ply file's local coordinates, anchoring them to the provided WGS84 reference point and applying rotation and scaling, converting all surfaces into a common, geo-referenced Cartesian coordinate system (e.g., UTM zone).

**FR-ID3.3**: The altitude (Z-coordinate) within all .ply files is assumed to be in feet, and no user specification for Z-unit is required. All Z-values will be treated as feet.

**FR-ID3.4**: The backend shall store parsed and transformed point cloud data for all surfaces in memory during processing to avoid disk I/O and minimize server-side state.

**FR-ID3.5**: The system shall check for substantial overlap between all provided surfaces within the user-defined analysis boundary. If significant non-overlap is detected (e.g., one surface entirely outside another's horizontal extent), a warning shall be displayed, indicating potential issues with reference coordinates or surface coverage.

**FR-ID3.6**: If surface files are of different sizes (e.g., differing extents or point densities), the backend will handle this by aligning them using the provided reference points, orientations, and scaling factors. Clipping to the common analysis boundary will further ensure consistency.

### 6.4. Analysis Boundary Definition
**FR-B4.1**: The system shall allow the user to define a rectangular analysis boundary using four WGS84 latitude/longitude coordinates.

**FR-B4.2**: The backend shall transform these WGS84 coordinates into the common Cartesian coordinate system of the processed .ply files using pyproj.

**FR-B4.3**: All subsequent volume, thickness calculations, and visualizations shall be clipped to this user-defined boundary. Data outside this boundary will be ignored.

### 6.5. Tonnage Input
**FR-T5.1**: The system shall allow the user to input the tonnage (in imperial tons, e.g., US short tons) of material added between each successive surface layer.

### 6.6. Core Calculations (Backend)

#### 6.6.1. Volume Calculation
**FR-C6.1.1**: The backend shall calculate the net volume difference between each successive pair of surfaces (e.g., Surface 1 relative to Surface 0, Surface 2 relative to Surface 1) within the defined analysis boundary.

**FR-C6.1.2 (ENHANCED)**: Volume calculations shall utilize Delaunay triangulation-based algorithms with the following hierarchy:
- **Primary**: PyVista's convex hull and mesh volume calculations using Delaunay triangulation (Note: PyVista provides native 3D Delaunay triangulation via VTK backend, superior to Trimesh's 2D-only approach)
- **Secondary**: CGAL-based exact geometric computations for precision-critical applications
- **Validation**: Cross-validation between methods with <1% tolerance requirements
- **Algorithm**: Delaunay Triangulation-Based Volume Calculation (DTVC) for irregular surfaces achieving <3% error for typical applications

**FR-C6.1.3**: Volume results shall be calculated in cubic yards. The backend shall handle necessary unit conversions from input units (feet) to cubic yards.

#### 6.6.2. Layer Thickness Calculation
**FR-C6.2.1**: For each successive pair of surfaces, the backend shall calculate the average, minimum, and maximum layer thickness within the defined analysis boundary.

**FR-C6.2.2**: Thickness shall be measured as the vertical distance (Z-difference) between corresponding points on the upper and lower surfaces, interpolated from their respective TINs.

**FR-C6.2.3**: Thickness results shall be in feet.

#### 6.6.3. Compaction Rate Calculation
**FR-C6.3.1**: For each successive layer where tonnage is provided, the backend shall calculate the compaction rate.

**FR-C6.3.2**: Compaction rate shall be calculated as (Input Tonnage * 2000 lbs/ton) / Calculated Volume (in cubic yards).

**FR-C6.3.3**: Compaction rate results shall be in lbs/cubic yard.

**FR-C6.3.4**: If tonnage is not provided for a specific layer, the compaction rate for that layer in the output table shall be displayed as "--".

### 6.7. 3D Visualization (Frontend leveraging Backend)
**FR-V7.1**: The Python backend, leveraging PyVista, shall process the surface .ply data and prepare it for 3D visualization. This preparation includes generating simplified meshes or extracting vertex data suitable for efficient transfer to the frontend. (Note: PyVista provides advanced mesh simplification and point cloud meshing via VTK backend, superior to Trimesh capabilities.)

**FR-V7.2**: The React frontend, using Three.js, shall render a 3D visualization of all selected surfaces, stacked vertically according to their actual Z-coordinates.

**FR-V7.3**: Each surface in the stack shall be represented by a solid, distinct color.

**FR-V7.4**: The 3D visualization shall be interactive, allowing users to:
- Rotate the view (orbit around the model).
- Pan the view (translate the model).
- Zoom in and out.

**FR-V7.5**: The visualization should only display the portion of the surfaces within the user-defined analysis boundary.

### 6.8. Interactive Point Analysis (Frontend with Backend Data)
**FR-I8.1**: When the user hovers or drags their mouse over any point on the 3D visualization (rendered via Three.js), the frontend shall raycast to determine the X-Y coordinates on the uppermost surface.

**FR-I8.2**: These X-Y coordinates shall be sent to a dedicated backend API endpoint.

**FR-I8.3**: The backend, using its processed TIN data, shall query the Z-value from all underlying surfaces at that specific X-Y location.

**FR-I8.4**: The backend shall then calculate the individual layer thickness at that point for each successive layer (e.g., (S1-S0) thickness, (S2-S1) thickness).

**FR-I8.5**: The calculated point thicknesses will be returned to the frontend, which shall display them in a dynamic, non-intrusive pop-up window.

### 6.9. Tabular Data Output (Frontend)
**FR-T9.1**: The system shall display a clear, organized table summarizing the analysis results.

**FR-T9.2**: The table shall include:
- Layer Designation: (e.g., "Surface 0 to Surface 1", "Surface 1 to Surface 2")
- Total Volume Added: (in cubic yards) for each layer.
- Compaction Rate: (in lbs/cubic yard) for each layer (or "--" if tonnage not provided).
- Thickest Layer Thickness: (in feet) between the two currently highlighted surfaces.
- Thinnest Layer Thickness: (in feet) between the two currently highlighted surfaces.
- Average Layer Thickness: (in feet) between the two currently highlighted surfaces.

**FR-T9.3**: The user shall be able to highlight or select any two surfaces in the 3D view or the table to populate the "thickest, thinnest, average" fields specifically for that pair.

### 6.10. User Interface Layout
**FR-UI10.1 (REVISED)**: Implement a wizard-based workflow:
- **Step 1**: Project Setup (boundary definition, surface count)
- **Step 2**: Surface Upload (with automatic validation)
- **Step 3**: Georeferencing (with manual parameter entry)
- **Step 4**: Material Input (tonnage per layer)
- **Step 5**: Analysis Review & Export

**FR-UI10.2**: Each step shall provide clear progress indicators and validation feedback.

**FR-UI10.3**: Users shall be able to navigate back to previous steps to modify inputs without losing subsequent work.

### 6.11. Algorithm Specifications
**FR-AS11.1**: All volume calculations shall provide statistical confidence intervals and uncertainty estimates.

**FR-AS11.2**: Thickness measurements shall include interpolation method documentation (e.g., linear, cubic, kriging).

**FR-AS11.3**: Compaction rate calculations shall account for material density variations and provide confidence ranges.

**FR-AS11.4**: Export capabilities for calculated intermediate results (TINs, point clouds, analysis reports).

**FR-AS11.5**: All algorithms shall maintain processing logs with performance metrics and validation results.

## 7. Non-Functional Requirements

### 7.1. Performance
**NFR-P1.1**: The system shall be able to process .ply files up to a reasonable size (e.g., tens of millions of points) within acceptable timeframes for calculations and rendering (e.g., calculations complete within minutes, 3D rendering smooth and responsive).

**NFR-P1.2**: The 3D visualization shall maintain a smooth frame rate (e.g., >30 FPS) during interaction on modern browsers.

**NFR-P1.3**: Processing benchmarks:
- **1M points**: <30 seconds for full analysis
- **10M points**: <5 minutes for full analysis
- **50M points**: <20 minutes for full analysis
- **Memory usage**: <8GB for 50M point datasets
- **File size limits**: 2GB per PLY file, 8GB total project size
- **Concurrent processing**: Support for 2-4 parallel surface processing tasks

### 7.2. Accuracy
**NFR-A2.1**: Volume and thickness calculations shall be accurate to within a defined tolerance (e.g., 1-5% of ground truth, depending on input data quality).

**NFR-A2.2**: Coordinate transformations using pyproj shall preserve positional accuracy.

### 7.3. Usability
**NFR-U3.1**: The user interface shall be intuitive and easy to navigate for users familiar with mapping and data analysis tools.

**NFR-U3.2**: Input forms shall provide clear instructions and validation feedback.

**NFR-U3.3**: Error messages shall be clear and actionable.

### 7.4. Maintainability
**NFR-M4.1**: The codebase shall be modular, well-commented, and follow established coding standards to facilitate future enhancements and bug fixes.

### 7.5. Quality Assurance
**NFR-QA5.1**: Volume calculations shall be validated against synthetic test datasets with known volumes (±1% accuracy target).

**NFR-QA5.2**: Coordinate transformations shall maintain <0.1m accuracy for surveying applications.

**NFR-QA5.3**: All calculations shall provide uncertainty estimates and confidence intervals.

**NFR-QA5.4**: Automated regression testing shall validate algorithm consistency across software updates.

**NFR-QA5.5**: Performance benchmarks shall be maintained and monitored for each release.

## 8. Development Methodology
**Test-Driven Development (TDD)**: All major tasks and minor tasks will have clear, defined tests written before any implementation code. These tests will be reviewed and approved prior to coding.

**Integration Testing**: Will be conducted after all individual components and unit tests are complete and functional.

**Performance Testing**: Continuous benchmarking against specified performance metrics with automated alerts for regression.

## 9. Technical Considerations / Assumptions
**Deployment Environment**: The entire application (backend and frontend) will run within a Docker container on a local machine (e.g., a laptop). There is no requirement for cloud server deployment initially.

**PLY File Structure**: Assumed to contain vertex data (X, Y, Z coordinates).

**Coordinate System Consistency**: While transformation parameters are provided, it is assumed that the .ply files themselves contain valid and consistent internal coordinate systems relative to their provided reference vertex.

**Surface Data Gaps/Noise**: The backend algorithms for volume and thickness calculation should gracefully handle minor gaps or noise inherent in drone-scanned data through interpolation or robust statistical methods.

**Algorithm Performance**: Primary algorithms shall be implemented using PyVista for optimal performance, with CGAL as fallback for precision-critical calculations requiring exact arithmetic. (Note: PyVista provides native 3D Delaunay triangulation and advanced mesh operations via VTK backend.)

## 10. Development Log

- All tasks once complete must append the summary of their changes, which tests passed and the open items to requirements/task_tracking.md

## Volume Calculation Method Recommendation

For production use, the mesh-based (PyVista/triangle) volume calculation method is the standard and should be used for all critical and irregular/rough surfaces. The prism method is only suitable for quick estimates or regular grid/planar surfaces, and may diverge significantly for rough or non-uniform surfaces. Strict cross-validation is not required for irregular surfaces; use mesh-based results for all reporting and analysis.