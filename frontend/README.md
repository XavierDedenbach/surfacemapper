# Surface Mapper Frontend

React-based frontend for the Surface Volume and Layer Thickness Analysis Tool.

## Overview

The frontend provides an interactive user interface for uploading PLY surface files, configuring georeferencing parameters, and visualizing analysis results. It implements a wizard-based workflow as specified in the PRD.

## Architecture

### Component Structure

```
src/
├── components/
│   ├── InputForms.js          # Wizard-based input forms
│   ├── ThreeDViewer.js        # Three.js 3D visualization
│   └── DataTable.js           # Results display tables
├── hooks/
│   └── useSurfaceData.js      # Custom hook for data management
├── api/
│   └── backendApi.js          # API client for backend communication
├── styles/
│   └── tailwind.css           # Tailwind CSS styles
└── utils/
    └── geometryUtils.js       # Geometric calculation utilities
```

### Key Features

- **Wizard-based Workflow**: 5-step process for surface analysis setup
- **3D Visualization**: Interactive Three.js rendering of surfaces
- **Real-time Analysis**: Live point analysis on surface hover
- **Responsive Design**: Mobile-friendly interface using Tailwind CSS
- **Data Export**: Export results in multiple formats

## Technology Stack

- **React 19.1.0**: Modern React with hooks and functional components
- **Three.js**: 3D graphics and visualization
- **React Three Fiber**: React integration for Three.js
- **React Three Drei**: Useful helpers for React Three Fiber
- **Tailwind CSS 3.4.17**: Utility-first CSS framework with plugins
- **Axios**: HTTP client for robust API communication
- **PostCSS**: CSS processing with autoprefixer

## Dependencies

### Core Dependencies
```json
{
  "react": "^19.1.0",
  "react-dom": "^19.1.0",
  "react-scripts": "5.0.1"
}
```

### 3D Visualization
```json
{
  "three": "^0.160.0",
  "@react-three/fiber": "^8.15.0",
  "@react-three/drei": "^9.99.0"
}
```

### Styling
```json
{
  "tailwindcss": "^3.4.17",
  "@tailwindcss/forms": "^0.5.10",
  "@tailwindcss/typography": "^0.5.10",
  "@tailwindcss/aspect-ratio": "^0.4.2",
  "autoprefixer": "^10.4.21",
  "postcss": "^8.5.6"
}
```

### API Communication
```json
{
  "axios": "^1.6.0"
}
```

### Development & Testing
```json
{
  "@testing-library/react": "^16.3.0",
  "@testing-library/jest-dom": "^6.6.3",
  "@testing-library/user-event": "^13.5.0",
  "web-vitals": "^2.1.4"
}
```

## Setup and Installation

### Prerequisites

- Node.js 16+ 
- npm or yarn
- Backend server running (see backend README)

### Installation

1. Install dependencies:
```bash
cd frontend
npm install
```

2. Configure environment variables:
```bash
# Create .env file
REACT_APP_API_BASE_URL=http://localhost:8081
```

3. Start development server:
```bash
npm start
```

The application will be available at `http://localhost:3000`

### Building for Production

```bash
npm run build
```

The build process includes:
- Tailwind CSS compilation with PostCSS
- Three.js optimization
- Code splitting and minification
- Asset optimization

## Configuration Files

### Tailwind CSS Configuration
```javascript
// tailwind.config.js
module.exports = {
  content: ["./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      // Custom theme extensions
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
    require('@tailwindcss/aspect-ratio'),
  ],
}
```

### PostCSS Configuration
```javascript
// postcss.config.js
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

## Component Documentation

### InputForms

Handles the 5-step wizard workflow:

1. **Project Setup**: Define number of surfaces
2. **Surface Upload**: Upload PLY files with validation
3. **Georeferencing**: Configure coordinate transformations
4. **Analysis Boundary**: Define rectangular analysis area
5. **Material Input**: Enter tonnage data (optional)

**Props:**
- `onSurfaceUpload`: Callback for file uploads
- `onGeoreferenceSubmit`: Callback for georeference data
- `onBoundarySubmit`: Callback for boundary coordinates
- `onTonnageSubmit`: Callback for tonnage inputs

### ThreeDViewer

Renders 3D surface visualization using Three.js:

**Features:**
- Interactive orbit controls
- Surface stacking by Z-coordinates
- Color-coded surface representation
- Point analysis on hover
- Responsive canvas sizing
- Raycasting for interactive point selection

**Props:**
- `surfaces`: Array of surface data
- `analysisBoundary`: Boundary coordinates
- `onPointHover`: Callback for point analysis
- `selectedSurfaces`: Currently selected surfaces

### DataTable

Displays analysis results in organized tables:

**Tabs:**
- **Summary**: Overview statistics
- **Volume Analysis**: Volume calculations per layer
- **Thickness Analysis**: Thickness measurements
- **Compaction Analysis**: Compaction rate data

**Features:**
- Sortable columns
- Interactive row selection
- Export functionality
- Print-friendly styling

## API Integration

### Backend Communication

The frontend communicates with the backend through the `backendApi.js` client using Axios:

```javascript
import { uploadSurface, processSurfaces, getProcessingStatus } from './api/backendApi';

// Upload surface file
const result = await uploadSurface(file);

// Process surfaces for analysis
const jobId = await processSurfaces(processingRequest);

// Poll for results
const status = await getProcessingStatus(jobId);
```

### API Endpoints

The frontend supports all backend endpoints:

- **Surface Management**: Upload, validate, process surfaces
- **Analysis**: Volume, thickness, compaction calculations
- **3D Visualization**: Mesh data retrieval for Three.js
- **Point Analysis**: Interactive point queries
- **Export**: Results export in multiple formats
- **Configuration**: Processing parameters and coordinate systems

### Error Handling

The API client includes comprehensive error handling:

- Network connectivity issues
- HTTP error responses (4xx, 5xx)
- Request timeouts (30-second timeout)
- Validation errors
- File upload errors

### Request/Response Interceptors

```javascript
// Request logging
apiClient.interceptors.request.use((config) => {
  console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`);
  return config;
});

// Response error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    // Comprehensive error processing
  }
);
```

## State Management

### useSurfaceData Hook

Custom hook for managing surface data and processing state:

```javascript
const {
  surfaces,
  processingStatus,
  analysisResults,
  uploadSurfaces,
  processSurfaceData,
  clearSurfaces
} = useSurfaceData();
```

**State:**
- `surfaces`: Uploaded surface files
- `processingStatus`: Current processing state
- `analysisResults`: Calculated results
- `error`: Error messages

## Styling

### Tailwind CSS

The application uses Tailwind CSS for styling with custom components:

```css
/* Custom button styles */
.btn-primary {
  @apply bg-blue-600 text-white hover:bg-blue-700 px-4 py-2 rounded-md;
}

/* Custom form styles */
.form-input {
  @apply w-full px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500;
}
```

### Responsive Design

The interface is fully responsive with Tailwind's responsive utilities:

```css
/* Mobile-first responsive design */
.container {
  @apply px-4 sm:px-6 lg:px-8;
}

.grid {
  @apply grid-cols-1 md:grid-cols-2 lg:grid-cols-3;
}
```

## Testing

### Unit Tests

```bash
npm test
```

Tests cover:
- Component rendering
- User interactions
- API integration
- State management

### Build Testing

```bash
npm run build
```

Validates:
- Tailwind CSS compilation
- Three.js integration
- Production optimization
- Asset bundling

## Performance

### Optimization Features

- **Code Splitting**: Automatic code splitting by React
- **Lazy Loading**: Components loaded on demand
- **Asset Optimization**: Images and static assets optimized
- **Bundle Analysis**: Webpack bundle analyzer available

### 3D Performance

- **Level of Detail**: Configurable mesh detail levels
- **Frustum Culling**: Only render visible objects
- **Geometry Instancing**: Efficient rendering of repeated geometries
- **Memory Management**: Proper disposal of Three.js resources

## Browser Support

- **Chrome**: 90+
- **Firefox**: 88+
- **Safari**: 14+
- **Edge**: 90+

## Development Workflow

1. **Feature Development**: Create feature branches
2. **Testing**: Write tests for new features
3. **Code Review**: Submit pull requests
4. **Integration**: Merge to main branch
5. **Deployment**: Build and deploy to production

## Troubleshooting

### Common Issues

1. **Build Failures**: Ensure all Tailwind plugins are installed
2. **API Errors**: Check backend server status and CORS configuration
3. **3D Performance**: Reduce mesh complexity for large datasets
4. **Memory Issues**: Monitor Three.js resource disposal

### Debug Mode

```bash
# Enable debug logging
REACT_APP_DEBUG=true npm start
```

## Contributing

1. Follow the existing code style
2. Write tests for new features
3. Update documentation
4. Ensure all tests pass
5. Submit pull requests

## License

See the main project LICENSE file.

## Environment Variables

Create a `.env` file in the frontend directory with the following variables:

```env
REACT_APP_API_BASE_URL=http://localhost:8081
```
