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

- **React 18**: Modern React with hooks and functional components
- **Three.js**: 3D graphics and visualization
- **Tailwind CSS**: Utility-first CSS framework
- **Fetch API**: Modern HTTP client for API communication

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
REACT_APP_API_BASE_URL=http://localhost:8000
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

The frontend communicates with the backend through the `backendApi.js` client:

```javascript
import { uploadSurface, processSurfaces, getProcessingStatus } from './api/backendApi';

// Upload surface file
const result = await uploadSurface(file);

// Process surfaces for analysis
const jobId = await processSurfaces(processingRequest);

// Poll for results
const status = await getProcessingStatus(jobId);
```

### Error Handling

The API client includes comprehensive error handling:

- Network connectivity issues
- HTTP error responses
- Request timeouts
- Validation errors

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
  @apply bg-blue-600 text-white hover:bg-blue-700;
}

/* Custom form styles */
.form-input {
  @apply w-full px-3 py-2 border border-gray-300 rounded-md;
}
```

### Responsive Design

- Mobile-first approach
- Breakpoint-specific layouts
- Touch-friendly interactions
- Print-optimized styles

## Development Guidelines

### Code Style

- Use functional components with hooks
- Implement proper error boundaries
- Follow React best practices
- Use TypeScript for type safety (future)

### Testing

```bash
# Run tests
npm test

# Run tests with coverage
npm test -- --coverage
```

### Performance

- Lazy load components where appropriate
- Optimize Three.js rendering
- Implement proper cleanup in useEffect
- Use React.memo for expensive components

## Deployment

### Docker

The frontend is containerized using nginx:

```dockerfile
FROM nginx:alpine
COPY build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf
```

### Environment Configuration

Configure environment variables for different deployments:

```bash
# Development
REACT_APP_API_BASE_URL=http://localhost:8000

# Production
REACT_APP_API_BASE_URL=https://api.surfacemapper.com
```

## Troubleshooting

### Common Issues

1. **CORS Errors**: Ensure backend CORS configuration
2. **Three.js Performance**: Reduce surface complexity for large datasets
3. **Memory Leaks**: Properly dispose Three.js resources
4. **API Timeouts**: Increase timeout values for large file uploads

### Debug Mode

Enable debug logging:

```javascript
localStorage.setItem('debug', 'surfacemapper:*');
```

## Contributing

1. Follow the established code style
2. Add tests for new features
3. Update documentation
4. Ensure responsive design
5. Test with various screen sizes

## License

See project LICENSE file for details.
