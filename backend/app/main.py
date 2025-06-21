from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from app.routes import surfaces, analysis

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Surface Mapper Backend API")
    yield
    # Shutdown
    logger.info("Shutting down Surface Mapper Backend API")

app = FastAPI(
    title="Surface Mapper API", 
    version="1.0.0",
    description="Surface Volume and Layer Thickness Analysis Tool API",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React development server
        "http://127.0.0.1:3000",
        "http://localhost:80",    # Production nginx
        "http://127.0.0.1:80"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Include routers
app.include_router(surfaces.router, prefix="/api/v1")
app.include_router(analysis.router)

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker health checks and monitoring"""
    return {
        "status": "healthy", 
        "service": "surface-mapper-backend",
        "version": "1.0.0",
        "timestamp": "2024-12-19T00:00:00Z"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Surface Mapper Backend API", 
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_base": "/api/v1"
    }

@app.get("/api-info")
async def api_info():
    """Detailed API information"""
    return {
        "name": "Surface Mapper API",
        "version": "1.0.0",
        "description": "Surface Volume and Layer Thickness Analysis Tool",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "surfaces": "/api/v1/surfaces",
            "coordinate_systems": "/api/v1/surfaces/coordinate-systems",
            "coordinate_transform": "/api/v1/surfaces/coordinate-transform"
        },
        "features": [
            "PLY file upload and validation",
            "Surface processing and analysis",
            "Volume and thickness calculations",
            "3D visualization data preparation",
            "Coordinate system transformations"
        ]
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url)
        }
    ) 