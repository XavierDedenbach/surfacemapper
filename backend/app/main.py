"""
Main FastAPI application for Surface Mapper Backend API
"""
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import json
import traceback
from contextlib import asynccontextmanager
from app.routes import surfaces, analysis
from app.utils.serialization import make_json_serializable, validate_json_serializable

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Patch FastAPI's jsonable_encoder for debugging
from fastapi import encoders as fastapi_encoders
orig_jsonable_encoder = fastapi_encoders.jsonable_encoder

def debug_jsonable_encoder(obj, *args, **kwargs):
    try:
        return orig_jsonable_encoder(obj, *args, **kwargs)
    except Exception as e:
        logger.error(f"jsonable_encoder failed on type {type(obj)}: {repr(obj)}")
        logger.error(traceback.format_exc())
        raise

fastapi_encoders.jsonable_encoder = debug_jsonable_encoder

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown events."""
    # Startup
    logger.info("Starting Surface Mapper Backend API")
    yield
    # Shutdown
    logger.info("Shutting down Surface Mapper Backend API")

app = FastAPI(
    title="Surface Mapper Backend API",
    description="Backend API for surface analysis and processing",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception logging middleware
@app.middleware("http")
async def log_exceptions_middleware(request: Request, call_next):
    try:
        response = await call_next(request)
        return response
    except Exception as exc:
        logger.error(f"Exception in {request.method} {request.url.path}: {exc}")
        logger.error(traceback.format_exc())
        raise

# Include routers
app.include_router(surfaces.router, prefix="/api/surfaces", tags=["surfaces"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Surface Mapper Backend API"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Surface Mapper Backend API", 
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "api_base": "/api"
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
            "surfaces": "/api/surfaces",
            "coordinate_systems": "/api/surfaces/coordinate-systems",
            "coordinate_transform": "/api/surfaces/coordinate-transform"
        },
        "features": [
            "PLY file upload and validation",
            "Surface processing and analysis",
            "Volume and thickness calculations",
            "3D visualization data preparation",
            "Coordinate system transformations"
        ]
    }

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