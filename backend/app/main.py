from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routes import surfaces

app = FastAPI(
    title="Surface Mapper API", 
    version="1.0.0",
    description="Surface Volume and Layer Thickness Analysis Tool API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(surfaces.router)

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker health checks"""
    return {"status": "healthy", "service": "surface-mapper-backend"}

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Surface Mapper Backend API", "version": "1.0.0"}

@app.get("/docs")
async def api_docs():
    """API documentation endpoint"""
    return {"message": "API documentation available at /docs"} 