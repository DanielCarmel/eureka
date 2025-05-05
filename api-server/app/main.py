"""
Eureka API Server Main Application
"""
import os
from typing import Any, Dict, Optional

from app.services.eureka_service import EurekaService
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Eureka API",
    description="API Server for the Eureka Enterprise RAG System",
    version="0.1.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Authentication
API_KEY = os.getenv("API_KEY", "")
api_key_header = APIKeyHeader(name="X-API-Key")

def verify_api_key(api_key: str = Depends(api_key_header)):
    if API_KEY and api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )
    return api_key

# Import routers
from app.routers import admin, ingest, query

# Register routers
app.include_router(query.router, prefix="/api/v1", tags=["Query"], dependencies=[Depends(verify_api_key)])
app.include_router(ingest.router, prefix="/api/v1", tags=["Ingest"], dependencies=[Depends(verify_api_key)])
app.include_router(admin.router, prefix="/api/v1", tags=["Admin"], dependencies=[Depends(verify_api_key)])

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint providing basic info about the API"""
    return {
        "name": "Eureka API Server",
        "description": "API Server for the Eureka Enterprise RAG System",
        "version": "0.1.0",
        "status": "running",
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for the API server"""
    try:
        # Get the service instance and check health
        service = await EurekaService.get_instance()
        health_data = await service.check_health()
        
        return {
            "status": health_data.get("status", "healthy"),
            "components": health_data.get("components", {"api_server": "up"}),
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "degraded",
            "components": {
                "api_server": "up",
                "error": str(e)
            }
        }

# Lifecycle event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    try:
        logger.info("Initializing Eureka services...")
        # This will initialize the service and connect to components
        await EurekaService.get_instance()
        logger.info("Eureka services initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")
        # Don't raise - let the app start anyway
        # Components will be initialized on first request

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    logger.info("Shutting down Eureka API Server...")
    # Any cleanup can be done here

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred"},
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=9000, reload=True)