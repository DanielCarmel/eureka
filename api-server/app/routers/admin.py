"""
Admin router for the Eureka API Server
"""
import os
from typing import Any, Dict, List, Optional

from app.models.api_models import (
    CollectionListResponse,
    CollectionStatsResponse,
    DeleteCollectionResponse,
    HealthResponse,
)
from app.services.eureka_service import EurekaService
from fastapi import APIRouter, HTTPException, status
from loguru import logger
from pydantic import BaseModel


# Router
router = APIRouter()


@router.get("/admin/health", response_model=HealthResponse)
async def check_system_health():
    """
    Check the health of all system components
    """
    # Get the service instance
    service = await EurekaService.get_instance()
    
    # Check health directly using the service
    health_data = await service.check_health()
    
    return HealthResponse(
        status=health_data.get("status", "unknown"),
        components=health_data.get("components", {}),
        details=health_data.get("details", {})
    )


@router.get("/admin/collections", response_model=CollectionListResponse)
async def list_collections():
    """
    List all collections in the vector database
    """
    try:
        # Get the service instance
        service = await EurekaService.get_instance()

        # Get collections directly using the service
        collections = await service.get_collections()
        return CollectionListResponse(collections=collections)

    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {str(e)}"
        )


@router.get("/admin/collections/{collection_name}", response_model=CollectionStatsResponse)
async def get_collection_stats(collection_name: str):
    """
    Get statistics for a specific collection
    """
    try:
        # Get the service instance
        service = await EurekaService.get_instance()
        
        # Get collection stats directly using the service
        stats = await service.get_collection_stats(collection_name)
        
        return CollectionStatsResponse(
            collection_name=collection_name,
            document_count=stats.get("document_count", 0),
            embedding_count=stats.get("embedding_count", 0),
            metadata=stats.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Error getting collection stats: {str(e)}")
        
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found"
            )
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting collection stats: {str(e)}"
        )


@router.delete("/admin/collections/{collection_name}", response_model=DeleteCollectionResponse)
async def delete_collection(collection_name: str):
    """
    Delete a collection from the vector database
    """
    try:
        # Get the service instance
        service = await EurekaService.get_instance()
        
        # Delete collection directly using the service
        result = await service.delete_collection(collection_name)
        
        return DeleteCollectionResponse(
            status=result.get("status", "unknown"),
            message=result.get("message", "")
        )
        
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        
        if "not found" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{collection_name}' not found"
            )
            
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection: {str(e)}"
        )
