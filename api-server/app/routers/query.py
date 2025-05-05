"""
Query router for the Eureka API Server
"""
import os
from typing import Any, Dict, List, Optional

import httpx
from app.models.api_models import QueryRequest, QueryResponse, Source
from app.services.eureka_service import EurekaService
from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel

# Router
router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Send a natural language query to the RAG system
    """
    try:
        # Get the service instance
        service = await EurekaService.get_instance()
        
        # Process the query using direct component integration
        logger.info(f"Processing query: {request.question}")
        
        result = await service.process_query(
            question=request.question,
            max_context_chunks=request.max_context_chunks,
            temperature=request.temperature,
            collection_name=request.collection_name
        )
        
        # Convert sources to the expected format if needed
        sources = []
        for source in result.get("sources", []):
            sources.append(Source(
                document_id=source.get("id", ""),
                source_type=source.get("source_type", "unknown"),
                metadata=source.get("metadata", {}),
                text_snippet=source.get("text", ""),
                url=source.get("url", "")
            ))
            
        return QueryResponse(
            answer=result.get("answer", "No answer provided"),
            sources=sources,
            metadata=result.get("metadata", {})
        )
            
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )