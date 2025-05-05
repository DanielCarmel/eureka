"""
Ingest router for the Eureka API Server
"""
import json
import os
from typing import Any, Dict, List, Optional, Union

from app.models.api_models import (
    IngestConfluenceRequest,
    IngestJiraRequest,
    IngestResponse,
    IngestS3Request,
)
from app.services.eureka_service import EurekaService
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from loguru import logger
from pydantic import BaseModel


# Router
router = APIRouter()


@router.post("/ingest/s3", response_model=IngestResponse)
async def ingest_s3(request: IngestS3Request):
    """
    Ingest data from S3
    """
    try:
        # Get the service instance
        service = await EurekaService.get_instance()

        # Ingest data directly using the service
        logger.info(f"Ingesting S3 data from bucket: {request.bucket}")

        result = await service.ingest_documents(
            source_type="s3",
            bucket=request.bucket,
            prefix=request.prefix,
            max_files=request.max_files
        )

        return IngestResponse(
            status=result.get("status", "success"),
            message=result.get("message", ""),
            document_ids=result.get("document_ids", []),
            errors=result.get("errors", [])
        )

    except Exception as e:
        logger.error(f"Error ingesting S3 data: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting S3 data: {str(e)}"
        )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    collection_name: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Ingest a file directly via the API
    """
    try:
        # Get the service instance
        service = await EurekaService.get_instance()

        # Read the file content
        content = await file.read()
        file_metadata = {}

        if metadata:
            try:
                file_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                logger.warning(f"Invalid metadata JSON: {metadata}")

        if not file_metadata:
            file_metadata = {
                "filename": file.filename,
                "content_type": file.content_type
            }

        # Ingest file directly using the service
        logger.info(f"Ingesting file: {file.filename}")

        result = await service.ingest_documents(
            source_type="file",
            file_content=content,
            file_name=file.filename,
            content_type=file.content_type,
            metadata=file_metadata,
            collection_name=collection_name or "default"
        )

        return IngestResponse(
            status=result.get("status", "success"),
            message=result.get("message", ""),
            document_ids=result.get("document_ids", []),
            errors=result.get("errors", [])
        )

    except Exception as e:
        logger.error(f"Error ingesting file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error ingesting file: {str(e)}"
        )
