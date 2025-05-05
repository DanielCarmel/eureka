"""
Utility functions for the API server
"""
import os
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException, status
from loguru import logger


async def make_service_request(
    url: str,
    method: str = "GET",
    json_data: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    expected_status_code: int = 200,
) -> Dict[str, Any]:
    """
    Make a request to a service and handle common errors
    
    Args:
        url: The URL to make the request to
        method: HTTP method (GET, POST, etc.)
        json_data: Optional JSON data to send
        headers: Optional headers to include
        timeout: Request timeout in seconds
        expected_status_code: Expected HTTP status code
        
    Returns:
        JSON response data
        
    Raises:
        HTTPException: If an error occurs
    """
    if headers is None:
        headers = {}
        
    # Include API key if set
    api_key = os.getenv("API_KEY")
    if api_key and "X-API-Key" not in headers:
        headers["X-API-Key"] = api_key
        
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                url,
                json=json_data,
                headers=headers
            )
            
            if response.status_code != expected_status_code:
                logger.error(f"Service error: {response.status_code}, {response.text}")
                status_code = status.HTTP_502_BAD_GATEWAY
                detail = f"Error calling service: {response.text}"
                
                # Map some common status codes to appropriate FastAPI status codes
                if response.status_code == 404:
                    status_code = status.HTTP_404_NOT_FOUND
                    detail = "Resource not found"
                elif response.status_code == 401 or response.status_code == 403:
                    status_code = status.HTTP_403_FORBIDDEN
                    detail = "Authentication or permission error"
                elif response.status_code == 400:
                    status_code = status.HTTP_400_BAD_REQUEST
                    detail = "Invalid request"
                
                # Try to get details from response
                try:
                    error_data = response.json()
                    if "detail" in error_data:
                        detail = error_data["detail"]
                    elif "message" in error_data:
                        detail = error_data["message"]
                except Exception:
                    pass
                    
                raise HTTPException(
                    status_code=status_code,
                    detail=detail
                )
                
            # Return the JSON response or empty dict if no content
            if response.content:
                return response.json()
            return {}
            
    except httpx.TimeoutException:
        logger.error(f"Timeout when calling {url}")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Request timed out"
        )
    except httpx.HTTPError as e:
        logger.error(f"HTTP error when calling {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"HTTP error: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error when calling {url}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}"
        )