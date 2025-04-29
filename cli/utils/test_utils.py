import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional

logger = logging.getLogger("cli.test_utils")

def test_query(question: str, sources: Optional[List[str]] = None, 
               collection_name: str = "documents", max_context_chunks: int = 5) -> Dict[str, Any]:
    """
    Test a query against the RAG Orchestrator
    
    Args:
        question: Question to ask
        sources: Optional list of specific sources to query (jira, confluence, s3)
        collection_name: Vector DB collection to query
        max_context_chunks: Number of context chunks to retrieve
        
    Returns:
        Response from the RAG Orchestrator
    """
    try:
        # Get RAG orchestrator URL from environment or use default
        rag_url = os.environ.get("RAG_ORCHESTRATOR_URL", "http://localhost:8888")
        api_key = os.environ.get("API_KEY", "")
        
        # Prepare the request
        url = f"{rag_url}/query"
        headers = {"X-API-Key": api_key}
        payload = {
            "question": question,
            "collection_name": collection_name,
            "max_context_chunks": max_context_chunks,
            "include_sources": True
        }
        
        # Add sources if specified
        if sources:
            payload["sources"] = sources
        
        # Send the request
        logger.info(f"Sending query to RAG Orchestrator: {question}")
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        # Return the response
        result = response.json()
        logger.info(f"Received response from RAG Orchestrator: {len(result.get('answer', ''))}")
        return result
        
    except Exception as e:
        logger.error(f"Error testing query: {e}")
        raise