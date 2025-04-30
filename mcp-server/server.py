import json
import logging
import os
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import jsonrpcserver
from jsonrpcserver.result import Success
import uvicorn

# Import connectors
from connectors.confluence_connector import ConfluenceConnector
from connectors.jira_connector import JiraConnector
from connectors.s3_connector import S3Connector
from fastapi import Body, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("logs/mcp-server.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("mcp-server")

app = FastAPI(
    title="Model Context Protocol Server",
    description="MCP Implementation for internal data sources",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize connectors with environment variables
jira_connector = JiraConnector(
    url=os.environ.get("JIRA_URL"),
    username=os.environ.get("JIRA_USERNAME"),
    token=os.environ.get("JIRA_TOKEN"),
)

confluence_connector = ConfluenceConnector(
    url=os.environ.get("CONFLUENCE_URL"),
    username=os.environ.get("CONFLUENCE_USERNAME"),
    token=os.environ.get("CONFLUENCE_TOKEN"),
)

s3_connector = S3Connector(
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_DEFAULT_REGION"),
    endpoint_url=os.environ.get("AWS_S3_ENDPOINT_URL")
)

# Register all connectors
connectors = {
    "jira": jira_connector,
    "confluence": confluence_connector,
    "s3": s3_connector,
}


# Define MCP request model
class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any]
    id: Optional[str] = None


# Define MCP methods
@jsonrpcserver.method
def query_context(
    query: str, sources: List[str] = None, max_results: int = 10
) -> Dict[str, Any]:
    """
    Query context from all registered sources or specified sources
    """
    logger.info(f"Processing context query: {query}")

    if not sources:
        sources = list(connectors.keys())

    results = []
    for source in sources:
        if source not in connectors:
            logger.warning(f"Unknown source: {source}")
            continue

        try:
            logger.info(f"Querying source: {source}")
            source_results = connectors[source].query(query, max_results=max_results)
            for result in source_results:
                result["source"] = source
            results.extend(source_results)
        except Exception as e:
            logger.error(f"Error querying {source}: {str(e)}")

    # Sort by relevance if results have a score
    if results and "score" in results[0]:
        results.sort(key=lambda x: x.get("score", 0), reverse=True)

    # Limit total results
    results = results[:max_results]

    # Return a proper jsonrpcserver.Success object
    return Success({"results": results, "count": len(results)})


@jsonrpcserver.method
def get_document(source: str, document_id: str) -> Dict[str, Any]:
    """
    Get a specific document by ID from a source
    """
    if source not in connectors:
        raise ValueError(f"Unknown source: {source}")

    document = connectors[source].get_document(document_id)
    # Return a proper jsonrpcserver.Success object
    return Success({"document": document})


@jsonrpcserver.method
def list_sources() -> Dict[str, Any]:
    """
    List all available data sources
    """
    # Return a proper jsonrpcserver.Success object
    return Success({"sources": list(connectors.keys())})


@jsonrpcserver.method
def health() -> Dict[str, Any]:
    """
    Check health of all connectors
    """
    status = {}
    all_healthy = True

    for name, connector in connectors.items():
        try:
            connector_status = connector.health_check()
            status[name] = connector_status
            if not connector_status.get("healthy", False):
                all_healthy = False
        except Exception as e:
            logger.error(f"Error checking health of {name}: {str(e)}")
            status[name] = {"healthy": False, "error": str(e)}
            all_healthy = False

    # Return a proper jsonrpcserver.Success object
    return Success({"healthy": all_healthy, "status": status})


# FastAPI endpoint that handles MCP requests
@app.post("/")
async def handle_mcp_request(request: Dict[str, Any] = Body(...)):
    logger.debug(f"Received request: {request}")

    # Process request with jsonrpcserver
    response = jsonrpcserver.dispatch(json.dumps(request))

    # Parse response
    if response:
        return json.loads(response)
    return {}


@app.get("/health")
async def health_check():
    """Simple HTTP health check endpoint"""
    return {"status": "ok"}


if __name__ == "__main__":
    logger.info("Starting MCP Server")
    uvicorn.run("server:app", host="0.0.0.0", port=5001, log_level="info")
