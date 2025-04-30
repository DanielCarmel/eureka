import logging
import os
import time
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            "logs/rag-orchestrator.log", maxBytes=10485760, backupCount=5
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("rag-orchestrator")

app = FastAPI(
    title="RAG Orchestration API", description="Retrieval Augmented Generation System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key Security
API_KEY = os.environ.get("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")


async def get_api_key(api_key: str = Security(api_key_header)):
    if not API_KEY:
        logger.warning("API KEY environment variable not set")
        return api_key

    if api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )
    return api_key


# Service URLs from environment variables
LLM_ENGINE_URL = os.environ.get("LLM_ENGINE_URL", "http://localhost:8080")
MCP_SERVER_URL = os.environ.get("MCP_SERVER_URL", "http://localhost:5001")
VECTOR_DB_URL = os.environ.get("VECTOR_DB_URL", "http://localhost:8000")


# Define request and response models
class QueryRequest(BaseModel):
    question: str
    sources: Optional[
        List[str]
    ] = None  # Optional list of sources to query (jira, confluence, s3)
    max_context_chunks: int = Field(default=5)
    collection_name: str = Field(default="documents")
    include_sources: bool = Field(default=True)


class QueryResponse(BaseModel):
    question: str
    answer: str
    context_chunks: Optional[List[Dict[str, Any]]] = None
    processing_time: float


# Helper classes for services
class LLMService:
    @staticmethod
    async def generate_answer(question: str, context: str):
        """Generate an answer using the LLM Engine"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Create a prompt with context
                prompt = f"""I need you to answer the question based on the given context.
If the answer cannot be determined from the context, say "I don't have enough information to answer this question."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                # Use the OpenAI-compatible chat completion endpoint
                response = await client.post(
                    f"{LLM_ENGINE_URL}/v1/chat/completions",
                    json={
                        "model": "local",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 512,
                        "temperature": 0.7,
                    },
                )

                if response.status_code != 200:
                    logger.error(f"Error from LLM Engine: {response.text}")
                    return "Sorry, I couldn't generate an answer at this time."

                response_data = response.json()
                return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM service error: {str(e)}")
            return "Sorry, I encountered an error while generating your answer."


class MCPService:
    @staticmethod
    async def query_context(
        question: str, sources: Optional[List[str]] = None, max_results: int = 5
    ):
        """Query context from MCP Server"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create JSON-RPC request
                request_data = {
                    "jsonrpc": "2.0",
                    "method": "query_context",
                    "params": {"query": question, "max_results": max_results},
                    "id": str(int(time.time())),
                }

                if sources:
                    request_data["params"]["sources"] = sources

                # Send request to MCP Server
                response = await client.post(MCP_SERVER_URL, json=request_data)

                if response.status_code != 200:
                    logger.error(f"Error from MCP Server: {response.text}")
                    return []

                # Parse response
                response_data = response.json()
                if "result" in response_data and "results" in response_data["result"]:
                    return response_data["result"]["results"]
                else:
                    logger.error(f"Unexpected MCP response format: {response_data}")
                    return []
        except Exception as e:
            logger.error(f"MCP service error: {str(e)}")
            return []


class VectorDBService:
    @staticmethod
    async def query_similar_documents(
        question: str, collection_name: str, top_k: int = 5
    ):
        """Query similar documents from Vector DB"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Create query request
                request_data = {
                    "query": question,
                    "collection_name": collection_name,
                    "top_k": top_k,
                }

                # Send request to Vector DB
                response = await client.post(
                    f"{VECTOR_DB_URL}/query", json=request_data
                )

                if response.status_code != 200:
                    logger.error(f"Error from Vector DB: {response.text}")
                    return []

                # Parse response
                response_data = response.json()
                if "documents" in response_data:
                    return response_data["documents"]
                else:
                    logger.error(
                        f"Unexpected Vector DB response format: {response_data}"
                    )
                    return []
        except Exception as e:
            logger.error(f"Vector DB service error: {str(e)}")
            return []


# API Endpoints
@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, api_key: str = Security(get_api_key)):
    """
    Process a question with RAG:
    1. Query for context from internal sources
    2. Retrieve similar documents from vector store
    3. Format context for the LLM
    4. Generate answer using the LLM
    """
    start_time = time.time()

    try:
        logger.info(f"Processing question: {request.question}")

        # Step 1: Query MCP for context
        mcp_results = await MCPService.query_context(
            request.question,
            sources=request.sources,
            max_results=request.max_context_chunks,
        )

        # Step 2: Query Vector DB for similar documents
        vector_results = await VectorDBService.query_similar_documents(
            request.question,
            collection_name=request.collection_name,
            top_k=request.max_context_chunks,
        )

        # Step 3: Format context
        context_chunks = []
        context_text = ""

        # Format MCP results
        for i, result in enumerate(mcp_results):
            chunk = {
                "source": f"{result.get('source', 'unknown')}-{result.get('id', i)}",
                "title": result.get("title", "Untitled"),
                "content": result.get("content", ""),
                "url": result.get("url", ""),
                "source_type": "mcp",
            }
            context_chunks.append(chunk)
            context_text += f"\n\n[MCP DOCUMENT {i + 1}]\nTitle: {chunk['title']}\nSource: {chunk['source']}\n{chunk['content']}"

        # Format Vector DB results
        for i, result in enumerate(vector_results):
            chunk = {
                "source": f"vector-{result.get('id', i)}",
                "title": result.get("metadata", {}).get("title", "Untitled"),
                "content": result.get("text", ""),
                "metadata": result.get("metadata", {}),
                "source_type": "vector",
            }
            context_chunks.append(chunk)
            context_text += f"\n\n[VECTOR DOCUMENT {i + 1}]\nTitle: {chunk['title']}\nSource: {chunk['source']}\n{chunk['content']}"

        # Step 4: Generate answer using LLM
        answer = await LLMService.generate_answer(request.question, context_text)

        # Calculate processing time
        processing_time = time.time() - start_time

        response = {
            "question": request.question,
            "answer": answer,
            "processing_time": processing_time,
        }

        # Include context chunks if requested
        if request.include_sources:
            response["context_chunks"] = context_chunks

        logger.info(f"Answered question in {processing_time:.2f}s")
        return response

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Check if all services are healthy
    """
    services_status = {}
    all_healthy = True

    # Check LLM Engine
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{LLM_ENGINE_URL}/health")
            if response.status_code == 200:
                services_status["llm_engine"] = {"status": "healthy"}
            else:
                services_status["llm_engine"] = {
                    "status": "unhealthy",
                    "reason": response.text,
                }
                all_healthy = False
    except Exception as e:
        services_status["llm_engine"] = {"status": "unhealthy", "reason": str(e)}
        all_healthy = False

    # Check MCP Server
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{MCP_SERVER_URL}/health")
            if response.status_code == 200:
                services_status["mcp_server"] = {"status": "healthy"}
            else:
                services_status["mcp_server"] = {
                    "status": "unhealthy",
                    "reason": response.text,
                }
                all_healthy = False
    except Exception as e:
        services_status["mcp_server"] = {"status": "unhealthy", "reason": str(e)}
        all_healthy = False

    # Check Vector DB
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{VECTOR_DB_URL}/health")
            if response.status_code == 200:
                services_status["vector_db"] = {"status": "healthy"}
            else:
                services_status["vector_db"] = {
                    "status": "unhealthy",
                    "reason": response.text,
                }
                all_healthy = False
    except Exception as e:
        services_status["vector_db"] = {"status": "unhealthy", "reason": str(e)}
        all_healthy = False

    status = "healthy" if all_healthy else "unhealthy"
    return {"status": status, "services": services_status}


if __name__ == "__main__":
    logger.info("Starting RAG Orchestrator")
    uvicorn.run("main:app", host="0.0.0.0", port=8001, log_level="info")
