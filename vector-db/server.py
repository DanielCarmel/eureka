import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Dict, List, Optional, Any

import uvicorn
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler(
            "logs/vector-db.log", maxBytes=10485760, backupCount=5
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("vector-db")

app = FastAPI(title="Vector Database API", description="ChromaDB Vector Database for RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ChromaDB
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "/app/data")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# Ensure data directory exists
os.makedirs(CHROMA_DB_PATH, exist_ok=True)

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Load sentence transformer model
try:
    logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

# Define API models
class Document(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any] = {}

class QueryRequest(BaseModel):
    query: str
    collection_name: str
    top_k: int = Field(default=5)
    filter: Optional[Dict[str, Any]] = None

class AddDocumentsRequest(BaseModel):
    collection_name: str
    documents: List[Document]

class DeleteRequest(BaseModel):
    collection_name: str
    ids: Optional[List[str]] = None
    filter: Optional[Dict[str, Any]] = None

# Helper functions
def get_or_create_collection(name: str):
    """Get an existing collection or create a new one"""
    try:
        return chroma_client.get_collection(name=name)
    except ValueError:
        return chroma_client.create_collection(name=name)

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts"""
    if not embedding_model:
        raise Exception("Embedding model not initialized")
    
    embeddings = embedding_model.encode(texts)
    return embeddings.tolist()

# API Endpoints
@app.post("/add")
async def add_documents(request: AddDocumentsRequest):
    """
    Add documents to a collection
    """
    try:
        collection = get_or_create_collection(request.collection_name)
        
        # Extract document fields
        ids = [doc.id for doc in request.documents]
        texts = [doc.text for doc in request.documents]
        metadatas = [doc.metadata for doc in request.documents]
        
        # Generate embeddings
        embeddings = generate_embeddings(texts)
        
        # Add to collection
        collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        
        return {"status": "success", "count": len(ids)}
    
    except Exception as e:
        logger.error(f"Error adding documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    """
    Query similar documents from a collection
    """
    try:
        collection = get_or_create_collection(request.collection_name)
        
        # Generate query embedding
        query_embedding = generate_embeddings([request.query])[0]
        
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=request.top_k,
            where=request.filter
        )
        
        # Format results
        documents = []
        for i in range(len(results["ids"][0])):
            documents.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": float(results["distances"][0][i]) if "distances" in results else None
            })
        
        return {
            "query": request.query,
            "documents": documents
        }
    
    except Exception as e:
        logger.error(f"Error querying: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/delete")
async def delete_documents(request: DeleteRequest):
    """
    Delete documents from a collection
    """
    try:
        collection = get_or_create_collection(request.collection_name)
        
        # Delete by IDs if provided
        if request.ids:
            collection.delete(ids=request.ids)
        # Delete by filter if provided
        elif request.filter:
            collection.delete(where=request.filter)
        # Otherwise raise error
        else:
            raise HTTPException(status_code=400, detail="Either ids or filter must be provided")
        
        return {"status": "success"}
    
    except Exception as e:
        logger.error(f"Error deleting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collections")
async def list_collections():
    """
    List all collections
    """
    try:
        collections = chroma_client.list_collections()
        return {"collections": [c.name for c in collections]}
    
    except Exception as e:
        logger.error(f"Error listing collections: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collection/{name}")
async def get_collection_info(name: str):
    """
    Get information about a collection
    """
    try:
        collection = get_or_create_collection(name)
        return {
            "name": name,
            "count": collection.count()
        }
    
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Check if the vector database is healthy
    """
    try:
        # Check if ChromaDB is working
        chroma_client.list_collections()
        
        # Check if embedding model is loaded
        if not embedding_model:
            return {"status": "unhealthy", "reason": "Embedding model not loaded"}
        
        return {"status": "healthy"}
    
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "unhealthy", "reason": str(e)}

@app.post("/reset/{name}")
async def reset_collection(name: str):
    """
    Reset (delete and recreate) a collection
    """
    try:
        try:
            chroma_client.delete_collection(name=name)
        except ValueError:
            pass  # Collection doesn't exist
        
        chroma_client.create_collection(name=name)
        return {"status": "success", "message": f"Collection {name} has been reset"}
    
    except Exception as e:
        logger.error(f"Error resetting collection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting Vector Database server")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, log_level="info")