"""
Core services for the API server to directly interact with Eureka components
"""
import os
import asyncio
from typing import Dict, Any, List, Optional
from loguru import logger

# Import components directly
from llm.llm_client import LLMClient
from orchestrator.main import MCPOrchestrator
import mcp.s3 as s3
import mcp.slack as slack


class EurekaService:
    """
    Main service class that integrates with all Eureka components
    """
    _instance = None

    def __init__(self):
        self.llm_client = None
        self.orchestrator = None
        self.initialized = False

    @classmethod
    async def get_instance(cls):
        """Singleton pattern to get the service instance"""
        if cls._instance is None:
            cls._instance = EurekaService()

        if not cls._instance.initialized:
            await cls._instance.initialize()

        return cls._instance

    async def initialize(self):
        """Initialize all components"""
        try:
            # Initialize LLM client
            model_path = os.getenv("MODEL_PATH", "/app/models/mistral-7b-instruct.gguf")
            model_type = os.getenv("MODEL_TYPE", "llama")

            logger.info(f"Initializing LLM client with {model_type} model from {model_path}")

            self.llm_client = await LLMClient.create(
                provider="llamacpp",
                model=model_type,
                model_path=model_path,
                temperature=float(os.getenv("TEMPERATURE", "0.7")),
                max_tokens=int(os.getenv("MAX_TOKENS", "512")),
                n_ctx=int(os.getenv("N_CTX", "4096")),
                n_gpu_layers=int(os.getenv("N_GPU_LAYERS", "-1")),
            )
            
            # Initialize orchestrator
            self.orchestrator = MCPOrchestrator(llm_client=self.llm_client)
            
            # Load configuration from environment
            config = self._load_config_from_env()
            
            # Connect to all available servers
            logger.info("Connecting to MCP servers...")
            await self._connect_to_servers(config)
            
            # Collect capabilities
            await self.orchestrator.collect_capabilities()
            
            self.initialized = True
            logger.info("Eureka service initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Eureka service: {str(e)}")
            raise
    
    def _load_config_from_env(self) -> Dict[str, Dict[str, Any]]:
        """Load server configuration from environment variables"""
        config = {
            "servers": {}
        }
        
        # Jira configuration
        if os.getenv("JIRA_URL"):
            config["servers"]["jira"] = {
                "type": "jira",
                "url": os.getenv("JIRA_URL"),
                "username": os.getenv("JIRA_USERNAME"),
                "token": os.getenv("JIRA_TOKEN"),
            }
            
        # Confluence configuration
        if os.getenv("CONFLUENCE_URL"):
            config["servers"]["confluence"] = {
                "type": "confluence",
                "url": os.getenv("CONFLUENCE_URL"),
                "username": os.getenv("CONFLUENCE_USERNAME"),
                "token": os.getenv("CONFLUENCE_TOKEN"),
            }
            
        # S3 configuration
        if os.getenv("AWS_ACCESS_KEY_ID"):
            config["servers"]["s3"] = {
                "type": "s3",
                "region": os.getenv("AWS_DEFAULT_REGION", "us-east-1"),
                "access_key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "endpoint_url": os.getenv("AWS_S3_ENDPOINT_URL", ""),
            }
            
        # Slack configuration (if enabled)
        if os.getenv("SLACK_BOT_TOKEN"):
            config["servers"]["slack"] = {
                "type": "slack",
                "bot_token": os.getenv("SLACK_BOT_TOKEN"),
            }
            
        return config
    
    async def _connect_to_servers(self, config: Dict[str, Dict[str, Any]]) -> bool:
        """Connect to all configured servers"""
        success = True
        
        for server_name, server_config in config.get("servers", {}).items():
            try:
                logger.info(f"Connecting to {server_name}...")
                
                server_type = server_config.get("type", "").lower()
                
                if server_type == "jira":
                    # Connect to Jira
                    await self.orchestrator.connect_to_jira(
                        server_config["url"],
                        server_config["username"],
                        server_config["token"],
                        session_name=server_name,
                    )
                elif server_type == "confluence":
                    # Connect to Confluence
                    await self.orchestrator.connect_to_confluence(
                        server_config["url"],
                        server_config["username"],
                        server_config["token"],
                        session_name=server_name,
                    )
                elif server_type == "s3":
                    # Connect to S3
                    await self.orchestrator.connect_to_s3(
                        server_config.get("access_key", ""),
                        server_config.get("secret_key", ""),
                        server_config.get("region", "us-east-1"),
                        server_config.get("endpoint_url", ""),
                        session_name=server_name,
                    )
                elif server_type == "slack":
                    # Connect to Slack
                    await self.orchestrator.connect_to_slack(
                        server_config["bot_token"],
                        session_name=server_name,
                    )
                else:
                    logger.warning(f"Unknown server type '{server_type}' for {server_name}")
                    success = False
                    continue
                    
                logger.info(f"Connected to {server_name} successfully")
                
            except Exception as e:
                logger.error(f"Error connecting to {server_name}: {str(e)}")
                success = False
                
        return success
    
    async def process_query(self, question: str, max_context_chunks: int = 5, 
                           temperature: float = 0.7, collection_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a query using the RAG orchestrator
        """
        if not self.initialized:
            await self.initialize()
            
        # Set temperature if different from default
        if temperature != 0.7:
            self.llm_client.temperature = temperature
            
        # Process the query
        response = await self.orchestrator.process_query(
            question, 
            max_context_chunks=max_context_chunks,
            collection_name=collection_name
        )
        
        # Get sources if available
        sources = self.orchestrator.get_last_sources()
        
        result = {
            "answer": response,
            "sources": sources,
            "metadata": {
                "model": self.llm_client.model,
                "temperature": self.llm_client.temperature,
            }
        }
        
        return result
    
    async def ingest_documents(self, source_type: str, **kwargs) -> Dict[str, Any]:
        """
        Ingest documents from various sources
        """
        if not self.initialized:
            await self.initialize()
            
        result = {
            "status": "success",
            "document_ids": [],
            "errors": []
        }
        
        try:
            if source_type == "jira":
                # Ingest from Jira
                project_key = kwargs.get("project_key")
                max_issues = kwargs.get("max_issues", 100)
                
                document_ids = await self.orchestrator.ingest_jira_issues(
                    project_key, max_issues=max_issues
                )
                result["document_ids"] = document_ids
                result["message"] = f"Successfully ingested {len(document_ids)} Jira issues"
                
            elif source_type == "confluence":
                # Ingest from Confluence
                space_key = kwargs.get("space_key")
                max_pages = kwargs.get("max_pages", 100)
                
                document_ids = await self.orchestrator.ingest_confluence_pages(
                    space_key, max_pages=max_pages
                )
                result["document_ids"] = document_ids
                result["message"] = f"Successfully ingested {len(document_ids)} Confluence pages"
                
            elif source_type == "s3":
                # Ingest from S3
                bucket = kwargs.get("bucket")
                prefix = kwargs.get("prefix")
                max_files = kwargs.get("max_files", 50)
                
                document_ids = await self.orchestrator.ingest_s3_files(
                    bucket, prefix=prefix, max_files=max_files
                )
                result["document_ids"] = document_ids
                result["message"] = f"Successfully ingested {len(document_ids)} S3 documents"
                
            elif source_type == "file":
                # Ingest from file
                file_content = kwargs.get("file_content")
                file_name = kwargs.get("file_name")
                content_type = kwargs.get("content_type")
                metadata = kwargs.get("metadata", {})
                collection_name = kwargs.get("collection_name", "default")
                
                document_id = await self.orchestrator.ingest_file(
                    file_content, file_name, content_type, 
                    metadata=metadata, collection_name=collection_name
                )
                result["document_ids"] = [document_id]
                result["message"] = f"Successfully ingested file: {file_name}"
                
            else:
                result["status"] = "error"
                result["message"] = f"Unknown source type: {source_type}"
                result["errors"].append(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            logger.error(f"Error ingesting from {source_type}: {str(e)}")
            result["status"] = "error"
            result["message"] = f"Error ingesting from {source_type}"
            result["errors"].append(str(e))
            
        return result
    
    async def get_collections(self) -> List[str]:
        """Get all collections from the vector database"""
        if not self.initialized:
            await self.initialize()
            
        return await self.orchestrator.get_collections()
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics for a specific collection"""
        if not self.initialized:
            await self.initialize()
            
        return await self.orchestrator.get_collection_stats(collection_name)
    
    async def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """Delete a collection from the vector database"""
        if not self.initialized:
            await self.initialize()
            
        success = await self.orchestrator.delete_collection(collection_name)
        
        result = {
            "status": "success" if success else "error",
            "message": f"Collection '{collection_name}' deleted successfully" if success else f"Failed to delete collection '{collection_name}'"
        }
        
        return result
    
    async def check_health(self) -> Dict[str, Any]:
        """Check the health of all components"""
        try:
            if not self.initialized:
                # Try to initialize
                await self.initialize()
                
            overall_status = "healthy"
            components = {
                "api_server": "up",
                "llm_engine": "up" if self.llm_client else "down",
                "orchestrator": "up" if self.orchestrator else "down",
            }
            
            # Check vector DB connection
            try:
                collections = await self.orchestrator.get_collections()
                components["vector_db"] = "up"
            except Exception as e:
                logger.error(f"Vector DB health check failed: {str(e)}")
                components["vector_db"] = "down"
                overall_status = "degraded"
            
            # Check server connections
            connected_servers = list(self.orchestrator.sessions.keys()) if self.orchestrator else []
            components["connected_servers"] = connected_servers
            
            if not connected_servers:
                components["mcp_server"] = "degraded"
                overall_status = "degraded"
            else:
                components["mcp_server"] = "up"
            
            return {
                "status": overall_status,
                "components": components,
                "details": {
                    "llm_model": self.llm_client.model if self.llm_client else "unknown",
                    "connected_servers": connected_servers,
                }
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "status": "unhealthy",
                "components": {
                    "api_server": "up",
                    "llm_engine": "unknown",
                    "mcp_server": "unknown",
                    "vector_db": "unknown",
                    "orchestrator": "unknown",
                },
                "details": {
                    "error": str(e)
                }
            }