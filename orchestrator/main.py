import asyncio
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from llm.llm_client import LLMClient, ToolCall
from mcp.client import ClientSession
from mcp.client.stdio import stdio_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MCPOrchestrator:
    """
    Orchestrator for connecting to and managing multiple MCP servers.
    Provides a unified interface for knowledge base queries across different sources
    using LLM guidance.
    """

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize the MCP orchestrator.

        Args:
            llm_client: Optional LLM client for query interpretation and response generation
        """
        self.sessions: Dict[str, ClientSession] = {}
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        self.llm_client = llm_client or LLMClient()

    async def connect_to_server(self, name: str, command: str, args: List[str],
                                env: Optional[Dict[str, str]] = None) -> ClientSession:
        """
        Connect to an MCP server using stdio transport.

        Args:
            name: A friendly name for the server connection
            command: The executable command to run the server
            args: Command-line arguments for the server
            env: Optional environment variables
            
        Returns:
            An initialized ClientSession
        """
        logger.info(f"Connecting to server {name} with command: {command} {' '.join(args)}")

        try:
            read, write = await stdio_client(command=command, args=args, env=env)
            session = ClientSession(read, write)
            await session.initialize()

            self.sessions[name] = session
            logger.info(f"Successfully connected to {name}")

            return session
        except Exception as e:
            logger.error(f"Failed to connect to server {name}: {str(e)}")
            raise

    async def disconnect_all(self):
        """Close all server connections."""
        for name, session in self.sessions.items():
            try:
                await session.shutdown()
                logger.info(f"Disconnected from {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from {name}: {str(e)}")

    async def collect_capabilities(self):
        """
        Collect tools and resources from all connected servers.
        """
        self.capabilities = {}

        for name, session in self.sessions.items():
            try:
                tools = await session.list_tools()
                resources = await session.list_resources()

                self.capabilities[name] = {
                    "tools": tools,
                    "resources": resources,
                    "session": session
                }

                logger.info(f"Collected capabilities from {name}: {len(tools)} tools, {len(resources)} resources")
            except Exception as e:
                logger.error(f"Error collecting capabilities from {name}: {str(e)}")

    async def process_query(self, query: str) -> str:
        """
        Process a user query using the LLM-guided workflow:
        1. Send query to LLM for interpretation
        2. Execute tool calls on appropriate servers
        3. Send results back to LLM for response generation

        Args:
            query: The user's query

        Returns:
            Generated response to the user query
        """
        logger.info(f"Processing query: {query}")

        try:
            # Step 1: Interpret the query using LLM
            tool_calls = await self.llm_client.interpret_query(query, self.capabilities)
            logger.info(f"LLM determined {len(tool_calls)} tool calls to execute")

            # Step 2: Execute the tool calls
            results = {}
            for i, tool_call in enumerate(tool_calls):
                try:
                    server_name = tool_call.server_name
                    tool_name = tool_call.tool_name
                    arguments = tool_call.arguments

                    # Make sure the server exists
                    if server_name not in self.sessions:
                        logger.warning(f"Unknown server: {server_name}")
                        results[f"{tool_name}({server_name})_{i}"] = f"Error: Unknown server '{server_name}'"
                        continue

                    # Call the tool
                    logger.info(f"Calling {tool_name} on {server_name} with arguments: {arguments}")
                    session = self.sessions[server_name]
                    response = await session.call_tool(tool_name, arguments)

                    # Store the result
                    results[f"{tool_name}({server_name})_{i}"] = response

                except Exception as e:
                    logger.error(f"Error executing tool call {tool_name} on {server_name}: {str(e)}")
                    results[f"{tool_name}({server_name})_{i}"] = f"Error: {str(e)}"

            # Step 3: Generate a response using the results
            final_response = await self.llm_client.generate_response(query, results)

            return final_response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I encountered an error processing your query: {str(e)}"

    async def get_resource(self, resource_uri: str, server_name: Optional[str] = None) -> Tuple[str, str]:
        """
        Retrieve a resource from a specific server.

        Args:
            resource_uri: The URI of the resource to retrieve
            server_name: Optional server name to target

        Returns:
            Tuple of (content, mime_type)
        """
        if server_name and server_name in self.sessions:
            session = self.sessions[server_name]
            return await session.read_resource(resource_uri)

        # Try each server if not specified
        for name, session in self.sessions.items():
            try:
                content, mime_type = await session.read_resource(resource_uri)
                return content, mime_type
            except Exception:
                # Skip to next server if this one can't provide the resource
                continue

        raise ValueError(f"No server could provide resource: {resource_uri}")

    async def call_tool(self, tool_name: str, server_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a specific tool on a specific server.

        Args:
            tool_name: The name of the tool to call
            server_name: The name of the server to call the tool on
            arguments: Arguments to pass to the tool

        Returns:
            Tool response
        """
        if server_name not in self.sessions:
            raise ValueError(f"Unknown server: {server_name}")

        session = self.sessions[server_name]
        return await session.call_tool(tool_name, arguments)
