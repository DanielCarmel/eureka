"""
LLM Client Data Models

This module defines the data models used by the LLM clients.
"""
from typing import Any, Dict

from pydantic import BaseModel


class ToolCall(BaseModel):
    """Represents a tool call for MCP server execution"""
    tool_name: str
    server_name: str
    arguments: Dict[str, Any]
    description: str = ""


class Message(BaseModel):
    """Represents a message in the conversation with the LLM"""
    role: str
    content: str
