import asyncio
import json
import os
from typing import Any, Dict, List, Optional, Union


def format_search_results(results: Dict[str, Any]) -> str:
    """
    Format search results from multiple servers into a human-readable format.
    
    Args:
        results: Dictionary of server name to search results
        
    Returns:
        Formatted results as a string
    """
    output = []
    
    for server_name, result in results.items():
        output.append(f"=== {server_name} ===")
        output.append(str(result))
        output.append("")  # Empty line for spacing
    
    return "\n".join(output)

def save_results_to_file(results: Dict[str, Any], filename: str) -> str:
    """
    Save search results to a file.
    
    Args:
        results: Dictionary of search results
        filename: Name of the file to save to
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs("output", exist_ok=True)
    
    # Determine file path
    file_path = os.path.join("output", filename)
    
    # Save results
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return file_path

def load_server_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load server configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary of server configurations
    """
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config

def create_default_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Create a default configuration file if none exists.
    
    Args:
        config_path: Path to create the configuration file
        
    Returns:
        Default configuration dictionary
    """
    default_config = {
        "confluence": {
            "command": "python",
            "args": ["servers/confluence_server.py"],
            "env": {}
        },
        "notion": {
            "command": "python",
            "args": ["servers/notion_server.py"],
            "env": {}
        },
        "postgres": {
            "command": "python",
            "args": ["servers/postgres_server.py"],
            "env": {}
        }
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Write default configuration to file
    with open(config_path, "w") as f:
        json.dump(default_config, f, indent=2)
    
    return default_config

async def run_with_timeout(coroutine, timeout_seconds: float = 30.0):
    """
    Run a coroutine with a timeout.
    
    Args:
        coroutine: Coroutine to run
        timeout_seconds: Timeout in seconds
        
    Returns:
        Result of the coroutine
        
    Raises:
        asyncio.TimeoutError: If the coroutine times out
    """
    return await asyncio.wait_for(coroutine, timeout=timeout_seconds)