#!/usr/bin/env python3
"""
MCP Knowledge Base Client with LLM Integration

This application connects to multiple MCP servers (S3, Slack)
and provides a unified interface for knowledge base queries guided by an LLM.
"""

import argparse
import asyncio
import logging
import os
import sys
from typing import Any, Dict

from helper_functions import (
    create_default_config,
    load_server_config,
    save_results_to_file,
)
from rich.console import Console
from rich.panel import Panel

from llm.llm_client import LLMClient
from orchestrator import MCPOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("mcp-kb-client.log")],
)
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = "config/servers.json"
DEFAULT_TIMEOUT = 60.0  # seconds

# Initialize Rich console for prettier output
console = Console()


async def connect_to_servers(
    orchestrator: MCPOrchestrator, config: Dict[str, Dict[str, Any]]
) -> bool:
    """
    Connect to all configured MCP servers.

    Args:
        orchestrator: The MCP orchestrator instance
        config: Server configuration dictionary

    Returns:
        True if all servers connected successfully, False otherwise
    """
    success = True

    for server_name, server_config in config.items():
        try:
            command = server_config.get("command", "python")
            args = server_config.get("args", [])
            env = server_config.get("env", {})

            await orchestrator.connect_to_server(server_name, command, args, env)
        except Exception as e:
            logger.error(f"Failed to connect to {server_name}: {str(e)}")
            success = False

    return success


async def run_interactive_mode(orchestrator: MCPOrchestrator):
    """
    Run the client in interactive mode, prompting the user for queries.

    Args:
        orchestrator: The MCP orchestrator instance
    """
    console.print(
        Panel.fit(
            "[bold blue]MCP Knowledge Base Client with LLM Integration[/bold blue]\n\n"
            "Type [bold green]'exit'[/bold green] or [bold green]'quit'[/bold green] to terminate\n"
            "Type [bold green]'help'[/bold green] for available commands",
            title="Interactive Mode",
        )
    )

    while True:
        try:
            command = console.input(
                "\n[bold yellow]Enter query:[/bold yellow] "
            ).strip()

            if command.lower() in ("exit", "quit"):
                break

            if command.lower() == "help":
                print_help()
                continue

            if command.lower() == "servers":
                console.print("\n[bold]Connected servers:[/bold]")
                for server_name in orchestrator.sessions.keys():
                    console.print(f"- {server_name}")
                continue

            if command.lower() == "reset":
                orchestrator.llm_client.reset_conversation()
                console.print("[green]Conversation history reset.[/green]")
                continue

            # Process the query using LLM-guided workflow
            console.print(f"\n[bold]Processing query:[/bold] {command}")

            with console.status("[bold green]Thinking...[/bold green]"):
                response = await orchestrator.process_query(command)

            # Display the response
            console.print(Panel(response, title="Response", title_align="left"))

        except KeyboardInterrupt:
            console.print("\n[yellow]Operation cancelled.[/yellow]")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")


def print_help():
    """Print available commands for interactive mode."""
    console.print(
        Panel.fit(
            "help                      - Show this help message\n"
            "servers                   - List connected servers\n"
            "reset                     - Reset conversation history\n"
            "exit, quit                - Exit the program",
            title="Available Commands",
        )
    )


async def main():
    """Main entry point for the MCP Knowledge Base Client."""
    parser = argparse.ArgumentParser(
        description="MCP Knowledge Base Client with LLM Integration"
    )
    parser.add_argument(
        "query", nargs="?", help="The search query (omit for interactive mode)"
    )
    parser.add_argument("--output", "-o", help="Save results to a file")
    parser.add_argument(
        "--config", "-c", default=CONFIG_FILE, help="Path to server configuration file"
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help="Timeout in seconds for server operations",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    # LLM provider and model arguments
    llm_group = parser.add_argument_group("LLM Options")
    llm_group.add_argument(
        "--provider",
        default="ollama",
        choices=["llamacpp", "ollama"],
        help="LLM provider to use (default: ollama)",
    )
    llm_group.add_argument(
        "--model",
        default="llama2",
        help="Model to use with the selected provider (default: llama2)",
    )
    llm_group.add_argument(
        "--model-path", help="Path to model file (required for llamacpp provider)"
    )
    llm_group.add_argument(
        "--api-base", help="Base URL for the API (for local/custom endpoints)"
    )
    llm_group.add_argument("--api-key", help="API key if needed by the provider")
    llm_group.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperature for response generation (0.0-1.0, default: 0.2)",
    )
    llm_group.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate in responses (default: 2048)",
    )

    # Additional args for local models
    local_group = parser.add_argument_group("Local Model Options")
    local_group.add_argument(
        "--n-ctx",
        type=int,
        default=4096,
        help="Context window size for llama.cpp models (default: 4096)",
    )
    local_group.add_argument(
        "--n-gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU for llama.cpp (-1 = all, default: -1)",
    )
    local_group.add_argument(
        "--verbose", action="store_true", help="Enable verbose output for local models"
    )

    args = parser.parse_args()

    # Set debug logging if requested
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create appropriate LLM client based on provider
    try:
        console.print(f"[bold]Initializing {args.provider} LLM client...[/bold]")

        # Create LLM client with the specified provider
        llm_client = await LLMClient.create(
            provider=args.provider,
            model=args.model,
            model_path=args.model_path,
            api_base=args.api_base,
            api_key=args.api_key,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=args.verbose,
        )

        # Log successful initialization
        console.print(
            f"[green]Successfully initialized {args.provider} with model {args.model}[/green]"
        )

    except Exception as e:
        console.print(f"[bold red]Error initializing LLM client:[/bold red] {str(e)}")
        return 1

    # Create orchestrator with LLM client
    orchestrator = MCPOrchestrator(llm_client=llm_client)

    try:
        # Load or create configuration
        if os.path.exists(args.config):
            config = load_server_config(args.config)
        else:
            logger.info(
                f"Configuration file not found. Creating default at {args.config}"
            )
            config = create_default_config(args.config)

        # Connect to servers
        console.print("[bold]Connecting to MCP servers...[/bold]")
        success = await connect_to_servers(orchestrator, config)
        if not success:
            logger.warning(
                "Some servers failed to connect. Continuing with available servers."
            )

        # Collect capabilities
        await orchestrator.collect_capabilities()

        if args.query:
            # Run in command-line mode with a single query
            console.print(f"[bold]Processing query:[/bold] {args.query}")

            # Process the query using LLM-guided workflow
            with console.status("[bold green]Thinking...[/bold green]"):
                response = await asyncio.wait_for(
                    orchestrator.process_query(args.query), args.timeout
                )

            # Display the response
            console.print(Panel(response, title="Response", title_align="left"))

            if args.output:
                file_path = save_results_to_file(
                    {"query": args.query, "response": response}, args.output
                )
                console.print(f"[green]Results saved to {file_path}[/green]")
        else:
            # Run in interactive mode
            await run_interactive_mode(orchestrator)

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1
    finally:
        # Disconnect from all servers
        await orchestrator.disconnect_all()

    return 0


if __name__ == "__main__":
    # Run the async main function
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
