"""
Base LLM Client Interface

This module defines the abstract base class for LLM client implementations,
allowing for multiple backends (OpenAI, local models, etc.)
"""
import abc
from typing import Any, Dict, List, Optional

from llm.models import Message, ToolCall


class BaseLLMClient(abc.ABC):
    """
    Abstract base class for LLM clients.

    Implementations should provide concrete implementations for calling
    specific LLM providers like OpenAI, local models, etc.
    """
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        system_prompt: Optional[str] = None
    ):
        """
        Initialize the base LLM client.

        Args:
            model: The model identifier to use
            temperature: Controls randomness in response generation
            system_prompt: Optional system prompt to guide the LLM
        """
        self.model = model
        self.temperature = temperature

        # Set default system prompt if none provided
        if system_prompt is None:
            system_prompt = (
                "You are an AI assistant that helps users find information from various data sources. "
                "Based on the user's query, determine which tools need to be called to retrieve relevant information. "
                "You have access to tools from S3 and Slack servers. "
                "Think step by step about what information is needed to answer the query completely.")

        self.system_prompt = system_prompt
        self.conversation_history = [{"role": "system", "content": system_prompt}]

    async def interpret_query(self, query: str, server_capabilities: Dict[str, Dict[str, Any]]) -> List[ToolCall]:
        """
        Interpret a user query and determine which tools to call.

        Args:
            query: The user's query
            server_capabilities: Dictionary mapping server names to their capabilities

        Returns:
            List of tool calls to execute
        """
        # Prepare a detailed prompt with available tools
        tools_prompt = "Available tools:\n\n"

        for server_name, capabilities in server_capabilities.items():
            tools_prompt += f"Server: {server_name}\n"
            for tool in capabilities.get("tools", []):
                tools_prompt += f"- {tool['name']}: {tool['description']}\n"
                tools_prompt += f"  Parameters: {', '.join([p['name'] for p in tool.get('parameters', [])])}\n"
            tools_prompt += "\n"

        # Add the user's query and tools information to the conversation
        user_prompt = f"User query: {query}\n\n{tools_prompt}\n\nDetermine which tools \
                        I should call to answer this query. For each tool, specify the \
                        server name, tool name, and parameters."

        self.conversation_history.append({"role": "user", "content": user_prompt})

        try:
            # Call the LLM to interpret the query
            response = await self._call_llm(self.conversation_history)

            # Add the assistant's response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            # Parse the response to extract tool calls
            tool_calls = self._parse_tool_calls(response, server_capabilities)

            return tool_calls

        except Exception as e:
            raise RuntimeError(f"Error interpreting query with LLM: {str(e)}")

    async def generate_response(self, query: str, tool_results: Dict[str, Any]) -> str:
        """
        Generate a final response based on the results of tool calls.

        Args:
            query: The original user query
            tool_results: Results from each tool call

        Returns:
            Generated response to the user
        """
        # Prepare a prompt with the tool call results
        results_prompt = "Results from tool calls:\n\n"

        for tool_call, result in tool_results.items():
            results_prompt += f"Tool: {tool_call}\n"
            results_prompt += f"Result: {result}\n\n"

        # Add the results and request a response to the conversation
        user_prompt = f"Original query: {query}\n\n{results_prompt}\n\nPlease generate a comprehensive response to the \
                        original query based on these results."

        self.conversation_history.append({"role": "user", "content": user_prompt})

        try:
            # Call the LLM to generate the final response
            response = await self._call_llm(self.conversation_history)

            # Add the response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": response})

            return response

        except Exception as e:
            raise RuntimeError(f"Error generating response with LLM: {str(e)}")

    @abc.abstractmethod
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM to generate a response. Must be implemented by subclasses.

        Args:
            messages: List of message dictionaries representing the conversation

        Returns:
            Generated response from the LLM
        """
        pass

    def _parse_tool_calls(self, llm_response: str, server_capabilities: Dict[str, Dict[str, Any]]) -> List[ToolCall]:
        """
        Parse the LLM response to extract tool calls.

        Args:
            llm_response: The response from the LLM
            server_capabilities: Dictionary of server capabilities

        Returns:
            List of ToolCall objects
        """
        # Simple parsing for demonstration - in a real system, you might use a more robust approach
        tool_calls = []
        lines = llm_response.split('\n')

        current_tool = None
        current_server = None
        current_args = {}
        current_description = ""

        for line in lines:
            line = line.strip()

            # Check for tool name
            if line.startswith("Tool:") or line.startswith("- Tool:"):
                # Save previous tool if exists
                if current_tool and current_server:
                    tool_calls.append(ToolCall(
                        tool_name=current_tool,
                        server_name=current_server,
                        arguments=current_args,
                        description=current_description
                    ))

                # Start new tool
                current_tool = line.split(":", 1)[1].strip()
                current_args = {}
                current_description = ""

            # Check for server name
            elif line.startswith("Server:") or line.startswith("- Server:"):
                current_server = line.split(":", 1)[1].strip()

            # Check for arguments
            elif line.startswith("Arguments:") or line.startswith("- Arguments:") or line.startswith("Parameters:"):
                # Arguments might be on this line or following lines
                arg_part = line.split(":", 1)[1].strip()
                if arg_part:
                    # Process arguments on this line
                    self._process_argument_line(arg_part, current_args)

            # Check for individual argument
            elif ":" in line and not line.startswith("-") and not line.startswith("#"):
                key, value = line.split(":", 1)
                current_args[key.strip()] = value.strip()

            # Collect description lines
            elif current_tool and current_server and not line.startswith("-") and line:
                if current_description:
                    current_description += " " + line
                else:
                    current_description = line

        # Add the last tool if exists
        if current_tool and current_server:
            tool_calls.append(ToolCall(
                tool_name=current_tool,
                server_name=current_server,
                arguments=current_args,
                description=current_description
            ))

        return tool_calls

    def _process_argument_line(self, arg_line: str, args_dict: Dict[str, str]):
        """
        Process a line containing arguments.

        Args:
            arg_line: Line containing arguments
            args_dict: Dictionary to update with parsed arguments
        """
        # Split by commas outside of quotes
        import re
        arg_pairs = re.findall(r'([^,]+?):\s*([^,]+?)(?:,|$)', arg_line)

        for key, value in arg_pairs:
            args_dict[key.strip()] = value.strip().strip('"\'')

    def reset_conversation(self):
        """Reset the conversation history, keeping only the system prompt."""
        self.conversation_history = [
            {"role": "system", "content": self.system_prompt}
        ]
