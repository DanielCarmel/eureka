"""
Local LLM Clients

This module provides implementations for local LLM clients, including:
- LlamaCppClient: For running models with llama.cpp
- OllamaClient: For interfacing with Ollama
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional

import httpx
from llama_cpp import Llama

from llm.base_client import BaseLLMClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LlamaCppClient(BaseLLMClient):
    """
    Client for running models locally using llama.cpp

    This client loads models directly into memory using llama.cpp Python bindings.
    Good for running models without additional dependencies or services.
    """
    def __init__(
        self,
        model_path: str,
        model_name: str = "local_model",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the llama.cpp client.

        Args:
            model_path: Path to the model file (.gguf format)
            model_name: Name identifier for the model
            temperature: Controls randomness in response generation
            max_tokens: Maximum number of tokens to generate
            n_ctx: Context window size
            n_gpu_layers: Number of layers to offload to GPU (-1 = all)
            system_prompt: Optional system prompt to guide the LLM
            verbose: Whether to print verbose output
        """
        super().__init__(model_name, temperature, system_prompt)

        self.model_path = model_path
        self.max_tokens = max_tokens
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose

        # Validate that the model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load the model
        try:
            logger.info(f"Loading model from {model_path}...")
            self.llm = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=verbose
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the local LLM to generate a response.

        Args:
            messages: List of message dictionaries representing the conversation

        Returns:
            Generated response from the LLM
        """
        try:
            # Create a prompt from the messages
            # Note: This runs in a separate thread to avoid blocking the async event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.llm.create_chat_completion(
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            )

            # Extract and return the response content
            content = response["choices"][0]["message"]["content"]

            if self.verbose:
                logger.info(f"Generated response: {content}")

            return content

        except Exception as e:
            logger.error(f"Error calling local LLM: {str(e)}")
            raise


class OllamaClient(BaseLLMClient):
    """
    Client for running models using Ollama

    This client interfaces with the Ollama service to run models.
    Requires Ollama to be installed and running.
    """
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        timeout: float = 120.0
    ):
        """
        Initialize the Ollama client.

        Args:
            model_name: Name of the model in Ollama
            base_url: URL of the Ollama API server
            temperature: Controls randomness in response generation
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the LLM
            timeout: Timeout for API requests in seconds
        """
        super().__init__(model_name, temperature, system_prompt)

        self.base_url = base_url
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the Ollama API to generate a response.

        Args:
            messages: List of message dictionaries representing the conversation

        Returns:
            Generated response from the LLM
        """
        try:
            # Set up the API request
            url = f"{self.base_url}/api/chat"

            payload = {
                "model": self.model,
                "messages": messages,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens
                },
                "stream": False
            }

            # Make the API request
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    url,
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )

                response.raise_for_status()
                result = response.json()

            # Extract and return the response content
            content = result["message"]["content"]
            return content

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling Ollama: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Error calling Ollama: {str(e)}")
            raise
