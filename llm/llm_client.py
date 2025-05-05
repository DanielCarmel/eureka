"""
LLM Client Factory Module

This module provides a unified interface for creating LLM clients
with support for local models.
"""
import logging
from typing import Optional

from llm.base_client import BaseLLMClient
from llm.local_llm import LlamaCppClient, OllamaClient
from llm.models import Message, ToolCall

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """
    Factory class for creating LLM clients based on the selected provider.

    This class provides a unified interface for working with different LLM providers,
    focusing on local model implementations.
    """

    @staticmethod
    async def create(
        provider: str = "ollama",
        model: str = "llama2",
        model_path: Optional[str] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client for the specified provider.

        Args:
            provider: The LLM provider to use ('llamacpp', 'ollama')
            model: The model to use
            model_path: Path to the model file (for local models)
            api_base: Base URL for the API (for local/custom endpoints)
            api_key: API key for the provider (if needed)
            temperature: Controls randomness in response generation
            max_tokens: Maximum number of tokens to generate
            system_prompt: Optional system prompt to guide the LLM
            **kwargs: Additional provider-specific parameters

        Returns:
            An instance of BaseLLMClient for the specified provider
        """
        # Select the appropriate client class based on the provider
        if provider.lower() == "llamacpp":
            # Validate model_path for local model
            if not model_path:
                raise ValueError("model_path is required for the llamacpp provider")

            # Create and return LlamaCpp client
            return LlamaCppClient(
                model_path=model_path,
                model_name=model,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                n_ctx=kwargs.get("n_ctx", 4096),
                n_gpu_layers=kwargs.get("n_gpu_layers", -1),
                verbose=kwargs.get("verbose", False)
            )

        elif provider.lower() == "ollama":
            # Create and return Ollama client
            return OllamaClient(
                model_name=model,
                base_url=api_base or "http://localhost:11434",
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                timeout=kwargs.get("timeout", 120.0)
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
