import logging
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("cli.vector_utils")


class VectorDBClient:
    """
    Client for interacting with the Vector Database service
    """

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the Vector DB client

        Args:
            base_url: Base URL of the Vector DB service (defaults to environment variable)
        """
        self.base_url = base_url or os.environ.get(
            "VECTOR_DB_URL", "http://localhost:8000"
        )
        logger.debug(f"Initialized Vector DB client with base URL: {self.base_url}")

    def add_documents(
        self, collection_name: str, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Add documents to a collection

        Args:
            collection_name: Name of the collection
            documents: List of documents to add (each with id, text, and metadata)

        Returns:
            Response from the Vector DB service
        """
        try:
            url = f"{self.base_url}/add"
            payload = {"collection_name": collection_name, "documents": documents}

            response = requests.post(url, json=payload)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error adding documents to collection {collection_name}: {e}")
            raise

    def query_documents(
        self, collection_name: str, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query documents from a collection

        Args:
            collection_name: Name of the collection
            query: Query string
            top_k: Number of results to return

        Returns:
            List of matching documents
        """
        try:
            url = f"{self.base_url}/query"
            payload = {
                "collection_name": collection_name,
                "query": query,
                "top_k": top_k,
            }

            response = requests.post(url, json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("documents", [])

        except Exception as e:
            logger.error(f"Error querying collection {collection_name}: {e}")
            raise

    def delete_documents(
        self,
        collection_name: str,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Delete documents from a collection

        Args:
            collection_name: Name of the collection
            ids: Optional list of document IDs to delete
            filter: Optional filter to select documents to delete

        Returns:
            Response from the Vector DB service
        """
        try:
            url = f"{self.base_url}/delete"
            payload = {"collection_name": collection_name}

            if ids:
                payload["ids"] = ids

            if filter:
                payload["filter"] = filter

            response = requests.post(url, json=payload)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(
                f"Error deleting documents from collection {collection_name}: {e}"
            )
            raise

    def reset_collection(self, collection_name: str) -> bool:
        """
        Reset a collection (delete and recreate)

        Args:
            collection_name: Name of the collection

        Returns:
            True if successful, False otherwise
        """
        try:
            url = f"{self.base_url}/reset/{collection_name}"

            response = requests.post(url)
            response.raise_for_status()

            return True

        except Exception as e:
            logger.error(f"Error resetting collection {collection_name}: {e}")
            return False

    def list_collections(self) -> List[str]:
        """
        List all collections

        Returns:
            List of collection names
        """
        try:
            url = f"{self.base_url}/collections"

            response = requests.get(url)
            response.raise_for_status()

            result = response.json()
            return result.get("collections", [])

        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection

        Args:
            collection_name: Name of the collection

        Returns:
            Collection information
        """
        try:
            url = f"{self.base_url}/collection/{collection_name}"

            response = requests.get(url)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error getting collection info for {collection_name}: {e}")
            raise
