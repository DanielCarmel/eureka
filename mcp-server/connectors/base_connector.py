from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseConnector(ABC):
    """
    Base class for all MCP connectors to implement
    """

    @abstractmethod
    def query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Query the data source for relevant information

        Args:
            query: The search query string
            max_results: Maximum number of results to return

        Returns:
            List of dictionaries, each containing at least:
            - id: A unique identifier for the document
            - title: Document title
            - content: Document content or snippet
            - url: URL to the document (if applicable)
            - score: Relevance score (optional)
        """
        pass

    @abstractmethod
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Retrieve a specific document by ID

        Args:
            document_id: The document's unique identifier

        Returns:
            Dictionary with document details
        """
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """
        Check if the connector is healthy and can connect to its data source

        Returns:
            Dictionary with at least:
            - healthy: Boolean indicating if connector is healthy
            - details: Any additional details about the health status
        """
        pass
