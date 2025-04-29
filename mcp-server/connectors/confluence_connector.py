import logging
from typing import Any, Dict, List

from atlassian import Confluence
from bs4 import BeautifulSoup
from connectors.base_connector import BaseConnector

logger = logging.getLogger("mcp-server.confluence")


class ConfluenceConnector(BaseConnector):
    """
    MCP Connector for Confluence
    """

    def __init__(self, url: str, username: str, token: str):
        """
        Initialize the Confluence connector

        Args:
            url: Confluence instance URL
            username: Confluence username
            token: Confluence API token
        """
        self.url = url
        self.username = username
        self.token = token
        self.client = None

        if url and username and token:
            try:
                self.client = Confluence(url=url, username=username, password=token)
                logger.info(f"Confluence connector initialized for {url}")
            except Exception as e:
                logger.error(f"Failed to initialize Confluence connector: {e}")
        else:
            logger.warning("Confluence connector initialized with missing credentials")

    def _extract_text_from_html(self, html_content: str) -> str:
        """Convert HTML to plain text"""
        if not html_content:
            return ""

        try:
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text(separator=" ", strip=True)
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {e}")
            return html_content

    def query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search Confluence pages based on the query
        """
        if not self.client:
            logger.error("Confluence client not initialized")
            return []

        try:
            # Search Confluence content
            search_results = self.client.cql(f'text ~ "{query}"', limit=max_results)
            results = []

            for result in search_results.get("results", []):
                content_id = result.get("content", {}).get("id")
                if not content_id:
                    continue

                # Get basic content info
                title = result.get("content", {}).get("title", "")
                content_type = result.get("content", {}).get("type", "")
                space_key = result.get("content", {}).get("space", {}).get("key", "")

                # Get content details if it's a page
                content = ""
                if content_type == "page":
                    try:
                        page = self.client.get_page_by_id(
                            content_id, expand="body.storage"
                        )
                        body = page.get("body", {}).get("storage", {}).get("value", "")
                        content = self._extract_text_from_html(body)
                    except Exception as e:
                        logger.error(f"Error getting page content: {e}")

                # Build the URL
                url = f"{self.url}/display/{space_key}/{result.get('content', {}).get('title', '').replace(' ', '+')}"

                # Create result
                result = {
                    "id": content_id,
                    "title": title,
                    "content": content[:1000],  # Limit content length
                    "url": url,
                    "type": content_type,
                    "space": space_key,
                    "last_modified": result.get("lastModified", ""),
                }

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error querying Confluence: {e}")
            return []

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a specific Confluence page by ID
        """
        if not self.client:
            raise Exception("Confluence client not initialized")

        try:
            # Get page with content
            page = self.client.get_page_by_id(
                document_id, expand="body.storage,version,space,history,ancestors"
            )

            # Extract content
            body_html = page.get("body", {}).get("storage", {}).get("value", "")
            content = self._extract_text_from_html(body_html)

            # Get metadata
            title = page.get("title", "")
            space_key = page.get("space", {}).get("key", "")

            # Build the URL
            url = f"{self.url}/display/{space_key}/{title.replace(' ', '+')}"

            # Get ancestors for breadcrumbs
            ancestors = []
            for ancestor in page.get("ancestors", []):
                ancestors.append(
                    {"id": ancestor.get("id", ""), "title": ancestor.get("title", "")}
                )

            # Create document
            document = {
                "id": document_id,
                "title": title,
                "content": content,
                "url": url,
                "space": space_key,
                "space_name": page.get("space", {}).get("name", ""),
                "version": page.get("version", {}).get("number", 1),
                "created": page.get("history", {}).get("createdDate", ""),
                "updated": page.get("version", {}).get("when", ""),
                "creator": page.get("history", {})
                .get("createdBy", {})
                .get("displayName", ""),
                "last_updater": page.get("version", {})
                .get("by", {})
                .get("displayName", ""),
                "ancestors": ancestors,
            }

            return document

        except Exception as e:
            logger.error(f"Error getting Confluence document {document_id}: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Check if Confluence connection is healthy
        """
        if not self.client:
            return {"healthy": False, "details": "Confluence client not initialized"}

        try:
            # Get current user as a health check
            current_user = self.client.get_current_user()
            return {
                "healthy": True,
                "details": {
                    "username": current_user.get("username", ""),
                    "display_name": current_user.get("displayName", ""),
                },
            }
        except Exception as e:
            logger.error(f"Confluence health check failed: {e}")
            return {"healthy": False, "details": str(e)}
