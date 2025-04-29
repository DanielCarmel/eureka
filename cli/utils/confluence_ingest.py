import logging
import os
from typing import Any, Dict, List

from atlassian import Confluence
from bs4 import BeautifulSoup

logger = logging.getLogger("cli.confluence_ingest")


class ConfluenceIngestor:
    """
    Utility for ingesting data from Confluence
    """

    def __init__(self):
        """
        Initialize the Confluence ingestor with credentials
        from environment variables
        """
        self.confluence_url = os.environ.get("CONFLUENCE_URL")
        self.username = os.environ.get("CONFLUENCE_USERNAME")
        self.token = os.environ.get("CONFLUENCE_TOKEN")

        if not all([self.confluence_url, self.username, self.token]):
            logger.warning("Confluence environment variables not set properly")

        self.client = None
        if all([self.confluence_url, self.username, self.token]):
            try:
                self.client = Confluence(
                    url=self.confluence_url, username=self.username, password=self.token
                )
                logger.info(f"Initialized Confluence client for {self.confluence_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Confluence client: {e}")

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

    def fetch_space_pages(
        self, space_key: str, max_results: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Fetch pages from a Confluence space

        Args:
            space_key: Confluence space key
            max_results: Maximum number of pages to fetch

        Returns:
            List of pages with content and metadata
        """
        if not self.client:
            raise ValueError("Confluence client not initialized")

        try:
            logger.info(f"Fetching pages for space {space_key}")

            # Get space information
            space = self.client.get_space(space_key)

            if not space:
                logger.error(f"Space {space_key} not found")
                return []

            # Fetch pages from space
            pages = self.client.get_all_pages_from_space(space_key, limit=max_results)

            logger.info(f"Fetched {len(pages)} pages from Confluence space {space_key}")

            # Process pages
            processed_pages = []
            for page in pages:
                page_id = page.get("id")
                page_title = page.get("title", "")

                try:
                    # Get page content with body
                    page_with_content = self.client.get_page_by_id(
                        page_id, expand="body.storage,version,ancestors"
                    )

                    # Extract body HTML and convert to text
                    body_html = (
                        page_with_content.get("body", {})
                        .get("storage", {})
                        .get("value", "")
                    )
                    body_text = self._extract_text_from_html(body_html)

                    # Get page URL
                    page_url = f"{self.confluence_url}/display/{space_key}/{page_title.replace(' ', '+')}"

                    # Get parent page if available
                    ancestors = page_with_content.get("ancestors", [])
                    parent_title = ancestors[-1].get("title") if ancestors else None

                    # Create processed page
                    processed_page = {
                        "id": page_id,
                        "title": page_title,
                        "content": body_text,
                        "url": page_url,
                        "space": space_key,
                        "space_name": space.get("name", ""),
                        "parent": parent_title,
                        "version": page_with_content.get("version", {}).get(
                            "number", 1
                        ),
                    }

                    processed_pages.append(processed_page)

                except Exception as e:
                    logger.error(f"Error processing page {page_id} - {page_title}: {e}")

            return processed_pages

        except Exception as e:
            logger.error(f"Error fetching Confluence pages: {e}")
            raise
