import logging
from typing import Dict, List, Any, Optional

from atlassian import Jira
import markdown
from bs4 import BeautifulSoup

from connectors.base_connector import BaseConnector

logger = logging.getLogger("mcp-server.jira")

class JiraConnector(BaseConnector):
    """
    MCP Connector for Jira
    """
    
    def __init__(self, url: str, username: str, token: str):
        """
        Initialize the Jira connector
        
        Args:
            url: Jira instance URL
            username: Jira username
            token: Jira API token
        """
        self.url = url
        self.username = username
        self.token = token
        self.client = None
        
        if url and username and token:
            try:
                self.client = Jira(
                    url=url,
                    username=username,
                    password=token
                )
                logger.info(f"Jira connector initialized for {url}")
            except Exception as e:
                logger.error(f"Failed to initialize Jira connector: {e}")
        else:
            logger.warning("Jira connector initialized with missing credentials")
    
    def _extract_text_from_markup(self, content: str) -> str:
        """Convert Jira markup to plain text"""
        if not content:
            return ""
        
        # First try to convert from markdown
        try:
            html = markdown.markdown(content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text(separator=" ", strip=True)
        except:
            # Fallback to simple string cleanup
            return content.replace("{code}", "").replace("{noformat}", "")
    
    def query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search Jira issues based on the query
        """
        if not self.client:
            logger.error("Jira client not initialized")
            return []
        
        try:
            # Convert natural language query to Jira JQL
            jql = f'text ~ "{query}" ORDER BY updated DESC'
            
            issues = self.client.jql(jql, limit=max_results)
            results = []
            
            for issue in issues.get('issues', []):
                issue_id = issue['key']
                issue_summary = issue.get('fields', {}).get('summary', '')
                issue_description = issue.get('fields', {}).get('description', '')
                
                # Extract plain text from description
                plain_description = self._extract_text_from_markup(issue_description)
                
                # Create result
                result = {
                    "id": issue_id,
                    "title": f"JIRA-{issue_id}: {issue_summary}",
                    "content": plain_description,
                    "url": f"{self.url}/browse/{issue_id}",
                    "created": issue.get('fields', {}).get('created', ''),
                    "updated": issue.get('fields', {}).get('updated', ''),
                    "status": issue.get('fields', {}).get('status', {}).get('name', ''),
                    "issue_type": issue.get('fields', {}).get('issuetype', {}).get('name', '')
                }
                
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error querying Jira: {e}")
            return []
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a specific Jira issue by ID
        """
        if not self.client:
            raise Exception("Jira client not initialized")
        
        try:
            issue = self.client.issue(document_id)
            
            # Extract all fields
            fields = issue.get('fields', {})
            
            # Extract comments if available
            comments = []
            if 'comment' in fields:
                for comment in fields['comment'].get('comments', []):
                    comment_text = self._extract_text_from_markup(comment.get('body', ''))
                    comments.append({
                        "author": comment.get('author', {}).get('displayName', ''),
                        "created": comment.get('created', ''),
                        "content": comment_text
                    })
            
            # Create document
            document = {
                "id": issue['key'],
                "title": fields.get('summary', ''),
                "content": self._extract_text_from_markup(fields.get('description', '')),
                "url": f"{self.url}/browse/{issue['key']}",
                "created": fields.get('created', ''),
                "updated": fields.get('updated', ''),
                "status": fields.get('status', {}).get('name', ''),
                "issue_type": fields.get('issuetype', {}).get('name', ''),
                "priority": fields.get('priority', {}).get('name', ''),
                "reporter": fields.get('reporter', {}).get('displayName', ''),
                "assignee": fields.get('assignee', {}).get('displayName', '') if fields.get('assignee') else '',
                "comments": comments,
                "labels": fields.get('labels', [])
            }
            
            return document
        
        except Exception as e:
            logger.error(f"Error getting Jira document {document_id}: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if Jira connection is healthy
        """
        if not self.client:
            return {"healthy": False, "details": "Jira client not initialized"}
        
        try:
            # Try to get server info as a simple health check
            server_info = self.client.get_server_info()
            return {
                "healthy": True,
                "details": {
                    "version": server_info.get('version', 'unknown'),
                    "server_title": server_info.get('serverTitle', 'Jira')
                }
            }
        except Exception as e:
            logger.error(f"Jira health check failed: {e}")
            return {
                "healthy": False,
                "details": str(e)
            }