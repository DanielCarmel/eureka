import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from atlassian import Jira
from bs4 import BeautifulSoup

logger = logging.getLogger("cli.jira_ingest")

class JiraIngestor:
    """
    Utility for ingesting data from Jira
    """
    
    def __init__(self):
        """Initialize the Jira ingestor with credentials from environment variables"""
        self.jira_url = os.environ.get("JIRA_URL")
        self.username = os.environ.get("JIRA_USERNAME")
        self.token = os.environ.get("JIRA_TOKEN")
        
        if not all([self.jira_url, self.username, self.token]):
            logger.warning("Jira environment variables not set properly")
        
        self.client = None
        if all([self.jira_url, self.username, self.token]):
            try:
                self.client = Jira(
                    url=self.jira_url,
                    username=self.username,
                    password=self.token
                )
                logger.info(f"Initialized Jira client for {self.jira_url}")
            except Exception as e:
                logger.error(f"Failed to initialize Jira client: {e}")
    
    def _extract_text_from_markup(self, content: str) -> str:
        """Convert Jira markup to plain text"""
        if not content:
            return ""
        
        # First try to convert from HTML if possible
        try:
            soup = BeautifulSoup(content, 'html.parser')
            return soup.get_text(separator=" ", strip=True)
        except:
            # Fallback to simple string cleanup
            return content.replace("{code}", "").replace("{noformat}", "")
    
    def fetch_project_issues(self, project_key: str, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch issues from a Jira project
        
        Args:
            project_key: Jira project key
            max_results: Maximum number of issues to fetch
            
        Returns:
            List of issues with content and metadata
        """
        if not self.client:
            raise ValueError("Jira client not initialized")
        
        try:
            logger.info(f"Fetching issues for project {project_key}")
            
            # Build JQL query
            jql = f"project = {project_key} ORDER BY updated DESC"
            
            # Fetch issues
            response = self.client.jql(jql, limit=max_results)
            issues = response.get('issues', [])
            
            logger.info(f"Fetched {len(issues)} issues from Jira")
            
            # Process issues
            processed_issues = []
            for issue in issues:
                issue_key = issue.get('key')
                issue_id = issue.get('id')
                
                try:
                    # Extract fields
                    fields = issue.get('fields', {})
                    summary = fields.get('summary', '')
                    description = fields.get('description', '')
                    status = fields.get('status', {}).get('name', '')
                    issue_type = fields.get('issuetype', {}).get('name', '')
                    
                    # Extract comments if available
                    comments_text = ""
                    if 'comment' in fields and 'comments' in fields['comment']:
                        for comment in fields['comment']['comments']:
                            author = comment.get('author', {}).get('displayName', '')
                            body = comment.get('body', '')
                            created = comment.get('created', '')
                            comments_text += f"\nComment by {author} on {created}:\n{body}\n"
                    
                    # Build full content
                    content = f"Summary: {summary}\n\nDescription:\n{description}\n\n"
                    content += f"Status: {status}\nType: {issue_type}\n"
                    
                    if comments_text:
                        content += f"\nComments:\n{comments_text}"
                    
                    # Extract plain text
                    plain_content = self._extract_text_from_markup(content)
                    
                    # Create result
                    processed_issue = {
                        "id": issue_key,
                        "title": f"{issue_key}: {summary}",
                        "content": plain_content,
                        "url": f"{self.jira_url}/browse/{issue_key}",
                        "project": project_key,
                        "type": issue_type,
                        "status": status
                    }
                    
                    processed_issues.append(processed_issue)
                    
                except Exception as e:
                    logger.error(f"Error processing issue {issue_key}: {e}")
            
            return processed_issues
            
        except Exception as e:
            logger.error(f"Error fetching Jira issues: {e}")
            raise