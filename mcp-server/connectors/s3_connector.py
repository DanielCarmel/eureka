import io
import logging
import os
from typing import Any, Dict, List, Optional

import boto3
import docx
import PyPDF2
from connectors.base_connector import BaseConnector

logger = logging.getLogger("mcp-server.s3")


class S3Connector(BaseConnector):
    """
    MCP Connector for AWS S3 documents
    """

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        endpoint_url: Optional[str] = None,
    ):
        """
        Initialize the S3 connector

        Args:
            aws_access_key_id: AWS access key
            aws_secret_access_key: AWS secret key
            region_name: AWS region
            endpoint_url: Optional VPC endpoint URL for S3
        """
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.region_name = region_name
        self.endpoint_url = endpoint_url
        self.client = None
        self.resource = None

        if aws_access_key_id and aws_secret_access_key and region_name:
            try:
                session = boto3.Session(
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    region_name=region_name,
                )

                self.client = session.client("s3", endpoint_url=endpoint_url)
                self.resource = session.resource("s3", endpoint_url=endpoint_url)
                logger.info(f"S3 connector initialized for region {region_name}")
                if endpoint_url:
                    logger.info(f"Using custom endpoint: {endpoint_url}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 connector: {e}")
        else:
            logger.warning("S3 connector initialized with missing credentials")

    def _extract_text_from_object(self, bucket: str, key: str) -> str:
        """Extract text from S3 objects based on file type"""
        if not self.client:
            return ""

        try:
            obj = self.resource.Object(bucket, key)
            content_type = obj.content_type if hasattr(obj, "content_type") else ""

            # Get the file extension
            _, ext = os.path.splitext(key.lower())

            # Get the object
            response = obj.get()
            data = response["Body"].read()

            # Process based on file type
            if ext == ".pdf" or "application/pdf" in content_type:
                return self._extract_text_from_pdf(data)
            elif (
                ext in [".doc", ".docx"]
                or "application/vnd.openxmlformats-officedocument.wordprocessingml"
                in content_type
            ):
                return self._extract_text_from_docx(data)
            elif ext in [
                ".txt",
                ".md",
                ".py",
                ".java",
                ".js",
                ".html",
                ".css",
                ".json",
                ".xml",
                ".csv",
            ]:
                return data.decode("utf-8", errors="ignore")
            else:
                logger.warning(f"Unsupported file type: {ext} for {bucket}/{key}")
                return f"[Unsupported file type: {ext}]"

        except Exception as e:
            logger.error(f"Error extracting text from S3 object {bucket}/{key}: {e}")
            return f"[Error extracting text: {str(e)}]"

    def _extract_text_from_pdf(self, data: bytes) -> str:
        """Extract text from PDF file"""
        try:
            pdf_file = io.BytesIO(data)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            for page_num in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page_num].extract_text() or ""

            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return f"[Error extracting PDF text: {str(e)}]"

    def _extract_text_from_docx(self, data: bytes) -> str:
        """Extract text from DOCX file"""
        try:
            docx_file = io.BytesIO(data)
            doc = docx.Document(docx_file)
            text = ""

            for para in doc.paragraphs:
                text += para.text + "\n"

            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {e}")
            return f"[Error extracting DOCX text: {str(e)}]"

    def query(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search S3 objects based on the query
        Note: This is a simple implementation that lists objects and filters by name.
        In a production environment, you might want to integrate with a proper search service.
        """
        if not self.client:
            logger.error("S3 client not initialized")
            return []

        results = []

        try:
            # List all buckets
            response = self.client.list_buckets()

            for bucket in response["Buckets"]:
                bucket_name = bucket["Name"]

                try:
                    # List objects in the bucket
                    paginator = self.client.get_paginator("list_objects_v2")
                    pages = paginator.paginate(Bucket=bucket_name, Prefix="")

                    for page in pages:
                        if "Contents" not in page:
                            continue

                        for obj in page["Contents"]:
                            key = obj["Key"]

                            # Simple filtering based on query in filename
                            if query.lower() in key.lower():
                                # Create result
                                result = {
                                    "id": f"{bucket_name}/{key}",
                                    "title": key,
                                    "content": f"S3 Object: {bucket_name}/{key}",
                                    "url": f"s3://{bucket_name}/{key}",
                                    "size": obj.get("Size", 0),
                                    "last_modified": obj.get(
                                        "LastModified", ""
                                    ).strftime("%Y-%m-%d %H:%M:%S")
                                    if obj.get("LastModified")
                                    else "",
                                }

                                results.append(result)

                                # Limit results
                                if len(results) >= max_results:
                                    return results

                except Exception as e:
                    logger.error(f"Error listing objects in bucket {bucket_name}: {e}")

            return results

        except Exception as e:
            logger.error(f"Error querying S3: {e}")
            return []

    def get_document(self, document_id: str) -> Dict[str, Any]:
        """
        Get a specific S3 object by ID (bucket_name/key)
        """
        if not self.client:
            raise Exception("S3 client not initialized")

        try:
            # Parse bucket and key
            parts = document_id.split("/", 1)
            if len(parts) != 2:
                raise ValueError(
                    f"Invalid S3 document ID format: {document_id}. Expected 'bucket/key'"
                )

            bucket_name, key = parts

            # Get object metadata
            obj = self.resource.Object(bucket_name, key)
            metadata = obj.metadata if hasattr(obj, "metadata") else {}

            # Extract content
            content = self._extract_text_from_object(bucket_name, key)

            # Create document
            document = {
                "id": document_id,
                "title": key,
                "content": content,
                "url": f"s3://{bucket_name}/{key}",
                "bucket": bucket_name,
                "key": key,
                "size": obj.content_length if hasattr(obj, "content_length") else 0,
                "last_modified": obj.last_modified.strftime("%Y-%m-%d %H:%M:%S")
                if hasattr(obj, "last_modified")
                else "",
                "metadata": metadata,
            }

            return document

        except Exception as e:
            logger.error(f"Error getting S3 document {document_id}: {e}")
            raise

    def health_check(self) -> Dict[str, Any]:
        """
        Check if S3 connection is healthy
        """
        if not self.client:
            return {"healthy": False, "details": "S3 client not initialized"}

        try:
            # Try to list buckets as a health check
            response = self.client.list_buckets()
            buckets = [bucket["Name"] for bucket in response["Buckets"]]

            return {
                "healthy": True,
                "details": {
                    "bucket_count": len(buckets),
                    "region": self.region_name,
                    "endpoint": self.endpoint_url or "default",
                },
            }
        except Exception as e:
            logger.error(f"S3 health check failed: {e}")
            return {"healthy": False, "details": str(e)}
