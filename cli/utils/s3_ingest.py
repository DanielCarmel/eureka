import os
import io
import logging
import boto3
from typing import List, Dict, Any, Optional
import PyPDF2
import docx

logger = logging.getLogger("cli.s3_ingest")

class S3Ingestor:
    """
    Utility for ingesting data from S3
    """
    
    def __init__(self):
        """Initialize the S3 ingestor with credentials from environment variables"""
        self.aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        self.region_name = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self.endpoint_url = os.environ.get("AWS_S3_ENDPOINT_URL")
        
        if not all([self.aws_access_key_id, self.aws_secret_access_key]):
            logger.warning("AWS environment variables not set properly")
        
        self.client = None
        self.resource = None
        
        if all([self.aws_access_key_id, self.aws_secret_access_key]):
            try:
                session = boto3.Session(
                    aws_access_key_id=self.aws_access_key_id,
                    aws_secret_access_key=self.aws_secret_access_key,
                    region_name=self.region_name
                )
                
                self.client = session.client('s3', endpoint_url=self.endpoint_url)
                self.resource = session.resource('s3', endpoint_url=self.endpoint_url)
                
                endpoint_info = f" using endpoint {self.endpoint_url}" if self.endpoint_url else ""
                logger.info(f"Initialized S3 client for region {self.region_name}{endpoint_info}")
            except Exception as e:
                logger.error(f"Failed to initialize S3 client: {e}")
    
    def _extract_text_from_object(self, bucket: str, key: str) -> str:
        """Extract text from S3 objects based on file type"""
        if not self.client:
            return ""
        
        try:
            obj = self.resource.Object(bucket, key)
            
            # Get the file extension
            _, ext = os.path.splitext(key.lower())
            
            # Get the object data
            response = obj.get()
            data = response['Body'].read()
            
            # Process based on file type
            if ext == '.pdf':
                return self._extract_text_from_pdf(data)
            elif ext in ['.doc', '.docx']:
                return self._extract_text_from_docx(data)
            elif ext in ['.txt', '.md', '.py', '.java', '.js', '.html', '.css', '.json', '.xml', '.csv']:
                return data.decode('utf-8', errors='ignore')
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
    
    def fetch_s3_documents(self, bucket_name: str, prefix: str = "", max_files: int = 100) -> List[Dict[str, Any]]:
        """
        Fetch documents from an S3 bucket
        
        Args:
            bucket_name: S3 bucket name
            prefix: S3 object prefix/folder
            max_files: Maximum number of files to fetch
            
        Returns:
            List of documents with content and metadata
        """
        if not self.client:
            raise ValueError("S3 client not initialized")
        
        try:
            logger.info(f"Fetching objects from bucket {bucket_name} with prefix {prefix}")
            
            # List objects in the bucket with prefix
            paginator = self.client.get_paginator('list_objects_v2')
            pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)
            
            files = []
            for page in pages:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    key = obj['Key']
                    if key.endswith('/'):  # Skip folders
                        continue
                    
                    files.append({
                        'bucket': bucket_name,
                        'key': key,
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
                    
                    if len(files) >= max_files:
                        break
                        
                if len(files) >= max_files:
                    break
            
            logger.info(f"Found {len(files)} files in bucket {bucket_name}")
            
            # Process files
            processed_files = []
            for file_info in files:
                bucket = file_info['bucket']
                key = file_info['key']
                
                try:
                    # Extract content
                    content = self._extract_text_from_object(bucket, key)
                    
                    # Get filename for the title
                    filename = os.path.basename(key)
                    
                    # Create processed file
                    processed_file = {
                        "id": f"{bucket}/{key}",
                        "title": filename,
                        "content": content,
                        "url": f"s3://{bucket}/{key}",
                        "bucket": bucket,
                        "key": key,
                        "size": file_info['size'],
                        "last_modified": file_info['last_modified'].strftime('%Y-%m-%d %H:%M:%S')
                    }
                    
                    processed_files.append(processed_file)
                    
                except Exception as e:
                    logger.error(f"Error processing file {bucket}/{key}: {e}")
            
            return processed_files
            
        except Exception as e:
            logger.error(f"Error fetching S3 documents: {e}")
            raise