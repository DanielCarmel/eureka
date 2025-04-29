#!/usr/bin/env python3
import argparse
import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Import utilities
from utils.chunking import chunk_document
from utils.confluence_ingest import ConfluenceIngestor
from utils.jira_ingest import JiraIngestor
from utils.s3_ingest import S3Ingestor
from utils.test_utils import test_query
from utils.vector_utils import VectorDBClient

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        RotatingFileHandler("logs/cli.log", maxBytes=10485760, backupCount=5),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("cli")

# Main parser
parser = argparse.ArgumentParser(description="Eureka RAG System Management CLI")
subparsers = parser.add_subparsers(dest="command", help="Command to execute")

# Add Jira ingestion command
jira_parser = subparsers.add_parser(
    "ingest-jira", help="Ingest Jira issues into the vector database"
)
jira_parser.add_argument("--project", required=True, help="Jira project key")
jira_parser.add_argument(
    "--max-issues", type=int, default=100, help="Maximum number of issues to ingest"
)
jira_parser.add_argument(
    "--collection", default="documents", help="Vector DB collection name"
)
jira_parser.add_argument(
    "--chunk-size", type=int, default=512, help="Size of text chunks"
)
jira_parser.add_argument(
    "--chunk-overlap", type=int, default=128, help="Overlap between chunks"
)
jira_parser.add_argument(
    "--force", action="store_true", help="Force re-ingestion even if exists"
)

# Add Confluence ingestion command
confluence_parser = subparsers.add_parser(
    "ingest-confluence", help="Ingest Confluence spaces into the vector database"
)
confluence_parser.add_argument("--space", required=True, help="Confluence space key")
confluence_parser.add_argument(
    "--max-pages", type=int, default=100, help="Maximum number of pages to ingest"
)
confluence_parser.add_argument(
    "--collection", default="documents", help="Vector DB collection name"
)
confluence_parser.add_argument(
    "--chunk-size", type=int, default=512, help="Size of text chunks"
)
confluence_parser.add_argument(
    "--chunk-overlap", type=int, default=128, help="Overlap between chunks"
)
confluence_parser.add_argument(
    "--force", action="store_true", help="Force re-ingestion even if exists"
)

# Add S3 ingestion command
s3_parser = subparsers.add_parser(
    "ingest-s3", help="Ingest S3 documents into the vector database"
)
s3_parser.add_argument("--bucket", required=True, help="S3 bucket name")
s3_parser.add_argument("--prefix", default="", help="S3 object prefix")
s3_parser.add_argument(
    "--collection", default="documents", help="Vector DB collection name"
)
s3_parser.add_argument(
    "--chunk-size", type=int, default=512, help="Size of text chunks"
)
s3_parser.add_argument(
    "--chunk-overlap", type=int, default=128, help="Overlap between chunks"
)
s3_parser.add_argument(
    "--max-files", type=int, default=100, help="Maximum number of files to ingest"
)
s3_parser.add_argument(
    "--force", action="store_true", help="Force re-ingestion even if exists"
)

# Add rebuild index command
rebuild_parser = subparsers.add_parser(
    "rebuild-index", help="Rebuild vector database collection"
)
rebuild_parser.add_argument(
    "--collection", default="documents", help="Collection name to rebuild"
)

# Add test query command
query_parser = subparsers.add_parser(
    "test-query", help="Test a query against the RAG system"
)
query_parser.add_argument("question", help="Question to ask")
query_parser.add_argument(
    "--sources", nargs="+", help="Specific sources to query (jira, confluence, s3)"
)
query_parser.add_argument(
    "--collection", default="documents", help="Vector DB collection to query"
)
query_parser.add_argument(
    "--context-chunks", type=int, default=5, help="Number of context chunks to retrieve"
)


def handle_ingest_jira(args):
    """Handle Jira ingestion command"""
    logger.info(f"Starting Jira ingestion for project {args.project}")
    ingestor = JiraIngestor()
    vector_db = VectorDBClient()

    try:
        # Ingest issues
        issues = ingestor.fetch_project_issues(args.project, args.max_issues)
        logger.info(f"Fetched {len(issues)} issues from Jira project {args.project}")

        # Process and add to vector store
        success_count = 0
        for issue in issues:
            try:
                # Chunk the issue content
                chunks = chunk_document(
                    text=issue["content"],
                    metadata={
                        "id": issue["id"],
                        "title": issue["title"],
                        "url": issue["url"],
                        "source": "jira",
                    },
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )

                # Add chunks to vector DB
                vector_db.add_documents(args.collection, chunks)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process issue {issue['id']}: {e}")

        logger.info(
            f"Successfully ingested {success_count} out of {len(issues)} Jira issues"
        )

    except Exception as e:
        logger.error(f"Error during Jira ingestion: {e}")
        sys.exit(1)


def handle_ingest_confluence(args):
    """Handle Confluence ingestion command"""
    logger.info(f"Starting Confluence ingestion for space {args.space}")
    ingestor = ConfluenceIngestor()
    vector_db = VectorDBClient()

    try:
        # Ingest pages
        pages = ingestor.fetch_space_pages(args.space, args.max_pages)
        logger.info(f"Fetched {len(pages)} pages from Confluence space {args.space}")

        # Process and add to vector store
        success_count = 0
        for page in pages:
            try:
                # Chunk the page content
                chunks = chunk_document(
                    text=page["content"],
                    metadata={
                        "id": page["id"],
                        "title": page["title"],
                        "url": page["url"],
                        "space": page["space"],
                        "source": "confluence",
                    },
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )

                # Add chunks to vector DB
                vector_db.add_documents(args.collection, chunks)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process page {page['id']}: {e}")

        logger.info(
            f"Successfully ingested {success_count} out of {len(pages)} Confluence pages"
        )

    except Exception as e:
        logger.error(f"Error during Confluence ingestion: {e}")
        sys.exit(1)


def handle_ingest_s3(args):
    """Handle S3 ingestion command"""
    logger.info(
        f"Starting S3 ingestion for bucket {args.bucket} with prefix {args.prefix}"
    )
    ingestor = S3Ingestor()
    vector_db = VectorDBClient()

    try:
        # Ingest files
        files = ingestor.fetch_s3_documents(args.bucket, args.prefix, args.max_files)
        logger.info(f"Fetched {len(files)} files from S3 bucket {args.bucket}")

        # Process and add to vector store
        success_count = 0
        for file in files:
            try:
                # Chunk the file content
                chunks = chunk_document(
                    text=file["content"],
                    metadata={
                        "id": file["id"],
                        "title": file["title"],
                        "bucket": file["bucket"],
                        "key": file["key"],
                        "url": file["url"],
                        "source": "s3",
                    },
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                )

                # Add chunks to vector DB
                vector_db.add_documents(args.collection, chunks)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process file {file['id']}: {e}")

        logger.info(
            f"Successfully ingested {success_count} out of {len(files)} S3 files"
        )

    except Exception as e:
        logger.error(f"Error during S3 ingestion: {e}")
        sys.exit(1)


def handle_rebuild_index(args):
    """Handle rebuild index command"""
    logger.info(f"Rebuilding vector database collection: {args.collection}")
    vector_db = VectorDBClient()

    try:
        # Reset collection
        success = vector_db.reset_collection(args.collection)
        if success:
            logger.info(f"Successfully reset collection {args.collection}")
        else:
            logger.error(f"Failed to reset collection {args.collection}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during index rebuild: {e}")
        sys.exit(1)


def handle_test_query(args):
    """Handle test query command"""
    logger.info(f"Testing query: {args.question}")

    try:
        # Execute the query
        response = test_query(
            question=args.question,
            sources=args.sources,
            collection_name=args.collection,
            max_context_chunks=args.context_chunks,
        )

        # Print the response
        print("\n" + "=" * 80)
        print(f"QUESTION: {response['question']}")
        print("=" * 80)
        print(f"ANSWER: {response['answer']}")
        print("-" * 80)
        print(f"Processing time: {response['processing_time']:.2f}s")

        # Print context chunks if available
        if "context_chunks" in response and response["context_chunks"]:
            print("\nSOURCES:")
            for i, chunk in enumerate(response["context_chunks"]):
                print(f"\n{i+1}. {chunk['title']} ({chunk['source_type']})")
                if "url" in chunk and chunk["url"]:
                    print(f"   URL: {chunk['url']}")

        print("=" * 80)

    except Exception as e:
        logger.error(f"Error testing query: {e}")
        sys.exit(1)


def main():
    """Main CLI entrypoint"""
    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate handler
    if args.command == "ingest-jira":
        handle_ingest_jira(args)
    elif args.command == "ingest-confluence":
        handle_ingest_confluence(args)
    elif args.command == "ingest-s3":
        handle_ingest_s3(args)
    elif args.command == "rebuild-index":
        handle_rebuild_index(args)
    elif args.command == "test-query":
        handle_test_query(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
