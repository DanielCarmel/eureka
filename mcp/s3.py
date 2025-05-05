from mcp.server.fastmcp import Context, FastMCP

# Create an MCP server for S3
mcp = FastMCP("S3Server")


@mcp.tool()
async def search_s3(query: str, bucket: str = None, prefix: str = None, ctx: Context = None) -> str:
    """
    Search S3 objects for the given query.

    Args:
        query: The search query string
        bucket: Optional bucket name to search in
        prefix: Optional prefix to filter objects
        ctx: MCP context object

    Returns:
        Search results from S3 as formatted text
    """
    # Here you would implement actual S3 API integration
    # For example, using boto3
    if ctx:
        ctx.info(f"Searching S3 {bucket or 'all buckets'} with prefix '{prefix or ''}' for: {query}")

    # Simulated search results
    return f"Search results from S3 for query: {query}\n- s3://{bucket or 'example-bucket'}/documents/file1.pdf\n- \
            s3://{bucket or 'example-bucket'}/images/image1.jpg"


@mcp.tool()
async def get_s3_object(bucket: str, key: str, ctx: Context = None) -> str:
    """
    Retrieve metadata or contents of a specific S3 object.

    Args:
        bucket: The S3 bucket name
        key: The object key
        ctx: MCP context object

    Returns:
        Object metadata or content as text
    """
    if ctx:
        ctx.info(f"Retrieving S3 object: s3://{bucket}/{key}")

    # Simulated object metadata
    return f"# S3 Object: s3://{bucket}/{key}\n\nSize: 1024 bytes\nLast Modified: \
            2025-05-03T10:15:30Z\nContent Type: application/pdf"


@mcp.resource()
async def list_s3_buckets(ctx: Context = None):
    """
    List all available S3 buckets.

    Returns:
        A list of S3 buckets
    """
    if ctx:
        ctx.info("Listing S3 buckets")

    # Simulated buckets list
    buckets = ["documents-bucket", "images-bucket", "data-bucket", "backups-bucket"]
    return "\n".join(buckets), "text/plain"


@mcp.tool()
async def list_s3_objects(bucket: str, prefix: str = "", delimiter: str = "/", ctx: Context = None) -> str:
    """
    List objects in an S3 bucket with optional prefix and delimiter.

    Args:
        bucket: The S3 bucket name
        prefix: Optional prefix to filter objects
        delimiter: Optional delimiter for hierarchical listing
        ctx: MCP context object

    Returns:
        List of objects as formatted text
    """
    if ctx:
        ctx.info(f"Listing objects in bucket '{bucket}' with prefix '{prefix}'")

    # Simulated objects list
    objects = [
        f"s3://{bucket}/documents/report1.pdf",
        f"s3://{bucket}/documents/report2.pdf",
        f"s3://{bucket}/images/logo.png",
        f"s3://{bucket}/data/users.csv"
    ]

    # Filter by prefix if provided
    if prefix:
        objects = [obj for obj in objects if obj.replace(f"s3://{bucket}/", "").startswith(prefix)]

    return "\n".join(objects)

if __name__ == "__main__":
    mcp.run()
