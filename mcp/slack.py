from mcp.server.fastmcp import Context, FastMCP

# Create an MCP server for Slack
mcp = FastMCP("SlackServer")


@mcp.tool()
async def search_slack(query: str, channels: str = None, users: str = None, ctx: Context = None) -> str:
    """
    Search Slack messages and content for the given query.

    Args:
        query: The search query string
        channels: Optional comma-separated list of channel names to search in
        users: Optional comma-separated list of usernames to filter by
        ctx: MCP context object

    Returns:
        Search results from Slack as formatted text
    """
    # Here you would implement actual Slack API integration
    # For example, using the slack_sdk package
    if ctx:
        channels_info = f" in channels: {channels}" if channels else ""
        users_info = f" from users: {users}" if users else ""
        ctx.info(f"Searching Slack{channels_info}{users_info} for: {query}")

    # Simulated search results
    return f"Search results from Slack for query: {query}\n- Channel: general, User: user1, Date: 2025-05-01: \
            Message matching your query\n- Channel: dev-team, User: user2, Date: 2025-05-02: Another relevant message"


@mcp.tool()
async def get_slack_conversation_history(channel: str, limit: int = 10, ctx: Context = None) -> str:
    """
    Retrieve recent conversation history from a Slack channel.

    Args:
        channel: The channel name or ID
        limit: Maximum number of messages to retrieve
        ctx: MCP context object

    Returns:
        Conversation history as formatted text
    """
    if ctx:
        ctx.info(f"Retrieving conversation history from Slack channel: {channel} (limit: {limit})")

    # Simulated conversation history
    messages = [
        {"user": "user1", "text": "Has anyone tested the new feature?", "timestamp": "2025-05-03T14:22:00Z"},
        {"user": "user2", "text": "Yes, it works well!", "timestamp": "2025-05-03T14:24:30Z"},
        {"user": "user3", "text": "I found a small bug in the error handling.", "timestamp": "2025-05-03T14:26:45Z"}
    ]

    result = f"# Conversation History for #{channel}\n\n"
    for msg in messages:
        result += f"**{msg['user']}** ({msg['timestamp']}): {msg['text']}\n\n"

    return result


@mcp.resource()
async def list_slack_channels(ctx: Context = None):
    """
    List all available Slack channels.

    Returns:
        A list of Slack channels
    """
    if ctx:
        ctx.info("Listing Slack channels")

    # Simulated channels list
    channels = ["general", "random", "dev-team", "product", "announcements"]
    return "\n".join(channels), "text/plain"


@mcp.tool()
async def post_slack_message(channel: str, message: str, thread_ts: str = None, ctx: Context = None) -> str:
    """
    Post a message to a Slack channel or thread.

    Args:
        channel: The channel name or ID to post to
        message: The message text to post
        thread_ts: Optional thread timestamp to reply to
        ctx: MCP context object

    Returns:
        Confirmation of message posting
    """
    if ctx:
        if thread_ts:
            ctx.info(f"Posting message to Slack channel: {channel} in thread {thread_ts}")
        else:
            ctx.info(f"Posting message to Slack channel: {channel}")

    # Here you would implement actual message posting with slack_sdk
    return f"Message posted to #{channel}" + (f" in thread {thread_ts}" if thread_ts else "")

if __name__ == "__main__":
    mcp.run()
