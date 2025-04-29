# Eureka: Enterprise RAG System

Eureka is a powerful Retrieval Augmented Generation (RAG) system designed for enterprise use, allowing you to query your organization's knowledge base using natural language. It combines the power of LLMs with efficient document retrieval across multiple data sources.

## Features

- **Multiple Data Sources**: Query information from Jira, Confluence, S3 documents, and more
- **Vector Search**: Semantic document search using embeddings
- **Local LLM**: Run everything on-premises with local language models
- **Secure API**: API key authentication for secure access
- **Dockerized**: Easy deployment with Docker Compose
- **Extensible**: Modular design makes it easy to add new data sources

## Architecture

Eureka consists of several microservices:

1. **LLM Engine**: Local LLM inference service with OpenAI-compatible API
2. **Vector DB**: ChromaDB vector database for semantic search
3. **MCP Server**: Model Context Protocol service for data source connections
4. **RAG Orchestrator**: Main API service that orchestrates the RAG workflow
5. **CLI Tools**: Command-line tools for data ingestion and management

```
┌─────────────┐     ┌─────────────┐     ┌───────────────┐
│ Command-Line│     │     API     │     │  Web Client   │
│    Tools    │     │ Consumers   │     │   (Future)    │
└──────┬──────┘     └──────┬──────┘     └───────┬───────┘
       │                   │                    │
       │                   ▼                    │
       │           ┌──────────────┐             │
       └──────────►│      RAG     │◄────────────┘
                   │ Orchestrator │
                   └──┬───────┬───┘
                      │       │
          ┌───────────┘       └────────────┐
          ▼                                ▼
┌──────────────────┐              ┌────────────────┐
│    LLM Engine    │              │   MCP Server   │
└──────────────────┘              └────────┬───────┘
                                           │
                                           ▼
                                  ┌────────────────┐
                                  │   Vector DB    │
                                  └────────────────┘
                                           │
                                           ▼
                                  ┌────────────────┐
                                  │  Data Sources  │
                                  │ Jira Confluence│
                                  │ S3 Documents   │
                                  └────────────────┘
```

## Requirements

- Docker and Docker Compose
- 16GB+ RAM (varies based on LLM model size)
- GPU recommended but not required
- Python 3.9+ for CLI tools

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/eureka.git
   cd eureka
   ```

2. Download an LLM model:
   ```bash
   cd scripts
   chmod +x download_models.sh
   ./download_models.sh
   ```

3. Create and configure environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. Build and start the services:
   ```bash
   docker-compose up -d
   ```

5. Verify the services are running:
   ```bash
   docker-compose ps
   ```

## Usage

### Ingesting Data

Use the CLI tools to ingest data from various sources:

**Jira issues:**
```bash
cd cli
python manage.py ingest-jira --project PROJECT_KEY --max-issues 100
```

**Confluence pages:**
```bash
python manage.py ingest-confluence --space SPACE_KEY --max-pages 100
```

**S3 documents:**
```bash
python manage.py ingest-s3 --bucket BUCKET_NAME --prefix docs/ --max-files 50
```

### Querying the System

You can query the system using the RAG Orchestrator API:

```bash
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was discussed in the last project meeting?",
    "max_context_chunks": 5,
    "include_sources": true
  }'
```

### Testing Queries with CLI

For quick testing, use the CLI tool:

```bash
cd cli
python manage.py test-query "What is the current status of the XYZ project?"
```

## Connectors

Eureka supports the following data connectors:

| Connector   | Status | Configuration                   | Notes                                |
|-------------|--------|--------------------------------|--------------------------------------|
| Jira        | ✅     | JIRA_URL, JIRA_USERNAME, JIRA_TOKEN | Requires Atlassian API token        |
| Confluence  | ✅     | CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_TOKEN | Requires Atlassian API token |
| S3          | ✅     | AWS_* environment variables     | Works with S3-compatible services    |

## Customization

### Adding New Data Sources

To add a new data source:

1. Create a new connector in `mcp-server/connectors/`
2. Implement the required methods from `base_connector.py`
3. Register the connector in `mcp-server/server.py`
4. Add corresponding ingestor in `cli/utils/`

### Using Different LLM Models

You can use any GGUF-format LLM model with the system. Download the model using the provided script or manually place it in the `models/` directory, then update the `MODEL_PATH` in your `.env` file.

## Troubleshooting

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues and solutions.

## Development

### Project Structure

```
eureka/
├── cli/                   # Command-line tools
│   ├── manage.py         # Main CLI script
│   └── utils/            # CLI utilities
├── config/                # Configuration files
├── llm-engine/            # LLM inference service
├── mcp-server/            # Model Context Protocol server
├── models/                # LLM and embedding models
├── rag-orchestrator/      # Main RAG API service
├── scripts/               # Utility scripts
└── vector-db/             # Vector database service
```

### Adding Features

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Built with FastAPI, ChromaDB, and various open-source LLM tools
- Uses GGUF models from HuggingFace
