# Eureka Troubleshooting Guide

This guide covers common issues and solutions for the Eureka RAG system.

## Service Health Checks

If you're experiencing issues, first check if all services are running correctly:

```bash
docker-compose ps
```

All services should show as "Up". If a service is listed as "Exit" or "Restarting", check its logs:

```bash
docker-compose logs [service-name]
```

You can also use the health endpoints directly:

```bash
curl http://localhost:8000/health  # RAG Orchestrator health check
curl http://localhost:8080/health  # LLM Engine health check
curl http://localhost:5001/health  # MCP Server health check
curl http://localhost:8000/health  # Vector DB health check
```

## Common Issues

### LLM Engine Issues

#### Model Loading Failures

**Problem**: "Model path does not exist" errors or LLM Engine container crashes on startup.

**Solution**:
1. Make sure you've downloaded a model using the `scripts/download_models.sh` script
2. Check that the `MODEL_PATH` in your `.env` file points to a valid model file
3. Verify the model file is in the correct location in the `models/` directory

#### Out of Memory Errors

**Problem**: "CUDA out of memory" errors or container crashes when loading model.

**Solution**:
1. Use a smaller or more quantized model (e.g., switch to a 4-bit quantized version)
2. Allocate more memory to Docker in Docker Desktop settings
3. If on a server, ensure you have enough GPU memory for the selected model

### Vector Database Issues

#### Collection Not Found

**Problem**: "Collection not found" errors when querying.

**Solution**:
1. Check if the collection exists: `curl http://localhost:8000/collections`
2. Create the collection by ingesting documents using the CLI tools
3. Use the `rebuild-index` command if needed: `python manage.py rebuild-index --collection documents`

#### Embedding Model Download Failures

**Problem**: Vector DB fails to start with model download errors.

**Solution**:
1. Ensure you have internet access when first starting the Vector DB (required for model download)
2. If air-gapped, pre-download the embedding model and add it to a custom Docker image
3. Check logs for specific error messages: `docker-compose logs vector-db`

### MCP Server Issues

#### Connection Errors to Data Sources

**Problem**: MCP Server can't connect to Jira, Confluence, or S3.

**Solution**:
1. Verify your credentials in the `.env` file
2. Check network connectivity from the container to your data sources
3. If using a private network, ensure proper DNS resolution

#### JSON-RPC Errors

**Problem**: "Method not found" or parsing errors when querying the MCP Server.

**Solution**:
1. Ensure you're using the correct method name in your requests
2. Check the format of your parameters
3. Verify the MCP Server logs for detailed error information

### RAG Orchestrator Issues

#### Authorization Failures

**Problem**: "401 Unauthorized" errors when calling the API.

**Solution**:
1. Ensure you're providing the correct API key in the `X-API-Key` header
2. Check that the `API_KEY` environment variable is set correctly

#### No Results from Queries

**Problem**: Queries return empty responses or "not enough information" answers.

**Solution**:
1. Verify that you've ingested relevant documents
2. Check if the Vector DB collection exists and contains documents
3. Try increasing the `max_context_chunks` parameter in your query
4. Check if your data sources (Jira, Confluence, S3) are properly connected

### CLI Tool Issues

#### Connection Refused Errors

**Problem**: "Connection refused" errors when using CLI tools.

**Solution**:
1. Ensure all services are running: `docker-compose ps`
2. Check that the service URLs in `.env` are correct
3. Verify network connectivity between your CLI environment and the services

#### Authentication Errors

**Problem**: "Authentication failed" errors when ingesting from data sources.

**Solution**:
1. Verify credentials in the `.env` file
2. For Jira/Confluence, ensure your API token has the correct permissions
3. For S3, check that your access key and secret key are valid

## Docker and Infrastructure Issues

### Docker Disk Space Issues

**Problem**: "No space left on device" errors.

**Solution**:
1. Clean up unused Docker resources: `docker system prune -af`
2. Remove unused Docker volumes: `docker volume prune`
3. Allocate more disk space to Docker

### Network Issues

**Problem**: Services can't communicate with each other.

**Solution**:
1. Check that all services are on the same Docker network
2. Verify service names match the hostnames used in configuration
3. Ensure ports are correctly exposed and mapped

### Permission Issues

**Problem**: "Permission denied" errors in logs.

**Solution**:
1. Check file permissions on mounted volumes
2. Ensure Docker has appropriate permissions to access mounted directories
3. Try running `chmod -R 777` on data directories (for development only)

## Data and Ingestion Issues

### Document Chunking Issues

**Problem**: Documents are not properly chunked or indexed.

**Solution**:
1. Experiment with different chunk sizes: `--chunk-size 512 --chunk-overlap 128`
2. Check if documents are being properly parsed (e.g., PDF extraction issues)
3. Look for error messages in the CLI logs

### Embedding Generation Issues

**Problem**: "Failed to generate embeddings" errors.

**Solution**:
1. Verify the embedding model is available
2. Check if text chunks are within the token limits
3. Ensure there's enough memory for embedding generation

## Performance Optimization

### Slow Query Response Times

**Problem**: Queries take a long time to complete.

**Solution**:
1. Reduce the `max_context_chunks` parameter for faster responses
2. Use a faster LLM or more optimized version
3. Ensure you're running on a GPU if available
4. Consider adding more specific metadata to limit document search space

### High Resource Usage

**Problem**: System uses excessive CPU, memory, or GPU resources.

**Solution**:
1. Use more quantized models (e.g., 4-bit instead of 8-bit)
2. Reduce parallel operations in the system
3. Optimize Docker resource allocation
4. Consider scaling horizontally (running multiple inference servers)

## Advanced Troubleshooting

### Debugging Mode

For more verbose logging, set `LOG_LEVEL=DEBUG` in your `.env` file and restart the services:

```bash
docker-compose down
docker-compose up -d
```

### Interactive Debugging

To access a running container for debugging:

```bash
docker-compose exec [service-name] bash
```

### Checking Service Logs in Real-Time

To monitor logs in real-time:

```bash
docker-compose logs -f [service-name]
```

## Still Having Problems?

If you've tried the solutions above and are still experiencing issues:

1. Check the GitHub issues for similar problems and solutions
2. Ensure all your dependencies and Docker images are up to date
3. Try rebuilding the system with `docker-compose build --no-cache`
4. Create a detailed issue report with logs, environment information, and steps to reproduce