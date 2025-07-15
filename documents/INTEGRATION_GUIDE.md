# 🚀 MCP Server Integration Guide

This guide explains how to integrate the standalone MCP server with your existing FastAPI application.

## 📋 Overview

The integration consists of two separate services:

1. **MCP Server** (Port 3001) - Handles research processing, PDF analysis, and PPT generation
2. **Your FastAPI Server** (Your chosen port) - Handles your existing API and communicates with MCP server

## 🏗️ Architecture

```
Your Frontend/Client
        ↓
Your FastAPI Server (Port 8000)
        ↓ HTTP calls
MCP Server (Port 3001)
        ↓
Research Processing Components
```

## 🚀 Quick Start

### Step 1: Start the MCP Server

```bash
# Navigate to the demo_prompt copy directory
cd "demo_prompt copy"

# Install dependencies
pip install -r requirements.txt

# Start the MCP server
python start_mcp_server.py --host localhost --port 3001
```

The MCP server will start and show:
```
🚀 Starting MCP Research Server
Host: localhost
Port: 3001
Server URL: http://localhost:3001
Available endpoints:
  Health Check: http://localhost:3001/health
  Tool Call:    http://localhost:3001/mcp/call
  Upload:       http://localhost:3001/mcp/upload-and-process
  List Tools:   http://localhost:3001/mcp/list-tools
  Downloads:    http://localhost:3001/presentations/{filename}
```

### Step 2: Integrate with Your FastAPI Server

Add these imports to your existing FastAPI application:

```python
from fastapi import FastAPI
import sys
from pathlib import Path

# Add the demo_prompt copy directory to Python path
mcp_dir = Path("path/to/demo_prompt copy")
sys.path.insert(0, str(mcp_dir))

# Import MCP integration
from api_integration.fastapi_routes import router as mcp_router, cleanup_mcp_client

# Your existing FastAPI app
app = FastAPI()

# Include MCP routes
app.include_router(mcp_router)

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await cleanup_mcp_client()

# Your existing routes continue here...
```

### Step 3: Test the Integration

```bash
# Test MCP server health
curl http://localhost:3001/health

# Test through your FastAPI server
curl http://localhost:8000/api/v1/mcp/health
```

## 📡 Available API Endpoints

Once integrated, your FastAPI server will have these new endpoints:

### Health & Status
- `GET /api/v1/mcp/health` - Check MCP server health
- `GET /api/v1/mcp/tools` - List available tools
- `GET /api/v1/mcp/status` - System status

### Paper Processing
- `POST /api/v1/mcp/papers/upload` - Upload and process PDF papers
- `GET /api/v1/mcp/papers/{paper_id}` - Get paper information

### Search
- `POST /api/v1/mcp/search/web` - Web search (Google, Scholar, News)
- `POST /api/v1/mcp/search/semantic` - Semantic search within papers

### Presentations
- `POST /api/v1/mcp/presentations/generate` - Generate PowerPoint presentations
- `GET /api/v1/mcp/presentations/{filename}/download` - Download presentations

### Analysis
- `POST /api/v1/mcp/analysis/research` - Research intelligence analysis
- `POST /api/v1/mcp/insights/generate` - Generate research insights

## 🔧 Usage Examples

### Upload and Process a Paper

```python
import httpx

async def upload_paper():
    async with httpx.AsyncClient() as client:
        with open("research_paper.pdf", "rb") as f:
            files = {"file": ("research_paper.pdf", f, "application/pdf")}
            data = {
                "paper_id": "my_paper_001",
                "enable_research_analysis": True,
                "enable_vector_storage": True,
                "analysis_depth": "comprehensive"
            }
            
            response = await client.post(
                "http://localhost:8000/api/v1/mcp/papers/upload",
                files=files,
                data=data
            )
            
            return response.json()
```

### Search the Web

```python
async def search_web():
    async with httpx.AsyncClient() as client:
        payload = {
            "query": "machine learning healthcare applications",
            "search_type": "scholar",
            "num_results": 10,
            "enhance_results": True
        }
        
        response = await client.post(
            "http://localhost:8000/api/v1/mcp/search/web",
            json=payload
        )
        
        return response.json()
```

### Generate a Presentation

```python
async def generate_presentation():
    async with httpx.AsyncClient() as client:
        payload = {
            "paper_id": "my_paper_001",
            "user_prompt": "Create a presentation focusing on methodology and results for medical professionals",
            "title": "Research Findings Presentation",
            "theme": "academic_professional",
            "slide_count": 15,
            "audience_type": "academic"
        }
        
        response = await client.post(
            "http://localhost:8000/api/v1/mcp/presentations/generate",
            json=payload
        )
        
        return response.json()
```

### Semantic Search

```python
async def semantic_search():
    async with httpx.AsyncClient() as client:
        payload = {
            "query": "What were the statistical results and p-values?",
            "paper_id": "my_paper_001",
            "max_results": 5,
            "similarity_threshold": 0.7
        }
        
        response = await client.post(
            "http://localhost:8000/api/v1/mcp/search/semantic",
            json=payload
        )
        
        return response.json()
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the "demo_prompt copy" directory:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# SerpAPI Configuration (for web search)
SERPAPI_API_KEY=your_serpapi_key_here

# Pinecone Configuration (for vector storage)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment

# LlamaParse Configuration (for advanced PDF parsing)
LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here

# MCP Server Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=3001
```

### MCP Server Configuration

The MCP server can be configured via command line arguments:

```bash
# Custom host and port
python start_mcp_server.py --host 0.0.0.0 --port 3002

# Enable debug logging
python start_mcp_server.py --debug

# Help
python start_mcp_server.py --help
```

## 🐳 Docker Deployment

### Docker Compose Setup

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  mcp-server:
    build:
      context: .
      dockerfile: Dockerfile.mcp
    ports:
      - "3001:3001"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SERPAPI_API_KEY=${SERPAPI_API_KEY}
      - PINECONE_API_KEY=${PINECONE_API_KEY}
    volumes:
      - ./presentations:/app/presentations
      - ./cache:/app/cache
    restart: unless-stopped

  your-fastapi-app:
    build:
      context: .
      dockerfile: Dockerfile.api
    ports:
      - "8000:8000"
    depends_on:
      - mcp-server
    environment:
      - MCP_SERVER_URL=http://mcp-server:3001
    restart: unless-stopped
```

### Dockerfile for MCP Server

Create `Dockerfile.mcp`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p presentations cache logs temp exports

# Expose port
EXPOSE 3001

# Start MCP server
CMD ["python", "start_mcp_server.py", "--host", "0.0.0.0", "--port", "3001"]
```

## 🔍 Troubleshooting

### Common Issues

1. **MCP Server Won't Start**
   ```bash
   # Check if port is already in use
   lsof -i :3001
   
   # Try a different port
   python start_mcp_server.py --port 3002
   ```

2. **Connection Refused Errors**
   ```bash
   # Verify MCP server is running
   curl http://localhost:3001/health
   
   # Check logs for errors
   python start_mcp_server.py --debug
   ```

3. **Import Errors**
   ```python
   # Make sure the path is correct in your FastAPI app
   mcp_dir = Path("correct/path/to/demo_prompt copy")
   sys.path.insert(0, str(mcp_dir))
   ```

4. **API Key Errors**
   ```bash
   # Verify your .env file has all required keys
   cat .env
   
   # Test individual services
   curl -X POST http://localhost:3001/mcp/call \
     -H "Content-Type: application/json" \
     -d '{"tool": "advanced_search_web", "arguments": {"query": "test"}}'
   ```

### Debugging

Enable debug mode for detailed logging:

```bash
python start_mcp_server.py --debug
```

Check server health:

```bash
curl http://localhost:3001/health | jq
```

Test tool availability:

```bash
curl -X POST http://localhost:3001/mcp/list-tools | jq
```

## 📊 Monitoring

### Health Checks

Set up health check monitoring:

```python
import asyncio
import httpx

async def monitor_mcp_health():
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:3001/health")
            health = response.json()
            print(f"MCP Server Status: {health['status']}")
            print(f"Uptime: {health['uptime']} seconds")
            print(f"Memory Usage: {health['memory_usage']}")
            return health['status'] == 'healthy'
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

# Run health check
healthy = asyncio.run(monitor_mcp_health())
```

### Performance Monitoring

Monitor request performance:

```python
import time
import httpx

async def benchmark_mcp_call():
    async with httpx.AsyncClient() as client:
        start_time = time.time()
        
        response = await client.post(
            "http://localhost:3001/mcp/call",
            json={
                "tool": "advanced_search_web",
                "arguments": {"query": "test query"}
            }
        )
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        print(f"Request took: {execution_time:.2f} seconds")
        return response.json()
```

## 🚀 Production Deployment

### Scaling

For production, consider:

1. **Multiple MCP Server Instances**
   ```bash
   # Start multiple instances
   python start_mcp_server.py --port 3001 &
   python start_mcp_server.py --port 3002 &
   python start_mcp_server.py --port 3003 &
   ```

2. **Load Balancer Configuration**
   ```nginx
   upstream mcp_servers {
       server localhost:3001;
       server localhost:3002;
       server localhost:3003;
   }
   
   server {
       location /mcp/ {
           proxy_pass http://mcp_servers;
       }
   }
   ```

3. **Process Management**
   ```bash
   # Using systemd
   sudo systemctl start mcp-server
   sudo systemctl enable mcp-server
   ```

## 🎯 Next Steps

1. **Test the Integration**: Start both servers and test the endpoints
2. **Update Your Frontend**: Modify your frontend to use the new API endpoints
3. **Configure Environment**: Set up your API keys and configuration
4. **Deploy**: Use Docker or your preferred deployment method
5. **Monitor**: Set up health checks and monitoring
6. **Scale**: Add more MCP server instances as needed

## 📞 Support

If you encounter issues:

1. Check the logs with `--debug` flag
2. Verify all API keys are configured
3. Test individual components separately
4. Check network connectivity between services

The integration provides a clean separation between your main application and the research processing capabilities, making it easy to scale and maintain both services independently. 