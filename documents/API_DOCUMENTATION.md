# 📚 Complete API Documentation

## 🚀 Server Information
- **Base URL**: `http://localhost:3003`
- **Protocol**: HTTP/JSON
- **Total Tools**: 19 available tools
- **Version**: 1.0.0

## 🌟 Configuration Settings
- **Knowledge Base Index**: `optimized-kb-index`
- **Knowledge Base Namespace**: `knowledge-base`
- **Research Papers Index**: `all-pdf-index`

---

## 📋 API Endpoints Overview

### 1. Health Check
**Method**: `GET`  
**Endpoint**: `/health`  
**Purpose**: Check server status

#### Request:
```bash
curl -X GET http://localhost:3003/health
```

#### Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": 2656.127779,
  "active_connections": 1,
  "tools_count": 19,
  "memory_usage": "20.8MB"
}
```

### 2. MCP Tool Call
**Method**: `POST`  
**Endpoint**: `/mcp/call`  
**Purpose**: Execute any MCP tool

#### Request Format:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "TOOL_NAME",
    "arguments": {
      "param1": "value1",
      "param2": "value2"
    }
  }'
```

#### Response Format:
```json
{
  "success": true/false,
  "result": {
    "content": "Tool output content",
    "type": "text"
  },
  "error": null,
  "request_id": "unique-id",
  "execution_time": 1.234,
  "timestamp": "2025-07-10T16:10:11.696487"
}
```

---

## 🔧 Knowledge Base APIs

### 1. Search Knowledge Base
**Tool**: `search_knowledge_base`  
**Purpose**: Search through the knowledge base with enhanced semantic search

#### Input Parameters:
```json
{
  "query": "string (required) - Search query",
  "search_type": "string (optional) - 'enhanced' or 'basic', default: 'enhanced'",
  "max_results": "integer (optional) - Max results to return, default: 5",
  "namespace": "string (optional) - default: 'knowledge-base'",
  "index_name": "string (optional) - default: 'optimized-kb-index'"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search_knowledge_base",
    "arguments": {
      "query": "machine learning algorithms",
      "search_type": "enhanced",
      "max_results": 3,
      "namespace": "knowledge-base",
      "index_name": "optimized-kb-index"
    }
  }'
```

#### Expected Output:
```json
{
  "success": true,
  "query": "machine learning algorithms",
  "search_type": "enhanced",
  "results": "Formatted search results with source information, confidence scores, and content snippets",
  "namespace": "knowledge-base",
  "index_name": "optimized-kb-index"
}
```

### 2. Get Knowledge Base Inventory
**Tool**: `get_knowledge_base_inventory`  
**Purpose**: Get complete inventory of knowledge base content

#### Input Parameters:
```json
{
  "namespace": "string (optional) - default: 'knowledge-base'",
  "index_name": "string (optional) - default: 'optimized-kb-index'"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "get_knowledge_base_inventory",
    "arguments": {
      "namespace": "knowledge-base",
      "index_name": "optimized-kb-index"
    }
  }'
```

#### Expected Output:
```json
{
  "success": true,
  "inventory": {
    "books": ["List of book titles"],
    "total_chunks": 1240,
    "books_structure": {
      "BookTitle": {
        "chapters": ["Chapter list"],
        "sections": ["Section list"],
        "chunk_count": 458,
        "mathematical_content": 445,
        "total_words": 173690.0,
        "chunk_types": {
          "mixed": 184,
          "formula": 210,
          "text": 13,
          "table": 51
        }
      }
    }
  },
  "namespace": "knowledge-base",
  "index_name": "optimized-kb-index"
}
```

### 3. Find Books Covering Topic
**Tool**: `find_books_covering_topic`  
**Purpose**: Find which books in knowledge base cover a specific topic

#### Input Parameters:
```json
{
  "topic": "string (required) - Topic to search for",
  "namespace": "string (optional) - default: 'knowledge-base'",
  "index_name": "string (optional) - default: 'optimized-kb-index'"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "find_books_covering_topic",
    "arguments": {
      "topic": "neural networks",
      "namespace": "knowledge-base",
      "index_name": "optimized-kb-index"
    }
  }'
```

#### Expected Output:
```json
{
  "success": true,
  "topic": "neural networks",
  "books": ["List of relevant book titles"],
  "namespace": "knowledge-base",
  "index_name": "optimized-kb-index"
}
```

---

## 🧠 Universal Document Processing APIs

### 1. Process Research Paper
**Tool**: `process_research_paper_universal`  
**Purpose**: Process research papers with academic analysis

#### Input Parameters:
```json
{
  "file_content": "string (required) - Hex-encoded PDF content",
  "filename": "string (required) - Original filename",
  "paper_id": "string (optional) - Unique paper identifier",
  "enable_analysis": "boolean (optional) - Enable deep analysis, default: true",
  "analysis_depth": "string (optional) - 'basic' or 'comprehensive', default: 'comprehensive'"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "process_research_paper_universal",
    "arguments": {
      "file_content": "hex_encoded_pdf_content",
      "filename": "research_paper.pdf",
      "paper_id": "paper_001",
      "enable_analysis": true,
      "analysis_depth": "comprehensive"
    }
  }'
```

#### Expected Output:
```json
{
  "success": true,
  "processing_type": "research_paper",
  "index_used": "all-pdf-index",
  "analysis_results": "Academic analysis and insights",
  "chunks_created": 25,
  "paper_id": "paper_001"
}
```

### 2. Process Knowledge Base
**Tool**: `process_knowledge_base`  
**Purpose**: Process documents for knowledge base with optimized extraction

#### Input Parameters:
```json
{
  "file_content": "string (required) - Hex-encoded PDF content",
  "filename": "string (required) - Original filename", 
  "book_name": "string (optional) - Book/document name",
  "enable_llamaparse": "boolean (optional) - Use LlamaParse, default: true",
  "extraction_mode": "string (optional) - Extraction mode, default: 'knowledge_extraction'"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "process_knowledge_base",
    "arguments": {
      "file_content": "hex_encoded_pdf_content",
      "filename": "knowledge_book.pdf",
      "book_name": "Data Science Handbook",
      "enable_llamaparse": true,
      "extraction_mode": "knowledge_extraction"
    }
  }'
```

#### Expected Output:
```json
{
  "success": true,
  "processing_type": "knowledge_base",
  "index_used": "optimized-kb-index",
  "extraction_results": "Knowledge extraction results",
  "chunks_created": 35,
  "book_name": "Data Science Handbook"
}
```

---

## 📊 System Management APIs

### 1. System Status
**Tool**: `system_status`  
**Purpose**: Get detailed system status and configuration

#### Input Parameters:
```json
{
  "include_config": "boolean (optional) - Include configuration details, default: false",
  "run_health_check": "boolean (optional) - Run comprehensive health check, default: false"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "system_status",
    "arguments": {
      "include_config": true,
      "run_health_check": true
    }
  }'
```

#### Expected Output:
```json
{
  "success": true,
  "status": "# 🚀 Perfect Research System Status\n\n## 🔧 Core Components...",
  "health_check": "All systems operational",
  "tools_count": 19,
  "uptime": "43 minutes"
}
```

---

## 📄 Document Management APIs

### 1. List Documents
**Tool**: `list_documents`  
**Purpose**: List all processed documents

#### Input Parameters:
```json
{}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "list_documents",
    "arguments": {}
  }'
```

### 2. Semantic Search
**Tool**: `semantic_search`  
**Purpose**: General semantic search across all documents

#### Input Parameters:
```json
{
  "query": "string (required) - Search query",
  "max_results": "integer (optional) - Max results, default: 10"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "semantic_search",
    "arguments": {
      "query": "machine learning",
      "max_results": 5
    }
  }'
```

### 3. Web Search
**Tool**: `web_search`  
**Purpose**: Search the web using SerpAPI

#### Input Parameters:
```json
{
  "query": "string (required) - Search query",
  "max_results": "integer (optional) - Max results, default: 10"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "web_search",
    "arguments": {
      "query": "artificial intelligence trends 2025",
      "max_results": 3
    }
  }'
```

---

## 🎨 Presentation Generation APIs

### 1. Create Presentation
**Tool**: `create_presentation`  
**Purpose**: Generate presentations from processed content

#### Input Parameters:
```json
{
  "paper_id": "string (required) - Paper/document ID",
  "user_prompt": "string (required) - User requirements",
  "title": "string (optional) - Presentation title",
  "slide_count": "integer (optional) - Number of slides, default: 10",
  "theme": "string (optional) - Presentation theme, default: 'professional'"
}
```

#### Example Request:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "create_presentation",
    "arguments": {
      "paper_id": "paper_001",
      "user_prompt": "Create a technical presentation about machine learning algorithms",
      "title": "ML Algorithms Overview",
      "slide_count": 12,
      "theme": "academic_professional"
    }
  }'
```

---

## 🛠️ Additional Tool APIs

### 1. Extract Images
**Tool**: `extract_images`  
**Purpose**: Extract images from PDF documents

### 2. Process PDF
**Tool**: `process_pdf`  
**Purpose**: Basic PDF processing

### 3. Analyze Research Paper
**Tool**: `analyze_research_paper`  
**Purpose**: Deep analysis of research papers

### 4. Generate Summary
**Tool**: `generate_summary`  
**Purpose**: Generate summaries of processed content

### 5. Fact Check Content
**Tool**: `fact_check_content`  
**Purpose**: Verify facts in processed content

---

## 🔍 Usage Examples

### Basic Knowledge Base Search:
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search_knowledge_base",
    "arguments": {
      "query": "deep learning"
    }
  }'
```

### Upload and Process Research Paper:
```bash
# First convert PDF to hex
xxd -p document.pdf | tr -d '\n' > hex_content.txt

# Then process
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "process_research_paper_universal",
    "arguments": {
      "file_content": "'$(cat hex_content.txt)'",
      "filename": "document.pdf",
      "paper_id": "my_paper_001"
    }
  }'
```

### Get System Health:
```bash
curl -X GET http://localhost:3003/health
```

---

## 🚨 Error Handling

### Common Error Response:
```json
{
  "success": false,
  "error": "Error description",
  "request_id": "unique-id",
  "timestamp": "2025-07-10T16:10:11.696487"
}
```

### HTTP Status Codes:
- `200` - Success
- `400` - Bad Request (invalid parameters)
- `404` - Tool not found
- `500` - Internal server error

---

## 📝 Notes

1. **File Upload**: All file content must be hex-encoded
2. **Async Processing**: Some operations may take time
3. **Rate Limiting**: No current limits, but monitor usage
4. **Authentication**: Currently no authentication required
5. **CORS**: Enabled for cross-origin requests

## 🔗 Quick Reference

| API Type | Tool Count | Primary Use |
|----------|------------|-------------|
| Knowledge Base | 3 tools | Search, inventory, topic discovery |
| Document Processing | 2 tools | Research papers, knowledge base |
| System Management | 1 tool | Status, health, configuration |
| Document Management | 3 tools | List, search, web search |
| Presentation | 1+ tools | PPT generation |
| Additional Tools | 9+ tools | Images, analysis, summaries |

**Total: 19 tools available** 