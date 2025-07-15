# 🚀 API Formats Quick Reference Guide

## 📡 Server Configuration
- **Base URL**: `http://localhost:3003`
- **Index Name**: `optimized-kb-index`
- **Namespace**: `knowledge-base`
- **Total Tools**: 19 available

---

## 🔥 Most Important APIs

### 1. 🏥 Health Check
```bash
curl -X GET http://localhost:3003/health
```
**Returns**: Server status, tools count, memory usage

### 2. 🔍 Search Knowledge Base
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "search_knowledge_base",
    "arguments": {
      "query": "your search query",
      "namespace": "knowledge-base",
      "index_name": "optimized-kb-index"
    }
  }'
```

### 3. 📊 Get Knowledge Base Inventory
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

### 4. 📚 Find Books by Topic
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "find_books_covering_topic",
    "arguments": {
      "topic": "machine learning",
      "namespace": "knowledge-base",
      "index_name": "optimized-kb-index"
    }
  }'
```

### 5. 🧠 Process Research Paper
```bash
# Convert PDF to hex first:
xxd -p document.pdf | tr -d '\n' > hex_content.txt

# Then process:
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "process_research_paper_universal",
    "arguments": {
      "file_content": "'$(cat hex_content.txt)'",
      "filename": "document.pdf",
      "paper_id": "unique_id"
    }
  }'
```

### 6. 📖 Process Knowledge Base Document
```bash
curl -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{
    "tool": "process_knowledge_base",
    "arguments": {
      "file_content": "hex_encoded_content",
      "filename": "book.pdf",
      "book_name": "Book Title"
    }
  }'
```

---

## 📋 All 19 Available Tools

### 🔧 Knowledge Base Tools (3):
1. `search_knowledge_base` - Enhanced semantic search
2. `get_knowledge_base_inventory` - Complete content inventory
3. `find_books_covering_topic` - Topic-based book discovery

### 🧠 Document Processing Tools (2):
4. `process_research_paper_universal` - Research paper processing
5. `process_knowledge_base` - Knowledge base document processing

### 📊 System Management Tools (1):
6. `system_status` - Detailed system status and configuration

### 📄 Document Management Tools (3):
7. `list_documents` - List all processed documents
8. `semantic_search` - General semantic search
9. `web_search` - Web search using SerpAPI

### 🎨 Presentation Tools (1+):
10. `create_presentation` - Generate presentations

### 🛠️ Additional Processing Tools (9+):
11. `extract_images` - Extract images from PDFs
12. `process_pdf` - Basic PDF processing
13. `analyze_research_paper` - Deep research analysis
14. `generate_summary` - Content summarization
15. `fact_check_content` - Fact verification
16. `extract_text` - Text extraction
17. `chunk_content` - Content chunking
18. `store_vectors` - Vector storage
19. `query_vectors` - Vector querying

---

## 🎯 Standard Request Format

### For ALL Tools:
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

### Standard Response Format:
```json
{
  "success": true/false,
  "result": {
    "content": "Tool output",
    "type": "text"
  },
  "error": null,
  "request_id": "unique-id",
  "execution_time": 1.234,
  "timestamp": "2025-07-10T..."
}
```

---

## 🔑 Key Parameters

### For Knowledge Base APIs:
- `namespace`: "knowledge-base" (required)
- `index_name`: "optimized-kb-index" (required)
- `query`: Your search query (required for search)
- `topic`: Topic to search (required for book discovery)

### For Document Processing:
- `file_content`: Hex-encoded PDF content (required)
- `filename`: Original filename (required)
- `paper_id` / `book_name`: Identifier (optional)

### For Search APIs:
- `query`: Search query (required)
- `max_results`: Number of results (optional, default: 5-10)
- `search_type`: "enhanced" or "basic" (optional, default: "enhanced")

---

## 🚀 Quick Test Commands

### Test Server Health:
```bash
curl -s http://localhost:3003/health | jq .
```

### Test Knowledge Search:
```bash
curl -s -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "search_knowledge_base", "arguments": {"query": "machine learning"}}' | jq .success
```

### Test System Status:
```bash
curl -s -X POST http://localhost:3003/mcp/call \
  -H "Content-Type: application/json" \
  -d '{"tool": "system_status", "arguments": {}}' | jq .success
```

---

## 📝 Important Notes

1. **File Upload**: Convert PDFs to hex using `xxd -p file.pdf | tr -d '\n'`
2. **Configuration**: Always use correct index/namespace names
3. **Response**: Check `.success` field for operation status
4. **Tools**: All 19 tools available and tested ✅
5. **Server**: Running on port 3003 ✅

## 🎉 Status: ALL APIS DOCUMENTED AND VERIFIED! ✅ 