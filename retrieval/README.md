# HybridRetriever - Advanced Knowledge Base Retrieval System

## Overview

The HybridRetriever system provides sophisticated knowledge base search and retrieval capabilities with intelligent query routing and specialized response generation.

## Features

### 🧠 Intelligent Query Routing
- **Study Plan Queries**: Automatically detects requests for learning curricula and generates structured study plans
- **Topic Location**: Finds specific topics within books with precise page references
- **Chapter Analysis**: Provides comprehensive book structure analysis
- **Concept Explanation**: Offers detailed explanations with contextual examples
- **Knowledge Base Meta**: Handles inventory and capability queries

### 🔍 Hybrid Search Technology
- **Dense Retrieval**: Uses Pinecone vector database with text-embedding-3-large
- **Sparse Retrieval**: BM25 keyword-based search for exact term matching
- **Reranking**: Cross-encoder model for result quality optimization
- **Smart Fusion**: Combines multiple search strategies for optimal results

### 📚 Comprehensive Book Analysis
- **Real Chapter Extraction**: Identifies actual book chapters vs. references/citations
- **Hardcoded Knowledge Base**: Curated information for 4 data science books
- **Topic Coverage**: Maps topics across all available books
- **Page References**: Provides specific page numbers for content location

### 🎯 Specialized Prompt Templates
- **Study Plan Generation**: Creates structured learning curricula
- **Chapter Analysis**: Analyzes book structure and organization
- **Topic Location**: Pinpoints content locations with citations
- **Concept Explanation**: Provides educational explanations
- **Comparison Analysis**: Offers technical comparisons
- **Knowledge Base Meta**: Handles system capability questions

## Integration with MCP Server

### Implementation Details

The HybridRetriever is integrated into the Perfect MCP Server through the `search_knowledge_base` tool:

```python
# Initialization in PerfectMCPServer
if HYBRID_RETRIEVER_AVAILABLE:
    self.hybrid_retriever = HybridRetriever()

# Smart query routing in _handle_search_knowledge_base
if any(indicator in query_lower for indicator in special_indicators):
    response_text = self.hybrid_retriever.search_knowledge_base_contents(query)
else:
    response_text = await self._run_hybrid_retriever_async(query, max_results)
```

### Key Methods

1. **`search_knowledge_base_contents(query)`**: Handles specialized queries (study plans, topic location, inventory)
2. **`answer_question(query, top_k)`**: Processes regular questions using enhanced RAG
3. **`get_comprehensive_book_analysis()`**: Provides complete knowledge base analysis
4. **`generate_study_plan(book_name, duration)`**: Creates structured learning plans

### Query Types Detected

- **Study Plans**: "study plan", "learning plan", "curriculum", "study schedule"
- **Topic Location**: "where is", "find topic", "location of", "covered in"
- **Chapter Analysis**: "chapters", "table of contents", "book structure"
- **Knowledge Base Meta**: "what books", "inventory", "available content"
- **Regular RAG**: All other queries use enhanced retrieval with AI response generation

## Usage Examples

### Study Plan Generation
```
Query: "Give me a 90-day study plan for machine learning"
Response: Comprehensive curriculum with daily tasks, page references, and milestones
```

### Topic Location
```
Query: "Where is linear regression covered in the 40 Algorithms book?"
Response: Specific chapter, page numbers, and content previews
```

### Knowledge Base Inventory
```
Query: "What books do you have available?"
Response: Complete list of books with chapters, topics, and descriptions
```

### Concept Explanation
```
Query: "Explain gradient descent algorithm"
Response: Educational explanation with examples, formulas, and applications
```

## Technical Architecture

### Core Components

1. **HybridRetriever Class**: Main orchestrator for all retrieval operations
2. **QueryProcessor**: Analyzes query intent and expands search terms
3. **Prompt Templates**: Specialized system prompts for different query types
4. **Chapter Extraction**: Identifies real book structure vs. artifacts

### Data Sources

- **Pinecone Vector Database**: Stores document embeddings and metadata
- **BM25 Index**: Keyword-based sparse retrieval
- **Hardcoded Knowledge Base**: Curated chapter information for 4 books
- **Enhanced Metadata**: Chapter titles, page references, and topic mappings

### Response Generation

- Uses OpenAI GPT-4 with specialized prompts
- Removes confidence scores from responses
- Includes proper academic citations
- Provides structured, educational content

## Configuration

### Required Dependencies
```
openai
pinecone-client
sentence-transformers
rank-bm25
tiktoken
numpy
tqdm
```

### Environment Variables
```
OPENAI_API_KEY=your_openai_key
PINECONE_API_KEY=your_pinecone_key
```

## Testing

Run the integration test:
```bash
python perfect_mcp_server.py --test
```

This will test:
- HybridRetriever initialization
- Query type detection
- Specialized content search
- Regular RAG functionality

## Error Handling

The system includes comprehensive error handling:

1. **Fallback to Basic Search**: If HybridRetriever fails, falls back to vector storage
2. **Parameter Validation**: Detects and logs incorrect parameter usage
3. **Graceful Degradation**: Continues operation even if advanced features fail
4. **Detailed Logging**: Provides debug information for troubleshooting

## Bug Fixes

### Parameter Bug Resolution
- Added parameter validation to prevent `use_chain_of_thought` being passed to `search_knowledge_base`
- Fixed tool signature to accept `**kwargs` and filter unexpected parameters
- Added explicit error detection and logging for parameter misuse

### Integration Improvements
- Async wrapper for synchronous HybridRetriever methods
- Proper import handling with fallback options
- Comprehensive error handling and logging

## Future Enhancements

1. **Dynamic Knowledge Base Updates**: Real-time addition of new documents
2. **Personalized Learning Paths**: User-specific study plan generation
3. **Advanced Analytics**: Usage patterns and learning progress tracking
4. **Multi-modal Support**: Integration with images, videos, and interactive content 