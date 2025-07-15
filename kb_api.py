from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
import asyncio

from knowledge_base_retrieval import AdvancedKnowledgeBaseRetriever

router = APIRouter()

# Global retriever instance
retriever = None

class QueryRequest(BaseModel):
    """Single query request model"""
    query: str
    context: Optional[Dict[str, Any]] = None

class UnifiedResponse(BaseModel):
    """Unified response model that can handle any type of response"""
    success: bool
    query: str
    response_type: str  # 'search', 'study_plan', 'book_analysis', 'chapters'
    data: Dict[str, Any]
    execution_time: float
    timestamp: str

def get_retriever():
    """Get or initialize the retriever"""
    global retriever
    if retriever is None:
        retriever = AdvancedKnowledgeBaseRetriever()
    return retriever

@router.post("/query", response_model=UnifiedResponse)
async def intelligent_query(
    request: QueryRequest,
    background_tasks: BackgroundTasks
) -> UnifiedResponse:
    """
    🧠 INTELLIGENT KNOWLEDGE BASE QUERY ENDPOINT
    
    This single endpoint handles ALL types of knowledge base queries:
    - Search queries: "What is machine learning?"
    - Study plans: "Create a study plan for deep learning"
    - Book analysis: "Analyze books about algorithms"
    - Chapter extraction: "Show chapters from machine learning book"
    
    The API automatically determines query intent and responds appropriately.
    
    Args:
        request: QueryRequest containing the user query and optional context
        
    Returns:
        UnifiedResponse with appropriate data based on query type
    """
    start_time = time.time()
    
    try:
        # Get retriever instance
        kb_retriever = get_retriever()
        
        # Perform intelligent search
        result = await kb_retriever.intelligent_search(
            query=request.query,
            top_k=10,
            namespace="knowledge-base",
            index_name="optimized-kb-index"
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Determine response type
        response_type = result.get("response_type", "search")
        
        # Format unified response
        unified_response = UnifiedResponse(
            success=result.get("success", True),
            query=request.query,
            response_type=response_type,
            data=result,
            execution_time=execution_time,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
        
        return unified_response
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Query processing failed: {str(e)}",
                "query": request.query,
                "execution_time": execution_time
            }
        )

@router.get("/health")
async def health_check():
    """Health check for the knowledge base retrieval API"""
    try:
        # Test retriever initialization
        kb_retriever = get_retriever()
        
        return {
            "status": "healthy",
            "service": "Knowledge Base Retrieval API",
            "version": "1.0.0",
            "retriever_initialized": kb_retriever is not None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

@router.get("/stats")
async def get_stats():
    """Get statistics about the knowledge base"""
    try:
        kb_retriever = get_retriever()
        
        # Get knowledge base inventory
        inventory = await kb_retriever.vector_storage.get_knowledge_base_inventory()
        
        return {
            "success": True,
            "stats": {
                "total_books": len(inventory.get("books", [])),
                "total_chunks": inventory.get("total_chunks", 0),
                "books_available": inventory.get("books", []),
                "namespace": inventory.get("namespace", "knowledge-base"),
                "index_name": inventory.get("index_name", "optimized-kb-index")
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to get stats: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

@router.get("/books")
async def get_books():
    """Get list of available books in the knowledge base"""
    try:
        kb_retriever = get_retriever()
        
        # Get knowledge base inventory
        inventory = await kb_retriever.vector_storage.get_knowledge_base_inventory()
        
        # Get book structure info
        book_details = {}
        for book in inventory.get("books", []):
            # Try to get chapters from hardcoded structure
            chapters = await kb_retriever._get_book_chapters(book)
            book_details[book] = {
                "chapters": chapters,
                "chapter_count": len(chapters)
            }
        
        return {
            "success": True,
            "books": inventory.get("books", []),
            "book_details": book_details,
            "total_books": len(inventory.get("books", [])),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Failed to get books: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        )

# Example usage documentation
@router.get("/examples")
async def get_examples():
    """Get example queries for different use cases"""
    return {
        "examples": {
            "search_queries": [
                "What is machine learning?",
                "Explain neural networks",
                "How do clustering algorithms work?",
                "What is the difference between supervised and unsupervised learning?"
            ],
            "study_plan_queries": [
                "Create a study plan for machine learning",
                "I want to learn data science from scratch",
                "Help me plan my deep learning studies",
                "Design a curriculum for algorithms"
            ],
            "book_analysis_queries": [
                "Analyze books about machine learning",
                "Which books cover neural networks?",
                "Compare available data science books",
                "Show me books suitable for beginners"
            ],
            "chapter_queries": [
                "Show chapters from machine learning book",
                "What chapters are in the algorithms book?",
                "List table of contents for data science books",
                "Get chapter structure of available books"
            ]
        },
        "usage_tips": [
            "Use natural language - the API understands intent",
            "Be specific about your learning goals",
            "Ask for study plans if you want structured learning",
            "Request book analysis for recommendations",
            "Query chapters for detailed content structure"
        ]
    }

# Advanced search endpoint for power users
@router.post("/advanced-search")
async def advanced_search(
    query: str,
    search_type: str = "hybrid",
    top_k: int = 10,
    namespace: str = "knowledge-base",
    index_name: str = "optimized-kb-index"
):
    """
    Advanced search endpoint for power users who want more control
    
    Args:
        query: Search query
        search_type: Type of search ('hybrid', 'dense', 'sparse')
        top_k: Number of results to return
        namespace: Vector database namespace
        index_name: Vector database index name
    """
    start_time = time.time()
    
    try:
        kb_retriever = get_retriever()
        
        if search_type == "hybrid":
            # Use the full hybrid search
            result = await kb_retriever._hybrid_search(
                query=query,
                query_analysis={"type": "factual", "difficulty": "intermediate"},
                top_k=top_k,
                namespace=namespace,
                index_name=index_name
            )
        else:
            # Use basic enhanced search
            results = await kb_retriever.vector_storage.enhanced_knowledge_base_search(
                query=query,
                namespace=namespace,
                top_k=top_k,
                index_name=index_name
            )
            result = {
                "success": True,
                "query": query,
                "search_type": search_type,
                "results": results,
                "total_results": len(results)
            }
        
        execution_time = time.time() - start_time
        
        return {
            "success": result.get("success", True),
            "query": query,
            "search_type": search_type,
            "results": result.get("results", []),
            "total_results": result.get("total_results", 0),
            "execution_time": execution_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        raise HTTPException(
            status_code=500,
            detail={
                "error": f"Advanced search failed: {str(e)}",
                "query": query,
                "execution_time": execution_time
            }
        )

# Diagnostic endpoint
@router.get("/diagnostics")
async def run_diagnostics():
    """Run diagnostic tests on the knowledge base retrieval system"""
    try:
        kb_retriever = get_retriever()
        
        diagnostics = {
            "retriever_initialized": kb_retriever is not None,
            "vector_storage_available": kb_retriever.vector_storage is not None,
            "openai_client_available": kb_retriever.openai_client is not None,
            "query_processor_available": kb_retriever.query_processor is not None,
            "reranker_available": kb_retriever.reranker is not None,
            "bm25_initialized": kb_retriever.bm25 is not None
        }
        
        # Test basic functionality
        try:
            test_query = "test query"
            query_analysis = kb_retriever.query_processor.understand_query(test_query)
            diagnostics["query_analysis_working"] = query_analysis is not None
        except Exception as e:
            diagnostics["query_analysis_working"] = False
            diagnostics["query_analysis_error"] = str(e)
        
        # Test vector storage
        try:
            inventory = await kb_retriever.vector_storage.get_knowledge_base_inventory()
            diagnostics["vector_storage_working"] = inventory is not None
            diagnostics["books_available"] = len(inventory.get("books", []))
        except Exception as e:
            diagnostics["vector_storage_working"] = False
            diagnostics["vector_storage_error"] = str(e)
        
        return {
            "success": True,
            "diagnostics": diagnostics,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        } 