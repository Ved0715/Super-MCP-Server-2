"""
FastAPI Routes for MCP Integration
Add these routes to your existing FastAPI server to integrate with the MCP server
"""

from fastapi import APIRouter, HTTPException, File, UploadFile, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import tempfile
import os
import io
from pathlib import Path
import time

from .mcp_client import MCPClient, MCPClientManager

logger = logging.getLogger(__name__)

# ============================================================================
# PYDANTIC MODELS FOR API
# ============================================================================

class PaperProcessRequest(BaseModel):
    """Request model for paper processing"""
    paper_id: Optional[str] = None
    enable_research_analysis: bool = True
    enable_vector_storage: bool = True
    analysis_depth: str = "comprehensive"

class SearchRequest(BaseModel):
    """Request model for web search"""
    query: str
    search_type: str = "web"
    num_results: int = 10
    location: str = "United States"
    enhance_results: bool = True

class SemanticSearchRequest(BaseModel):
    """Request model for semantic search"""
    query: str
    paper_id: Optional[str] = None
    search_type: str = "general"
    max_results: int = 10
    similarity_threshold: float = 0.7

class PresentationRequest(BaseModel):
    """Request model for presentation generation"""
    query: str
    user_prompt: str
    title: Optional[str] = None
    author: str = "AI Research Assistant"
    theme: str = "academic_professional"
    slide_count: int = 12
    audience_type: str = "academic"
    include_web_references: bool = False
    reference_query: Optional[str] = None

class AnalysisRequest(BaseModel):
    """Request model for research analysis"""
    paper_id: str
    analysis_types: Optional[List[str]] = None
    provide_recommendations: bool = True

class InsightsRequest(BaseModel):
    """Request model for insights generation"""
    paper_id: str
    focus_area: str = "future_research"
    insight_depth: str = "detailed"
    include_citations: bool = True

class NamespacePresentationRequest(BaseModel):
    """Request model for namespace-based presentation generation"""
    user_id: str  # Required - User identifier
    doc_id: str   # Required - Document UUID
    prompt: str   # Required - User's presentation requirements
    title: Optional[str] = None
    author: str = "AI Research Assistant"
    theme: str = "academic_professional"
    slide_count: int = 12
    audience_type: str = "academic"
    search_query: Optional[str] = None

class KnowledgeBaseRequest(BaseModel):
    """Request model for knowledge base content processing"""
    book_name: Optional[str] = None
    enable_llamaparse: bool = True
    extraction_mode: str = "knowledge_extraction"

class KnowledgeBaseSearchRequest(BaseModel):
    """Request model for knowledge base search"""
    query: str
    search_type: str = "enhanced"
    max_results: int = 5
    namespace: str = "knowledge-base"
    index_name: str = "optimized-kb-index"

class QuizGenerationRequest(BaseModel):
    """Request model for quiz generation"""
    user_id: str
    document_uuid: str
    number_of_questions: int = 10
    difficulty_level: str = "mixed"
    include_explanations: bool = True
    question_categories: List[str] = ["conceptual", "methodological", "factual", "analytical"]

# ============================================================================
# MCP CLIENT DEPENDENCY
# ============================================================================

# Create a global MCP client manager
mcp_manager = MCPClientManager()

async def get_mcp_client() -> MCPClient:
    """Dependency to get MCP client"""
    return await mcp_manager.get_client()

# ============================================================================
# FASTAPI ROUTER
# ============================================================================

# Create router that you can include in your main FastAPI app
router = APIRouter(prefix="/api/v1/mcp", tags=["MCP Research"])

# Create knowledge base retrieval router
from kb_api import router as kb_router
kb_router.prefix = "/api/v1/kb"
kb_router.tags = ["Knowledge Base Retrieval"]

# ============================================================================
# HEALTH AND STATUS ENDPOINTS
# ============================================================================

@router.get("/health")
async def check_mcp_health(mcp_client: MCPClient = Depends(get_mcp_client)):
    """
    Check MCP server health
    
    Add this endpoint to your FastAPI server to monitor MCP server status.
    """
    try:
        health = await mcp_client.health_check()
        return health
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="MCP server unavailable")

@router.get("/tools")
async def list_available_tools(mcp_client: MCPClient = Depends(get_mcp_client)):
    """
    List all available MCP tools
    
    Your frontend can call this to know what capabilities are available.
    """
    try:
        tools = await mcp_client.list_tools()
        return tools
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve tools")

# ============================================================================
# PAPER PROCESSING ENDPOINTS
# ============================================================================

@router.post("/papers/upload")
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    paper_id: Optional[str] = None,
    enable_research_analysis: bool = True,
    enable_vector_storage: bool = True,
    analysis_depth: str = "comprehensive",
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Upload and process a research paper
    
    This endpoint handles PDF uploads and processes them through the MCP server.
    Your frontend can call this to upload papers.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        file_content = await file.read()
        
        # Process the paper
        result = await mcp_client.process_paper_from_bytes(
            file_content=file_content,
            filename=file.filename,
            paper_id=paper_id,
            enable_research_analysis=enable_research_analysis,
            enable_vector_storage=enable_vector_storage,
            analysis_depth=analysis_depth
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Paper upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Paper processing failed: {str(e)}")

@router.get("/papers/{paper_id}")
async def get_paper_info(
    paper_id: str,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Get information about a processed paper
    
    Your frontend can call this to get paper details and processing status.
    """
    try:
        # You might want to implement a tool in your MCP server to get paper info
        # For now, we'll use a generic tool call
        result = await mcp_client.call_tool("get_paper_info", {"paper_id": paper_id})
        return result
    except Exception as e:
        logger.error(f"Failed to get paper info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve paper information")

# ============================================================================
# NEW UNIVERSAL PROCESSING ENDPOINTS
# ============================================================================

@router.post("/process/research-paper")
async def process_research_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    paper_id: Optional[str] = None,
    user_id: Optional[str] = None,
    document_uuid: Optional[str] = None,
    enable_research_analysis: bool = True,
    analysis_depth: str = "comprehensive",
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Process research papers using enhanced PDF processor
    
    This endpoint specifically handles research papers with academic analysis,
    citation extraction, and storage in the all-pdf-index.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        file_content = await file.read()
        
        # Use the universal processor via MCP tool
        result = await mcp_client.call_tool("process_research_paper", {
            "file_content": file_content.hex(),  # Convert bytes to hex string
            "filename": file.filename,
            "paper_id": paper_id,
            "user_id": user_id,
            "document_uuid": document_uuid,
            "enable_research_analysis": enable_research_analysis,
            "analysis_depth": analysis_depth
        })
        
        # Generate namespace for response
        namespace = None
        if user_id and document_uuid:
            namespace = f"user_{user_id}_doc_{document_uuid}"
        elif user_id:
            namespace = f"user_{user_id}_doc_{paper_id}"
        elif document_uuid:
            namespace = f"doc_{document_uuid}"
        else:
            namespace = paper_id
            
        return {
            "success": True,
            "message": "Research paper processed successfully",
            "data": result,
            "processing_type": "research_paper",
            "index_used": "all-pdf-index",
            "namespace": namespace,
            "user_id": user_id,
            "document_uuid": document_uuid
        }
        
    except Exception as e:
        logger.error(f"Research paper processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research paper processing failed: {str(e)}")

@router.post("/process/knowledge-base")
async def process_knowledge_base(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    book_name: Optional[str] = None,
    enable_llamaparse: bool = True,
    extraction_mode: str = "knowledge_extraction",
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Process knowledge base content (books, manuals, etc.)
    
    This endpoint specifically handles knowledge base content with optimized
            chunking, mathematical entity extraction, and storage in optimized-kb-index.
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read file content
        file_content = await file.read()
        
        # Extract book name from filename if not provided
        if not book_name:
            book_name = Path(file.filename).stem.replace('_', ' ').replace('-', ' ').title()
        
        # Use the universal processor via MCP tool
        result = await mcp_client.call_tool("process_knowledge_base", {
            "file_content": file_content.hex(),  # Convert bytes to hex string
            "filename": file.filename,
            "book_name": book_name,
            "enable_llamaparse": enable_llamaparse,
            "extraction_mode": extraction_mode
        })
        
        return {
            "success": True,
            "message": "Knowledge base content processed successfully",
            "data": result,
            "processing_type": "knowledge_base",
            "index_used": "optimized-kb-index",
            "book_name": book_name
        }
        
    except Exception as e:
        logger.error(f"Knowledge base processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Knowledge base processing failed: {str(e)}")

# Legacy /search/knowledge-base endpoint removed - use /kb/query for intelligent knowledge base queries

# Legacy /knowledge-base/inventory endpoint removed - use /kb/stats for knowledge base statistics

# Legacy /knowledge-base/books/{topic} endpoint removed - use /kb/query for intelligent book analysis
# Example: POST /kb/query {"query": "Analyze books about {topic}"}

# ============================================================================
# SEARCH ENDPOINTS
# ============================================================================

@router.post("/search/web")
async def search_web(
    request: SearchRequest,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Perform web search
    
    Your frontend can call this to search Google, Scholar, or News.
    """
    try:
        result = await mcp_client.search_web(
            query=request.query,
            search_type=request.search_type,
            num_results=request.num_results,
            location=request.location,
            enhance_results=request.enhance_results
        )
        return result
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@router.post("/search/semantic")
async def semantic_search(
    request: SemanticSearchRequest,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Perform semantic search within processed papers
    
    Your frontend can call this to search within uploaded papers using AI.
    """
    try:
        result = await mcp_client.semantic_search(
            query=request.query,
            paper_id=request.paper_id,
            search_type=request.search_type,
            max_results=request.max_results,
            similarity_threshold=request.similarity_threshold
        )
        return result
    except Exception as e:
        logger.error(f"Semantic search failed: {e}")
        raise HTTPException(status_code=500, detail="Semantic search failed")

# ============================================================================
# PRESENTATION ENDPOINTS
# ============================================================================

@router.post("/presentations/generate")
async def generate_presentation(
    request: PresentationRequest,
    background_tasks: BackgroundTasks,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Generate a research presentation from knowledge base using Chain-of-Thought reasoning
    
    Your frontend can call this to create PowerPoint presentations from knowledge base content.
    """
    try:
        result = await mcp_client.create_presentation(
            query=request.query,
            user_prompt=request.user_prompt,
            title=request.title,
            author=request.author,
            theme=request.theme,
            slide_count=request.slide_count,
            audience_type=request.audience_type,
            include_web_references=request.include_web_references,
            reference_query=request.reference_query
        )
        return result
    except Exception as e:
        logger.error(f"Presentation generation failed: {e}")
        raise HTTPException(status_code=500, detail="Presentation generation failed")

@router.get("/presentations/{filename}/download")
async def download_presentation(
    filename: str,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Download a generated presentation
    
    Your frontend can provide download links using this endpoint.
    """
    try:
        file_content = await mcp_client.download_presentation(filename)
        
        return StreamingResponse(
            io.BytesIO(file_content),
            media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        logger.error(f"Presentation download failed: {e}")
        raise HTTPException(status_code=404, detail="Presentation not found")

@router.post("/presentations/generate-from-namespace")
async def generate_presentation_from_namespace(
    request: NamespacePresentationRequest,
    background_tasks: BackgroundTasks,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Generate presentation from namespace-based vector search
    
    Takes user_id and doc_id, constructs namespace, searches for information,
    and generates PPT based on found content.
    """
    start_time = time.time()
    
    try:
        logger.info(f"🌟 NEW REQUEST: Namespace-based presentation generation")
        logger.info(f"👤 User ID: {request.user_id}")
        logger.info(f"📄 Document ID: {request.doc_id}")
        logger.info(f"💭 User prompt: {request.prompt}")
        logger.info(f"🎨 Settings: theme={request.theme}, slides={request.slide_count}, audience={request.audience_type}")
        
        # Construct namespace: user_{user_id}_doc_{doc_id}
        namespace = f"user_{request.user_id}_doc_{request.doc_id}"
        logger.info(f"📁 Constructed namespace: {namespace}")
        
        # Call MCP server for namespace-based PPT generation
        logger.info(f"🔗 Calling MCP server for presentation generation...")
        mcp_start_time = time.time()
        
        result = await mcp_client.create_presentation_from_namespace(
            namespace=namespace,
            user_prompt=request.prompt,
            title=request.title,
            author=request.author,
            theme=request.theme,
            slide_count=request.slide_count,
            audience_type=request.audience_type,
            search_query=request.search_query
        )
        
        mcp_duration = time.time() - mcp_start_time
        total_duration = time.time() - start_time
        
        logger.info(f"✅ MCP server call completed in {mcp_duration:.2f}s")
        logger.info(f"🎉 Total API request completed in {total_duration:.2f}s")
        
        # Log result summary
        if result.get("success"):
            logger.info(f"✅ Presentation generation successful")
        else:
            logger.warning(f"⚠️  Presentation generation had issues: {result.get('error', 'Unknown error')}")
        
        logger.info(f"📤 Returning result to client")
        return result
        
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"❌ FastAPI endpoint error after {total_duration:.2f}s: {e}")
        logger.error(f"🔍 Error details: {str(e)}")
        import traceback
        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Presentation generation failed: {str(e)}")

# ============================================================================
# ANALYSIS ENDPOINTS
# ============================================================================

@router.post("/analysis/research")
async def analyze_research(
    request: AnalysisRequest,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Perform research intelligence analysis
    
    Your frontend can call this to get detailed analysis of research papers.
    """
    try:
        result = await mcp_client.analyze_research(
            paper_id=request.paper_id,
            analysis_types=request.analysis_types,
            provide_recommendations=request.provide_recommendations
        )
        return result
    except Exception as e:
        logger.error(f"Research analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Research analysis failed")

@router.post("/insights/generate")
async def generate_insights(
    request: InsightsRequest,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Generate research insights
    
    Your frontend can call this to get AI-generated insights from papers.
    """
    try:
        result = await mcp_client.generate_insights(
            paper_id=request.paper_id,
            focus_area=request.focus_area,
            insight_depth=request.insight_depth,
            include_citations=request.include_citations
        )
        return result
    except Exception as e:
        logger.error(f"Insights generation failed: {e}")
        raise HTTPException(status_code=500, detail="Insights generation failed")

# ============================================================================
# QUIZ GENERATION ENDPOINTS
# ============================================================================

@router.post("/quiz/generate")
async def generate_research_quiz(
    request: QuizGenerationRequest,
    background_tasks: BackgroundTasks,
    mcp_client: MCPClient = Depends(get_mcp_client)
):
    """
    Generate a comprehensive MCQ quiz from a specific research paper
    
    This endpoint generates multiple-choice questions from a research paper
    stored in the user's namespace (user_{user_id}_doc_{document_uuid}).
    
    Parameters:
    - user_id: User identifier who uploaded the PDF
    - document_uuid: Unique identifier for the research paper
    - number_of_questions: Number of MCQ questions to generate (5-50)
    - difficulty_level: Difficulty level (easy/medium/hard/mixed)
    - include_explanations: Whether to include answer explanations
    - question_categories: Types of questions to generate
    
    Returns:
    - JSON with quiz questions, metadata, and content analysis
    """
    start_time = time.time()
    
    try:
        logger.info(f"🧠 NEW REQUEST: Quiz generation for research paper")
        logger.info(f"👤 User ID: {request.user_id}")
        logger.info(f"📄 Document UUID: {request.document_uuid}")
        logger.info(f"❓ Questions: {request.number_of_questions}")
        logger.info(f"🎯 Difficulty: {request.difficulty_level}")
        logger.info(f"📚 Categories: {request.question_categories}")
        
        # Construct namespace
        namespace = f"user_{request.user_id}_doc_{request.document_uuid}"
        logger.info(f"📁 Target namespace: {namespace}")
        
        # Call MCP server for quiz generation
        logger.info(f"🔗 Calling MCP server for quiz generation...")
        mcp_start_time = time.time()
        
        result = await mcp_client.call_tool("generate_research_quiz", {
            "user_id": request.user_id,
            "document_uuid": request.document_uuid,
            "number_of_questions": request.number_of_questions,
            "difficulty_level": request.difficulty_level,
            "include_explanations": request.include_explanations,
            "question_categories": request.question_categories
        })
        
        mcp_duration = time.time() - mcp_start_time
        total_duration = time.time() - start_time
        
        logger.info(f"✅ MCP server call completed in {mcp_duration:.2f}s")
        logger.info(f"🎉 Total API request completed in {total_duration:.2f}s")
        
        # Parse and validate result
        if isinstance(result, list) and len(result) > 0:
            result_text = result[0].get("text", "{}")
            import json
            try:
                parsed_result = json.loads(result_text)
                if parsed_result.get("success"):
                    logger.info(f"✅ Quiz generation successful - {len(parsed_result.get('questions', []))} questions generated")
                    return {
                        "success": True,
                        "message": "Quiz generated successfully",
                        "data": parsed_result,
                        "performance": {
                            "total_time": total_duration,
                            "mcp_time": mcp_duration,
                            "processing_time": total_duration - mcp_duration
                        }
                    }
                else:
                    logger.warning(f"⚠️  Quiz generation failed: {parsed_result.get('error', 'Unknown error')}")
                    return {
                        "success": False,
                        "error": parsed_result.get("error", "Quiz generation failed"),
                        "data": parsed_result
                    }
            except json.JSONDecodeError as e:
                logger.error(f"❌ Failed to parse MCP response: {e}")
                return {
                    "success": False,
                    "error": "Invalid response format from MCP server",
                    "raw_response": result_text
                }
        else:
            logger.error(f"❌ Invalid MCP response format")
            return {
                "success": False,
                "error": "Invalid response format from MCP server",
                "raw_response": str(result)
            }
        
    except Exception as e:
        total_duration = time.time() - start_time
        logger.error(f"❌ Quiz generation API error after {total_duration:.2f}s: {e}")
        logger.error(f"🔍 Error details: {str(e)}")
        import traceback
        logger.error(f"📋 Full traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500, 
            detail=f"Quiz generation failed: {str(e)}"
        )

# ============================================================================
# UTILITY ENDPOINTS
# ============================================================================

@router.get("/status")
async def get_system_status(mcp_client: MCPClient = Depends(get_mcp_client)):
    """
    Get comprehensive system status
    
    Your frontend can call this to check overall system health and capabilities.
    """
    try:
        # Get MCP server health
        health = await mcp_client.health_check()
        
        # Get available tools
        tools = await mcp_client.list_tools()
        
        # Get server info
        server_info = await mcp_client.get_server_info()
        
        return {
            "mcp_server": health,
            "available_tools": tools.get("count", 0),
            "server_info": server_info,
            "integration_status": "active"
        }
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return {
            "mcp_server": {"status": "unhealthy"},
            "available_tools": 0,
            "integration_status": "failed",
            "error": str(e)
        }

# ============================================================================
# CLEANUP
# ============================================================================

async def cleanup_mcp_client():
    """Cleanup function to close MCP client connections"""
    await mcp_manager.close()

# ============================================================================
# HOW TO INTEGRATE WITH YOUR FASTAPI APP
# ============================================================================

"""
To integrate these routes with your existing FastAPI app, add this to your main FastAPI file:

```python
from fastapi import FastAPI
from api_integration.fastapi_routes import router as mcp_router, cleanup_mcp_client

# Your existing FastAPI app
app = FastAPI()

# Include the MCP routes
app.include_router(mcp_router)

# Add cleanup on shutdown
@app.on_event("shutdown")
async def shutdown_event():
    await cleanup_mcp_client()

# Your existing routes...
```

Then your frontend can call:
- POST /api/v1/mcp/papers/upload - Upload papers
- POST /api/v1/mcp/search/web - Web search  
- POST /api/v1/mcp/presentations/generate - Generate PPTs
- GET /api/v1/mcp/health - Check MCP server health
- And all other endpoints defined above
""" 