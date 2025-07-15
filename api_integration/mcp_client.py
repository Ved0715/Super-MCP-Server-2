"""
MCP Client for FastAPI Integration
This client allows your FastAPI server to communicate with the standalone MCP server
"""

import httpx
import asyncio
import logging
from typing import Dict, Any, List, Optional, Union
import json
from pathlib import Path
import aiofiles

logger = logging.getLogger(__name__)

class MCPClient:
    """
    HTTP client for communicating with the standalone MCP server
    
    Your FastAPI server will use this client to:
    1. Send requests to the MCP server
    2. Handle responses and errors
    3. Manage file uploads
    4. Check server health
    """
    
    def __init__(self, base_url: str = "http://localhost:3001", timeout: float = 300.0):
        """
        Initialize MCP client
        
        Args:
            base_url: Base URL of the MCP server (e.g., http://localhost:3001)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
        
        logger.info(f"MCP Client initialized for {base_url}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    # ========================================================================
    # HEALTH AND STATUS METHODS
    # ========================================================================
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check if MCP server is healthy
        
        Returns:
            Health status information
        """
        try:
            response = await self.client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def get_server_info(self) -> Dict[str, Any]:
        """
        Get basic server information
        
        Returns:
            Server information and available endpoints
        """
        try:
            response = await self.client.get(f"{self.base_url}/")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get server info: {e}")
            return {"error": str(e)}
    
    async def list_tools(self) -> Dict[str, Any]:
        """
        Get list of available MCP tools
        
        Returns:
            List of tools with their descriptions and schemas
        """
        try:
            response = await self.client.post(f"{self.base_url}/mcp/list-tools")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return {"error": str(e), "tools": [], "count": 0}
    
    # ========================================================================
    # TOOL EXECUTION METHODS
    # ========================================================================
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call an MCP tool
        
        Args:
            tool_name: Name of the tool to call
            arguments: Tool arguments
        
        Returns:
            Tool execution result
        """
        try:
            payload = {
                "tool": tool_name,
                "arguments": arguments
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp/call",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name}: {e}")
            return {
                "success": False,
                "error": {
                    "code": "CLIENT_ERROR",
                    "message": str(e)
                }
            }
    
    # ========================================================================
    # RESEARCH PAPER METHODS
    # ========================================================================
    
    async def upload_and_process_paper(
        self,
        file_path: Union[str, Path],
        paper_id: Optional[str] = None,
        enable_research_analysis: bool = True,
        enable_vector_storage: bool = True,
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Upload and process a research paper
        
        Args:
            file_path: Path to the PDF file
            paper_id: Unique identifier for the paper (auto-generated if None)
            enable_research_analysis: Whether to perform research analysis
            enable_vector_storage: Whether to store in vector database
            analysis_depth: Depth of analysis (basic, standard, comprehensive)
        
        Returns:
            Processing result
        """
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {
                    "success": False,
                    "error": {"code": "FILE_NOT_FOUND", "message": f"File not found: {file_path}"}
                }
            
            # Prepare form data
            files = {"file": (file_path.name, open(file_path, "rb"), "application/pdf")}
            data = {
                "paper_id": paper_id or "auto",
                "enable_research_analysis": enable_research_analysis,
                "enable_vector_storage": enable_vector_storage,
                "analysis_depth": analysis_depth
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp/upload-and-process",
                files=files,
                data=data
            )
            
            # Close the file
            files["file"][1].close()
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Paper processing failed: {e}")
            return {
                "success": False,
                "error": {
                    "code": "PROCESSING_ERROR",
                    "message": str(e)
                }
            }
    
    async def process_paper_from_bytes(
        self,
        file_content: bytes,
        filename: str,
        paper_id: Optional[str] = None,
        enable_research_analysis: bool = True,
        enable_vector_storage: bool = True,
        analysis_depth: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Process a research paper from bytes content
        
        Args:
            file_content: PDF file content as bytes
            filename: Original filename
            paper_id: Unique identifier for the paper
            enable_research_analysis: Whether to perform research analysis
            enable_vector_storage: Whether to store in vector database
            analysis_depth: Depth of analysis
        
        Returns:
            Processing result
        """
        try:
            # Prepare form data
            files = {"file": (filename, file_content, "application/pdf")}
            data = {
                "paper_id": paper_id or "auto",
                "enable_research_analysis": enable_research_analysis,
                "enable_vector_storage": enable_vector_storage,
                "analysis_depth": analysis_depth
            }
            
            response = await self.client.post(
                f"{self.base_url}/mcp/upload-and-process",
                files=files,
                data=data
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Paper processing failed: {e}")
            return {
                "success": False,
                "error": {
                    "code": "PROCESSING_ERROR",
                    "message": str(e)
                }
            }
    
    # ========================================================================
    # SEARCH METHODS
    # ========================================================================
    
    async def search_web(
        self,
        query: str,
        search_type: str = "web",
        num_results: int = 10,
        location: str = "United States",
        enhance_results: bool = True
    ) -> Dict[str, Any]:
        """
        Perform web search
        
        Args:
            query: Search query
            search_type: Type of search (web, scholar, news)
            num_results: Number of results to return
            location: Search location
            enhance_results: Whether to enhance results with AI
        
        Returns:
            Search results
        """
        arguments = {
            "query": query,
            "search_type": search_type,
            "num_results": num_results,
            "location": location,
            "enhance_results": enhance_results
        }
        
        return await self.call_tool("advanced_search_web", arguments)
    
    async def semantic_search(
        self,
        query: str,
        paper_id: Optional[str] = None,
        search_type: str = "general",
        max_results: int = 10,
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        Perform semantic search within processed papers
        
        Args:
            query: Search query
            paper_id: Specific paper to search (optional)
            search_type: Type of search (general, methodology, results, etc.)
            max_results: Maximum number of results
            similarity_threshold: Minimum similarity score
        
        Returns:
            Search results
        """
        arguments = {
            "query": query,
            "paper_id": paper_id,
            "search_type": search_type,
            "max_results": max_results,
            "similarity_threshold": similarity_threshold
        }
        
        return await self.call_tool("semantic_paper_search", arguments)
    
    # ========================================================================
    # PRESENTATION METHODS
    # ========================================================================
    
    async def create_presentation(
        self,
        query: str,
        user_prompt: str,
        title: Optional[str] = None,
        author: str = "AI Research Assistant",
        theme: str = "academic_professional",
        slide_count: int = 12,
        audience_type: str = "academic",
        include_web_references: bool = False,
        reference_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a research presentation from knowledge base using Chain-of-Thought reasoning
        
        Args:
            query: Topic/query for presentation content
            user_prompt: User requirements for the presentation
            title: Presentation title
            author: Author name
            theme: Presentation theme
            slide_count: Number of slides
            audience_type: Target audience
            include_web_references: Whether to include web search for reference links
            reference_query: Query for additional reference links
        
        Returns:
            Presentation creation result
        """
        arguments = {
            "query": query,
            "user_prompt": user_prompt,
            "title": title,
            "author": author,
            "theme": theme,
            "slide_count": slide_count,
            "audience_type": audience_type,
            "include_web_references": include_web_references,
            "reference_query": reference_query
        }
        
        return await self.call_tool("create_perfect_presentation", arguments)
    
    async def download_presentation(self, filename: str) -> bytes:
        """
        Download a generated presentation
        
        Args:
            filename: Name of the presentation file
        
        Returns:
            File content as bytes
        """
        try:
            response = await self.client.get(f"{self.base_url}/presentations/{filename}")
            response.raise_for_status()
            return response.content
        except Exception as e:
            logger.error(f"Failed to download presentation {filename}: {e}")
            raise
    
    async def create_presentation_from_namespace(
        self,
        namespace: str,
        user_prompt: str,
        title: Optional[str] = None,
        author: str = "AI Research Assistant",
        theme: str = "academic_professional",
        slide_count: int = 12,
        audience_type: str = "academic",
        search_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create presentation from namespace-based vector search
        
        Args:
            namespace: Vector database namespace (user_{user_id}_doc_{doc_id})
            user_prompt: User's presentation requirements
            title: Presentation title
            author: Presentation author
            theme: Presentation theme
            slide_count: Number of slides
            audience_type: Target audience
            search_query: Additional search context
        
        Returns:
            Presentation generation result
        """
        try:
            result = await self.call_tool("create_presentation_from_namespace", {
                "namespace": namespace,
                "user_prompt": user_prompt,
                "title": title,
                "author": author,
                "theme": theme,
                "slide_count": slide_count,
                "audience_type": audience_type,
                "search_query": search_query
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Namespace presentation creation failed: {e}")
            return {
                "success": False,
                "error": {
                    "code": "NAMESPACE_PRESENTATION_ERROR",
                    "message": str(e)
                }
            }
    
    # ========================================================================
    # ANALYSIS METHODS
    # ========================================================================
    
    async def analyze_research(
        self,
        paper_id: str,
        analysis_types: Optional[List[str]] = None,
        provide_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Perform research intelligence analysis
        
        Args:
            paper_id: ID of the paper to analyze
            analysis_types: Types of analysis to perform
            provide_recommendations: Whether to provide recommendations
        
        Returns:
            Analysis results
        """
        arguments = {
            "paper_id": paper_id,
            "analysis_types": analysis_types or ["methodology", "contributions", "quality"],
            "provide_recommendations": provide_recommendations
        }
        
        return await self.call_tool("research_intelligence_analysis", arguments)
    
    async def generate_insights(
        self,
        paper_id: str,
        focus_area: str = "future_research",
        insight_depth: str = "detailed",
        include_citations: bool = True
    ) -> Dict[str, Any]:
        """
        Generate research insights
        
        Args:
            paper_id: ID of the paper
            focus_area: Area to focus on
            insight_depth: Depth of insights
            include_citations: Whether to include citations
        
        Returns:
            Generated insights
        """
        arguments = {
            "paper_id": paper_id,
            "focus_area": focus_area,
            "insight_depth": insight_depth,
            "include_citations": include_citations
        }
        
        return await self.call_tool("generate_research_insights", arguments)

# ============================================================================
# CONVENIENCE FUNCTIONS FOR YOUR FASTAPI SERVER
# ============================================================================

async def get_mcp_client(base_url: str = "http://localhost:3001") -> MCPClient:
    """
    Get an MCP client instance
    
    Use this in your FastAPI dependency injection:
    
    @app.get("/api/papers")
    async def list_papers(mcp_client: MCPClient = Depends(get_mcp_client)):
        # Use mcp_client here
    """
    return MCPClient(base_url=base_url)

class MCPClientManager:
    """
    Manager for MCP client connections
    
    Use this in your FastAPI app to manage client connections efficiently.
    """
    
    def __init__(self, base_url: str = "http://localhost:3001"):
        self.base_url = base_url
        self._client = None
    
    async def get_client(self) -> MCPClient:
        """Get or create MCP client"""
        if self._client is None:
            self._client = MCPClient(self.base_url)
        return self._client
    
    async def close(self):
        """Close client connections"""
        if self._client:
            await self._client.close()
            self._client = None 