"""
HTTP Transport Layer for MCP Server
This file handles all HTTP communication for the MCP server
Converts your existing MCP server to work over HTTP for FastAPI integration
"""

import asyncio
import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid
import base64

from fastapi import FastAPI, HTTPException, Request, Depends, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class MCPToolCallRequest(BaseModel):
    """Request model for MCP tool calls"""
    tool: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique request ID")

class MCPToolCallResponse(BaseModel):
    """Response model for MCP tool calls"""
    success: bool = Field(..., description="Whether the call succeeded")
    result: Optional[Dict[str, Any]] = Field(None, description="Tool execution result")
    error: Optional[Dict[str, Any]] = Field(None, description="Error information if failed")
    request_id: str = Field(..., description="Request ID from the request")
    execution_time: float = Field(..., description="Execution time in seconds")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class MCPListToolsResponse(BaseModel):
    """Response model for listing tools"""
    tools: List[Dict[str, Any]] = Field(..., description="Available tools")
    count: int = Field(..., description="Number of tools")

class MCPHealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Server status")
    version: str = Field(..., description="Server version")
    uptime: float = Field(..., description="Server uptime in seconds")
    active_connections: int = Field(..., description="Number of active connections")
    tools_count: int = Field(..., description="Number of available tools")
    memory_usage: str = Field(..., description="Current memory usage")

# ============================================================================
# HTTP TRANSPORT CLASS
# ============================================================================

class MCPHTTPTransport:
    """
    HTTP Transport layer for MCP Server
    
    This class wraps your existing MCP server and provides HTTP endpoints
    for external clients (like your FastAPI server) to communicate with it.
    """
    
    def __init__(self, mcp_server, host: str = "localhost", port: int = 3001):
        """
        Initialize HTTP transport
        
        Args:
            mcp_server: Your existing PerfectMCPServer instance
            host: Host to bind to
            port: Port to listen on
        """
        self.mcp_server = mcp_server
        self.host = host
        self.port = port
        self.start_time = datetime.now()
        self.active_connections = 0
        
        # Create FastAPI app for MCP server
        self.app = FastAPI(
            title="MCP Research Server",
            description="HTTP interface for MCP Research Server - Integrates with your main FastAPI app",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure this properly in production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self._setup_routes()
        
        logger.info(f"MCP HTTP Transport initialized on {host}:{port}")
    
    def _setup_routes(self):
        """Setup all HTTP routes for MCP communication"""
        
        @self.app.middleware("http")
        async def connection_counter(request: Request, call_next):
            """Middleware to count active connections"""
            self.active_connections += 1
            try:
                response = await call_next(request)
                return response
            finally:
                self.active_connections -= 1
        
        @self.app.post("/mcp/call", response_model=MCPToolCallResponse)
        async def call_tool(request: MCPToolCallRequest):
            """
            Call an MCP tool
            
            This endpoint receives tool calls from your FastAPI server and forwards them
            to your existing MCP server.
            """
            start_time = datetime.now()
            
            try:
                logger.info(f"Calling tool: {request.tool} with request_id: {request.request_id}")
                
                # Call the tool on your MCP server
                result = await self._execute_tool(request.tool, request.arguments)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return MCPToolCallResponse(
                    success=True,
                    result=result,
                    request_id=request.request_id,
                    execution_time=execution_time
                )
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_details = {
                    "code": "TOOL_EXECUTION_ERROR",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                logger.error(f"Tool execution failed: {error_details}")
                
                return MCPToolCallResponse(
                    success=False,
                    error=error_details,
                    request_id=request.request_id,
                    execution_time=execution_time
                )
        
        @self.app.post("/mcp/upload-and-process", response_model=MCPToolCallResponse)
        async def upload_and_process_paper(
            file: UploadFile = File(...),
            paper_id: str = "auto",
            enable_research_analysis: bool = True,
            enable_vector_storage: bool = True,
            analysis_depth: str = "comprehensive"
        ):
            """
            Upload and process PDF paper
            
            This endpoint handles file uploads and processes them through your MCP server.
            Your FastAPI server can call this endpoint to process PDFs.
            """
            start_time = datetime.now()
            
            try:
                # Read file content
                file_content = await file.read()
                file_content_b64 = base64.b64encode(file_content).decode()
                
                # Generate paper ID if auto
                if paper_id == "auto":
                    paper_id = f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Call MCP tool
                arguments = {
                    "file_content": file_content_b64,
                    "file_name": file.filename,
                    "paper_id": paper_id,
                    "enable_research_analysis": enable_research_analysis,
                    "enable_vector_storage": enable_vector_storage,
                    "analysis_depth": analysis_depth
                }
                
                result = await self._execute_tool("process_research_paper", arguments)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return MCPToolCallResponse(
                    success=True,
                    result=result,
                    request_id=str(uuid.uuid4()),
                    execution_time=execution_time
                )
                
            except Exception as e:
                execution_time = (datetime.now() - start_time).total_seconds()
                error_details = {
                    "code": "FILE_PROCESSING_ERROR",
                    "message": str(e),
                    "traceback": traceback.format_exc()
                }
                
                logger.error(f"File processing failed: {error_details}")
                
                return MCPToolCallResponse(
                    success=False,
                    error=error_details,
                    request_id=str(uuid.uuid4()),
                    execution_time=execution_time
                )
        
        @self.app.post("/mcp/list-tools", response_model=MCPListToolsResponse)
        async def list_tools():
            """
            List all available MCP tools
            
            Returns information about all tools that your FastAPI server can call.
            """
            try:
                # Get tools from your MCP server
                tools = await self._get_available_tools()
                
                return MCPListToolsResponse(
                    tools=tools,
                    count=len(tools)
                )
                
            except Exception as e:
                logger.error(f"Failed to list tools: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health", response_model=MCPHealthResponse)
        async def health_check():
            """
            Health check endpoint
            
            Your FastAPI server can call this to check if MCP server is running.
            """
            import psutil
            import os
            
            uptime = (datetime.now() - self.start_time).total_seconds()
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            tools_count = len(await self._get_available_tools())
            
            return MCPHealthResponse(
                status="healthy",
                version="1.0.0",
                uptime=uptime,
                active_connections=self.active_connections,
                tools_count=tools_count,
                memory_usage=f"{memory_mb:.1f}MB"
            )
        
        @self.app.get("/presentations/{filename}")
        async def download_presentation(filename: str):
            """
            Download generated presentation files
            
            Your FastAPI server can provide download links to users through this endpoint.
            """
            import os
            presentations_dir = os.path.join(os.path.dirname(__file__), "../../presentations")
            file_path = os.path.join(presentations_dir, filename)
            
            if os.path.exists(file_path):
                return FileResponse(
                    file_path,
                    media_type='application/vnd.openxmlformats-officedocument.presentationml.presentation',
                    filename=filename
                )
            else:
                raise HTTPException(status_code=404, detail="Presentation not found")
        
        @self.app.get("/")
        async def root():
            """Root endpoint with basic info"""
            return {
                "service": "MCP Research Server",
                "version": "1.0.0",
                "status": "running",
                "description": "HTTP interface for MCP server - integrates with your FastAPI app",
                "endpoints": {
                    "call_tool": "POST /mcp/call",
                    "upload_process": "POST /mcp/upload-and-process",
                    "list_tools": "POST /mcp/list-tools", 
                    "health": "GET /health",
                    "download": "GET /presentations/{filename}"
                }
            }
    
    async def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool on the MCP server
        
        This method bridges HTTP requests to your existing MCP server.
        """
        try:
            # Call the appropriate method based on tool name
            if tool_name == "advanced_search_web":
                result = await self.mcp_server._handle_advanced_search(**arguments)
            elif tool_name == "process_research_paper":
                result = await self.mcp_server._handle_process_paper(**arguments)
            elif tool_name == "create_perfect_presentation":
                result = await self.mcp_server._handle_create_presentation(**arguments)
            elif tool_name == "create_presentation_from_namespace":
                result = await self.mcp_server._handle_create_presentation_from_namespace(**arguments)
            elif tool_name == "semantic_paper_search":
                result = await self.mcp_server._handle_semantic_search(**arguments)
            elif tool_name == "research_intelligence_analysis":
                result = await self.mcp_server._handle_research_analysis(**arguments)
            elif tool_name == "generate_research_insights":
                result = await self.mcp_server._handle_generate_insights(**arguments)
            elif tool_name == "list_processed_papers":
                result = await self.mcp_server._handle_list_papers(**arguments)
            elif tool_name == "system_status":
                result = await self.mcp_server._handle_system_status(**arguments)
            else:
                raise ValueError(f"Unknown tool: {tool_name}")
            
            # Convert MCP result to HTTP format
            if isinstance(result, list) and len(result) > 0:
                # Extract content from MCP TextContent objects
                content = result[0].text if hasattr(result[0], 'text') else str(result[0])
                return {"content": content, "type": "text"}
            else:
                return {"content": str(result), "type": "text"}
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise
    
    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools from MCP server
        """
        try:
            if hasattr(self.mcp_server, 'server') and hasattr(self.mcp_server.server, '_list_tools_handler'):
                # Get tools from MCP server
                tools_result = await self.mcp_server.server._list_tools_handler()
                
                # Convert to HTTP format
                tools = []
                for tool in tools_result:
                    tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                
                return tools
            else:
                # Fallback: return hardcoded tool list
                return [
                    {
                        "name": "advanced_search_web",
                        "description": "Advanced web search with academic focus",
                        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
                    },
                    {
                        "name": "process_research_paper",
                        "description": "Process PDF research paper with advanced extraction",
                        "input_schema": {"type": "object", "properties": {"file_content": {"type": "string"}}}
                    },
                    {
                        "name": "create_perfect_presentation",
                        "description": "Create perfect research presentation",
                        "input_schema": {"type": "object", "properties": {"paper_id": {"type": "string"}}}
                    },
                    {
                        "name": "semantic_paper_search",
                        "description": "Perform semantic search within processed papers",
                        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}}
                    }
                ]
                
        except Exception as e:
            logger.error(f"Error getting tools: {e}")
            return []
    
    async def start(self):
        """Start the HTTP server"""
        config = uvicorn.Config(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        
        logger.info(f"Starting MCP HTTP server on {self.host}:{self.port}")
        await server.serve()
    
    def run(self):
        """Run the server (blocking)"""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        ) 