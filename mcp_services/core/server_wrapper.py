"""
MCP Server Wrapper
This file wraps your existing PerfectMCPServer and makes it work with HTTP transport
"""

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import your existing modules
current_dir = Path(__file__).parent
parent_dir = current_dir.parent.parent
sys.path.insert(0, str(parent_dir))

# Import your existing MCP server and components
from perfect_mcp_server import PerfectMCPServer
from mcp_services.transports.http_transport import MCPHTTPTransport

logger = logging.getLogger(__name__)

class MCPServerWrapper:
    """
    Wrapper for your existing MCP server that adds HTTP transport capability
    
    This class:
    1. Initializes your existing PerfectMCPServer
    2. Wraps it with HTTP transport
    3. Provides easy start/stop methods
    4. Handles server lifecycle
    """
    
    def __init__(self, host: str = "localhost", port: int = 3001):
        """
        Initialize the MCP server wrapper
        
        Args:
            host: Host to bind the HTTP server to
            port: Port to bind the HTTP server to
        """
        self.host = host
        self.port = port
        self.mcp_server = None
        self.http_transport = None
        self.is_running = False
        
        logger.info(f"MCP Server Wrapper initialized for {host}:{port}")
    
    async def initialize(self):
        """
        Initialize the MCP server and HTTP transport
        
        This method:
        1. Creates your PerfectMCPServer instance
        2. Sets up the HTTP transport layer
        3. Prepares everything for serving
        """
        try:
            logger.info("Initializing MCP server...")
            
            # Initialize your existing MCP server
            self.mcp_server = PerfectMCPServer()
            
            # Create HTTP transport wrapper
            self.http_transport = MCPHTTPTransport(
                mcp_server=self.mcp_server,
                host=self.host,
                port=self.port
            )
            
            logger.info("MCP server and HTTP transport initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    async def start(self):
        """
        Start the MCP server with HTTP transport
        
        This method starts the server and makes it available for your FastAPI app to call.
        """
        if not self.http_transport:
            await self.initialize()
        
        try:
            logger.info(f"Starting MCP server on {self.host}:{self.port}")
            self.is_running = True
            
            # Start the HTTP server
            await self.http_transport.start()
            
        except Exception as e:
            logger.error(f"Failed to start MCP server: {e}")
            self.is_running = False
            raise
    
    def run(self):
        """
        Run the server (blocking call)
        
        Use this method when you want to run the MCP server as a standalone service.
        """
        if not self.http_transport:
            asyncio.run(self.initialize())
        
        logger.info(f"Running MCP server on {self.host}:{self.port}")
        self.is_running = True
        
        try:
            self.http_transport.run()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal")
            self.is_running = False
        except Exception as e:
            logger.error(f"Server error: {e}")
            self.is_running = False
            raise
    
    async def stop(self):
        """
        Stop the MCP server
        """
        logger.info("Stopping MCP server...")
        self.is_running = False
        
        # Add any cleanup logic here if needed
        logger.info("MCP server stopped")
    
    def get_server_info(self):
        """
        Get information about the server
        
        Returns basic info that your FastAPI server can use.
        """
        return {
            "host": self.host,
            "port": self.port,
            "is_running": self.is_running,
            "server_url": f"http://{self.host}:{self.port}",
            "endpoints": {
                "health": f"http://{self.host}:{self.port}/health",
                "call_tool": f"http://{self.host}:{self.port}/mcp/call",
                "upload_process": f"http://{self.host}:{self.port}/mcp/upload-and-process",
                "list_tools": f"http://{self.host}:{self.port}/mcp/list-tools",
                "download": f"http://{self.host}:{self.port}/presentations/{{filename}}"
            }
        }

# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def create_mcp_server(host: str = "localhost", port: int = 3001) -> MCPServerWrapper:
    """
    Convenience function to create and initialize an MCP server
    
    Args:
        host: Host to bind to
        port: Port to bind to
    
    Returns:
        Initialized MCPServerWrapper instance
    """
    wrapper = MCPServerWrapper(host=host, port=port)
    await wrapper.initialize()
    return wrapper

def run_mcp_server(host: str = "localhost", port: int = 3001):
    """
    Convenience function to run MCP server as standalone service
    
    This is what you'll use when running the MCP server separately from your FastAPI app.
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    wrapper = MCPServerWrapper(host=host, port=port)
    wrapper.run()

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for running MCP server as standalone service
    
    You can run this file directly to start the MCP server:
    python mcp_services/core/server_wrapper.py
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Run MCP Research Server")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=3001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    logger.info(f"Starting MCP Research Server on {args.host}:{args.port}")
    
    try:
        run_mcp_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server failed: {e}")
        sys.exit(1) 