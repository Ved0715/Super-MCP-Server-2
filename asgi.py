"""
ASGI entrypoint for running the MCP HTTP server with uvicorn (multi-worker).

Usage:
    uvicorn asgi:app --host 0.0.0.0 --port 3001 --workers 4

Environment variables (optional):
    MCP_HOST: host reported by the app (default: 0.0.0.0)
    MCP_PORT: port reported by the app (default: 3001)
"""

import os

from perfect_mcp_server import PerfectMCPServer
from mcp_services.transports.http_transport import MCPHTTPTransport


_host = os.getenv("MCP_HOST", "0.0.0.0")
_port = int(os.getenv("MCP_PORT", "3001"))

# Build the underlying MCP server and HTTP transport, then expose FastAPI app
_mcp_server = PerfectMCPServer()
_http_transport = MCPHTTPTransport(mcp_server=_mcp_server, host=_host, port=_port)

# This is the ASGI application uvicorn will import
app = _http_transport.app


