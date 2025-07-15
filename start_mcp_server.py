#!/usr/bin/env python3
"""
Start MCP Server Script
This script starts the MCP server as a standalone HTTP service
"""

import sys
import os
import logging
import argparse
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import MCP server wrapper
from mcp_services.core.server_wrapper import run_mcp_server

def main():
    """Main entry point for starting MCP server"""
    parser = argparse.ArgumentParser(
        description="Start MCP Research Server as standalone HTTP service"
    )
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=3001, 
        help="Port to bind to (default: 3001)"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    # Print startup info
    print("=" * 60)
    print("🚀 Starting MCP Research Server")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Debug: {args.debug}")
    print(f"Server URL: http://{args.host}:{args.port}")
    print("=" * 60)
    print()
    print("🔧 Core MCP Endpoints:")
    print(f"  📊 Health Check:     http://{args.host}:{args.port}/health")
    print(f"  🛠️  Tool Call:        http://{args.host}:{args.port}/mcp/call")
    print(f"  📤 Upload & Process: http://{args.host}:{args.port}/mcp/upload-and-process")
    print(f"  📋 List Tools:       http://{args.host}:{args.port}/mcp/list-tools")
    print(f"  📥 Downloads:        http://{args.host}:{args.port}/presentations/{{filename}}")
    print()
    print("🧠 Knowledge Base API (Intelligent):")
    print(f"  🔍 Smart Query:      http://{args.host}:{args.port}/kb/query")
    print(f"  📊 KB Statistics:    http://{args.host}:{args.port}/kb/stats") 
    print(f"  📚 Books Inventory:  http://{args.host}:{args.port}/kb/books")
    print(f"  ⚡ KB Health:        http://{args.host}:{args.port}/kb/health")
    print(f"  📝 Query Examples:   http://{args.host}:{args.port}/kb/examples")
    print(f"  🔧 Diagnostics:      http://{args.host}:{args.port}/kb/diagnostics")
    print()
    print("💾 Vector Databases:")
    
    # Import config to get actual index names
    try:
        from config import AdvancedConfig
        config = AdvancedConfig()
        research_index = config.PINECONE_INDEX_NAME or "all-pdfs-index"
        kb_index = config.PINECONE_KB_INDEX_NAME or "optimized-kb-index"
        print(f"  📄 Research Papers:  {research_index}")
        print(f"  📚 Knowledge Base:   {kb_index}")
    except Exception:
        # Fallback to default names if config fails
        print(f"  📄 Research Papers:  all-pdfs-index")
        print(f"  📚 Knowledge Base:   optimized-kb-index")
    
    print("=" * 60)
    print()
    print("Press Ctrl+C to stop the server")
    print()
    
    try:
        # Start the MCP server
        run_mcp_server(host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        print("🛑 Server stopped by user")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ Server failed: {e}")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main() 