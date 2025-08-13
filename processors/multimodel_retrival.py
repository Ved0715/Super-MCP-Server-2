#!/usr/bin/env python3
"""
Server-Ready Multimodal Retrieval Function
Comprehensive function that reuses existing project components for server deployment
"""

import os
import json
import time
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import your existing project components
from .multimodal_integrator import MultimodalIntegrator
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServerMultimodalRetrieval:
    """Server-ready class for multimodal content retrieval using existing project components."""
    
    def __init__(self):
        """Initialize the retrieval system using existing MultimodalIntegrator."""
        try:
            self.multimodal_integrator = MultimodalIntegrator()
            logger.info("✅ Server multimodal retrieval system initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize multimodal integrator: {e}")
            raise
    
    async def search_multimodal_content(self, 
                                      query: str, 
                                      paper_id: str,
                                      max_images: int = 6,        # Increased from 3 to 6 for smaller chunks
                                      max_tables: int = 4,        # Increased from 3 to 4 for smaller chunks  
                                      max_text_chunks: int = 25) -> Dict[str, Any]:  # Increased from 8 to 25 for smaller chunks
        """
        Search for multimodal content (images, tables, text) in a processed paper.
        
        Args:
            query: Search query for finding relevant multimodal content
            paper_id: ID of the processed paper to search within
            max_images: Maximum number of images to return (default: 6, updated for smaller chunks)
            max_tables: Maximum number of tables to return (default: 4, updated for smaller chunks)
            max_text_chunks: Maximum number of text chunks to return (default: 25, updated for smaller chunks)
            
        Returns:
            Dict containing multimodal search results with inline elements
        """
        try:
            logger.info(f"🔍 Searching multimodal content for query: '{query}' in paper: {paper_id}")
            
            # Calculate total chunks needed (images + tables + text)
            total_chunks = max_images + max_tables + max_text_chunks
            
            # Call the multimodal integrator to get results
            results = self.multimodal_integrator.query_multimodal_content(
                query=query,
                document_uuid=paper_id,
                max_chunks=total_chunks
            )
            
            # Process and filter results by content type
            if results and 'inline_elements' in results:
                inline_elements = results['inline_elements']
                
                # Separate elements by type
                images = []
                tables = []
                text_chunks = []
                
                for element in inline_elements:
                    element_type = element.get('type', '').lower()
                    
                    if element_type == 'image' and len(images) < max_images:
                        images.append(element)
                    elif element_type == 'table' and len(tables) < max_tables:
                        tables.append(element)
                    elif element_type in ['text', 'composite'] and len(text_chunks) < max_text_chunks:
                        text_chunks.append(element)
                
                # Rebuild filtered inline elements
                filtered_elements = images + tables + text_chunks
                
                # Update results with filtered elements
                results['inline_elements'] = filtered_elements
                results['element_counts'] = {
                    'images': len(images),
                    'tables': len(tables),
                    'text_chunks': len(text_chunks),
                    'total': len(filtered_elements)
                }
            
            # Add metadata about the search
            enhanced_results = {
                "search_metadata": {
                    "query": query,
                    "paper_id": paper_id,
                    "max_images": max_images,
                    "max_tables": max_tables,
                    "max_text_chunks": max_text_chunks,
                    "timestamp": time.time()
                },
                "multimodal_results": results,
                "success": True
            }
            
            element_count = len(results.get('inline_elements', [])) if results else 0
            logger.info(f"✅ Found multimodal content with {element_count} elements")
            return enhanced_results
            
        except Exception as e:
            logger.error(f"❌ Error in multimodal search: {str(e)}")
            return {
                "search_metadata": {
                    "query": query,
                    "paper_id": paper_id,
                    "error": str(e)
                },
                "multimodal_results": None,
                "success": False,
                "error": str(e)
            }
    
    def process_pdf_for_server(self, 
                              pdf_path: str, 
                              original_filename: Optional[str] = None,
                              force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process a PDF file for server use (wrapper around existing functionality).
        
        Args:
            pdf_path: Path to the PDF file
            original_filename: Original filename if different from path
            force_reprocess: Whether to force reprocessing even if already processed
            
        Returns:
            Dict containing processing results with pdf_id
        """
        try:
            logger.info(f"📄 Processing PDF for server: {original_filename or pdf_path}")
            
            # Use existing multimodal integrator processing
            result = self.multimodal_integrator.process_pdf_complete(
                pdf_path=pdf_path,
                force_reprocess=force_reprocess,
                original_filename=original_filename
            )
            
            if result and 'pdf_id' in result:
                logger.info(f"✅ PDF processed successfully with ID: {result['pdf_id']}")
                return {
                    "success": True,
                    "pdf_id": result['pdf_id'],
                    "processing_results": result,
                    "timestamp": time.time()
                }
            else:
                logger.error("❌ PDF processing failed - no pdf_id returned")
                return {
                    "success": False,
                    "error": "PDF processing failed - no pdf_id returned",
                    "timestamp": time.time()
                }
                
        except Exception as e:
            logger.error(f"❌ Error processing PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "timestamp": time.time()
            }
    
    def get_paper_info(self, paper_id: str) -> Dict[str, Any]:
        """
        Get information about a processed paper.
        
        Args:
            paper_id: ID of the processed paper
            
        Returns:
            Dict containing paper information
        """
        try:
            # Query pinecone to get paper info
            # This is a basic info query - you can enhance it based on your needs
            sample_results = self.multimodal_integrator.index.query(
                vector=[0] * 3072,  # Dummy vector for metadata query
                top_k=1,
                namespace=paper_id,
                include_metadata=True,
                include_values=False
            )
            
            if sample_results.matches:
                metadata = sample_results.matches[0].metadata
                return {
                    "success": True,
                    "paper_id": paper_id,
                    "info": {
                        "original_filename": metadata.get('original_filename', 'Unknown'),
                        "processing_date": metadata.get('created_at', 'Unknown'),
                        "content_type": metadata.get('content_type', 'Unknown')
                    }
                }
            else:
                return {
                    "success": False,
                    "error": f"Paper with ID '{paper_id}' not found"
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting paper info: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_processed_papers(self) -> Dict[str, Any]:
        """
        List all processed papers (namespaces) in the system.
        
        Returns:
            Dict containing list of processed papers
        """
        try:
            # Get index stats to see all namespaces
            stats = self.multimodal_integrator.index.describe_index_stats()
            namespaces = list(stats.namespaces.keys()) if stats.namespaces else []
            
            return {
                "success": True,
                "processed_papers": namespaces,
                "total_count": len(namespaces),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Error listing processed papers: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

# Example usage function for your server controller
async def server_search_example(query: str, paper_id: str) -> Dict[str, Any]:
    """
    Example function showing how to use the retrieval system in your server.
    Copy this pattern into your controller.
    """
    # Initialize the retrieval system (do this once in your server startup)
    retrieval_system = ServerMultimodalRetrieval()
    
    # Perform the search with updated defaults for smaller chunks
    results = await retrieval_system.search_multimodal_content(
        query=query,
        paper_id=paper_id,
        max_images=6,        # Updated from 3 to 6
        max_tables=4,        # Updated from 3 to 4
        max_text_chunks=25   # Updated from 8 to 25
    )
    
    return results

# For server integration, you'll want to create a singleton instance
_retrieval_instance = None

def get_retrieval_instance() -> ServerMultimodalRetrieval:
    """Get singleton instance of the retrieval system for server use."""
    global _retrieval_instance
    if _retrieval_instance is None:
        _retrieval_instance = ServerMultimodalRetrieval()
    return _retrieval_instance