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
                
                # Collect all elements by type first
                all_images = []
                all_tables = []
                all_text_chunks = []
                
                for element in inline_elements:
                    element_type = element.get('type', '').lower()
                    
                    if element_type == 'image':
                        all_images.append(element)
                    elif element_type == 'table':
                        all_tables.append(element)
                    elif element_type in ['text', 'composite']:
                        all_text_chunks.append(element)
                
                # Apply intelligent selection for images (existing behavior)
                images = all_images[:max_images]
                
                # Apply intelligent table selection with quality filtering
                tables = self._select_best_tables(all_tables, query, max_tables)
                
                # Apply basic selection for text chunks
                text_chunks = all_text_chunks[:max_text_chunks]
                
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
    
    def _select_best_tables(self, all_tables: list, query: str, max_tables: int) -> list:
        """Intelligently select the best tables based on content quality and relevance."""
        if not all_tables:
            return []
        
        if len(all_tables) <= max_tables:
            return all_tables
        
        # Score each table for intelligent selection
        scored_tables = []
        query_lower = query.lower()
        query_keywords = set(query_lower.split())
        
        for table in all_tables:
            score = 0.0
            
            # Base relevance score from table structure (not nested under 'data')
            base_relevance = table.get('relevance_score', 0.0)
            score += base_relevance * 0.4
            
            # Table content quality assessment - access directly from table, not nested
            table_content_json = table.get('table_content_json', '')
            if table_content_json:
                try:
                    import json
                    table_content = json.loads(table_content_json)
                    metadata = table_content.get('_metadata', {})
                    
                    # Data richness scoring
                    total_rows = metadata.get('total_rows', 0)
                    total_columns = metadata.get('total_columns', 0)
                    
                    # Boost for larger, data-rich tables
                    if total_rows >= 4 and total_columns >= 3:
                        score += 0.3
                    elif total_rows >= 3 and total_columns >= 2:
                        score += 0.2
                    elif total_rows >= 2:
                        score += 0.1
                    
                    # Content density scoring
                    table_keys = [k for k in table_content.keys() if not k.startswith('_')]
                    non_empty_columns = 0
                    total_data_points = 0
                    
                    for key in table_keys:
                        column_data = table_content.get(key, [])
                        if isinstance(column_data, list):
                            non_empty_cells = sum(1 for cell in column_data if str(cell).strip() and len(str(cell).strip()) > 1)
                            total_data_points += len(column_data)
                            if non_empty_cells >= 2:
                                non_empty_columns += 1
                    
                    # Boost for content density
                    if non_empty_columns >= 3:
                        score += 0.25
                    elif non_empty_columns >= 2:
                        score += 0.15
                    elif non_empty_columns >= 1:
                        score += 0.08
                    
                except (json.JSONDecodeError, TypeError, KeyError):
                    # Fallback: minimal boost for having table content
                    score += 0.05
            
            # Summary relevance scoring using backend-generated summaries
            table_id = table.get('table_id', '')
            # Use enhanced table summary extraction for relevance scoring
            summary_text = self._get_table_summary_for_scoring(table).lower()
            
            if summary_text and len(summary_text) > 5:
                # Keyword matching in summary
                keyword_matches = sum(1 for kw in query_keywords if kw in summary_text)
                if keyword_matches > 0:
                    score += (keyword_matches / len(query_keywords)) * 0.2
                
                # Data-related keyword boosts
                data_keywords = ['data', 'statistics', 'results', 'analysis', 'comparison', 'values', 'measurements']
                data_matches = sum(1 for kw in data_keywords if kw in summary_text)
                if data_matches > 0 and any(kw in query_lower for kw in data_keywords):
                    score += min(data_matches * 0.05, 0.15)
                
                # Penalty for exercise/worksheet tables
                exercise_indicators = ['exercise', 'practice', 'worksheet', 'assignment', 'homework', 'question']
                if any(indicator in summary_text for indicator in exercise_indicators):
                    score *= 0.6
            
            # Page context scoring (prefer tables from relevant pages)
            page_number = table.get('page_number', 0)
            if page_number:
                # Slight boost for tables (assuming they're already contextually retrieved)
                score += 0.05
            
            scored_tables.append((score, table))
        
        # Sort by score (descending) and select top tables
        scored_tables.sort(key=lambda x: x[0], reverse=True)
        selected_tables = [table for score, table in scored_tables[:max_tables]]
        
        # Log selection for debugging
        if len(all_tables) > max_tables:
            selected_scores = [score for score, table in scored_tables[:max_tables]]
            logger.info(f"Selected {len(selected_tables)} best tables from {len(all_tables)} available. Top scores: {selected_scores[:3]}")
        
        return selected_tables
    
    def _get_table_summary_for_scoring(self, table: dict) -> str:
        """Extract table summary for relevance scoring."""
        # Try different possible summary fields in order of preference
        summary_fields = ['summary', 'table_summary', 'description', 'title']
        
        for field in summary_fields:
            if field in table and table[field]:
                summary = str(table[field]).strip()
                if len(summary) > 5:
                    return summary
        
        # Fallback: try to extract from table content if available
        table_content_json = table.get('table_content_json', '')
        if table_content_json:
            try:
                import json
                table_data = json.loads(table_content_json)
                metadata = table_data.get('_metadata', {})
                rows = metadata.get('total_rows', 0)
                cols = metadata.get('total_columns', 0)
                if rows > 0 and cols > 0:
                    return f"Data table with {rows} rows and {cols} columns"
            except (json.JSONDecodeError, TypeError, KeyError):
                pass
        
        return "structured data"

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