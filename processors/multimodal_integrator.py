#!/usr/bin/env python3
"""
Multimodal Integrator for LlamaParse Data
Main integration system that coordinates text, image, and table processing with unified namespace storage
"""

import os
import json
import hashlib
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import re
import logging
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import zipfile
import io

# Third-party imports
import pinecone
from pinecone import Pinecone
import openai
from openai import OpenAI
import streamlit as st
from llama_cloud_services import LlamaParse

# Local imports
from config import *
from processors.image_processor import ImageProcessor
from processors.table_processor import TableProcessor
from processors.inline_multimodal_generator import InlineMultimodalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalIntegrator:
    """Main integration system for processing and storing multimodal PDF content."""
    
    def __init__(self):
        """Initialize the multimodal integrator."""
        self.openai_client = None
        self.pinecone_client = None
        self.index = None
        self.llamaparse_parser = None
        
        # Specialized processors
        self.image_processor = None
        self.table_processor = None
        self.inline_generator = None
        
        # Settings
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
        self.chat_model = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        self.max_workers = 4
        self.batch_size = 50
        
        # Initialize all components
        self._initialize_clients()
        self._initialize_processors()
    
    def _initialize_clients(self):
        """Initialize OpenAI, Pinecone, and LlamaParse clients."""
        try:
            # OpenAI initialization
            openai_api_key = os.getenv('OPENAI_API_KEY')
            if not openai_api_key:
                raise ValueError("OPENAI_API_KEY not found")
            
            self.openai_client = OpenAI(api_key=openai_api_key)
            
            # Pinecone initialization
            pinecone_api_key = os.getenv('PINECONE_API_KEY')
            if not pinecone_api_key:
                raise ValueError("PINECONE_API_KEY not found")

            self.DEFAULT_TIMEOUT = 60000
            
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            
            index_name = os.getenv('PINECONE_INDEX_NAME_TEST')
            if not index_name:
                raise ValueError("PINECONE_INDEX_NAME_TEST not found")
            
            self.index = self.pinecone_client.Index(index_name)
            
            # LlamaParse initialization (optional)
            llamaparse_api_key = os.getenv('LLAMA_PARSE_API_KEY')
            if llamaparse_api_key:
                try:
                    self.llamaparse_parser = LlamaParse(
                        api_key=llamaparse_api_key,
                        result_type="markdown",
                        system_prompt="Extract all text content, images, and tables with complete metadata. Preserve document structure and formatting.",
                        max_timeout=self.DEFAULT_TIMEOUT,
                        split_by_page=True,
                        verbose=True
                    )
                    logger.info("✅ LlamaParse initialized successfully")
                except Exception as e:
                    logger.warning(f"⚠️ LlamaParse initialization failed: {e}")
                    self.llamaparse_parser = None
            else:
                logger.warning("⚠️ LLAMA_PARSE_API_KEY not found - LlamaParse will not be available")
                self.llamaparse_parser = None
                logger.warning("⚠️  LLAMA PARSE_API_KEY not found - document processing disabled, queries on existing documents will still work")
            
            logger.info("All clients initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing clients: {e}")
            raise
    
    def _initialize_processors(self):
        """Initialize specialized processors."""
        try:
            self.image_processor = ImageProcessor(
                self.openai_client, 
                self.pinecone_client, 
                self.index
            )  
            self.table_processor = TableProcessor(
                self.openai_client, 
                self.pinecone_client, 
                self.index
            )
            self.inline_generator = InlineMultimodalGenerator(
                self.openai_client, 
                self.chat_model
            )
            logger.info("Specialized processors initialized")
            
        except Exception as e:
            logger.error(f"Error initializing processors: {e}")
            raise
    
    def process_pdf_complete(self, pdf_path: str, force_reprocess: bool = False, original_filename: str = None) -> Dict[str, Any]:
        """Complete PDF processing with unified namespace storage."""
        try:
            start_time = time.time()
            
            # Use original filename if provided, otherwise extract from path
            document_name = original_filename if original_filename else Path(pdf_path).name
            
            # Generate document UUID and namespace using the original filename
            document_uuid = self._generate_pdf_id_from_name(document_name)
            
            # Check if already processed
            if not force_reprocess and self._is_pdf_processed(document_uuid, document_name):
                st.info(f"📋 PDF already processed: {document_name}")
                return {
                    'document_uuid': document_uuid,
                    'cached': True,
                    'processing_time': 0
                }
            
            st.info(f"🚀 Starting complete multimodal processing: {document_name}")
            
            with st.spinner("Parsing PDF with LlamaParse..."):
                # Parse PDF with LlamaParse and capture job ID
                if self.llamaparse_parser is None:
                    raise ValueError("LlamaParse is not available. Please set LLAMAPARSE_API_KEY environment variable.")
                
                json_objs = self.llamaparse_parser.get_json_result(pdf_path)
                
                if not json_objs:
                    raise ValueError("No content extracted from PDF")
                
                # Extract images from LlamaParse results using proven methods
                logger.info(f"Extracting images from LlamaParse JSON results using proven extraction")
                extracted_count = self._extract_images_from_results(json_objs, document_uuid, document_name)
                logger.info(f"Enhanced image extraction completed: {extracted_count} images processed")
                
                logger.info(f"LlamaParse extraction complete for {document_name}")
            
            # Process all content types in parallel
            with st.spinner("Processing multimodal content..."):
                processing_results = self._process_multimodal_content_parallel(
                    json_objs, document_name, document_uuid
                )
            
            # Create composite chunks combining all content types
            with st.spinner("Creating composite chunks..."):
                composite_results = self._process_composite_chunks(
                    json_objs, document_name, document_uuid, 
                    processing_results.get('images', {}), 
                    processing_results.get('tables', {})
                )
                processing_results['composite'] = composite_results
            
            # Display comprehensive results
            self._display_processing_results(processing_results, time.time() - start_time)
            
            return {
                'document_uuid': document_uuid,
                'cached': False,
                'processing_time': time.time() - start_time,
                'results': processing_results
            }
            
        except Exception as e:
            logger.error(f"Error in complete PDF processing: {e}")
            st.error(f"❌ Error processing PDF: {e}")
            raise
    
    def _process_multimodal_content_parallel(self, json_objs: List[Dict], 
                                           pdf_name: str, pdf_id: str) -> Dict[str, Any]:
        """Process content and create composite chunks with unified embeddings."""
        results = {}
        
        # Extract all content types first (skip individual Pinecone storage for composite chunks)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit processing tasks with skip_pinecone flag
            futures = {
                executor.submit(
                    self.image_processor.process_images_from_llamaparse, 
                    json_objs, pdf_name, pdf_id, skip_pinecone=True
                ): 'images',
                executor.submit(
                    self.table_processor.process_tables_from_llamaparse, 
                    json_objs, pdf_name, pdf_id, skip_pinecone=True
                ): 'tables'
            }
            
            # Collect results as they complete
            for future in as_completed(futures):
                content_type = futures[future]
                try:
                    results[content_type] = future.result()
                    logger.info(f"{content_type.title()} processing completed successfully")
                except Exception as e:
                    logger.error(f"Error processing {content_type}: {e}")
                    results[content_type] = {'error': str(e)}
        
        return results
    
    def _process_composite_chunks(self, json_objs: List[Dict], document_name: str, document_uuid: str, 
                                images_data: Dict, tables_data: Dict) -> Dict[str, Any]:
        """Create composite chunks combining text, images, and tables with unified embeddings."""
        try:
            if not json_objs:
                return {'processed_chunks': 0, 'pinecone_stored': 0}
            
            json_data = json_objs[0]
            pages = json_data.get('pages', [])
            
            # Extract tables directly from LlamaParse JSON structure
            tables_by_page = self._extract_tables_from_llamaparse_json(json_objs, document_uuid)
            
            # Organize images by page
            images_by_page = self._organize_content_by_page(images_data.get('processed_images', []))
            
            composite_chunks = []
            
            for page_data in pages:
                page_num = page_data.get('page', 0)
                
                # Extract text content
                text = page_data.get('text', '') or page_data.get('md', '')
                
                # Debug text content length
                text_length = len(text.strip()) if text else 0
                logger.info(f"Page {page_num}: text content length = {text_length}")
                
                if text and len(text.strip()) > 10:  # Lowered from 50 to 10
                    # Get associated images and tables for this page
                    page_images = images_by_page.get(page_num, [])
                    page_tables = tables_by_page.get(page_num, [])
                    
                    logger.info(f"Page {page_num}: {len(page_images)} images, {len(page_tables)} tables")
                    
                    # Create composite chunk
                    composite_chunk = self._create_composite_chunk(
                        text, page_images, page_tables, 
                        document_uuid, document_name, page_num
                    )
                    
                    if composite_chunk:
                        composite_chunks.append(composite_chunk)
                else:
                    logger.warning(f"Page {page_num}: Skipped due to insufficient text content ({text_length} chars)")
            
            # Store composite chunks in Pinecone
            stored_count = self._store_composite_chunks_in_pinecone(composite_chunks, document_uuid)
            
            logger.info(f"Composite chunk processing complete: {len(composite_chunks)} chunks, {stored_count} stored")
            
            return {
                'processed_chunks': len(composite_chunks),
                'pinecone_stored': stored_count,
                'composite_chunks': composite_chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing composite chunks: {e}")
            return {'error': str(e)}
    
    def _organize_content_by_page(self, content_list: List[Dict]) -> Dict[int, List[Dict]]:
        """Organize content items by page number."""
        by_page = {}
        for item in content_list:
            page_num = item.get('page_number', 0)
            if page_num not in by_page:
                by_page[page_num] = []
            by_page[page_num].append(item)
        return by_page
    
    def _extract_tables_from_llamaparse_json(self, json_objs: List[Dict], pdf_id: str) -> Dict[int, List[Dict]]:
        """Extract tables directly from LlamaParse JSON structure: pages->items->type:'table'."""
        tables_by_page = {}
        
        try:
            if not json_objs:
                return tables_by_page
            
            json_data = json_objs[0]  # LlamaParse returns single JSON object
            pages = json_data.get('pages', [])
            
            for page_data in pages:
                page_num = page_data.get('page', 0)
                items = page_data.get('items', [])
                
                page_tables = []
                table_count = 0
                
                for item in items:
                    if item.get('type') == 'table':
                        table_count += 1
                        
                        # Create structured table JSON from LlamaParse data
                        table_json = self._create_structured_table_json(item, page_num, table_count, pdf_id)
                        
                        if table_json:
                            page_tables.append({
                                'table_id': f"{pdf_id}_p{page_num}_table{table_count}",
                                'page_number': page_num,
                                'table_index': table_count,
                                'table_content_json': table_json,
                                'table_summary': self._extract_table_summary_from_llamaparse(item),
                                'raw_llamaparse_data': item
                            })
                
                if page_tables:
                    tables_by_page[page_num] = page_tables
                    logger.info(f"Extracted {len(page_tables)} tables from page {page_num}")
            
            return tables_by_page
            
        except Exception as e:
            logger.error(f"Error extracting tables from LlamaParse JSON: {e}")
            return tables_by_page
    
    def _create_structured_table_json(self, table_item: Dict, page_num: int, table_count: int, pdf_id: str) -> str:
        """Create structured table JSON from LlamaParse table item."""
        try:
            # Extract rows from LlamaParse table item
            rows = table_item.get('rows', [])
            if not rows or len(rows) < 2:
                logger.warning(f"Table on page {page_num} has insufficient data")
                return ""
            
            # Create structured table with headers as keys
            table_dict = {}
            
            # Use first row as headers
            headers = rows[0]
            data_rows = rows[1:] if len(rows) > 1 else []
            
            # Create columns with headers as keys
            for i, header in enumerate(headers):
                if header:  # Skip empty headers
                    column_data = []
                    for row in data_rows:
                        if i < len(row):
                            cell_value = row[i] if row[i] else ""
                            column_data.append(str(cell_value).strip())
                        else:
                            column_data.append("")
                    table_dict[str(header).strip()] = column_data
            
            # Add metadata
            table_dict["_metadata"] = {
                "total_rows": len(rows),
                "total_columns": len(headers),
                "has_headers": True,
                "source": "llamaparse_direct",
                "page_number": page_num,
                "table_id": f"{pdf_id}_p{page_num}_table{table_count}"
            }
            
            return json.dumps(table_dict)
            
        except Exception as e:
            logger.error(f"Error creating structured table JSON: {e}")
            return ""
    
    def _extract_table_summary_from_llamaparse(self, table_item: Dict) -> str:
        """Generate intelligent table summary using LLM."""
        try:
            # Extract table data for LLM processing
            rows = table_item.get('rows', [])
            if not rows or len(rows) < 2:
                return "Table with insufficient data"
            
            # Create table content for LLM
            table_content = self._create_table_content_for_llm(rows)
            
            # Generate summary using LLM
            summary = self._generate_table_summary_with_llm(table_content)
            
            return summary
            
        except Exception as e:
            logger.warning(f"Error generating table summary with LLM: {e}")
            # Fallback to basic summary
            rows = table_item.get('rows', [])
            if rows:
                row_count = len(rows)
                col_count = len(rows[0]) if rows else 0
                return f"Table with {row_count} rows and {col_count} columns"
            return "Table data"
    
    def _create_table_content_for_llm(self, rows: List[List]) -> str:
        """Create formatted table content for LLM processing."""
        try:
            if not rows:
                return ""
            
            # Create markdown table format
            table_lines = []
            
            # Add headers
            headers = rows[0]
            table_lines.append("| " + " | ".join(str(h) for h in headers) + " |")
            table_lines.append("|" + "|".join(["---"] * len(headers)) + "|")
            
            # Add data rows
            for row in rows[1:]:
                table_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
            
            return "\n".join(table_lines)
            
        except Exception as e:
            logger.error(f"Error creating table content for LLM: {e}")
            return ""
    
    def _generate_table_summary_with_llm(self, table_content: str) -> str:
        """Generate intelligent table summary using OpenAI LLM."""
        try:
            if not table_content or len(table_content.strip()) < 10:
                return "Table with minimal content"
            
            # Create prompt for table summary
            prompt = f"""Please provide a concise, intelligent summary of this table data. Focus on the key insights, patterns, and what the table represents.

Table:
{table_content}

Summary (2-3 sentences, focus on main insights):"""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing and summarizing table data. Provide clear, concise summaries that highlight the key information and insights."}, 
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Clean up summary
            if summary and len(summary) > 10:
                return summary
            else:
                return "Table data summary"
                
        except Exception as e:
            logger.warning(f"Error calling LLM for table summary: {e}")
            return "Table data summary"
    
    def _create_composite_chunk(self, text: str, page_images: List[Dict], 
                               page_tables: List[Dict], document_uuid: str, document_name: str, 
                               page_num: int) -> Optional[Dict]:
        """Create composite chunk with unified text embedding combining text + image OCR + table summaries. 
        
        Uses arrays for multiple images and tables per chunk.
        Implements comprehensive error handling with fallbacks.
        """
        try:
            # Generate unique chunk ID
            chunk_id = f"{document_uuid}_p{page_num}_c{int(time.time() * 1000) % 100000}"
            
            # Start building composite text with main content
            composite_text_parts = [text.strip()]
            
            # Initialize metadata with new array-based structure
            chunk_metadata = {
                "chunk_id": chunk_id,
                "document_name": document_name,
                "document_uuid": document_uuid,
                "page_number": page_num,
                "contains_image": False,
                "contains_table": False,
                "has_text": bool(text.strip()),
                # Arrays for multiple images
                "image_s3_urls": [],
                "image_ids": [],
                "image_summaries": [],
                "image_count": 0,
                # Arrays for multiple tables  
                "table_content_jsons": [],
                "table_ids": [],
                "table_summaries": [],
                "table_count": 0
            }
            
            # Process images with comprehensive error handling
            if page_images:
                try:
                    image_summaries_for_text = []
                    image_keywords = []
                    
                    for img_data in page_images:  # Process all images
                        try:
                            # Extract image summary with fallbacks
                            img_summary = self._extract_image_summary_with_fallbacks(img_data)
                            if img_summary:
                                image_summaries_for_text.append(img_summary)
                                chunk_metadata["image_summaries"].append(img_summary)
                            
                            # Extract keywords with fallbacks
                            img_keywords = self._extract_image_keywords_with_fallbacks(img_data)
                            if img_keywords:
                                image_keywords.extend(img_keywords)
                            
                            # Add image data to arrays
                            img_url = img_data.get('s3_url', '') or img_data.get('local_path', '')
                            img_id = img_data.get('image_id', '')
                            
                            if img_url:
                                chunk_metadata["image_s3_urls"].append(img_url)
                            if img_id:
                                chunk_metadata["image_ids"].append(img_id)
                                
                        except Exception as e:
                            logger.warning(f"Error processing image in chunk {chunk_id}: {e}")
                            continue
                    
                    # Update metadata flags and counts
                    if chunk_metadata["image_s3_urls"]:
                        chunk_metadata["contains_image"] = True
                        chunk_metadata["image_count"] = len(chunk_metadata["image_s3_urls"])
                    
                    # Add image information to composite text using exact format
                    if image_summaries_for_text:
                        composite_text_parts.append(f"[Image OCR: {' '.join(image_summaries_for_text)}]")
                    
                    if image_keywords:
                        # Remove duplicates and limit keywords
                        unique_keywords = list(set(image_keywords))[:10]
                        composite_text_parts.append(f"[Image Keywords: {', '.join(unique_keywords)}]")
                        
                except Exception as e:
                    logger.error(f"Error processing images for chunk {chunk_id}: {e}")
                    # Continue without image data - graceful degradation
            
            # Process tables with comprehensive error handling
            if page_tables:
                try:
                    table_summaries_for_text = []
                    
                    for table_data in page_tables:  # Process all tables
                        try:
                            # Use pre-extracted table data from LlamaParse
                            table_summary = table_data.get('table_summary', '')
                            table_json = table_data.get('table_content_json', '')
                            table_id = table_data.get('table_id', '')
                            
                            if table_summary:
                                table_summaries_for_text.append(table_summary)
                                chunk_metadata["table_summaries"].append(table_summary)
                            
                            if table_json:
                                chunk_metadata["table_content_jsons"].append(table_json)
                                
                            if table_id:
                                chunk_metadata["table_ids"].append(table_id)
                                
                            # Log table extraction for debugging
                            logger.info(f"Added table to chunk {chunk_id}: {table_id} with {len(table_json)} chars JSON")
                                
                        except Exception as e:
                            logger.warning(f"Error processing table in chunk {chunk_id}: {e}")
                            continue
                    
                    # Update metadata flags and counts
                    if chunk_metadata["table_content_jsons"]:
                        chunk_metadata["contains_table"] = True
                        chunk_metadata["table_count"] = len(chunk_metadata["table_content_jsons"])
                    
                    # Add table information to composite text using exact format
                    if table_summaries_for_text:
                        composite_text_parts.append(f"[Table Summary: {' | '.join(table_summaries_for_text)}]")
                        
                except Exception as e:
                    logger.error(f"Error processing tables for chunk {chunk_id}: {e}")
                    # Continue without table data - graceful degradation
            
            # Create final composite text
            composite_text = ' '.join(composite_text_parts)
            
            # Debug logging for composite text creation
            logger.info(f"Composite text for chunk {chunk_id}: {len(composite_text)} chars")
            logger.info(f"Text content length: {len(text.strip())}")
            logger.info(f"Composite text parts: {len(composite_text_parts)} parts")
            if len(composite_text.strip()) < 50:  # Show details for short text
                logger.info(f"Short composite text: '{composite_text[:100]}...'" )
            
            # Validate composite text meets minimum requirements (lowered threshold)
            if len(composite_text.strip()) < 10:  # Lowered from 20 to 10
                logger.warning(f"Composite text too short for chunk {chunk_id}, skipping")
                return None
            
            # Add composite text and timestamp to metadata
            chunk_metadata["composite_text"] = composite_text
            chunk_metadata["processed_timestamp"] = datetime.now().isoformat()
            
            logger.info(f"Created composite chunk {chunk_id}: {len(composite_text)} chars, "
                       f"images: {chunk_metadata['contains_image']}, tables: {chunk_metadata['contains_table']}")
            
            return chunk_metadata
            
        except Exception as e:
            logger.error(f"Critical error creating composite chunk for page {page_num}: {e}")
            # Return basic fallback chunk with just text
            try:
                return {
                    "chunk_id": f"{document_uuid}_p{page_num}_fallback",
                    "document_name": document_name,
                    "document_uuid": document_uuid,
                    "page_number": page_num,
                    "composite_text": text.strip(),
                    "contains_image": False,
                    "contains_table": False,
                    "has_text": bool(text.strip()),
                    # Empty arrays for fallback
                    "image_s3_urls": [],
                    "image_ids": [],
                    "image_summaries": [],
                    "image_count": 0,
                    "table_content_jsons": [],
                    "table_ids": [],
                    "table_summaries": [],
                    "table_count": 0,
                    "processed_timestamp": datetime.now().isoformat(),
                    "fallback_chunk": True
                }
            except Exception as fallback_error:
                logger.error(f"Even fallback chunk creation failed: {fallback_error}")
                return None
    
    def _extract_image_summary_with_fallbacks(self, img_data: Dict) -> str:
        """Generate intelligent image summary using LLM with OCR text and keywords."""
        try:
            # Extract OCR text and keywords for LLM processing
            ocr_text = ""
            keywords = []
            
            # Get OCR text
            for field in ['ocr_text', 'extracted_text', 'text']:
                if field in img_data and img_data[field]:
                    ocr_text = str(img_data[field]).strip()
                    break
            
            # Get keywords
            if 'keywords' in img_data and img_data['keywords']:
                if isinstance(img_data['keywords'], list):
                    keywords = img_data['keywords']
                elif isinstance(img_data['keywords'], str):
                    keywords = img_data['keywords'].split(',')
            
            # Generate summary using LLM
            if ocr_text or keywords:
                summary = self._generate_image_summary_with_llm(ocr_text, keywords, img_data)
                return summary
            
            # Fallback: Generate basic summary from metadata
            img_type = img_data.get('type', 'image')
            page_num = img_data.get('page_number', 0)
            return f"Visual content from page {page_num} ({img_type})"
            
        except Exception as e:
            logger.warning(f"Error generating image summary with LLM: {e}")
            # Fallback to basic summary
            img_type = img_data.get('type', 'image')
            page_num = img_data.get('page_number', 0)
            return f"Visual content from page {page_num} ({img_type})"
    
    def _generate_image_summary_with_llm(self, ocr_text: str, keywords: List[str], img_data: Dict) -> str:
        """Generate intelligent image summary using OpenAI LLM."""
        try:
            # Prepare content for LLM
            content_parts = []
            
            if ocr_text and len(ocr_text.strip()) > 10:
                content_parts.append(f"OCR Text: {ocr_text}")
            
            if keywords:
                content_parts.append(f"Keywords: {', '.join(keywords)}")
            
            if not content_parts:
                return "Image with minimal text content"
            
            # Add image metadata
            img_type = img_data.get('type', 'image')
            page_num = img_data.get('page_number', 0)
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            content_parts.append(f"Image Type: {img_type}")
            content_parts.append(f"Page: {page_num}")
            
            # Create prompt for image summary
            content = "\n".join(content_parts)
            prompt = f"""Please provide a concise, intelligent summary of this image based on its OCR text and keywords. Focus on what the image represents, its key elements, and any important information it conveys.

Image Content:
{content}

Summary (2-3 sentences, focus on main content and purpose):"""

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing and summarizing image content based on OCR text and keywords. Provide clear, concise summaries that highlight the key information and purpose of the image."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.3
            )
            
            summary = response.choices[0].message.content.strip()
            
            # Clean up summary
            if summary and len(summary) > 10:
                return summary
            else:
                return "Image with text content"
                
        except Exception as e:
            logger.warning(f"Error calling LLM for image summary: {e}")
            # Fallback to OCR text or basic summary
            if ocr_text and len(ocr_text.strip()) > 10:
                return f"OCR: {ocr_text[:200]}"
            return "Image with text content"
    
    def _extract_image_keywords_with_fallbacks(self, img_data: Dict) -> List[str]:
        """Extract image keywords with multiple fallback methods."""
        try:
            keywords = []
            
            # Method 1: Use existing keywords
            if 'keywords' in img_data and img_data['keywords']:
                if isinstance(img_data['keywords'], list):
                    keywords.extend(img_data['keywords'])
                elif isinstance(img_data['keywords'], str):
                    keywords.extend(img_data['keywords'].split(','))
            
            # Method 2: Extract from OCR text
            ocr_text = img_data.get('ocr_text', '') or img_data.get('extracted_text', '')
            if ocr_text and len(ocr_text) > 10:
                # Simple keyword extraction - get important words
                words = ocr_text.split()
                important_words = [w.strip('.,!?;:') for w in words 
                                 if len(w) > 3 and w.isalpha()][:5]
                keywords.extend(important_words)
            
            # Method 3: Use image type/metadata
            if img_data.get('type'):
                keywords.append(img_data['type'])
            
            return keywords[:10]  # Limit to 10 keywords
            
        except Exception as e:
            logger.warning(f"Error extracting image keywords: {e}")
            return []
    
    def _extract_table_summary_with_fallbacks(self, table_data: Dict) -> str:
        """Extract table summary with multiple fallback methods."""
        try:
            # Method 1: Use existing summary
            if 'summary' in table_data and table_data['summary']:
                return str(table_data['summary']).strip()
            
            # Method 2: Use table description
            for field in ['description', 'caption', 'title']:
                if field in table_data and table_data[field]:
                    return str(table_data[field]).strip()
            
            # Method 3: Generate summary from table structure
            rows_count = 0
            cols_count = 0
            
            # Try to get dimensions from different fields
            if 'rows' in table_data:
                if isinstance(table_data['rows'], list):
                    rows_count = len(table_data['rows'])
                    if rows_count > 0 and isinstance(table_data['rows'][0], list):
                        cols_count = len(table_data['rows'][0])
            
            if 'dimensions' in table_data:
                dims = table_data['dimensions']
                if isinstance(dims, dict):
                    rows_count = dims.get('rows', rows_count)
                    cols_count = dims.get('cols', cols_count)
            
            # Create basic summary
            page_num = table_data.get('page_number', 0)
            if rows_count > 0 and cols_count > 0:
                return f"Table from page {page_num} with {rows_count} rows and {cols_count} columns"
            else:
                return f"Data table from page {page_num}"
            
        except Exception as e:
            logger.warning(f"Error extracting table summary: {e}")
            return "Data table"
    
    def _extract_table_json_with_fallbacks(self, table_data: Dict) -> str:
        """Extract table JSON content with multiple fallback methods optimized for LlamaParse structure."""
        try:
            # Method 1: Use existing JSON content
            for field in ['table_content_json', 'json_content', 'data_json']:
                if field in table_data and table_data[field]:
                    content = table_data[field]
                    if isinstance(content, str):
                        return content
                    elif isinstance(content, dict):
                        return json.dumps(content)
            
            # Method 2: Convert LlamaParse rows to structured JSON (Primary method)
            if 'rows' in table_data and table_data['rows']:
                rows = table_data['rows']
                if isinstance(rows, list) and len(rows) > 0:
                    # Create structured table data
                    if isinstance(rows[0], list) and len(rows[0]) > 0:
                        # First row contains headers
                        headers = rows[0]
                        data_rows = rows[1:] if len(rows) > 1 else []
                        
                        # Create table structure with headers as keys
                        table_dict = {}
                        for i, header in enumerate(headers):
                            if header:  # Skip empty headers
                                column_data = []
                                for row in data_rows:
                                    if i < len(row):
                                        cell_value = row[i] if row[i] else ""
                                        column_data.append(cell_value)
                                    else:
                                        column_data.append("")
                                table_dict[str(header).strip()] = column_data
                        
                        # Add metadata
                        table_dict["_metadata"] = {
                            "total_rows": len(rows),
                            "total_columns": len(headers),
                            "has_headers": True,
                            "source": "llamaparse_rows"
                        }
                        
                        return json.dumps(table_dict)
            
            # Method 3: Use LlamaParse CSV content
            if 'csv' in table_data and table_data['csv']:
                csv_content = table_data['csv']
                if isinstance(csv_content, str) and len(csv_content.strip()) > 0:
                    try:
                        import csv
                        from io import StringIO
                        
                        # Parse CSV using proper CSV parser
                        csv_reader = csv.reader(StringIO(csv_content))
                        rows = list(csv_reader)
                        
                        if len(rows) > 1:
                            headers = rows[0]
                            data_rows = rows[1:]
                            
                            # Create table structure
                            table_dict = {}
                            for i, header in enumerate(headers):
                                if header:  # Skip empty headers
                                    column_data = []
                                    for row in data_rows:
                                        if i < len(row):
                                            cell_value = row[i] if row[i] else ""
                                            column_data.append(cell_value)
                                        else:
                                            column_data.append("")
                                    table_dict[str(header).strip()] = column_data
                            
                            # Add metadata
                            table_dict["_metadata"] = {
                                "total_rows": len(rows),
                                "total_columns": len(headers),
                                "has_headers": True,
                                "source": "llamaparse_csv"
                            }
                            
                            return json.dumps(table_dict)
                    except Exception as csv_error:
                        logger.warning(f"CSV parsing failed: {csv_error}")
            
            # Method 4: Use LlamaParse markdown table
            if 'md' in table_data and table_data['md']:
                md_content = table_data['md']
                if isinstance(md_content, str) and '|' in md_content:
                    try:
                        # Parse markdown table
                        lines = md_content.strip().split('\n')
                        table_lines = []
                        
                        for line in lines:
                            if '|' in line and line.strip():
                                # Remove markdown table formatting
                                clean_line = line.strip()
                                if clean_line.startswith('|'):
                                    clean_line = clean_line[1:]
                                if clean_line.endswith('|'):
                                    clean_line = clean_line[:-1]
                                
                                # Split by pipe and clean cells
                                cells = [cell.strip() for cell in clean_line.split('|')]
                                table_lines.append(cells)
                        
                        if len(table_lines) > 2:  # Headers + separator + data
                            headers = table_lines[0]
                            data_rows = table_lines[2:]  # Skip separator line
                            
                            # Create table structure
                            table_dict = {}
                            for i, header in enumerate(headers):
                                if header:  # Skip empty headers
                                    column_data = []
                                    for row in data_rows:
                                        if i < len(row):
                                            cell_value = row[i] if row[i] else ""
                                            column_data.append(cell_value)
                                        else:
                                            column_data.append("")
                                    table_dict[str(header).strip()] = column_data
                            
                            # Add metadata
                            table_dict["_metadata"] = {
                                "total_rows": len(table_lines) - 1,  # Exclude separator
                                "total_columns": len(headers),
                                "has_headers": True,
                                "source": "llamaparse_markdown"
                            }
                            
                            return json.dumps(table_dict)
                    except Exception as md_error:
                        logger.warning(f"Markdown parsing failed: {md_error}")
            
            # Method 5: Extract from text content if available
            if 'text' in table_data and table_data['text']:
                text = table_data['text']
                if isinstance(text, str) and len(text.strip()) > 10:
                    # Try to parse table from text content
                    lines = text.strip().split('\n')
                    if len(lines) > 1:
                        # Look for tabular structure
                        table_data_list = []
                        for line in lines:
                            # Split by common delimiters
                            cells = re.split(r'\s{2,}|\t|,', line.strip())
                            if len(cells) > 1:
                                table_data_list.append(cells)
                        
                        if len(table_data_list) > 1:
                            # Create table structure
                            headers = table_data_list[0]
                            data_rows = table_data_list[1:]
                            
                        table_dict = {}
                        for i, header in enumerate(headers):
                            if header:  # Skip empty headers
                                column_data = []
                                for row in data_rows:
                                    if i < len(row):
                                        cell_value = row[i] if row[i] else ""
                                        column_data.append(cell_value)
                                    else:
                                        column_data.append("")
                                table_dict[str(header).strip()] = column_data
                        
                        # Add metadata
                        table_dict["_metadata"] = {
                            "total_rows": len(table_data_list),
                            "total_columns": len(headers),
                            "has_headers": True,
                            "source": "llamaparse_text"
                        }
                        
                        return json.dumps(table_dict)
            
            # Method 6: Create enhanced JSON structure with available data
            enhanced_data = {
                "table_id": table_data.get('table_id', 'unknown'),
                "page": table_data.get('page_number', 0),
                "table_type": table_data.get('type', 'table'),
                "has_content": bool(table_data.get('text') or table_data.get('rows') or table_data.get('csv') or table_data.get('md')),
                "available_formats": []
            }
            
            # Add available formats
            if table_data.get('rows'):
                enhanced_data['available_formats'].append('rows')
                enhanced_data['row_count'] = len(table_data['rows'])
            
            if table_data.get('csv'):
                enhanced_data['available_formats'].append('csv')
            
            if table_data.get('md'):
                enhanced_data['available_formats'].append('markdown')
            
            if table_data.get('text'):
                enhanced_data['available_formats'].append('text')
                enhanced_data['text'] = table_data['text'][:500]  # Limit length
            
            return json.dumps(enhanced_data)
            
        except Exception as e:
            logger.warning(f"Error extracting table JSON: {e}")
            return "{}"
    
    def _store_composite_chunks_in_pinecone(self, composite_chunks: List[Dict], document_uuid: str) -> int:
        """Store composite chunks in Pinecone with unified embeddings and comprehensive error handling."""
        if not composite_chunks:
            logger.warning("No composite chunks to store")
            return 0
        
        try:
            vectors = []
            stored_count = 0
            
            logger.info(f"Processing {len(composite_chunks)} composite chunks for storage")
            
            for chunk in composite_chunks:
                try:
                    # Get composite text for embedding
                    composite_text = chunk.get('composite_text', '')
                    if not composite_text or len(composite_text.strip()) < 5:  # Lowered threshold
                        logger.warning(f"Skipping chunk {chunk.get('chunk_id')} - insufficient composite text ({len(composite_text.strip())} chars)")
                        continue
                    
                    # Generate embedding from composite text
                    embedding = self._get_embedding_with_retry(composite_text)
                    if not embedding or len(embedding) == 0:
                        logger.warning(f"Failed to generate embedding for chunk {chunk.get('chunk_id')}")
                        continue
                    
                    # Create metadata using new structure with arrays
                    metadata = {
                        'chunk_id': chunk.get('chunk_id', ''),
                        'document_uuid': document_uuid,
                        'document_name': chunk.get('document_name', ''),
                        'page_number': chunk.get('page_number', 0),
                        'has_text': chunk.get('has_text', True),
                        
                        # Image arrays
                        'contains_image': chunk.get('contains_image', False),
                        'image_count': chunk.get('image_count', 0),
                        'image_s3_urls': chunk.get('image_s3_urls', []),
                        'image_ids': chunk.get('image_ids', []),
                        'image_summaries': chunk.get('image_summaries', []),
                        
                        # Table arrays  
                        'contains_table': chunk.get('contains_table', False),
                        'table_count': chunk.get('table_count', 0),
                        'table_content_jsons': chunk.get('table_content_jsons', []),
                        'table_ids': chunk.get('table_ids', []),
                        'table_summaries': chunk.get('table_summaries', []),
                        
                        'processed_timestamp': chunk.get('processed_timestamp', ''),
                        
                        # Store composite text as 'text' field for compatibility
                        'text': composite_text[:10000]  # Store composite text for query access
                    }
                    
                    # Create vector for Pinecone
                    vector = {
                        'id': f"comp_{chunk.get('chunk_id', f'unknown_{len(vectors)}')}",
                        'values': embedding,
                        'metadata': metadata
                    }
                    
                    vectors.append(vector)
                    logger.debug(f"Prepared vector for chunk {chunk.get('chunk_id')}")
                    
                except Exception as e:
                    logger.error(f"Error preparing vector for chunk {chunk.get('chunk_id', 'unknown')}: {e}")
                    continue
            
            # Batch upsert vectors to Pinecone
            if vectors:
                try:
                    # Upsert in batches to avoid timeout
                    batch_size = 50
                    for i in range(0, len(vectors), batch_size):
                        batch = vectors[i:i + batch_size]
                        
                        try:
                            self.index.upsert(vectors=batch, namespace=document_uuid)
                            stored_count += len(batch)
                            logger.info(f"Stored batch {i//batch_size + 1}: {len(batch)} vectors in namespace {document_uuid}")
                            
                        except Exception as batch_error:
                            logger.error(f"Error storing batch {i//batch_size + 1}: {batch_error}")
                            # Try individual vectors in this batch
                            for vector in batch:
                                try:
                                    self.index.upsert(vectors=[vector], namespace=document_uuid)
                                    stored_count += 1
                                except Exception as single_error:
                                    logger.error(f"Error storing single vector {vector['id']}: {single_error}")
                    
                    logger.info(f"Successfully stored {stored_count}/{len(vectors)} composite chunk vectors in Pinecone namespace: {document_uuid}")
                    
                except Exception as e:
                    logger.error(f"Critical error during Pinecone upsert: {e}")
                    return 0
            else:
                logger.warning("No valid vectors prepared for storage")
                return 0
            
            return stored_count
            
        except Exception as e:
            logger.error(f"Critical error in _store_composite_chunks_in_pinecone: {e}")
            return 0
    
    def _get_embedding_with_retry(self, text: str, max_retries: int = 3) -> List[float]:
        """Generate embedding with retry logic and comprehensive error handling."""
        for attempt in range(max_retries):
            try:
                if not text or len(text.strip()) < 3:
                    logger.warning("Text too short for embedding, returning zeros")
                    return [0.0] * 3072
                
                # Truncate text to OpenAI limit
                truncated_text = text[:8191]
                
                response = self.openai_client.embeddings.create(
                    model=self.embedding_model,
                    input=truncated_text
                )
                
                embedding = response.data[0].embedding
                
                # Validate embedding
                if not embedding or len(embedding) == 0:
                    raise ValueError("Empty embedding received")
                
                return embedding
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    logger.error(f"All embedding attempts failed for text: {text[:100]}...")
                    return [0.0] * 3072
                else:
                    time.sleep(1)  # Brief pause before retry
        
        return [0.0] * 3072
    

    
    def _generate_pdf_id(self, pdf_path: str) -> str:
        """Generate PDF ID based on filename for reuse detection."""
        from config import generate_pdf_id_from_name
        pdf_name = Path(pdf_path).name
        return generate_pdf_id_from_name(pdf_name)
    
    def _generate_pdf_id_from_name(self, pdf_name: str) -> str:
        """Generate PDF ID directly from filename for reuse detection."""
        from config import generate_pdf_id_from_name
        return generate_pdf_id_from_name(pdf_name)
    
    def _is_pdf_processed(self, document_uuid: str, document_name: str = None) -> bool:
        """Check if PDF has been fully processed by checking BOTH S3 storage AND Pinecone vectors."""
        try:
            # Check Pinecone namespace to see if any content exists
            namespace_stats = self.index.describe_index_stats()
            namespaces = namespace_stats.get('namespaces', {})
            
            if document_uuid in namespaces:
                vector_count = namespaces[document_uuid].get('vector_count', 0)
                if vector_count > 0:
                    logger.info(f"Document {document_uuid} already processed ({vector_count} vectors in Pinecone)")
                    return True
            
            # If no vectors in Pinecone, document is not fully processed 
            logger.info(f"Document {document_uuid} not found in Pinecone namespaces - needs processing")
            return False
            
        except Exception as e:
            logger.error(f"Error checking if PDF processed: {e}")
            return False
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            if not text or len(text.strip()) < 3:
                return [0.0] * 3072
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8191]  # OpenAI token limit
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 3072
    
    def _handle_specific_content_requests(self, query: str, document_uuid: str) -> Optional[Dict[str, Any]]:
        """Handle specific table/image requests like 'show me table 1' or 'image on page 3'."""
        import re
        query_lower = query.lower().strip()
        
        # Pattern matching for specific content requests
        table_patterns = [
            r'(?:show|display|give|get|find)\s+(?:me\s+)?table\s+(\d+)',
            r'table\s+(\d+)',
            r'(?:show|display|give|get|find)\s+(?:me\s+)?(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+table',
            r'(?:show|display|give|get|find)\s+(?:me\s+)?(?:the\s+)?table\s+(?:on\s+)?page\s+(\d+)',
            r'table\s+(?:on\s+)?page\s+(\d+)'
        ]
        
        image_patterns = [
            r'(?:show|display|give|get|find)\s+(?:me\s+)?image\s+(\d+)',
            r'image\s+(\d+)',
            r'(?:show|display|give|get|find)\s+(?:me\s+)?(?:the\s+)?(\d+)(?:st|nd|rd|th)?\s+image',
            r'(?:show|display|give|get|find)\s+(?:me\s+)?(?:the\s+)?image\s+(?:on\s+)?page\s+(\d+)',
            r'image\s+(?:on\s+)?page\s+(\d+)'
        ]
        
        # Check for table requests
        for pattern in table_patterns:
            match = re.search(pattern, query_lower)
            if match:
                target_number = int(match.group(1))
                logger.info(f"Detected specific table request: Table {target_number}")
                return self._find_specific_table(document_uuid, target_number, query)
        
        # Check for image requests
        for pattern in image_patterns:
            match = re.search(pattern, query_lower)
            if match:
                target_number = int(match.group(1))
                logger.info(f"Detected specific image request: Image {target_number}")
                return self._find_specific_image(document_uuid, target_number, query)
        
        return None
    
    def _find_specific_table(self, document_uuid: str, table_number: int, query: str) -> Dict[str, Any]:
        """Find a specific table by number using metadata search."""
        try:
            # Search for tables with matching table IDs
            # Query all chunks to find the right table
            all_chunks = self.index.query(
                vector=[0] * 3072,  # Dummy vector - we only care about metadata
                top_k=1000,  # Get all chunks
                namespace=document_uuid,
                include_metadata=True,
                include_values=False
            )
            
            # Find tables that match the requested number
            candidate_tables = []
            for chunk in all_chunks.matches:
                metadata = chunk.metadata
                if metadata.get('contains_table', False):
                    table_ids = metadata.get('table_ids', [])
                    table_content_jsons = metadata.get('table_content_jsons', [])
                    table_summaries = metadata.get('table_summaries', [])
                    page_number = metadata.get('page_number', 0)
                    
                    # Check each table in the arrays
                    for i, table_id in enumerate(table_ids):
                        # Extract table number from table_id (e.g., "doc_p3_table1" -> 1)
                        table_match = re.search(r'table(\d+)', table_id)
                        if table_match:
                            table_num = int(table_match.group(1))
                            if table_num == table_number:
                                candidate_tables.append({
                                    'chunk': chunk,
                                    'table_number': table_num,
                                    'page_number': page_number,
                                    'table_id': table_id,
                                    'table_content_json': table_content_jsons[i] if i < len(table_content_jsons) else '',
                                    'table_summary': table_summaries[i] if i < len(table_summaries) else ''
                                })
            
            if not candidate_tables:
                # Fallback: try to find table by sequence (first table = table 1, etc.)
                logger.info(f"No exact table {table_number} found, trying sequence-based search")
                all_tables = []
                for chunk in all_chunks.matches:
                    if chunk.metadata.get('contains_table', False):
                        all_tables.append(chunk)
                
                # Sort by page number, then by position on page
                all_tables.sort(key=lambda x: (x.metadata.get('page_number', 0), x.metadata.get('chunk_id', '')))
                
                if table_number <= len(all_tables):
                    target_chunk = all_tables[table_number - 1]  # Convert to 0-based index
                    candidate_tables = [{
                        'chunk': target_chunk,
                        'table_number': table_number,
                        'page_number': target_chunk.metadata.get('page_number', 0),
                        'table_id': target_chunk.metadata.get('table_id', '')
                    }]
            
            if not candidate_tables:
                return {
                    'inline_elements': [{'type': 'text', 'content': f"Table {table_number} not found in the document."}],
                    'performance_metrics': {'query_time': 0.1, 'total_results': 0, 'relevant_chunks': 0, 'search_strategy': 'metadata'}
                }
            
            # Use the first matching table (or handle multiple matches)
            best_match = candidate_tables[0]
            chunk = best_match['chunk']
            
            page_num = best_match['page_number']
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            logger.info(f"Found Table {table_number} on page {page_num}")
            
            # Create response with ONLY the specific table (no images)
            chunk_data = {
                'chunk_id': chunk.metadata.get('chunk_id', ''),
                'relevance_score': 1.0,  # Perfect match for specific request
                'original_score': chunk.score,
                'page_number': chunk.metadata.get('page_number', 0),
                'text': f"Here is Table {table_number} from page {page_num}:",  # Minimal text
                'contains_image': False,  # Force no images
                'image_s3_urls': [],
                'image_ids': [],
                'image_summaries': [],
                'image_count': 0,
                'contains_table': True,
                'table_content_jsons': [best_match.get('table_content_json', '')],
                'table_ids': [best_match.get('table_id', '')],
                'table_summaries': [best_match.get('table_summary', '')],
                'table_count': 1,
                'document_name': chunk.metadata.get('document_name', ''),
                'document_uuid': chunk.metadata.get('document_uuid', ''),
                'has_text': True
            }
            
            # Generate response using the inline generator
            response_data = self._generate_composite_response(query, [chunk_data], {}, 'standard', 2500, None)
            response_data['performance_metrics'] = {
                'query_time': 0.5,
                'total_results': 1,
                'relevant_chunks': 1,
                'search_strategy': 'metadata'
            }
            
            return response_data

        except Exception as e:
            logger.error(f"Error finding specific table {table_number}: {e}")
            return {
                'inline_elements': [{'type': 'text', 'content': f"Error finding Table {table_number}: {str(e)}"}],
                'performance_metrics': {'query_time': 0.1, 'total_results': 0, 'relevant_chunks': 0, 'search_strategy': 'metadata'}
            }
    
    def _find_specific_image(self, document_uuid: str, image_number: int, query: str) -> Dict[str, Any]:
        """Find a specific image by number using metadata search."""
        try:
            # Similar logic to _find_specific_table but for images
            all_chunks = self.index.query(
                vector=[0] * 3072,
                top_k=1000,
                namespace=document_uuid,
                include_metadata=True,
                include_values=False
            )
            
            # Find images that match the requested number
            candidate_images = []
            for chunk in all_chunks.matches:
                metadata = chunk.metadata
                if metadata.get('contains_image', False):
                    image_id = metadata.get('image_id', '')
                    page_number = metadata.get('page_number', 0)
                    
                    # Extract image number from image_id (e.g., "doc_img_1" -> 1)
                    image_match = re.search(r'img_(\d+)', image_id)
                    if image_match:
                        img_num = int(image_match.group(1))
                        if img_num == image_number:
                            candidate_images.append({
                                'chunk': chunk,
                                'image_number': img_num,
                                'page_number': page_number,
                                'image_id': image_id
                            })
            
            if not candidate_images:
                # Fallback: sequence-based search
                logger.info(f"No exact image {image_number} found, trying sequence-based search")
                all_images = []
                for chunk in all_chunks.matches:
                    if chunk.metadata.get('contains_image', False):
                        all_images.append(chunk)
                
                all_images.sort(key=lambda x: (x.metadata.get('page_number', 0), x.metadata.get('chunk_id', '')))
                
                if image_number <= len(all_images):
                    target_chunk = all_images[image_number - 1]
                    candidate_images = [{
                        'chunk': target_chunk,
                        'image_number': image_number,
                        'page_number': target_chunk.metadata.get('page_number', 0),
                        'image_id': target_chunk.metadata.get('image_id', '')
                    }]
            
            if not candidate_images:
                return {
                    'inline_elements': [{'type': 'text', 'content': f"Image {image_number} not found in the document."}],
                    'performance_metrics': {'query_time': 0.1, 'total_results': 0, 'relevant_chunks': 0, 'search_strategy': 'metadata'}
                }
            
            # Use the first matching image
            best_match = candidate_images[0]
            chunk = best_match['chunk']
            
            page_num = best_match['page_number']
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            logger.info(f"Found Image {image_number} on page {page_num}")
            
            # Create chunk data with ONLY the specific image (no tables)
            chunk_data = {
                'chunk_id': chunk.metadata.get('chunk_id', ''),
                'relevance_score': 1.0,
                'original_score': chunk.score,
                'page_number': chunk.metadata.get('page_number', 0),
                'text': f"Here is Image {image_number} from page {page_num}:",  # Minimal text
                'contains_image': True,
                'image_url': chunk.metadata.get('image_url', ''),
                'image_summary': chunk.metadata.get('image_summary', ''),
                'contains_table': False,  # Force no tables
                'table_content_json': '',
                'table_summary': '',
                'source_document': chunk.metadata.get('source_document', '')
            }
            
            response_data = self._generate_composite_response(query, [chunk_data], {}, 'standard', 2500, None)
            response_data['performance_metrics'] = {
                'query_time': 0.5,
                'total_results': 1,
                'relevant_chunks': 1,
                'search_strategy': 'metadata'
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error finding specific image {image_number}: {e}")
            return {
                'inline_elements': [{'type': 'text', 'content': f"Error finding Image {image_number}: {str(e)}"}],
                'performance_metrics': {'query_time': 0.1, 'total_results': 0, 'relevant_chunks': 0, 'search_strategy': 'metadata'}
            }

    async def query_multimodal_content(self, query: str, document_uuid: str, 
                               max_chunks: int = 5, **kwargs) -> Dict[str, Any]:
        """Query composite chunks using single-pass retrieval with full content."""
        try:
            start_time = time.time()
            
            # Store query terms for enhanced image relevance scoring
            self._current_query_terms = query.lower().split()
            
            # Extract optional conversation context (string of last turns)
            conversation_context: Optional[str] = kwargs.get("conversation_context")

            # Check for specific table/image requests first
            specific_content_results = self._handle_specific_content_requests(query, document_uuid)
            if specific_content_results:
                logger.info(f"Found specific content match for query: {query}")
                return specific_content_results
            
            # Parallel processing: Generate embedding and search params simultaneously
            # Use asyncio.to_thread for CPU-bound tasks to run in parallel
            embedding_task = asyncio.to_thread(self._get_embedding, query)
            search_params_task = asyncio.to_thread(self._get_adaptive_search_params, query, conversation_context=conversation_context)
            
            # Run embedding generation and query analysis in parallel
            query_embedding, search_params = await asyncio.gather(
                embedding_task,
                search_params_task
            )

            # Primary retrieval: query-only
            primary_results = self.index.query(
                vector=query_embedding,
                top_k=search_params['top_k'],
                namespace=document_uuid,
                include_metadata=True,
                include_values=False
            )

            # Optional secondary retrieval: lightly augmented with top context keywords
            combined_matches = list(primary_results.matches)
            if conversation_context:
                try:
                    intent = search_params.get('query_intent', {})
                    context_keywords = intent.get('intent_keywords', [])
                    if context_keywords:
                        import os
                        max_kw = int(os.getenv('CTX_AUGMENT_KEYWORDS', '3'))
                        aug_keywords = context_keywords[:max_kw]
                        if aug_keywords:
                            augmented_query = f"{query} {' '.join(aug_keywords)}".strip()
                            aug_emb = self._get_embedding(augmented_query)
                            secondary_top_k = min(max(int(search_params['top_k'] * 0.5), 10), search_params['top_k'])
                            secondary_results = self.index.query(
                                vector=aug_emb,
                                top_k=secondary_top_k,
                                namespace=document_uuid,
                                include_metadata=True,
                                include_values=False
                            )
                            # Union by id, preserve primary ordering preference
                            seen_ids = {m.id for m in combined_matches}
                            for m in secondary_results.matches:
                                if m.id not in seen_ids:
                                    combined_matches.append(m)
                                    seen_ids.add(m.id)
                except Exception as _:
                    # Fail open to primary results only
                    pass

            logger.info(f"Composite chunk query returned {len(combined_matches)} combined matches for namespace '{document_uuid}'")
            
            # Parallel processing: Run filtering and validation operations simultaneously  
            filtering_task = asyncio.to_thread(
                self._filter_chunks_with_enhanced_scoring,
                combined_matches, query, search_params
            )
            
            # Wait for filtering to complete, then run validation in parallel with other tasks
            relevant_chunks = await filtering_task
            
            validation_task = asyncio.to_thread(
                self._filter_and_validate_content, 
                relevant_chunks, query
            )
            
            # Continue with validation asynchronously
            validated_chunks = await validation_task
            
            if not search_params.get('query_intent'):
                relevant_chunks = self._apply_content_aware_selection(relevant_chunks, query, max_chunks)
            
            logger.info(f"Selected {len(relevant_chunks)} relevant page-chunks for response generation.")

            # Generate response using the full original content (no distillation)
            query_intent = search_params.get('query_intent', {})
            conditional_instructions = query_intent.get('conditional_instructions', {})
            response_scope = query_intent.get('response_scope', 'standard')
            suggested_max_tokens = query_intent.get('suggested_max_tokens', 2500)
            response_data = self._generate_composite_response(query, validated_chunks, conditional_instructions, response_scope, suggested_max_tokens, query_intent)
            
            logger.debug(f"Generated response data type: {type(response_data)}")
            logger.debug(f"Response data keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")
            
            response_data['performance_metrics'] = {
                'query_time': time.time() - start_time,
                'total_results': len(combined_matches),
                'relevant_chunks': len(relevant_chunks),
                'search_strategy': search_params['strategy']
            }
            
            logger.info(f"Final response data: {response_data}")
            return response_data
            
        except Exception as e:
            logger.error(f"Error in composite chunk query: {e}")
            return {
                'response': f"Error processing query: {str(e)}",
                'inline_elements': [{'type': 'text', 'content': f"Error: {str(e)}"}],
                'error': str(e)
            }
    
    def _get_adaptive_search_params(self, query: str, conversation_context: Optional[str] = None) -> Dict[str, Any]:
        """Get intelligent search parameters using enhanced dynamic query intent analysis.
        The conversation_context is a plain string; it is used only to enrich intent keywords and conditional instructions.
        Retrieval embeddings remain based on the original query only.
        """
        # Use enhanced AI-powered query intent detection
        query_intent = self._analyze_query_intent_with_cot(query)

        # If a string context is provided, derive lightweight keywords, page hints, and focus cues
        if isinstance(conversation_context, str) and conversation_context.strip():
            try:
                context_text = conversation_context.strip()

                # 1) Parse latest Q/A blocks (simple heuristic)
                qa_blocks = re.findall(r"(?is)Q:\s*(.+?)(?:\n\s*A:\s*(.+?))?(?=(?:\n\s*Q:)|\Z)", context_text)
                recent_qs = [q.strip() for q, _ in qa_blocks[-5:]] if qa_blocks else []
                last_q = recent_qs[-1] if recent_qs else ""

                # 2) Extract page hints from context
                page_numbers = [int(n) for n in re.findall(r"page\s*(\d+)", context_text, flags=re.I)]
                preferred_pages = sorted(list({p for p in page_numbers}))[:5]

                # 3) Extract lightweight keywords from last ~500 chars
                context_slice = context_text[-500:]
                words = re.findall(r"[A-Za-z][A-Za-z0-9_-]+", context_slice.lower())
                stop = {
                    'the','and','for','with','that','this','from','have','has','were','been','into','your','their','about','also','only','over','when','which','while','will','would','could','should','there','here','such','than','then','them','they','what','how','why','where','does','did','now','next','please','more','less','very','much','like'
                }
                keywords = [w for w in words if len(w) > 3 and w not in stop]
                seen = set()
                context_keywords = [w for w in keywords if not (w in seen or seen.add(w))][:25]

                # 4) Detect advantage/disadvantage focus cues from last question
                pos_kw, neg_kw = [], []
                if last_q:
                    lq = last_q.lower()
                    if any(t in lq for t in ["disadvantage","limitations","drawbacks","cons","weaknesses"]):
                        pos_kw.extend(["limitation","drawback","disadvantage","challenge","cost","memory","latency","quadratic","n2","compute","bottleneck"])
                        neg_kw.extend(["advantage","benefit","pro"])
                    elif any(t in lq for t in ["advantage","benefits","pros","strengths"]):
                        pos_kw.extend(["advantage","benefit","strength","parallel","scalable","efficient","accuracy","bleu","speed"])
                        neg_kw.extend(["limitation","drawback"])

                # Merge into intent
                query_intent.setdefault("intent_keywords", [])
                query_intent["intent_keywords"].extend([k for k in context_keywords if k not in query_intent["intent_keywords"]])

                query_intent.setdefault("conditional_instructions", {})
                ci = query_intent["conditional_instructions"]
                ci["context_guidance"] = "Prioritize content aligned with recent conversational topics and terminology."
                if preferred_pages:
                    ci["preferred_pages"] = preferred_pages
                if pos_kw:
                    existing_pos = set(ci.get("positive_keywords", []))
                    ci["positive_keywords"] = list(existing_pos.union(set(pos_kw)))
                if neg_kw:
                    existing_neg = set(ci.get("negative_keywords", []))
                    ci["negative_keywords"] = list(existing_neg.union(set(neg_kw)))

                query_intent.setdefault("content_preferences", ["text", "images", "tables"])
            except Exception:
                pass
        
        # Extract search parameters from intent analysis
        search_params = query_intent.get('search_parameters', {})
        base_top_k = 100
        
        # Apply multipliers from intent analysis
        top_k_multiplier = search_params.get('top_k_multiplier', 1.0)
        min_score_adjustment = search_params.get('min_score_adjustment', 0.0)
        
        # Dynamic parameter adjustment based on enhanced intent analysis
        if query_intent['scope'] == 'comprehensive':
            top_k = int(base_top_k * max(top_k_multiplier, 1.5))
            min_score = max(0.01, 0.03 + min_score_adjustment)  # Very low threshold for comprehensive queries
        elif query_intent['scope'] == 'specific':
            top_k = int(base_top_k * max(top_k_multiplier, 0.8))
            min_score = max(0.03, 0.08 + min_score_adjustment)
        else:  # focused
            top_k = int(base_top_k * max(top_k_multiplier, 0.6))
            min_score = max(0.05, 0.12 + min_score_adjustment)
        
        # Adjust based on search strategy
        strategy = query_intent.get('search_strategy', 'balanced')
        if strategy == 'recall':
            top_k = int(top_k * 1.3)
            min_score *= 0.7
        elif strategy == 'precision':
            top_k = int(top_k * 0.7)
            min_score *= 1.4
        elif strategy == 'hybrid':
            # Use multiple search passes with different parameters
            top_k = int(top_k * 1.1)
        
        # Adjust based on complexity
        complexity = query_intent.get('complexity_level', 'medium')
        if complexity == 'complex':
            top_k = int(top_k * 1.2)
            min_score *= 0.8
        elif complexity == 'simple':
            top_k = int(top_k * 0.8)
            min_score *= 1.1
        
        # Ensure reasonable bounds
        top_k = min(max(top_k, 20), 200)
        min_score = min(max(min_score, 0.01), 0.3)
        
        return {
            'strategy': strategy,
            'top_k': top_k,
            'min_score': min_score,
            'query_intent': query_intent,
            'complexity': complexity,
            'diversity_requirement': search_params.get('diversity_requirement', 'medium'),
            'semantic_expansion': search_params.get('semantic_expansion', False),
            'cross_content_boost': search_params.get('cross_content_boost', True)
        }
    
    def _score_and_filter_match(self, match, query_keywords: set, search_params: Dict) -> Optional[Dict]:
        """Scores and filters a single chunk based on relevance and search parameters."""
        query_intent = search_params.get('query_intent', {})
        conditional_instructions = query_intent.get('conditional_instructions', {})
        content_preferences = query_intent.get('content_preferences', ['text', 'images', 'tables'])
        semantic_concepts = query_intent.get('semantic_concepts', [])
        intent_keywords = query_intent.get('intent_keywords', [])

        include_content = conditional_instructions.get('include_content', [])
        exclude_content = conditional_instructions.get('exclude_content', [])
        positive_keywords = conditional_instructions.get('positive_keywords', [])
        negative_keywords = conditional_instructions.get('negative_keywords', [])

        content_type_weights = {
            'text': 1.0 if 'text' in content_preferences else 0.7,
            'images': 1.2 if 'images' in content_preferences else 0.8,
            'tables': 1.1 if 'tables' in content_preferences else 0.8
        }

        if match.score < search_params['min_score']:
            return None

        if self._should_exclude_chunk(match, exclude_content, negative_keywords):
            return None

        if include_content and not self._should_include_chunk(match, include_content, positive_keywords):
            return None

        chunk_content_types = self._identify_chunk_content_types(match)
        content_type_boost = self._calculate_content_type_boost(chunk_content_types, content_type_weights)

        enhanced_score = self._calculate_enhanced_relevance_score_advanced(
            match, query_keywords, search_params['strategy'],
            positive_keywords, negative_keywords, semantic_concepts,
            intent_keywords, content_type_boost
        )

        if enhanced_score > 0.01:
            return {
                'chunk_id': match.metadata.get('chunk_id', ''),
                'relevance_score': enhanced_score,
                'original_score': match.score,
                'page_number': match.metadata.get('page_number', 0),
                'text': match.metadata.get('text', ''),
                'contains_image': match.metadata.get('contains_image', False),
                'image_count': match.metadata.get('image_count', 0),
                'image_s3_urls': match.metadata.get('image_s3_urls', []),
                'image_ids': match.metadata.get('image_ids', []),
                'image_summaries': match.metadata.get('image_summaries', []),
                'contains_table': match.metadata.get('contains_table', False),
                'table_count': match.metadata.get('table_count', 0),
                'table_content_jsons': match.metadata.get('table_content_jsons', []),
                'table_ids': match.metadata.get('table_ids', []),
                'table_summaries': match.metadata.get('table_summaries', []),
                'document_name': match.metadata.get('document_name', ''),
                'document_uuid': match.metadata.get('document_uuid', ''),
                'has_text': match.metadata.get('has_text', True)
            }
        return None

    def _filter_chunks_with_enhanced_scoring(self, matches, query: str, search_params: Dict) -> List[Dict]:
        """Enhanced filtering with intelligent content scoring and preference-based selection, processed in parallel."""
        query_keywords = set(query.lower().split())

        potential_chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_match = {
                executor.submit(self._score_and_filter_match, match, query_keywords, search_params): match
                for match in matches
            }

            for future in as_completed(future_to_match):
                result = future.result()
                if result:
                    potential_chunks.append(result)

        filtered_count = len(matches) - len(potential_chunks)
        logger.info(f"Enhanced filtering results: {len(matches)} initial matches, {filtered_count} filtered out, {len(potential_chunks)} potential chunks")

        if potential_chunks:
            relevant_chunks = self._ai_powered_content_selection(query, potential_chunks, search_params)
        else:
            logger.warning("No potential chunks found after enhanced filtering - using fallback with original scores")
            fallback_chunks = []
            for match in matches[:10]:
                if match.score >= search_params['min_score']:
                    fallback_chunks.append({
                        'chunk_id': match.metadata.get('chunk_id', ''),
                        'relevance_score': match.score,
                        'original_score': match.score,
                        'page_number': match.metadata.get('page_number', 0),
                        'text': match.metadata.get('text', ''),
                        'contains_image': match.metadata.get('contains_image', False),
                        'image_url': match.metadata.get('image_url', ''),
                        'image_summary': match.metadata.get('image_summary', ''),
                        'contains_table': match.metadata.get('contains_table', False),
                        'table_content_json': match.metadata.get('table_content_json', ''),
                        'table_summary': match.metadata.get('table_summary', ''),
                        'source_document': match.metadata.get('source_document', '')
                    })
            relevant_chunks = fallback_chunks

        logger.info(f"Final selection: {len(relevant_chunks)} chunks after AI selection")
        return relevant_chunks
    
    def _should_exclude_chunk(self, match, exclude_content: List[str], negative_keywords: List[str]) -> bool:
        """Check if chunk should be excluded based on conditional instructions."""
        # Check content type exclusions
        for content_type in exclude_content:
            if content_type == 'images' and match.metadata.get('contains_image'):
                return True
            elif content_type == 'tables' and match.metadata.get('contains_table'):
                return True
            elif content_type == 'text' and match.metadata.get('text'):
                return True
        
        # Check negative keywords
        text = match.metadata.get('text', '').lower()
        for keyword in negative_keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def _ai_powered_content_selection(self, query: str, potential_chunks: List[Dict], search_params: Dict) -> List[Dict]:
        """Use AI to intelligently select the best content for the response."""
        try:
            # Prepare content summary for AI analysis
            content_summary = self._prepare_content_summary_for_ai(potential_chunks)
            
            # Enhanced AI prompt for intelligent content selection
            selection_prompt = f"""You are an intelligent content curator. Analyze the user query and determine exactly what content types are needed for the best response.

USER QUERY: "{query}"

AVAILABLE CONTENT:
{content_summary}

INSTRUCTIONS:
1. ANALYZE THE QUERY INTENT:
   - Is this a simple question needing text explanation only?
   - Does the user want to see data/statistics (needs tables)?
   - Does the user want visual information (needs images)?
   - Is this a comprehensive analysis (needs all content types)?

2. DECIDE CONTENT REQUIREMENTS:
   - For simple "what/how/why/tell me about" questions → TEXT ONLY
   - For "show me data/statistics/results" → TEXT + TABLES
   - For "show me/visualize/images/figures" → TEXT + IMAGES
   - For "analyze/compare/comprehensive" → TEXT + IMAGES + TABLES

3. SET APPROPRIATE LIMITS:
   - Simple queries: 1-3 text sections, 0 images, 0 tables
   - Data queries: 2-4 text sections, 0-1 images, 1-2 tables
   - Visual queries: 2-4 text sections, 1-3 images, 0-1 tables
   - Complex queries: 3-5 text sections, 1-3 images, 1-2 tables

Respond with ONLY this JSON format:
{{
    "selection_strategy": "[comprehensive|focused|balanced|minimal]",
    "content_limits": {{
        "text_sections": [min, max],
        "images": [min, max], 
        "tables": [min, max]
    }},
    "priority_order": ["text", "images", "tables"],
    "reasoning": "Brief explanation of why these content types are needed"
}}"""

            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": selection_prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            import json
            selection_strategy = json.loads(response.choices[0].message.content)
            logger.info(f"AI content selection strategy: {selection_strategy}")
            
            # Apply the AI-determined selection strategy
            selected_chunks = self._apply_ai_selection_strategy(potential_chunks, selection_strategy)
            
            return selected_chunks
            
        except Exception as e:
            logger.error(f"Error in AI-powered content selection: {e}")
            # Fallback to top-scoring chunks
            return sorted(potential_chunks, key=lambda x: x['relevance_score'], reverse=True)[:10]
    
    def _prepare_content_summary_for_ai(self, chunks: List[Dict]) -> str:
        """Prepare a summary of available content for AI analysis."""
        text_count = sum(1 for c in chunks if c.get('text'))
        image_count = sum(1 for c in chunks if c.get('contains_image'))
        table_count = sum(1 for c in chunks if c.get('contains_table'))
        
        # Sample content descriptions
        text_samples = [c.get('text', '')[:100] for c in chunks if c.get('text')][:3]
        image_samples = [c.get('image_summary', '') for c in chunks if c.get('contains_image')][:3]
        table_samples = [c.get('table_summary', '') for c in chunks if c.get('contains_table')][:3]
        
        summary = f"""TOTAL CONTENT:
- Text sections: {text_count}
- Images: {image_count} 
- Tables: {table_count}

SAMPLE CONTENT:
- Text samples: {text_samples}
- Image descriptions: {image_samples}
- Table summaries: {table_samples}

RELEVANCE SCORES:
- Average score: {sum(c['relevance_score'] for c in chunks) / len(chunks):.3f}
- Score range: {min(c['relevance_score'] for c in chunks):.3f} - {max(c['relevance_score'] for c in chunks):.3f}"""
        
        return summary
    
    def _apply_ai_selection_strategy(self, chunks: List[Dict], strategy: Dict) -> List[Dict]:
        """Apply AI-determined content selection strategy based on query intent and content analysis."""
        # Sort chunks by relevance score
        sorted_chunks = sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)
        
        # Extract strategy parameters
        selection_strategy = strategy.get('selection_strategy', 'balanced')
        content_limits = strategy.get('content_limits', {})
        priority_order = strategy.get('priority_order', ['text', 'images', 'tables'])
        reasoning = strategy.get('reasoning', '')
        
        logger.info(f"Applying AI selection strategy: {selection_strategy} with limits {content_limits}")
        
        # Apply strategy-specific selection logic
        if selection_strategy == 'comprehensive':
            # Select more content for comprehensive queries - Updated fallback limits for smaller chunks
            text_limit = content_limits.get('text_sections', [15, 25])[1]  # Reduced from [25,40] to [15,25]
            image_limit = content_limits.get('images', [2, 4])[1]          # Reduced from [4,8] to [2,4]
            table_limit = content_limits.get('tables', [2, 3])[1]         # Reduced from [3,6] to [2,3]
            
            # Filter out invalid tables before selection
            sorted_chunks = self._filter_invalid_tables_from_chunks(sorted_chunks)
        elif selection_strategy == 'focused':
            # Select focused content for specific queries - Updated fallback limits for smaller chunks
            text_limit = content_limits.get('text_sections', [12, 18])[0]  # Updated from [3,6] to [12,18]
            image_limit = content_limits.get('images', [2, 4])[0]          # Updated from [1,2] to [2,4]
            table_limit = content_limits.get('tables', [1, 3])[0]         # Updated from [1,2] to [1,3]
        elif selection_strategy == 'visual':
            # Prioritize visual content - Updated fallback limits for smaller chunks
            text_limit = content_limits.get('text_sections', [8, 15])[0]   # Updated from [2,4] to [8,15]
            image_limit = content_limits.get('images', [6, 12])[1]         # Updated from [3,5] to [6,12]
            table_limit = content_limits.get('tables', [2, 4])[1]         # Updated from [2,3] to [2,4]
        elif selection_strategy == 'textual':
            # Prioritize text content - Updated fallback limits for smaller chunks
            text_limit = content_limits.get('text_sections', [20, 35])[1]  # Updated from [6,10] to [20,35]
            image_limit = content_limits.get('images', [2, 4])[0]          # Updated from [1,2] to [2,4]
            table_limit = content_limits.get('tables', [2, 4])[0]         # Updated from [1,2] to [2,4]
        else:  # balanced or default
            text_limit = content_limits.get('text_sections', [15, 25])[1]  # Updated from [4,6] to [15,25]
            image_limit = content_limits.get('images', [3, 6])[1]          # Updated from [2,3] to [3,6]
            table_limit = content_limits.get('tables', [2, 4])[1]         # Updated from [1,2] to [2,4]
        
        # Enhanced sorting: prioritize chunks with relevant images for image-focused queries
        if 'images' in priority_order and priority_order.index('images') <= 1:  # Images are high priority
            # Create separate lists for image chunks and others
            image_chunks = [c for c in sorted_chunks if c.get('contains_image')]
            other_chunks = [c for c in sorted_chunks if not c.get('contains_image')]
            
            # Sort image chunks by image relevance indicators
            image_chunks.sort(key=lambda x: (
                x.get('image_count', 0),  # Prefer multiple images
                len([s for s in x.get('image_summaries', []) if s and len(s) > 50]),  # Rich summaries
                x['relevance_score']  # Fallback to relevance score
            ), reverse=True)
            
            # Recombine with image chunks having priority
            sorted_chunks = image_chunks + other_chunks
        
        # Select content according to priority order and limits
        selected_chunks = []
        text_selected = 0
        image_selected = 0
        table_selected = 0
        
        for priority_type in priority_order:
            for chunk in sorted_chunks:
                if len(selected_chunks) >= (text_limit + image_limit + table_limit):
                    break
                    
                if chunk in selected_chunks:
                    continue
                
                # Select based on priority type and limits
                if priority_type == 'text' and chunk.get('text') and text_selected < text_limit:
                    selected_chunks.append(chunk)
                    text_selected += 1
                elif priority_type == 'images' and chunk.get('contains_image') and image_selected < image_limit:
                    # Enhanced image selection: prioritize chunks with multiple relevant images
                    image_count = chunk.get('image_count', 0)
                    image_summaries = chunk.get('image_summaries', [])
                    
                    # Boost selection if chunk has multiple images or highly relevant image summaries
                    should_prioritize = image_count > 1 or any(
                        summary and len(summary) > 50 and any(
                            keyword in summary.lower() 
                            for keyword in ['diagram', 'architecture', 'model', 'visualization', 'chart', 'figure']
                        ) for summary in image_summaries
                    )
                    
                    # Add chunk, prioritizing those with better image content
                    selected_chunks.append(chunk)
                    image_selected += 1
                elif priority_type == 'tables' and chunk.get('contains_table') and table_selected < table_limit:
                    selected_chunks.append(chunk)
                    table_selected += 1
        
        # Fill remaining slots if we haven't reached limits
        for chunk in sorted_chunks:
            if len(selected_chunks) >= (text_limit + image_limit + table_limit):
                break
                
            if chunk not in selected_chunks:
                if chunk.get('text') and text_selected < text_limit:
                    selected_chunks.append(chunk)
                    text_selected += 1
                elif chunk.get('contains_image') and image_selected < image_limit:
                    selected_chunks.append(chunk)
                    image_selected += 1
                elif chunk.get('contains_table') and table_selected < table_limit:
                    selected_chunks.append(chunk)
                    table_selected += 1
        
        logger.info(f"AI selection strategy '{selection_strategy}' applied: {len(selected_chunks)} chunks selected (Text: {text_selected}, Images: {image_selected}, Tables: {table_selected})")
        logger.debug(f"Selection reasoning: {reasoning}")
        
        # Phase 2 Enhancement: Apply semantic clustering to improve coherence
        clustered_chunks = self._apply_semantic_clustering(selected_chunks, selection_strategy)
        
        return clustered_chunks
    
    def _apply_semantic_clustering(self, selected_chunks: List[Dict], selection_strategy: str) -> List[Dict]:
        """Phase 2 Enhancement: Apply semantic clustering to group related chunks for better coherence."""
        if len(selected_chunks) <= 3:
            return selected_chunks  # No clustering needed for small sets
        
        try:
            # Group chunks by page number to maintain page-based coherence
            page_groups = {}
            for chunk in selected_chunks:
                page_num = chunk.get('page_number', 0)
                if page_num not in page_groups:
                    page_groups[page_num] = []
                page_groups[page_num].append(chunk)
            
            # Apply clustering logic based on strategy
            if selection_strategy in ['comprehensive', 'textual']:
                # For comprehensive queries, prioritize page-based grouping for narrative flow
                clustered_chunks = self._cluster_by_page_coherence(page_groups, selected_chunks)
            elif selection_strategy in ['focused', 'visual']:
                # For focused queries, prioritize relevance over page grouping
                clustered_chunks = self._cluster_by_relevance_similarity(selected_chunks)
            else:  # balanced
                # Balanced approach: moderate clustering with relevance consideration
                clustered_chunks = self._cluster_balanced_approach(page_groups, selected_chunks)
            
            logger.info(f"Semantic clustering applied ({selection_strategy}): {len(selected_chunks)} → {len(clustered_chunks)} chunks")
            return clustered_chunks
            
        except Exception as e:
            logger.error(f"Error in semantic clustering: {e}")
            return selected_chunks  # Fallback to original selection
    
    def _cluster_by_page_coherence(self, page_groups: Dict, all_chunks: List[Dict]) -> List[Dict]:
        """Cluster chunks prioritizing page-based narrative coherence."""
        clustered_result = []
        
        # Sort pages by average relevance score
        page_scores = {}
        for page_num, chunks in page_groups.items():
            avg_score = sum(c['relevance_score'] for c in chunks) / len(chunks)
            page_scores[page_num] = avg_score
        
        # Process pages in order of relevance
        for page_num in sorted(page_scores.keys(), key=page_scores.get, reverse=True):
            page_chunks = page_groups[page_num]
            
            # Limit chunks per page to avoid over-concentration (Phase 2 improvement)
            max_chunks_per_page = 4 if len(page_groups) > 3 else 6
            if len(page_chunks) > max_chunks_per_page:
                # Keep highest scoring chunks from this page
                page_chunks = sorted(page_chunks, key=lambda x: x['relevance_score'], reverse=True)[:max_chunks_per_page]
            
            clustered_result.extend(page_chunks)
        
        return clustered_result
    
    def _cluster_by_relevance_similarity(self, chunks: List[Dict]) -> List[Dict]:
        """Cluster chunks by relevance similarity for focused queries."""
        if len(chunks) <= 5:
            return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)
        
        # Apply more aggressive filtering for focused queries
        threshold_score = sorted([c['relevance_score'] for c in chunks], reverse=True)[len(chunks)//2]
        
        high_relevance_chunks = [c for c in chunks if c['relevance_score'] >= threshold_score]
        
        # Ensure diversity by limiting chunks from same page
        page_counts = {}
        result = []
        
        for chunk in sorted(high_relevance_chunks, key=lambda x: x['relevance_score'], reverse=True):
            page_num = chunk.get('page_number', 0)
            page_counts[page_num] = page_counts.get(page_num, 0) + 1
            
            # Limit to max 2 chunks per page for focused queries
            if page_counts[page_num] <= 2:
                result.append(chunk)
        
        return result
    
    def _cluster_balanced_approach(self, page_groups: Dict, all_chunks: List[Dict]) -> List[Dict]:
        """Balanced clustering approach combining page coherence and relevance."""
        if len(all_chunks) <= 8:
            return sorted(all_chunks, key=lambda x: x['relevance_score'], reverse=True)
        
        # Target about 60% of original chunks for better coherence
        target_count = max(6, int(len(all_chunks) * 0.6))
        
        result = []
        chunks_per_page = {}
        
        # Calculate chunks per page limit
        num_pages = len(page_groups)
        max_per_page = max(2, target_count // num_pages) if num_pages > 0 else 3
        
        # Sort all chunks by relevance first
        sorted_chunks = sorted(all_chunks, key=lambda x: x['relevance_score'], reverse=True)
        
        for chunk in sorted_chunks:
            if len(result) >= target_count:
                break
                
            page_num = chunk.get('page_number', 0)
            page_count = chunks_per_page.get(page_num, 0)
            
            if page_count < max_per_page:
                result.append(chunk)
                chunks_per_page[page_num] = page_count + 1
        
        return result
    
    def _should_include_chunk(self, match, include_content: List[str], positive_keywords: List[str]) -> bool:
        """Check if chunk should be included based on conditional instructions."""
        # Check content type inclusions
        for content_type in include_content:
            if content_type == 'images' and match.metadata.get('contains_image'):
                return True
            elif content_type == 'tables' and match.metadata.get('contains_table'):
                return True
            elif content_type == 'text' and match.metadata.get('text'):
                return True
        
        # Check positive keywords
        text = match.metadata.get('text', '').lower()
        for keyword in positive_keywords:
            if keyword.lower() in text:
                return True
        
        return False
    
    def _calculate_enhanced_relevance_score_with_conditionals(self, match, query_keywords: set, strategy: str, 
                                                           positive_keywords: List[str], negative_keywords: List[str]) -> float:
        """Calculate enhanced relevance score with conditional adjustments."""
        base_score = match.score
        
        # Content type boosting based on strategy
        content_boost = 0.0
        if strategy == 'table_focused' and match.metadata.get('contains_table'):
            content_boost += 0.2
        elif strategy == 'image_focused' and match.metadata.get('contains_image'):
            content_boost += 0.2
        elif strategy == 'text_focused' and match.metadata.get('text'):
            content_boost += 0.1
        
        # Keyword matching boost
        text = match.metadata.get('text', '').lower()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in text)
        keyword_boost = min(keyword_matches * 0.05, 0.3)
        
        # Conditional keyword adjustments
        conditional_boost = 0.0
        conditional_penalty = 0.0
        
        # Positive keyword boost
        for keyword in positive_keywords:
            if keyword.lower() in text:
                conditional_boost += 0.1
        
        # Negative keyword penalty
        for keyword in negative_keywords:
            if keyword.lower() in text:
                conditional_penalty += 0.2
        
        enhanced_score = base_score + content_boost + keyword_boost + conditional_boost - conditional_penalty
        return max(0.0, min(enhanced_score, 1.0))  # Ensure score is between 0 and 1
    
    def _calculate_enhanced_relevance_score(self, match, query_keywords: set, strategy: str) -> float:
        """Calculate enhanced relevance score based on content type and keyword matching."""
        base_score = match.score
        
        # Content type bonus
        content_bonus = 0.0
        if strategy == 'table_focused' and match.metadata.get('contains_table', False):
            content_bonus += 0.1
        elif strategy == 'image_focused' and match.metadata.get('contains_image', False):
            content_bonus += 0.1
        
        # Keyword matching bonus
        text = match.metadata.get('text', '').lower()
        keyword_matches = sum(1 for keyword in query_keywords if keyword in text)
        keyword_bonus = min(keyword_matches * 0.05, 0.2)  # Max 0.2 bonus
        
        enhanced_score = base_score + content_bonus + keyword_bonus
        return min(enhanced_score, 1.0)  # Cap at 1.0
    
    def _analyze_query_intent_with_cot(self, query: str) -> Dict[str, Any]:
        """Analyze query intent using chain-of-thought reasoning with AI, including conditional instructions."""
        try:
            cot_prompt = f"""You are an expert query analyst for a multimodal document retrieval system. Analyze this user query with deep understanding to optimize search and response generation.

QUERY: "{query}"

PERFORM COMPREHENSIVE ANALYSIS:

1. SEMANTIC INTENT ANALYSIS:
   - What is the user's underlying information need?
   - Is this factual, analytical, exploratory, comparative, or summarization?
   - What domain knowledge areas are involved?
   - What level of detail is expected?

2. CONTENT TYPE REQUIREMENTS:
   - Text: Does the user need textual explanations, definitions, or narratives?
   - Images: Are visual elements like figures, charts, diagrams, or photos needed?
   - Tables: Does the query require numerical data, statistics, or structured information?
   - Mixed: Would a combination provide the best answer?

3. SEARCH STRATEGY OPTIMIZATION:
   - Precision: User wants highly relevant, specific results
   - Recall: User wants comprehensive coverage, don't miss anything
   - Balanced: Standard approach balancing precision and recall
   - Semantic: Focus on conceptual similarity over keyword matching
   - Hybrid: Combine multiple approaches for complex queries

4. QUERY COMPLEXITY & SCOPE:
   - Simple: Single concept, direct question, factual lookup
   - Medium: Multiple related concepts, requires some analysis
   - Complex: Multi-faceted, requires synthesis, comparison, or deep analysis
   - Comprehensive: User wants complete coverage of topic
   - Specific: User wants targeted, focused information
   - Focused: User wants particular aspect or content type

5. RESPONSE SCOPE INTELLIGENCE (Critical - determines response size):
   - Minimal: Simple definition/fact questions (e.g., "What is decoder?", "What is X?", "Define Y", "How many?") - NO IMAGES/TABLES
   - Concise: Brief explanations (e.g., "Explain briefly", "Quick overview", "Summarize") - MINIMAL MULTIMEDIA  
   - Standard: Normal detailed response with multiple content types (most queries)
   - Detailed: Thorough analysis with multiple elements (e.g., "Compare", "Analyze deeply")  
   - Comprehensive: Complete coverage with all relevant content (e.g., "Everything about", "Complete guide")

6. CONDITIONAL LOGIC DETECTION:
   - Positive filters: "show", "include", "focus on", "with", "containing"
   - Negative filters: "without", "exclude", "not", "except", "avoid"
   - Content preferences: "only tables", "just images", "text only"
   - Quality requirements: "detailed", "brief", "comprehensive"

7. CONTEXTUAL UNDERSTANDING:
   - Temporal context: "recent", "historical", "current", "latest"
   - Comparative context: "compare", "versus", "difference", "similar"
   - Analytical context: "analyze", "evaluate", "assess", "interpret"
   - Visual context: "show me", "display", "visualize"

RESPOND WITH THIS EXACT JSON FORMAT:
{{
    "query_type": "factual|analytical|exploratory|comparative|summarization|visual|procedural",
    "search_strategy": "precision|recall|balanced|semantic|hybrid",
    "content_preferences": ["text", "images", "tables"],
    "complexity_level": "simple|medium|complex", 
    "scope": "comprehensive|specific|focused",
    "response_scope": "minimal|concise|standard|detailed|comprehensive",
    "semantic_concepts": ["concept1", "concept2", "concept3"],
    "intent_keywords": ["key1", "key2", "key3"],
    "conditional_instructions": {{
        "include_content": ["content_types_to_prioritize"],
        "exclude_content": ["content_types_to_avoid"],
        "positive_keywords": ["boost_these_terms"],
        "negative_keywords": ["downrank_these_terms"],
        "quality_requirements": ["detailed|brief|comprehensive"],
        "temporal_context": "recent|historical|current|null"
    }},
    "search_parameters": {{
        "top_k_multiplier": 1.0,
        "min_score_adjustment": 0.0,
        "diversity_requirement": "high|medium|low",
        "cross_content_boost": true,
        "semantic_expansion": true
    }},
    "confidence_score": 0.95,
    "reasoning": "Detailed explanation of analysis and strategy selection"
}}

Think step-by-step about what the user really wants and how to optimally retrieve it."""
            
            response = self.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": cot_prompt}],
                max_tokens=500,
                temperature=0.1
            )
            
            import json
            
            # Parse JSON response with better error handling
            response_content = response.choices[0].message.content.strip()
            logger.debug(f"LLM Query Intent Raw Response: {response_content[:300]}...")
            
            try:
                # Try to extract JSON if it's embedded in other text
                if '{' in response_content and '}' in response_content:
                    start_idx = response_content.find('{')
                    end_idx = response_content.rfind('}') + 1
                    json_content = response_content[start_idx:end_idx]
                    intent_analysis = json.loads(json_content)
                    logger.debug(f"LLM Intent Analysis Parsed: {json.dumps(intent_analysis, indent=2)}")
                else:
                    # Fallback if no JSON found
                    raise json.JSONDecodeError("No JSON found in response", response_content, 0)
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON from LLM response: {e}. Response: {response_content[:200]}...")
                logger.info("Using fallback heuristic query intent analysis instead of LLM")
                # Use enhanced fallback instead of simple fallback
                return self._fallback_intent_analysis_enhanced(query)
            
            # Validate and enrich the analysis
            intent_analysis = self._validate_and_enrich_intent(intent_analysis, query)
            
            logger.info(f"LLM-driven query intent analysis - Type: {intent_analysis.get('query_type')}, Strategy: {intent_analysis.get('search_strategy')}, Confidence: {intent_analysis.get('confidence_score')}, Response Scope: {intent_analysis.get('response_scope')}")
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Error in enhanced query intent analysis: {e}")
            # Fallback to advanced heuristic analysis
            return self._fallback_intent_analysis_enhanced(query)
    
    def _validate_and_enrich_intent(self, intent_analysis: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Validate and enrich the intent analysis with additional processing."""
        try:
            # Ensure all required fields are present
            required_fields = ['query_type', 'search_strategy', 'content_preferences', 'complexity_level', 'scope', 'response_scope']
            for field in required_fields:
                if field not in intent_analysis:
                    logger.warning(f"Missing required field {field} in intent analysis")
                    return self._fallback_intent_analysis_enhanced(query)
            
            # Validate search parameters and adjust if needed
            search_params = intent_analysis.get('search_parameters', {})
            
            # Auto-adjust parameters based on query characteristics
            query_length = len(query.split())
            if query_length > 15:  # Complex query
                search_params['top_k_multiplier'] = max(search_params.get('top_k_multiplier', 1.0), 1.2)
                search_params['diversity_requirement'] = 'high'
            elif query_length < 5:  # Simple query
                search_params['min_score_adjustment'] = max(search_params.get('min_score_adjustment', 0.0), 0.05)
                search_params['diversity_requirement'] = 'low'
            
            # Enrich with query-specific keywords if semantic_concepts is empty
            if not intent_analysis.get('semantic_concepts'):
                intent_analysis['semantic_concepts'] = [word for word in query.split() if len(word) > 3]
            
            if not intent_analysis.get('intent_keywords'):
                intent_analysis['intent_keywords'] = query.lower().split()
            
            # Ensure confidence score is reasonable
            if intent_analysis.get('confidence_score', 0.5) < 0.3:
                intent_analysis['confidence_score'] = 0.5
            
            intent_analysis['search_parameters'] = search_params
            
            return intent_analysis
            
        except Exception as e:
            logger.error(f"Error validating intent analysis: {e}")
            return self._fallback_intent_analysis_enhanced(query)
    
    def _fallback_intent_analysis_enhanced(self, query: str) -> Dict[str, Any]:
        """Enhanced fallback heuristic-based intent analysis with comprehensive classification."""
        query_lower = query.lower()
        words = query_lower.split()
        
        # Analyze query type based on patterns
        query_type = 'factual'  # default
        if any(word in query_lower for word in ['analyze', 'analysis', 'evaluate', 'assess', 'interpret']):
            query_type = 'analytical'
        elif any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'between']):
            query_type = 'comparative'
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview', 'brief']):
            query_type = 'summarization'
        elif any(word in query_lower for word in ['show', 'display', 'visualize', 'image', 'figure']):
            query_type = 'visual'
        elif any(word in query_lower for word in ['explore', 'discover', 'find out', 'investigate']):
            query_type = 'exploratory'
        
        # Determine search strategy
        search_strategy = 'balanced'  # default
        if any(word in query_lower for word in ['all', 'everything', 'complete', 'comprehensive']):
            search_strategy = 'recall'
        elif any(word in query_lower for word in ['exactly', 'precisely', 'specific', 'particular']):
            search_strategy = 'precision'
        elif query_type in ['analytical', 'comparative']:
            search_strategy = 'semantic'
        elif len(words) > 10:
            search_strategy = 'hybrid'
        
        # Let AI decide content preferences - provide all options by default
        content_preferences = ['text', 'images', 'tables']
        
        # Simple scope and complexity determination - let AI make detailed decisions
        if len(words) <= 4:
            scope = 'specific'
            complexity_level = 'simple'
        elif len(words) > 10:
            scope = 'comprehensive'
            complexity_level = 'complex'
        else:
            scope = 'balanced'
            complexity_level = 'medium'
        
        # Determine response scope for controlling response size
        response_scope = 'standard'  # default
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'how many', 'when']):
            response_scope = 'minimal'
        elif any(word in query_lower for word in ['briefly', 'quick', 'short', 'summary']):
            response_scope = 'concise'
        elif any(word in query_lower for word in ['detailed', 'thorough', 'comprehensive', 'analyze deeply']):
            response_scope = 'detailed'
        elif any(word in query_lower for word in ['everything about', 'complete guide', 'all aspects']):
            response_scope = 'comprehensive'
        elif len(words) <= 3:  # Very short queries default to minimal
            response_scope = 'minimal'
        elif complexity_level == 'simple':
            response_scope = 'concise'
        
        # Extract semantic concepts and keywords
        semantic_concepts = [word for word in words if len(word) > 3 and word not in ['what', 'how', 'why', 'when', 'where', 'which', 'show', 'tell', 'give']]
        intent_keywords = words
        
        # Build conditional instructions
        conditional_instructions = {
            'include_content': content_preferences,
            'exclude_content': [],
            'positive_keywords': semantic_concepts,
            'negative_keywords': [],
            'quality_requirements': ['detailed' if complexity_level == 'complex' else 'balanced'],
            'temporal_context': 'null'
        }
        
        # Detect negative instructions
        if any(word in query_lower for word in ['not', 'without', 'except', 'avoid']):
            # Simple negative keyword extraction
            negative_words = []
            for i, word in enumerate(words):
                if word in ['not', 'without', 'except'] and i + 1 < len(words):
                    negative_words.append(words[i + 1])
            conditional_instructions['negative_keywords'] = negative_words
        
        # Set search parameters
        search_parameters = {
            'top_k_multiplier': 1.2 if scope == 'comprehensive' else 0.8 if scope == 'focused' else 1.0,
            'min_score_adjustment': 0.05 if search_strategy == 'precision' else -0.02 if search_strategy == 'recall' else 0.0,
            'diversity_requirement': 'high' if scope == 'comprehensive' else 'low' if scope == 'focused' else 'medium',
            'cross_content_boost': True,
            'semantic_expansion': search_strategy in ['semantic', 'hybrid']
        }
        
        return {
            'query_type': query_type,
            'search_strategy': search_strategy,
            'content_preferences': content_preferences,
            'complexity_level': complexity_level,
            'scope': scope,
            'response_scope': response_scope,
            'semantic_concepts': semantic_concepts,
            'intent_keywords': intent_keywords,
            'conditional_instructions': conditional_instructions,
            'search_parameters': search_parameters,
            'confidence_score': 0.6,  # Lower confidence for fallback
            'reasoning': f'Fallback analysis based on heuristic patterns. Query type: {query_type}, Strategy: {search_strategy}, Response scope: {response_scope}'
        }
        
        logger.info(f"Heuristic-based query intent analysis - Type: {query_type}, Strategy: {search_strategy}, Response Scope: {response_scope}")
        return result
    
    def _fallback_intent_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback heuristic-based intent analysis."""
        query_lower = query.lower()
        
        # Determine scope
        if any(word in query_lower for word in ['all', 'everything', 'complete', 'comprehensive', 'entire', 'summarize', 'summary', 'overview']):
            scope = 'comprehensive'
        elif any(word in query_lower for word in ['what', 'how', 'why', 'explain', 'describe']):
            scope = 'specific'
        else:
            scope = 'focused'
        
        # Determine primary content type
        if any(word in query_lower for word in ['table', 'chart', 'data', 'metrics', 'numbers']):
            strategy = 'table_focused'
            primary_content = 'tables'
        elif any(word in query_lower for word in ['image', 'picture', 'visual', 'figure', 'diagram']):
            strategy = 'image_focused' 
            primary_content = 'images'
        elif scope == 'comprehensive':
            strategy = 'comprehensive_overview'
            primary_content = 'mixed'
        else:
            strategy = 'multimodal_balanced'
            primary_content = 'mixed'
        
        # Determine complexity
        word_count = len(query.split())
        if word_count <= 5:
            complexity = 'simple'
        elif word_count <= 12:
            complexity = 'medium'
        else:
            complexity = 'complex'
        
        return {
            'strategy': strategy,
            'scope': scope,
            'complexity': complexity,
            'primary_content': primary_content,
            'reasoning': 'Heuristic fallback analysis'
        }
    
    async def _get_llm_driven_content_limits(self, query: str, query_intent: Dict, available_content: Dict) -> Dict[str, List[int]]:
        """Use LLM to determine optimal content limits based on query analysis."""
        try:
            # Prepare content availability info
            content_info = {
                'available_text_chunks': available_content.get('text_count', 0),
                'available_images': available_content.get('image_count', 0), 
                'available_tables': available_content.get('table_count', 0)
            }
            
            prompt = f"""
            Analyze this query and determine optimal content limits for a comprehensive response:
            
            QUERY: "{query}"
            
            QUERY ANALYSIS:
            - Type: {query_intent.get('type', 'general')}
            - Strategy: {query_intent.get('strategy', 'balanced')}
            - Confidence: {query_intent.get('confidence', 0.8)}
            - Response Scope: {query_intent.get('response_scope', 'standard')}
            
            AVAILABLE CONTENT:
            - Text chunks: {content_info['available_text_chunks']}
            - Images: {content_info['available_images']}
            - Tables: {content_info['available_tables']}
            
            Based on the query complexity, type, and available content, determine the optimal number of:
            1. Text sections (min-max range)
            2. Images (min-max range)  
            3. Tables (min-max range)
            4. Suggested response length in tokens
            
            Consider:
            - Simple questions need fewer resources
            - Complex analytical questions need more text and tables
            - Visual/diagram questions need more images
            - Comparison questions need balanced content
            - Don't exceed available content
            
            Respond in JSON format:
            {{
                "text_sections": [min, max],
                "images": [min, max], 
                "tables": [min, max],
                "response_length_tokens": suggested_tokens,
                "reasoning": "Brief explanation for these limits"
            }}
            """
            
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1
            )
            
            # Parse LLM response with robust error handling
            import json
            import re
            
            response_content = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM content limits response: {response_content[:200]}...")
            
            # Try multiple parsing strategies
            try:
                # First try direct parsing
                llm_limits = json.loads(response_content)
            except json.JSONDecodeError:
                # Try to extract from markdown code blocks
                json_match = re.search(r'```(?:json)?\n(.*?)\n```', response_content, re.DOTALL)
                if json_match:
                    llm_limits = json.loads(json_match.group(1))
                else:
                    # Try to find JSON object pattern
                    json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
                    if json_match:
                        llm_limits = json.loads(json_match.group(0))
                    else:
                        raise ValueError("No valid JSON found in LLM content limits response")
            
            # Convert to expected format
            content_limits = {
                'text': llm_limits['text_sections'],
                'images': llm_limits['images'],
                'tables': llm_limits['tables']
            }
            
            # Add response length for later use
            content_limits['suggested_tokens'] = llm_limits.get('response_length_tokens', 2000)
            
            logger.info(f"LLM-determined content limits: {content_limits}")
            logger.info(f"LLM reasoning: {llm_limits.get('reasoning', 'No reasoning provided')}")
            
            return content_limits
            
        except Exception as e:
            logger.error(f"Failed to get LLM-driven content limits: {e}")
            # Fallback to reasonable defaults
            return {
                'text': [5, 15],
                'images': [1, 4], 
                'tables': [1, 3],
                'suggested_tokens': 2000
            }
    
    def _get_dynamic_content_limits(self, response_scope: str, is_comprehensive: bool, 
                                  is_analytical: bool, is_visual_focused: bool) -> Dict[str, List[int]]:
        """Legacy function - kept for backward compatibility."""
        # Base limits for different response scopes
        base_limits = {
            'minimal': {'text': [3, 6], 'images': [0, 2], 'tables': [0, 1]},
            'concise': {'text': [6, 12], 'images': [1, 2], 'tables': [0, 2]},
            'standard': {'text': [12, 20], 'images': [2, 4], 'tables': [1, 3]},
            'detailed': {'text': [18, 30], 'images': [3, 6], 'tables': [2, 4]},
            'comprehensive': {'text': [25, 40], 'images': [4, 8], 'tables': [3, 6]}
        }
        return base_limits.get(response_scope, base_limits['standard']).copy()
    
    def _apply_content_type_boosting(self, chunks: List[Dict], query_intent: Dict) -> List[Dict]:
        """Apply content-type specific boosting based on query intent."""
        strategy = query_intent.get('strategy', 'multimodal_balanced')
        
        for chunk in chunks:
            base_score = chunk['relevance_score']
            boost_factor = 1.0
            
            # Apply strategy-specific boosting
            if strategy == 'table_focused' and chunk['contains_table']:
                boost_factor = 1.5
            elif strategy == 'image_focused' and chunk['contains_image']:
                boost_factor = 1.5
            elif strategy == 'multimodal_balanced':
                if chunk['contains_image'] or chunk['contains_table']:
                    boost_factor = 1.2  # Slight preference for multimedia
            elif strategy == 'comprehensive_overview':
                if chunk['contains_image'] and chunk['contains_table']:
                    boost_factor = 1.8  # Strong preference for rich content
                elif chunk['contains_image'] or chunk['contains_table']:
                    boost_factor = 1.3
            
            chunk['boosted_score'] = base_score * boost_factor
        
        # Re-sort by boosted score
        chunks.sort(key=lambda x: x['boosted_score'], reverse=True)
        return chunks
    
    def _apply_diversity_filtering(self, chunks: List[Dict], query_intent: Dict) -> List[Dict]:
        """Apply diversity filtering to avoid redundant content."""
        diverse_chunks = []
        text_similarity_threshold = 0.8
        
        for chunk in chunks:
            page_num = chunk['page_number']
            
            # Skip if too many chunks from same page (unless comprehensive query)
            if query_intent.get('scope') != 'comprehensive':
                page_count = sum(1 for c in diverse_chunks if c['page_number'] == page_num)
                if page_count >= 2:  # Max 2 chunks per page for non-comprehensive queries
                    continue
            
            # Check text similarity with existing chunks
            is_duplicate = False
            chunk_text = chunk['text'][:200].lower()
            
            for existing in diverse_chunks:
                existing_text = existing['text'][:200].lower()
                if self._calculate_text_similarity(chunk_text, existing_text) > text_similarity_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                diverse_chunks.append(chunk)
        
        logger.info(f"Diversity filtering: {len(diverse_chunks)} unique chunks retained")
        return diverse_chunks
    
    def _apply_intelligent_selection(self, chunks: List[Dict], query: str, query_intent: Dict) -> List[Dict]:
        """Apply final intelligent selection based on query complexity and intent."""
        complexity = query_intent.get('complexity', 'medium')
        scope = query_intent.get('scope', 'specific')
        
        # Determine optimal chunk count based on complexity and scope
        if scope == 'comprehensive':
            max_chunks = min(25, len(chunks))  # Up to 25 for comprehensive
        elif complexity == 'complex':
            max_chunks = min(15, len(chunks))  # Up to 15 for complex queries
        elif complexity == 'medium':
            max_chunks = min(10, len(chunks))  # Up to 10 for medium queries
        else:
            max_chunks = min(6, len(chunks))   # Up to 6 for simple queries
        
        # Ensure content type balance for multimodal queries
        if query_intent.get('strategy') in ['multimodal_balanced', 'comprehensive_overview']:
            selected_chunks = self._ensure_content_balance(chunks[:max_chunks * 2], max_chunks)
        else:
            selected_chunks = chunks[:max_chunks]
        
        return selected_chunks
    
    def _ensure_content_balance(self, chunks: List[Dict], max_chunks: int) -> List[Dict]:
        """Ensure balanced representation of different content types."""
        text_chunks = [c for c in chunks if not c['contains_image'] and not c['contains_table']]
        image_chunks = [c for c in chunks if c['contains_image']]
        table_chunks = [c for c in chunks if c['contains_table']]
        mixed_chunks = [c for c in chunks if c['contains_image'] and c['contains_table']]
        
        # Prioritize mixed content, then distribute remaining slots
        selected = []
        
        # Add mixed content first (highest value)
        selected.extend(mixed_chunks[:min(3, len(mixed_chunks))])
        remaining_slots = max_chunks - len(selected)
        
        if remaining_slots > 0:
            # Distribute remaining slots proportionally
            image_slots = min(remaining_slots // 3, len(image_chunks))
            table_slots = min(remaining_slots // 3, len(table_chunks))
            text_slots = remaining_slots - image_slots - table_slots
            
            selected.extend(image_chunks[:image_slots])
            selected.extend(table_chunks[:table_slots])
            selected.extend(text_chunks[:text_slots])
        
        # Fill any remaining slots with highest scoring chunks
        if len(selected) < max_chunks:
            remaining_chunks = [c for c in chunks if c not in selected]
            selected.extend(remaining_chunks[:max_chunks - len(selected)])
        
        # Sort final selection by boosted score
        selected.sort(key=lambda x: x.get('boosted_score', x['relevance_score']), reverse=True)
        return selected[:max_chunks]
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings using simple word overlap."""
        if not text1 or not text2:
            return 0.0
        
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _apply_content_aware_selection(self, chunks: List[Dict], query: str, max_chunks: int) -> List[Dict]:
        """Apply content-aware chunk selection to ensure diverse content types."""
        if not chunks:
            return chunks
        
        # Check for "show all" type queries
        query_lower = query.lower()
        show_all_keywords = ['show all', 'all the', 'every', 'complete', 'entire', 'full', 'comprehensive']
        is_show_all_query = any(keyword in query_lower for keyword in show_all_keywords)
        
        if is_show_all_query:
            # For "show all" queries, return all chunks with relevant content
            logger.info(f"Show all query detected, returning all {len(chunks)} chunks")
            return chunks
        else:
            # Sort by enhanced relevance score for regular queries
            chunks.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            # Ensure diversity in content types
            selected_chunks = []
            content_types_seen = set()
            
            for chunk in chunks:
                content_type = 'text'
                if chunk['contains_table']:
                    content_type = 'table'
                elif chunk['contains_image']:
                    content_type = 'image'
                
                # Add if we haven't seen this content type or if it's highly relevant
                if content_type not in content_types_seen or chunk['relevance_score'] > 0.8:
                    selected_chunks.append(chunk)
                    content_types_seen.add(content_type)
                
                if len(selected_chunks) >= max_chunks:
                    break
            
            return selected_chunks
    
    def _generate_composite_response(self, query: str, relevant_chunks: List[Dict], conditional_instructions: Dict = None, response_scope: str = 'standard', suggested_max_tokens: int = 2500, query_intent: Dict = None) -> Dict[str, Any]:
        """Generate seamlessly integrated response using composite chunks and inline generator with conditional instructions."""
        try:
            if not relevant_chunks:
                return {
                    'inline_elements': [{"type": "text", "content": f"No relevant content found for '{query}'. Try rephrasing your question or using different keywords."}]
                }
            
            # Strengthen final generation with lightweight guardrails without changing pipeline logic
            guardrails = (
                "FINAL RESPONSE GUARDRAILS\n"
                "You must answer ONLY the current question using the retrieved evidence as the sole source of facts.\n\n"
                "ConversationIntent (non-factual): Use conversation context only to clarify scope/tone and focus (e.g., page cues like 'see page X', or focus like 'advantages'/'disadvantages'). Do NOT add facts from it.\n\n"
                "Grounding and page fidelity:\n"
                "- If the question includes 'see page X', prioritize evidence from page X.\n"
                "- If page X is not present in retrieved evidence, say so briefly and use the best nearby pages; mark clearly.\n"
                "- Cite inline as [Page N] after the sentences they support.\n\n"
                "Follow-up discipline:\n"
                "- If asked for 'advantages' or 'disadvantages', focus strictly on that and avoid the other unless explicitly requested.\n"
                "- If asked 'what were the previous questions?', list ONLY the last few questions (no answers), do not invent details.\n\n"
                "Tables and images:\n"
                "- Include a table/image only if it directly supports the answer and exists in retrieved evidence.\n"
            )
            
            # Skip redundant content strategy - AI limits already applied in main selection
            # Content is already filtered to 10 chunks by AI selection strategy (line 20 in logs)
            content_strategy = None
            logger.info("Skipping redundant content strategy - using AI-selected chunks directly")
            
            # Use the query intent passed from the main pipeline (avoid duplicate LLM call)
            if query_intent is None:
                logger.warning("Query intent not provided, will use basic defaults")
                query_intent = {}
            else:
                logger.info("Using query intent from main pipeline - avoiding duplicate LLM analysis")
            
            # Use inline generator to create seamless multimodal response
            return self.inline_generator.generate_inline_response_from_chunks(
                query=query,
                chunks=relevant_chunks,
                conditional_instructions=conditional_instructions,
                response_scope=response_scope,
                suggested_max_tokens=suggested_max_tokens,
                content_strategy=content_strategy,
                query_intent=query_intent
            )
            
        except Exception as e:
            logger.error(f"Error generating seamless composite response: {e}")
            return {
                'inline_elements': [{'type': 'text', 'content': f"Error generating response: {str(e)}"}]
            }
    
    async def _extract_multimodal_content_from_chunks_with_conditionals(self, chunks: List[Dict], query: str, conditional_instructions: Dict = None) -> Dict[str, List]:
        """Extract and organize multimodal content from composite chunks with AI-powered intelligent selection."""
        content = {
            'text': [],
            'images': [],
            'tables': []
        }
        
        used_image_ids = set()
        used_table_ids = set()
        
        # Extract conditional instructions
        include_content = []
        exclude_content = []
        if conditional_instructions:
            include_content = conditional_instructions.get('include_content', [])
            exclude_content = conditional_instructions.get('exclude_content', [])
        
        # AI-powered content organization and selection
        content_strategy = await self._determine_content_organization_strategy(query, chunks, conditional_instructions)
        
        # Apply AI-determined content organization
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i}: has_text={bool(chunk.get('text'))}, has_image={chunk.get('contains_image')}, has_image_url={bool(chunk.get('image_url'))}, has_table={chunk.get('contains_table')}, table_content={bool(chunk.get('table_content_json'))}")
            
            # Apply conditional filtering
            if exclude_content:
                if 'text' in exclude_content and chunk.get('text'):
                    continue
                if 'images' in exclude_content and chunk.get('contains_image'):
                    continue
                if 'tables' in exclude_content and chunk.get('contains_table'):
                    continue
            
            # Add text content with enhanced metadata
            if chunk.get('text') and self._should_include_text(chunk, content_strategy):
                content['text'].append({
                    'content': chunk['text'],
                    'page_number': chunk.get('page_number', 0),
                    'relevance_score': chunk.get('relevance_score', 0),
                    'chunk_type': 'composite',
                    'source': 'composite_chunk'
                })
            
            # Add image content if present and should be included
            if chunk.get('contains_image') and chunk.get('image_url'):
                should_include = self._should_include_image_content(chunk, content_strategy, used_image_ids)
                logger.info(f"Image chunk assessment: has_image={chunk.get('contains_image')}, has_url={bool(chunk.get('image_url'))}, should_include={should_include}, used_images={len(used_image_ids)}")
                if should_include:
                    image_id = f"img_{chunk['page_number']}_{hash(chunk['image_url'])}"
                    if image_id not in used_image_ids:
                        used_image_ids.add(image_id)
                        content['images'].append({
                        's3_url': chunk.get('image_url'),
                        'display_url': chunk.get('image_url'),
                        'page_number': chunk.get('page_number', 0),
                        'image_summary': chunk.get('image_summary', ''),
                        'ocr_text': '',  # Will be retrieved from original data if needed
                        'relevance_score': chunk.get('relevance_score', 0),
                        'source': 'composite_chunk'
                    })
            
            # Add table content if present and should be included
            if chunk.get('contains_table') and chunk.get('table_content_json'):
                should_include = self._should_include_table_content(chunk, content_strategy, used_table_ids)
                logger.info(f"Table chunk assessment: has_table={chunk.get('contains_table')}, has_content={bool(chunk.get('table_content_json'))}, should_include={should_include}, used_tables={len(used_table_ids)}")
                if should_include:
                    table_id = f"table_{chunk['page_number']}_{hash(chunk['table_content_json'])}"
                    if table_id not in used_table_ids:
                        used_table_ids.add(table_id)
                        content['tables'].append({
                        'table_content_json': chunk.get('table_content_json'),
                        'table_id': table_id,
                        'page_number': chunk.get('page_number', 0),
                        'relevance_score': chunk.get('relevance_score', 0)
                    })
        
        logger.info(f"Extracted multimodal content with AI selection: {len(content['text'])} text sections, {len(content['images'])} images, {len(content['tables'])} tables")
        return content
    
    def _should_include_image_content(self, chunk: Dict, content_strategy: Dict, used_image_ids: set) -> bool:
        """Enhanced image relevance scoring with OCR semantic matching and query-image scoring."""
        try:
            # Get basic image info
            image_summary = chunk.get('image_summary', '')
            ocr_text = chunk.get('ocr_text', '')
            relevance_score = chunk.get('relevance_score', 0)
            
            # Extract query from content_strategy context (if available)
            query_terms = getattr(self, '_current_query_terms', [])
            
            # 1. OCR Content Semantic Matching (LOW COMPLEXITY)
            ocr_relevance = 0.0
            if ocr_text and query_terms:
                ocr_words = set(ocr_text.lower().split())
                query_words = set(' '.join(query_terms).lower().split())
                
                # Calculate word overlap
                common_words = ocr_words.intersection(query_words)
                if query_words:
                    ocr_relevance = len(common_words) / len(query_words)
                    
            # 2. Query-Image Semantic Scoring (LOW COMPLEXITY) 
            summary_relevance = 0.0
            if image_summary and query_terms:
                summary_words = set(image_summary.lower().split())
                query_words = set(' '.join(query_terms).lower().split())
                
                # Calculate semantic overlap
                common_words = summary_words.intersection(query_words)
                if query_words:
                    summary_relevance = len(common_words) / len(query_words)
            
            # 3. Combined relevance score
            enhanced_score = (
                relevance_score * 0.4 +  # Original relevance
                ocr_relevance * 0.3 +    # OCR matching
                summary_relevance * 0.3   # Summary matching
            )
            
            # 4. Apply content strategy thresholds
            content_limits = content_strategy.get('content_limits', {})
            image_limits = content_limits.get('images', [1, 3])
            
            # Check if we've reached max images
            if len(used_image_ids) >= image_limits[1]:
                return False
            
            # Enhanced threshold based on combined scoring
            threshold = 0.15  # Lower threshold due to enhanced scoring
            should_include = enhanced_score > threshold
            
            if should_include:
                logger.debug(f"Image included: enhanced_score={enhanced_score:.3f} (original={relevance_score:.3f}, ocr={ocr_relevance:.3f}, summary={summary_relevance:.3f})")
            else:
                logger.debug(f"Image rejected: enhanced_score={enhanced_score:.3f} below threshold {threshold}")
                
            return should_include
            
        except Exception as e:
            logger.error(f"Error in image relevance scoring: {e}")
            # Fallback to basic relevance check
            return chunk.get('relevance_score', 0) > 0.2
    
    def _should_include_table_content(self, chunk: Dict, content_strategy: Dict, used_table_ids: set) -> bool:
        """Simple table inclusion logic - maintaining current filtering behavior."""
        try:
            # Apply content strategy limits
            content_limits = content_strategy.get('content_limits', {})
            table_limits = content_limits.get('tables', [1, 3])
            
            # Check if we've reached max tables
            if len(used_table_ids) >= table_limits[1]:
                return False
            
            # Use basic relevance threshold (keeping current logic)
            relevance_score = chunk.get('relevance_score', 0)
            return relevance_score > 0.2
            
        except Exception as e:
            logger.error(f"Error in table inclusion check: {e}")
            return chunk.get('relevance_score', 0) > 0.2
    
    async def _determine_content_organization_strategy(self, query: str, chunks: List[Dict], conditional_instructions: Dict = None) -> Dict:
        """Determine content organization strategy for seamless composite responses."""
        try:
            query_lower = query.lower()
            
            # Analyze query intent
            analytical_keywords = ['analysis', 'analyze', 'factors', 'results', 'findings', 'data', 'study']
            visual_keywords = ['show', 'display', 'image', 'picture', 'chart', 'graph', 'table']
            comprehensive_keywords = ['all', 'complete', 'entire', 'comprehensive', 'full']
            
            is_analytical = any(keyword in query_lower for keyword in analytical_keywords)
            is_visual_focused = any(keyword in query_lower for keyword in visual_keywords)
            is_comprehensive = any(keyword in query_lower for keyword in comprehensive_keywords)
            
            # Analyze available content
            has_tables = any(c.get('contains_table') for c in chunks)
            has_images = any(c.get('contains_image') for c in chunks)
            text_chunks = sum(1 for c in chunks if c.get('text'))
            
            # Get dynamic content limits based on response scope from query intent
            search_params = self._get_adaptive_search_params(query)
            query_intent = search_params.get('query_intent', {})
            response_scope = query_intent.get('response_scope', 'standard')
            
            # Count available content for LLM decision making
            available_content = {
                'text_count': sum(1 for c in chunks if c.get('text')),
                'image_count': sum(1 for c in chunks if c.get('contains_image')),
                'table_count': sum(1 for c in chunks if c.get('contains_table'))
            }
            
            # Use LLM to determine optimal content limits based on query analysis
            try:
                content_limits = await self._get_llm_driven_content_limits(query, query_intent, available_content)
                logger.info(f"LLM-driven content limits applied: {content_limits}")
            except Exception as e:
                logger.error(f"Failed to get LLM content limits, falling back to legacy method: {e}")
                content_limits = self._get_dynamic_content_limits(response_scope, is_comprehensive, is_analytical, is_visual_focused)
                logger.debug(f"Legacy content limits applied for response_scope '{response_scope}': {content_limits}")
            
            # Determine strategy based on query and content
            if is_comprehensive:
                selection_strategy = 'comprehensive'
                organization_strategy = 'comprehensive'
            elif is_analytical and text_chunks > 2:
                selection_strategy = 'focused'
                organization_strategy = 'analytical'
            elif is_visual_focused:
                selection_strategy = 'visual'
                organization_strategy = 'visual_priority'
            else:
                selection_strategy = 'balanced'
                organization_strategy = 'balanced'
            
            # Content priorities [min, max, threshold]
            content_priorities = {
                'text': [1, content_limits['text'][1], 0.3],
                'images': [0, content_limits['images'][1], 0.4],
                'tables': [0, content_limits['tables'][1], 0.4]
            }
            
            # Inclusion criteria [threshold, quality_focus]
            inclusion_criteria = {
                'text': [0.2 if organization_strategy == 'comprehensive' else 0.3, 'balanced'],
                'images': [0.3, 'balanced'],
                'tables': [0.4, 'balanced']
            }
            
            # Apply conditional instructions if provided
            if conditional_instructions:
                include_content = conditional_instructions.get('include_content', [])
                exclude_content = conditional_instructions.get('exclude_content', [])
                
                # Adjust priorities based on instructions
                if 'text' in exclude_content:
                    content_priorities['text'] = [0, 0, 1.0]
                elif 'text' in include_content:
                    content_priorities['text'][1] = max(content_priorities['text'][1], 3)
                    
                if 'images' in exclude_content:
                    content_priorities['images'] = [0, 0, 1.0]
                elif 'images' in include_content:
                    content_priorities['images'][1] = max(content_priorities['images'][1], 2)
                    
                if 'tables' in exclude_content:
                    content_priorities['tables'] = [0, 0, 1.0]
                elif 'tables' in include_content:
                    content_priorities['tables'][1] = max(content_priorities['tables'][1], 2)
            
            strategy = {
                'selection_strategy': selection_strategy,
                'organization_strategy': organization_strategy,
                'content_priorities': content_priorities,
                'inclusion_criteria': inclusion_criteria,
                'content_limits': content_limits,
                'suggested_max_tokens': content_limits.get('suggested_tokens', 2500)
            }
            
            logger.info(f"Content organization strategy: {selection_strategy}, priorities: {content_priorities}")
            return strategy
            
        except Exception as e:
            logger.error(f"Error determining content organization strategy: {e}")
            # Return default strategy
            return {
                'selection_strategy': 'balanced',
                'organization_strategy': 'balanced',
                'content_priorities': {
                    'text': [1, 3, 0.3],
                    'images': [0, 2, 0.4],
                    'tables': [0, 1, 0.4]
                },
                'inclusion_criteria': {
                    'text': [0.3, 'balanced'],
                    'images': [0.4, 'balanced'],
                    'tables': [0.4, 'balanced']
                },
                'content_limits': {'text': [2, 3], 'images': [1, 2], 'tables': [0, 1]}
            }
    
    def _determine_context_strategy(self, query: str, chunks: List[Dict]) -> str:
        """Determine the context strategy based on query type and available content."""
        try:
            query_lower = query.lower()
            
            # Analyze available content types
            has_tables = any(c.get('contains_table') for c in chunks)
            has_images = any(c.get('contains_image') for c in chunks)
            text_chunks = sum(1 for c in chunks if c.get('text'))
            
            # Check for comprehensive query indicators
            comprehensive_keywords = ['all', 'complete', 'entire', 'comprehensive', 'full', 'every', 'show all']
            is_comprehensive = any(keyword in query_lower for keyword in comprehensive_keywords)
            
            # Check for specific content type focus
            table_keywords = ['table', 'data', 'results', 'statistics', 'numbers', 'values']
            image_keywords = ['image', 'figure', 'chart', 'graph', 'visual', 'picture', 'diagram']
            summary_keywords = ['summary', 'overview', 'main points', 'key findings', 'brief']
            
            is_table_focused = any(keyword in query_lower for keyword in table_keywords) and has_tables
            is_image_focused = any(keyword in query_lower for keyword in image_keywords) and has_images
            is_summary_focused = any(keyword in query_lower for keyword in summary_keywords)
            
            # Determine strategy
            if is_comprehensive:
                return 'comprehensive_overview'
            elif is_table_focused:
                return 'table_focused'
            elif is_image_focused:
                return 'image_focused'
            elif is_summary_focused:
                return 'summary_focused'
            else:
                return 'balanced_narrative'
                
        except Exception as e:
            logger.error(f"Error determining context strategy: {e}")
            return 'balanced_narrative'  # Safe default
    
    def _select_optimal_chunks(self, chunks: List[Dict], strategy: str) -> List[Dict]:
        """Select optimal chunks based on enhanced context strategy for seamless integration."""
        if strategy == 'comprehensive_overview':
            # Include more chunks for comprehensive coverage
            return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)[:15]
        elif strategy == 'table_focused':
            # Prioritize chunks with tables
            return sorted(chunks, key=lambda x: (x.get('contains_table', False), x['relevance_score']), reverse=True)[:8]
        elif strategy == 'image_focused':
            # Prioritize chunks with images
            return sorted(chunks, key=lambda x: (x.get('contains_image', False), x['relevance_score']), reverse=True)[:8]
        elif strategy == 'summary_focused':
            # Take top chunks by relevance for summary
            return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)[:5]
        else:
            # Seamless narrative approach - balanced mix optimized for narrative flow
            # Sort by relevance but ensure mix of content types
            sorted_chunks = sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)
            selected = []
            
            # First, add highly relevant chunks regardless of type
            for chunk in sorted_chunks[:3]:
                selected.append(chunk)
            
            # Then add chunks with multimedia content for narrative richness
            remaining = sorted_chunks[3:]
            for chunk in remaining:
                if len(selected) >= 8:
                    break
                if chunk.get('contains_image') or chunk.get('contains_table'):
                    selected.append(chunk)
            
            # Fill remaining slots with text-only chunks if needed
            for chunk in remaining:
                if len(selected) >= 8:
                    break
                if chunk not in selected:
                    selected.append(chunk)
            
            return selected[:8]
    
    def _add_visual_content_to_response(self, chunks: List[Dict], inline_elements: List[Dict], used_content_ids: set, query: str):
        """Phase 2 Enhanced: Add visual content with improved deduplication for smaller chunks."""
        added_images = set()  # Track unique images by URL and ID
        added_tables = set()  # Track unique tables by content hash and ID
        processed_chunk_ids = set()  # Track processed chunks to avoid double processing
        
        # Determine if this is a table-focused query
        table_keywords = ['table', 'data', 'chart', 'graph', 'figure', 'statistics', 'numbers', 'values', 'columns', 'rows']
        query_lower = query.lower() if query else ''
        is_table_query = any(keyword in query_lower for keyword in table_keywords)
        
        logger.info(f"Query analysis: table_query={is_table_query}, query='{query_lower}'")
        
        # Check for "show all" type queries
        query_lower = query.lower()
        show_all_keywords = ['show all', 'all the', 'every', 'complete', 'entire', 'full', 'comprehensive']
        is_show_all_query = any(keyword in query_lower for keyword in show_all_keywords)
        
        # Phase 2 Enhancement: Determine chunk limit based on query type with better scaling
        if is_show_all_query:
            chunk_limit = len(chunks)
        elif is_table_query:
            chunk_limit = min(6, len(chunks))  # More chunks for table queries
        else:
            chunk_limit = min(4, len(chunks))  # Slightly increased default limit
        
        for chunk in chunks[:chunk_limit]:  # Use dynamic limit
            # Skip if we've already processed this chunk to avoid duplicates
            chunk_id = chunk.get('chunk_id', f"chunk_{chunk.get('page_number', 0)}")
            if chunk_id in processed_chunk_ids:
                continue
            processed_chunk_ids.add(chunk_id)
            # For table queries, prioritize tables over images
            if is_table_query and chunk['contains_table'] and chunk.get('table_content_json'):
                table_id = chunk.get('table_id', '')
                table_content_hash = hash(chunk['table_content_json'])
                if table_content_hash not in added_tables:
                    inline_elements.append({
                        'type': 'table',
                        'data': {
                            'page_number': chunk['page_number'],
                            'relevance_score': chunk['relevance_score'],
                            'table_content_json': chunk['table_content_json'],
                            'table_id': table_id,
                            'pdf_id': chunk.get('pdf_id', ''),
                            'table_markdown': '',  # Not stored in metadata
                            'table_html': ''  # Not stored in metadata
                        }
                    })
                    added_tables.add(table_content_hash)
                    logger.info(f"Added table to response: {table_id} from page {chunk['page_number']}")
                else:
                    logger.info(f"Skipped duplicate table: {table_id}")
            
            # For non-table queries, add images (Phase 2 enhanced deduplication)
            elif not is_table_query and chunk['contains_image']:
                # Handle both single image_url and multiple image URLs
                image_urls = []
                image_summaries = []
                image_ids = []
                
                # Check for single image format (legacy)
                if chunk.get('image_url'):
                    image_urls = [chunk.get('image_url')]
                    image_summaries = [chunk.get('image_summary', '')]
                    image_ids = [chunk.get('image_id', f"img_{chunk.get('page_number', 0)}")]
                
                # Check for multiple images format (new)
                elif chunk.get('image_s3_urls'):
                    image_urls = chunk.get('image_s3_urls', [])
                    image_summaries = chunk.get('image_summaries', [])
                    image_ids = chunk.get('image_ids', [])
                
                # Process each image with enhanced deduplication
                for i, image_url in enumerate(image_urls):
                    if not image_url:
                        continue
                        
                    image_summary = image_summaries[i] if i < len(image_summaries) else ''
                    image_id = image_ids[i] if i < len(image_ids) else f"img_{chunk.get('page_number', 0)}_{i}"
                    
                    # Create unique key combining URL and ID for better deduplication
                    unique_image_key = f"{image_url}_{image_id}_{chunk.get('page_number', 0)}"
                    
                    if unique_image_key not in added_images:
                        inline_elements.append({
                            'type': 'image',
                            'data': {
                                'display_url': image_url,
                                's3_url': image_url,
                                'local_path': '',
                                'page_number': chunk['page_number'],
                                'relevance_score': chunk['relevance_score'],
                                'image_summary': image_summary,
                                'image_id': image_id,
                                'ocr_text': '',
                                'pdf_id': chunk.get('pdf_id', '')
                            }
                        })
                        added_images.add(unique_image_key)
                        page_num = int(chunk['page_number']) if isinstance(chunk['page_number'], (int, float)) else chunk['page_number']
                        logger.info(f"Added unique image to response: {image_id} from page {page_num}")
                    else:
                        logger.info(f"Skipped duplicate image: {image_id} (key: {unique_image_key})")
            
            # For non-table queries, also add tables if available
            elif not is_table_query and chunk['contains_table'] and chunk.get('table_content_json'):
                table_id = chunk.get('table_id', '')
                table_content_hash = hash(chunk['table_content_json'])
                if table_content_hash not in added_tables:
                    inline_elements.append({
                        'type': 'table',
                        'data': {
                            'page_number': chunk['page_number'],
                            'relevance_score': chunk['relevance_score'],
                            'table_content_json': chunk['table_content_json'],
                            'table_id': table_id,
                            'pdf_id': chunk.get('pdf_id', ''),
                            'table_markdown': '',  # Not stored in metadata
                            'table_html': ''  # Not stored in metadata
                        }
                    })
                    added_tables.add(table_content_hash)
                    page_num = chunk['page_number']
                    # Convert page number to integer if it's a number
                    if isinstance(page_num, (int, float)):
                        page_num = int(page_num)
                    logger.info(f"Added table to response: {table_id} from page {page_num}")
                else:
                    logger.info(f"Skipped duplicate table: {table_id}")
    
    def _generate_ai_response_for_chunks(self, query: str, context: str, context_strategy: str = 'balanced', conditional_instructions: Dict = None) -> str:
        """Generate AI response using composite chunk context with adaptive strategy and conditional instructions."""
        try:
            # Build conditional instruction prompt
            conditional_prompt = ""
            if conditional_instructions:
                include_content = conditional_instructions.get('include_content', [])
                exclude_content = conditional_instructions.get('exclude_content', [])
                positive_keywords = conditional_instructions.get('positive_keywords', [])
                negative_keywords = conditional_instructions.get('negative_keywords', [])
                
                conditional_prompt = f"""

CONDITIONAL INSTRUCTIONS:
- INCLUDE: {', '.join(include_content) if include_content else 'All content types'}
- EXCLUDE: {', '.join(exclude_content) if exclude_content else 'None'}
- FOCUS ON: {', '.join(positive_keywords) if positive_keywords else 'All relevant content'}
- AVOID: {', '.join(negative_keywords) if negative_keywords else 'Nothing specific'}

IMPORTANT: Follow these conditional instructions strictly in your response."""

            # Adaptive system prompt based on context strategy
            if context_strategy == 'table_focused':
                system_prompt = f"""You are a document analysis assistant that analyzes tables and data STRICTLY from the provided document content.

ABSOLUTE CONSTRAINTS:
- Use ONLY the table data and text provided in the document chunks
- Do NOT add external knowledge about data analysis or general interpretations
- Every statement must reference specific values or text from the document

When responding to queries about tables and data:
1. Quote exact numbers and values from the document tables
2. Reference specific page numbers where data is found
3. Only describe patterns explicitly shown in the provided data
4. Use phrases like "The document shows the following data..." or "According to the table on page X..."
5. If asked about data not in the document, state it's not available

CRITICAL: Base all analysis solely on the document content provided.{conditional_prompt}"""
            
            elif context_strategy == 'image_focused':
                system_prompt = """You are a document analysis assistant that describes visual content STRICTLY from the provided document content.

ABSOLUTE CONSTRAINTS:
- Use ONLY the image descriptions and text provided in the document chunks
- Do NOT add external knowledge about visual analysis or general interpretations
- Every statement must reference specific descriptions or text from the document

When responding to queries about images and visual content:
1. Quote exact descriptions from the document about images
2. Reference specific page numbers where images are located
3. Only describe what is explicitly mentioned in the document text
4. Use phrases like "The document describes the image as..." or "According to the text on page X..."
5. If asked about visual elements not described in the document, state it's not available

CRITICAL: Base all descriptions solely on the document content provided."""
            
            elif context_strategy == 'summary_focused':
                system_prompt = """You are an expert research assistant creating comprehensive summaries of academic and technical documents. Your goal is to provide clear, well-structured summaries that capture the key information and insights from the document content.

SUMMARY QUALITY REQUIREMENTS:
1. **Comprehensive Coverage**: Include all major points and key information from the document
2. **Clear Organization**: Structure the summary logically with clear sections
3. **Specific Details**: Include important facts, numbers, and technical details
4. **Context and Relationships**: Show how different parts of the document relate to each other
5. **Accessible Language**: Write in clear, understandable language while maintaining technical accuracy

SUMMARY STRUCTURE:
1. **Overview**: Start with a high-level summary of the main topic or purpose
2. **Key Points**: Present the main findings, concepts, or arguments
3. **Technical Details**: Include specific technical information, methodologies, or data
4. **Visual Content**: Summarize what images and diagrams show and their significance
5. **Data Summary**: Highlight key data points from tables and their implications
6. **Conclusions**: Summarize main takeaways and implications

CONTENT GUIDELINES:
- Use information ONLY from the provided document chunks
- Reference specific page numbers when citing information
- Include important quotes when they add value
- Synthesize information from multiple sections
- Provide clear explanations of technical concepts
- Show relationships between different parts of the document

AVOID:
- Repetitive or robotic language
- Generic statements without specific details
- Overly formal or inaccessible language
- Vague summaries without concrete information

CRITICAL: Create informative, well-structured summaries that provide real value and understanding of the document content."""
            
            else:  # balanced
                system_prompt = """You are an expert research assistant analyzing academic and technical documents. Your goal is to provide clear, comprehensive, and insightful responses based on the document content provided.

RESPONSE QUALITY REQUIREMENTS:
1. **Comprehensive Analysis**: Provide thorough explanations that cover all relevant aspects of the query
2. **Clear Structure**: Organize your response logically with clear sections and flow
3. **Specific Details**: Include specific facts, numbers, and technical details from the document
4. **Contextual Understanding**: Explain concepts in context and show relationships between ideas
5. **Natural Language**: Write in a natural, engaging style that's easy to understand
6. **Technical Accuracy**: Maintain technical precision while being accessible

CONTENT GUIDELINES:
- Use information ONLY from the provided document chunks
- Reference specific page numbers when citing information
- Quote important passages when they add value to your explanation
- Synthesize information from multiple sections when relevant
- Provide clear explanations of technical concepts
- Connect related ideas and show how they fit together

RESPONSE STRUCTURE:
1. **Direct Answer**: Start with a clear, direct answer to the query
2. **Detailed Explanation**: Provide comprehensive details and context
3. **Technical Details**: Include specific technical information, facts, and numbers
4. **Visual References**: When images are mentioned, explain what they show and their significance
5. **Data References**: When tables are mentioned, explain key data points and their meaning
6. **Summary**: Conclude with key takeaways or implications

AVOID:
- Repetitive phrases like "According to the document..." in every sentence
- Generic statements without specific details
- Overly formal or robotic language
- Vague explanations without concrete information

CRITICAL: Create informative, engaging responses that provide real value to the user while staying grounded in the document content."""
            
            user_prompt = f"""DOCUMENT CONTENT TO ANALYZE:
{context}

QUERY: {query}

INSTRUCTIONS:
- Provide a comprehensive, well-structured response that directly answers the query
- Use ONLY information from the document content provided above
- Include specific technical details, facts, and numbers from the document
- Reference page numbers when citing specific information
- Quote important passages when they enhance your explanation
- Explain technical concepts clearly and in context
- Connect related ideas and show their relationships
- Write in a natural, engaging style that's easy to understand
- If information is not available in the document, clearly state this

RESPONSE REQUIREMENTS:
- Start with a clear, direct answer to the query
- Provide detailed explanations with specific examples from the document
- Include technical details and factual information
- Explain the significance and implications of the information
- Conclude with key insights or takeaways

Create a high-quality, informative response that provides real value to the user:"""
            
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000,  # Increased for more comprehensive responses
                temperature=0.3,  # Slightly higher for more creative and natural responses
                top_p=0.9
            )
            
            generated_response = response.choices[0].message.content
            
            # Verify the response is grounded in document content
            if self._verify_response_is_grounded(generated_response):
                return generated_response
            else:
                logger.warning("Response failed grounding check, using fallback")
                return f"According to the document content provided: {generated_response}"
            
        except Exception as e:
            logger.error(f"Error generating AI response for chunks: {e}")
            return f"Based on the document content, I can provide information about your query. The relevant sections have been identified and associated images and tables are displayed below."
    
    def _verify_response_is_grounded(self, response: str) -> bool:
        """Verify that the response is properly grounded in document content."""
        # Check for document-grounding phrases
        grounding_phrases = [
            "according to the document", "the document states", "the document shows",
            "the text states", "on page", "the provided content", "the document explains",
            "as mentioned in the document", "the document indicates", "based on the document",
            "the document content shows", "the provided document", "the text mentions"
        ]
        
        response_lower = response.lower()
        has_grounding_phrases = any(phrase in response_lower for phrase in grounding_phrases)
        
        # Check for warning signs of external knowledge
        warning_phrases = [
            "generally", "typically", "usually", "commonly", "in general",
            "it is known that", "research shows", "studies indicate", "experts believe",
            "this is because", "the reason is", "fundamentally", "essentially",
            "broadly speaking", "as we know", "it should be noted", "importantly"
        ]
        
        has_warning_phrases = any(phrase in response_lower for phrase in warning_phrases)
        
        # Response should have grounding phrases and minimal warning phrases
        is_grounded = has_grounding_phrases and not has_warning_phrases
        
        if not is_grounded:
            logger.info(f"Grounding verification: grounded_phrases={has_grounding_phrases}, warning_phrases={has_warning_phrases}")
        
        return is_grounded
    
    # Intent analysis removed as we now use composite chunks with unified retrieval
    
    # _generate_multimodal_response method removed - replaced by composite chunk strategy
    
    # _generate_ai_response method moved to _generate_ai_response_for_chunks
    def _identify_chunk_content_types(self, match) -> List[str]:
        """Identify what types of content a chunk contains."""
        content_types = []
        
        if not hasattr(match, 'metadata') or not match.metadata:
            return ['text']  # Default to text
        
        metadata = match.metadata
        
        # Always has text content if there's any content
        if metadata.get('text') or metadata.get('content'):
            content_types.append('text')
        
        # Check for images
        if metadata.get('contains_image') or metadata.get('image_url'):
            content_types.append('images')
        
        # Check for tables
        if metadata.get('contains_table') or metadata.get('table_content_json'):
            content_types.append('tables')
        
        return content_types if content_types else ['text']
    
    def _calculate_content_type_boost(self, chunk_content_types: List[str], content_type_weights: Dict[str, float]) -> float:
        """Calculate boost factor based on content type preferences."""
        if not chunk_content_types:
            return 1.0
        
        # Use the highest weight among the content types in this chunk
        max_weight = max(content_type_weights.get(content_type, 1.0) for content_type in chunk_content_types)
        return max_weight
    
    def _calculate_enhanced_relevance_score_advanced(self, match, query_keywords: set, strategy: str, 
                                                   positive_keywords: List[str], negative_keywords: List[str],
                                                   semantic_concepts: List[str], intent_keywords: List[str], 
                                                   content_type_boost: float) -> float:
        """Calculate advanced relevance score with multiple signals and content type preferences."""
        if not hasattr(match, 'metadata') or not match.metadata:
            return match.score
        
        base_score = match.score
        content = match.metadata.get('content', '').lower()
        
        # Keyword matching signals
        keyword_score = 0.0
        if content:
            # Original query keywords
            query_matches = sum(1 for kw in query_keywords if kw in content)
            keyword_score += (query_matches / max(len(query_keywords), 1)) * 0.3
            
            # Intent keywords
            intent_matches = sum(1 for kw in intent_keywords if kw.lower() in content)
            keyword_score += (intent_matches / max(len(intent_keywords), 1)) * 0.2
            
            # Semantic concepts
            concept_matches = sum(1 for concept in semantic_concepts if concept.lower() in content)
            keyword_score += (concept_matches / max(len(semantic_concepts), 1)) * 0.25
            
            # Positive keywords boost
            positive_matches = sum(1 for kw in positive_keywords if kw.lower() in content)
            keyword_score += (positive_matches / max(len(positive_keywords), 1)) * 0.15
            
            # Negative keywords penalty
            negative_matches = sum(1 for kw in negative_keywords if kw.lower() in content)
            keyword_score -= (negative_matches / max(len(negative_keywords), 1)) * 0.1
        
        # Content quality signals
        quality_score = 0.0
        if content:
            # Content length (longer content often more informative, but not always)
            content_length = len(content)
            if 100 <= content_length <= 1000:
                quality_score += 0.1
            elif content_length > 1000:
                quality_score += 0.05
            
            # Presence of structured information
            if any(indicator in content for indicator in ['table', 'figure', 'chart', 'data', 'result']):
                quality_score += 0.05
        
        # Image relevance enhancement using new metadata arrays
        image_relevance_boost = 0.0
        if match.metadata.get('contains_image') and match.metadata.get('image_summaries'):
            image_summaries = match.metadata.get('image_summaries', [])
            query_text = ' '.join(query_keywords).lower()
            
            # Calculate semantic similarity between query and image summaries
            for image_summary in image_summaries:
                if image_summary:
                    image_summary_lower = image_summary.lower()
                    
                    # Direct keyword matching in image summary
                    summary_keyword_matches = sum(1 for kw in query_keywords if kw in image_summary_lower)
                    if summary_keyword_matches > 0:
                        image_relevance_boost += (summary_keyword_matches / len(query_keywords)) * 0.15
                    
                    # Semantic concept matching in image summaries
                    summary_concept_matches = sum(1 for concept in semantic_concepts if concept.lower() in image_summary_lower)
                    if summary_concept_matches > 0:
                        image_relevance_boost += (summary_concept_matches / max(len(semantic_concepts), 1)) * 0.1
                    
                    # Intent keyword matching in image summaries
                    summary_intent_matches = sum(1 for intent_kw in intent_keywords if intent_kw.lower() in image_summary_lower)
                    if summary_intent_matches > 0:
                        image_relevance_boost += (summary_intent_matches / max(len(intent_keywords), 1)) * 0.08
                    
                    # Visual content type indicators boost
                    visual_indicators = ['diagram', 'chart', 'graph', 'illustration', 'figure', 'visualization', 'plot', 'architecture', 'model', 'flow', 'structure']
                    visual_matches = sum(1 for indicator in visual_indicators if indicator in image_summary_lower)
                    if visual_matches > 0 and any(kw in query_text for kw in ['show', 'illustrate', 'diagram', 'visual', 'architecture', 'model', 'structure']):
                        image_relevance_boost += min(visual_matches * 0.05, 0.2)
            
            # Cap image relevance boost
            image_relevance_boost = min(image_relevance_boost, 0.3)
            
            # Log significant image relevance boosts for debugging
            if image_relevance_boost > 0.1:
                logger.debug(f"Image relevance boost applied: {image_relevance_boost:.3f} for chunk with {len(image_summaries)} images")
        
        # Table relevance enhancement using new metadata arrays
        table_relevance_boost = 0.0
        if match.metadata.get('contains_table') and match.metadata.get('table_summaries'):
            table_summaries = match.metadata.get('table_summaries', [])
            table_content_jsons = match.metadata.get('table_content_jsons', [])
            query_text = ' '.join(query_keywords).lower()
            
            # Calculate semantic similarity between query and table summaries
            for i, table_summary in enumerate(table_summaries):
                if table_summary:
                    table_summary_lower = table_summary.lower()
                    
                    # Direct keyword matching in table summary
                    summary_keyword_matches = sum(1 for kw in query_keywords if kw in table_summary_lower)
                    if summary_keyword_matches > 0:
                        table_relevance_boost += (summary_keyword_matches / len(query_keywords)) * 0.18
                    
                    # Semantic concept matching in table summaries
                    summary_concept_matches = sum(1 for concept in semantic_concepts if concept.lower() in table_summary_lower)
                    if summary_concept_matches > 0:
                        table_relevance_boost += (summary_concept_matches / max(len(semantic_concepts), 1)) * 0.12
                    
                    # Intent keyword matching in table summaries
                    summary_intent_matches = sum(1 for intent_kw in intent_keywords if intent_kw.lower() in table_summary_lower)
                    if summary_intent_matches > 0:
                        table_relevance_boost += (summary_intent_matches / max(len(intent_keywords), 1)) * 0.1
                    
                    # Data content type indicators boost
                    data_indicators = ['data', 'table', 'statistics', 'results', 'values', 'numbers', 'measurements', 'comparison', 'analysis', 'findings']
                    data_matches = sum(1 for indicator in data_indicators if indicator in table_summary_lower)
                    if data_matches > 0 and any(kw in query_text for kw in ['data', 'table', 'statistics', 'results', 'analysis', 'comparison', 'values']):
                        table_relevance_boost += min(data_matches * 0.06, 0.25)
                    
                    # Table content quality assessment using JSON data
                    if i < len(table_content_jsons) and table_content_jsons[i]:
                        try:
                            import json
                            table_data = json.loads(table_content_jsons[i])
                            
                            # Assess table data richness
                            metadata = table_data.get('_metadata', {})
                            total_rows = metadata.get('total_rows', 0)
                            total_columns = metadata.get('total_columns', 0)
                            
                            # Boost for data-rich tables
                            if total_rows >= 3 and total_columns >= 2:
                                table_relevance_boost += 0.08
                            
                            # Boost for tables with meaningful data (not exercise tables)
                            table_keys = [k for k in table_data.keys() if not k.startswith('_')]
                            non_empty_columns = 0
                            for key in table_keys[:5]:  # Check first 5 columns
                                column_data = table_data.get(key, [])
                                if isinstance(column_data, list) and len(column_data) > 1:
                                    # Check if column has substantial non-empty data
                                    non_empty_cells = sum(1 for cell in column_data if str(cell).strip() and len(str(cell).strip()) > 1)
                                    if non_empty_cells >= 2:
                                        non_empty_columns += 1
                            
                            # Substantial data boost
                            if non_empty_columns >= 2:
                                table_relevance_boost += 0.12
                            elif non_empty_columns >= 1:
                                table_relevance_boost += 0.06
                            
                            # Penalty for likely exercise/worksheet tables
                            exercise_indicators = ['exercise', 'practice', 'worksheet', 'assignment', 'homework', 'question']
                            if any(indicator in table_summary_lower for indicator in exercise_indicators):
                                table_relevance_boost *= 0.7
                                
                        except (json.JSONDecodeError, TypeError, KeyError) as e:
                            logger.debug(f"Error parsing table content JSON for relevance scoring: {e}")
            
            # Cap table relevance boost
            table_relevance_boost = min(table_relevance_boost, 0.35)
            
            # Log significant table relevance boosts for debugging
            if table_relevance_boost > 0.1:
                logger.debug(f"Table relevance boost applied: {table_relevance_boost:.3f} for chunk with {len(table_summaries)} tables")
        
        # Strategy-specific adjustments
        strategy_multiplier = 1.0
        if strategy == 'precision':
            strategy_multiplier = 1.2 if keyword_score > 0.3 else 0.8
        elif strategy == 'recall':
            strategy_multiplier = 1.1  # Slight boost for broader coverage
        elif strategy == 'semantic':
            # Boost for semantic concept matches
            if concept_matches > 0:
                strategy_multiplier = 1.15
        
        # Combine all signals with better balance, including image and table relevance
        # Ensure base score dominates to prevent over-penalization
        final_score = (
            base_score * 0.6 +  # Reduced base score to accommodate both image and table boosts
            keyword_score * 0.2 +  # Keyword weight
            quality_score * 0.1 +   # Quality weight
            image_relevance_boost * 0.05 +  # Image relevance enhancement
            table_relevance_boost * 0.05    # Table relevance enhancement
        ) * content_type_boost * strategy_multiplier
        
        # Ensure minimum score is not too low
        final_score = max(final_score, base_score * 0.5)
        
        return min(final_score, 1.0)  # Cap at 1.0

    def display_multimodal_results(self, response_data: Dict[str, Any]):
        """Display intelligent inline multimodal results in Streamlit UI."""
        try:
            # Check if we have inline elements (new format)
            if 'inline_elements' in response_data:
                # Debug logging
                inline_elements = response_data['inline_elements']
                logger.info(f"Displaying {len(inline_elements)} inline elements")
                
                # Count different types
                text_count = sum(1 for elem in inline_elements if elem.get('type') == 'text')
                image_count = sum(1 for elem in inline_elements if elem.get('type') == 'image')
                table_count = sum(1 for elem in inline_elements if elem.get('type') == 'table')
                
                logger.info(f"Element breakdown: {text_count} text, {image_count} images, {table_count} tables")
                
                # New inline format
                self.inline_generator.display_inline_response(inline_elements)
            else:
                # Fallback display
                st.subheader("🤖 AI Response")
                st.write(response_data.get('response', 'No response available'))
                
            # Performance metrics removed from display
            
        except Exception as e:
            st.error(f"Error displaying multimodal results: {e}")
            logger.error(f"Display error: {e}")
    
    def _display_processing_results(self, results: Dict[str, Any], processing_time: float):
        """Display processing results summary."""
        try:
            st.success(f"✅ Multimodal processing completed in {processing_time:.2f}s")
            
            # Create metrics display
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                chunk_count = results.get('composite', {}).get('processed_chunks', 0)
                st.metric("Composite Chunks", chunk_count)
            
            with col2:
                image_count = results.get('images', {}).get('saved_count', 0)
                st.metric("Images Processed", image_count)
            
            with col3:
                table_count = results.get('tables', {}).get('saved_count', 0)
                st.metric("Tables Processed", table_count)
            
            with col4:
                st.metric("Processing Time", f"{processing_time:.1f}s")
            
            # Detailed processing results removed from display
                
        except Exception as e:
            logger.error(f"Error displaying processing results: {e}")
    
    def _extract_images_from_results(self, json_objs: List[Dict], pdf_id: str, pdf_name: str):
        """Extract and save images from LlamaParse JSON results with comprehensive detection."""
        try:
            # Check if S3 is available through image processor
            use_s3 = hasattr(self.image_processor, 'use_s3') and self.image_processor.use_s3
            
            if use_s3:
                from config import get_pdf_s3_prefix
                import tempfile
                import base64
                import re
                import requests
                
                # Use S3 storage
                s3_prefix = get_pdf_s3_prefix(pdf_name)
                # Create temporary directory for processing
                temp_dir = Path(tempfile.mkdtemp())
                pdf_image_dir = temp_dir
                logger.info(f"Using S3 storage with prefix: {s3_prefix}")
            else:
                from config import get_pdf_image_dir
                import base64
                import re
                import requests
                
                # Use local storage
                pdf_image_dir = get_pdf_image_dir(pdf_name)
                logger.info(f"Using local storage: {pdf_image_dir}")
            
            extracted_count = 0
            json_data = json_objs[0] if json_objs else {}
            
            logger.info(f"Analyzing LlamaParse JSON structure for image extraction")
            
            # Method 1: Look for base64 images in markdown/text content
            all_text = []
            
            # Collect all text/markdown content
            for page_data in json_data.get('pages', []):
                page_text = page_data.get('text', '') + page_data.get('md', '')
                all_text.append(page_text)
            
            full_content = '\n'.join(all_text)
            
            # More comprehensive base64 pattern matching
            base64_patterns = [
                r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)',  # Standard data URL
                r'!\(.*\[\(data:image/([^;]+);base64,([A-Za-z0-9+/=]+)\)\]',  # Markdown image
                r'<img[^>]*src="data:image/([^;]+);base64,([A-Za-z0-9+/=]+)"[^>]*>',  # HTML img tag
                r'([A-Za-z0-9+/]{50,}={0,2})'  # Standalone base64 strings (50+ chars)
            ]
            
            for pattern_idx, pattern in enumerate(base64_patterns):
                matches = re.findall(pattern, full_content)
                logger.info(f"Pattern {pattern_idx+1} found {len(matches)} matches")
                
                for i, match in enumerate(matches):
                    try:
                        if len(match) == 2:  # Format and data
                            format_type, base64_data = match
                        elif len(match) == 3:  # Markdown/HTML format
                            format_type, base64_data = match[0], match[1]
                        else:  # Standalone base64
                            format_type, base64_data = 'png', match[0] if isinstance(match, tuple) else match
                        
                        # Decode and save
                        image_data = base64.b64decode(base64_data)
                        
                        # Validate it's actually an image (check magic bytes)
                        if len(image_data) < 10:
                            continue
                            
                        # Determine extension
                        ext = f".{format_type.lower()}"
                        if ext not in ['.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp']:
                            ext = '.png'
                        
                        # Save image
                        image_path = pdf_image_dir / f"{pdf_id}_b64_{pattern_idx}_{i}{ext}"
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        
                        extracted_count += 1
                        logger.info(f"Extracted base64 image: {image_path}")
                        
                    except Exception as e:
                        logger.warning(f"Failed to decode base64 match {i}: {e}")
            
            # Method 2: Look for image objects in pages structure
            for page_data in json_data.get('pages', []):
                page_num = page_data.get('page', 0)
                
                # Check multiple possible image containers
                image_containers = [
                    page_data.get('images', []),
                    page_data.get('figures', []),
                    page_data.get('media', []),
                    page_data.get('elements', [])
                ]
                
                for container_idx, container in enumerate(image_containers):
                    if not container:
                        continue
                        
                    for img_idx, img_data in enumerate(container):
                        if not isinstance(img_data, dict):
                            continue
                            
                        # Try multiple image data fields
                        for data_field in ['data', 'image', 'base64', 'content', 'src', 'url', 'path']:
                            if data_field not in img_data or not img_data[data_field]:
                                continue
                                
                            try:
                                image_content = img_data[data_field]
                                image_bytes = None
                                
                                if isinstance(image_content, str):
                                    if image_content.startswith('data:image'):
                                        # Data URL
                                        _, b64_data = image_content.split(',', 1)
                                        image_bytes = base64.b64decode(b64_data)
                                    elif image_content.startswith('http'):
                                        # URL - try downloading
                                        try:
                                            response = requests.get(image_content, timeout=10)
                                            if response.status_code == 200:
                                                image_bytes = response.content
                                        except:
                                            pass
                                    elif len(image_content) > 100:
                                        # Try as base64
                                        try:
                                            image_bytes = base64.b64decode(image_content)
                                        except:
                                            pass
                                elif isinstance(image_content, bytes):
                                    image_bytes = image_content
                                
                                if image_bytes and len(image_bytes) > 10:
                                    # Save image
                                    image_path = pdf_image_dir / f"{pdf_id}_p{page_num}_c{container_idx}_i{img_idx}.png"
                                    with open(image_path, 'wb') as f:
                                        f.write(image_bytes)
                                    
                                    extracted_count += 1
                                    logger.info(f"Extracted page image: {image_path}")
                                    break
                                    
                            except Exception as e:
                                logger.debug(f"Failed to extract from {data_field}: {e}")
                                continue
            
            # Method 3: Create enhanced content placeholders with structured OCR data
            if extracted_count == 0:
                logger.info("No binary images found, creating enhanced content placeholders...")
                
                for page_data in json_data.get('pages', []):
                    page_num = page_data.get('page', 0)
                    
                    # Look for image references or OCR data
                    if 'images' in page_data:
                        for img_idx, img_data in enumerate(page_data['images']):
                            if isinstance(img_data, dict):
                                # Extract OCR text for processing but don't create text files
                                ocr_text = self._extract_ocr_from_image_data(img_data)
                                
                                # Just log that we found image metadata, don't create files
                                extracted_count += 1
                                logger.info(f"Found image metadata on page {page_num + 1}, image {img_idx}")
            
            return extracted_count
            
        except Exception as e:
            logger.error(f"Error extracting images from results: {e}")
            return 0
    
    def _extract_ocr_from_image_data(self, img_data: dict) -> str:
        """Extract clean OCR text from image data structure."""
        ocr_texts = []
        
        # Method 1: Direct text field
        if 'text' in img_data and img_data['text']:
            ocr_texts.append(img_data['text'])
        
        # Method 2: OCR array with confidence filtering
        if 'ocr' in img_data and isinstance(img_data['ocr'], list):
            for ocr_item in img_data['ocr']:
                if isinstance(ocr_item, dict):
                    text = ocr_item.get('text', '').strip()
                    confidence = ocr_item.get('confidence', 0)
                    
                    # Only include high-confidence text
                    if text and confidence > 0.5:
                        ocr_texts.append(text)
        
        # Method 3: Other possible text fields
        for field in ['extracted_text', 'content', 'ocr_text']:
            if field in img_data and img_data[field]:
                ocr_texts.append(str(img_data[field]))
        
        # Clean and combine
        if ocr_texts:
            combined = ' '.join(ocr_texts)
            # Remove excessive whitespace
            return ' '.join(combined.split())
        
        return ""
    
    def _get_current_job_id(self) -> Optional[str]:
        """Get the current job ID from LlamaParse parser with enhanced detection."""
        try:
            parser = self.llamaparse_parser
            
            # Method 1: Check if job_id is available in JSON results
            if hasattr(parser, '_last_result') and parser._last_result:
                if isinstance(parser._last_result, list) and len(parser._last_result) > 0:
                    result = parser._last_result[0]
                    if isinstance(result, dict) and 'job_id' in result:
                        job_id = result['job_id']
                        logger.info(f"Found job ID in last result: {job_id}")
                        return job_id
            
            # Method 2: Check parser attributes with comprehensive search
            job_id_attrs = [
                '_last_job_id', 'job_id', 'current_job_id', '_job_id',
                'latest_job_id', '_current_job_id', 'active_job_id',
                'last_job_id', 'recent_job_id', '_job_uuid'
            ]
            
            for attr in job_id_attrs:
                if hasattr(parser, attr):
                    job_id = getattr(parser, attr)
                    if job_id and isinstance(job_id, str):
                        logger.info(f"Found job ID in parser.{attr}: {job_id}")
                        return job_id
            
            # Method 3: Check if parser has a client with job info
            clients = []
            if hasattr(parser, 'client'):
                clients.append(parser.client)
            if hasattr(parser, '_client'):
                clients.append(parser._client)
            if hasattr(parser, 'api_client'):
                clients.append(parser.api_client)
            
            for client in clients:
                if client:
                    for attr in job_id_attrs:
                        if hasattr(client, attr):
                            job_id = getattr(client, attr)
                            if job_id and isinstance(job_id, str):
                                logger.info(f"Found job ID in client.{attr}: {job_id}")
                                return job_id
            
            # Method 4: Deep inspection of parser's __dict__
            if hasattr(parser, '__dict__'):
                for key, value in parser.__dict__.items():
                    if isinstance(value, str):
                        # Check for UUID format (typical for job IDs)
                        if len(value) >= 32 and '-' in value:
                            # Pattern like: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
                            parts = value.split('-')
                            if len(parts) >= 4 and all(len(p) >= 4 for p in parts):
                                logger.info(f"Found potential job ID in {key}: {value}")
                                return value
                        # Also check for other formats
                        elif 'job' in key.lower() and len(value) > 10:
                            logger.info(f"Found potential job ID in {key}: {value}")
                            return value
            
            # Method 5: Check recent activity or logs if available
            if hasattr(parser, '_history') or hasattr(parser, '_recent_jobs'):
                history = getattr(parser, '_history', None) or getattr(parser, '_recent_jobs', None)
                if history and isinstance(history, (list, dict)):
                    logger.info(f"Found history/recent_jobs, analyzing...")
                    if isinstance(history, list) and history:
                        last_item = history[-1]
                        if isinstance(last_item, dict) and 'job_id' in last_item:
                            return last_item['job_id']
                    elif isinstance(history, dict) and 'latest' in history:
                        return history['latest']
            
            logger.warning("Could not find job ID using any detection method")
            return None
            
        except Exception as e:
            logger.error(f"Error getting current job ID: {e}")
            return None
    
    def _download_images_official_method(self, job_id: str, pdf_id: str, pdf_name: str):
        """Download images using the official LlamaParse API method based on documentation."""
        try:
            import requests
            from config import IMAGES_DIR
            
            # Create PDF-specific image directory
            pdf_image_dir = IMAGES_DIR / pdf_id
            pdf_image_dir.mkdir(parents=True, exist_ok=True)
            
            # Get API key
            api_key = os.getenv('LLAMA_PARSE_API_KEY')
            if not api_key:
                logger.error("LLAMA_PARSE_API_KEY not found")
                return
            
            # Official LlamaParse API base URL
            base_url = "https://api.cloud.llamaindex.ai/api/v1/parsing"
            headers = {"Authorization": f"Bearer {api_key}"}
            
            logger.info(f"Downloading images from job {job_id} using official API")
            
            # First, get the job result to find available images
            try:
                job_result_url = f"{base_url}/job/{job_id}/result"
                logger.info(f"Getting job result from: {job_result_url}")
                
                response = requests.get(job_result_url, headers=headers, timeout=60)
                if response.status_code != 200:
                    logger.warning(f"Could not get job result: {response.status_code}")
                    return
                
                result_data = response.json()
                logger.info(f"Job result keys: {list(result_data.keys()) if isinstance(result_data, dict) else 'array'}")
                
                # Extract image names from the result
                image_names = []
                
                # Look for images in the pages structure
                pages = result_data.get('pages', [])
                for page in pages:
                    page_images = page.get('images', [])
                    for img in page_images:
                        if isinstance(img, dict) and 'name' in img:
                            image_names.append(img['name'])
                            logger.info(f"Found image: {img['name']} (type: {img.get('type', 'unknown')})")
                
                logger.info(f"Total images found in job result: {len(image_names)}")
                
                # Download each image using the official API endpoint format
                images_downloaded = 0
                for image_name in image_names:
                    try:
                        # Official endpoint format from documentation
                        image_url = f"{base_url}/job/{job_id}/result/image/{image_name}"
                        logger.info(f"Downloading image: {image_url}")
                        
                        img_response = requests.get(image_url, headers=headers, timeout=30)
                        
                        if img_response.status_code == 200:
                            # Determine file extension from image name or content type
                            file_extension = Path(image_name).suffix
                            if not file_extension:
                                content_type = img_response.headers.get('content-type', '')
                                if 'jpeg' in content_type or 'jpg' in content_type:
                                    file_extension = '.jpg'
                                elif 'png' in content_type:
                                    file_extension = '.png'
                                else:
                                    file_extension = '.png'  # default
                            
                            # Save with clean filename
                            clean_name = Path(image_name).stem
                            image_path = pdf_image_dir / f"{pdf_id}_{clean_name}{file_extension}"
                            
                            with open(image_path, 'wb') as f:
                                f.write(img_response.content)
                            
                            images_downloaded += 1
                            logger.info(f"Downloaded: {image_path} ({len(img_response.content)} bytes)")
                            
                        else:
                            logger.warning(f"Failed to download {image_name}: {img_response.status_code}")
                            
                    except Exception as e:
                        logger.error(f"Error downloading image {image_name}: {e}")
                        continue
                
                if images_downloaded > 0:
                    logger.info(f"Successfully downloaded {images_downloaded} images using official API")
                else:
                    logger.warning("No images were successfully downloaded")
                    # Create debug info to help troubleshoot
                    self._create_download_debug_info(job_id, pdf_image_dir, pdf_id, result_data)
                
            except Exception as e:
                logger.error(f"Error getting job result: {e}")
                
        except Exception as e:
            logger.error(f"Error in official image download: {e}")
    
    def _create_download_debug_info(self, job_id: str, save_dir: Path, pdf_id: str, result_data: dict):
        """Create debug information for image download troubleshooting."""
        try:
            debug_path = save_dir / f"{pdf_id}_download_debug.json"
            
            debug_info = {
                "job_id": job_id,
                "pdf_id": pdf_id,
                "timestamp": datetime.now().isoformat(),
                "result_structure": {
                    "top_level_keys": list(result_data.keys()) if isinstance(result_data, dict) else [],
                    "pages_count": len(result_data.get('pages', [])) if isinstance(result_data, dict) else 0
                },
                "images_found": []
            }
            
            # Analyze image structure
            if isinstance(result_data, dict) and 'pages' in result_data:
                for page_idx, page in enumerate(result_data['pages']):
                    if 'images' in page:
                        for img_idx, img in enumerate(page['images']):
                            if isinstance(img, dict):
                                debug_info["images_found"].append({
                                    "page": page_idx,
                                    "index": img_idx,
                                    "name": img.get('name', 'unknown'),
                                    "type": img.get('type', 'unknown'),
                                    "keys": list(img.keys())
                                })
            
            with open(debug_path, 'w', encoding='utf-8') as f:
                json.dump(debug_info, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Created download debug info: {debug_path}")
            
        except Exception as e:
            logger.error(f"Error creating download debug info: {e}")

    def _filter_and_validate_content(self, chunks: List[Dict], query: str) -> List[Dict]:
        """Centralized function to filter and validate content chunks before response generation."""
        validated_chunks = []
        for chunk in chunks:
            # If chunk contains tables, validate them
            if chunk.get('contains_table') and chunk.get('table_content_jsons'):
                valid_tables = []
                table_jsons = chunk.get('table_content_jsons', [])
                for table_json in table_jsons:
                    if self._is_table_valid(table_json) and self._has_meaningful_table_content(table_json, query):
                        valid_tables.append(table_json)
                    else:
                        logger.info(f"Filtering out invalid or irrelevant table from chunk {chunk.get('chunk_id')}")

                # If there are valid tables left, update the chunk
                if valid_tables:
                    chunk['table_content_jsons'] = valid_tables
                    chunk['table_count'] = len(valid_tables)
                    validated_chunks.append(chunk)
                else:
                    # If all tables in the chunk are invalid, we might still keep the chunk for its text
                    chunk['contains_table'] = False
                    chunk['table_count'] = 0
                    validated_chunks.append(chunk)
            else:
                # If no tables, the chunk is valid as is
                validated_chunks.append(chunk)
        
        logger.info(f"Content validation complete: {len(validated_chunks)} chunks remain after filtering.")
        return validated_chunks
    
    def _is_table_valid(self, table_json: str) -> bool:
        """Validate table content to filter out empty or distorted tables."""
        try:
            import json
            
            # Parse table JSON with robust error handling
            if isinstance(table_json, str):
                try:
                    # First try direct parsing
                    table_data = json.loads(table_json)
                except json.JSONDecodeError:
                    # Try to extract from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\n(.*?)\n```', table_json, re.DOTALL)
                    if json_match:
                        table_data = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON object pattern
                        json_match = re.search(r'\{.*\}', table_json, re.DOTALL)
                        if json_match:
                            table_data = json.loads(json_match.group(0))
                        else:
                            logger.warning(f"No valid JSON found in table data: {table_json[:100]}...")
                            return False
            else:
                table_data = table_json
            
            if not isinstance(table_data, dict):
                return False
            
            # Get non-metadata keys
            keys = [k for k in table_data.keys() if not k.startswith('_')]
            if not keys:
                return False
            
            # Check for empty columns - STRICTER: Even one empty column makes table invalid
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    # Count non-empty cells in this column
                    non_empty_cells = sum(1 for v in values if v and str(v).strip() and len(str(v).strip()) > 1)
                    
                    # NEW: If ANY column has empty cells, reject the entire table
                    if len(values) > 0 and non_empty_cells == 0:
                        logger.info(f"Column '{key}' has header but ALL cells are empty - rejecting table")
                        return False
                    # If more than 50% of cells in a column are empty, reject the table
                    elif len(values) > 0 and (non_empty_cells / len(values)) < 0.5:
                        logger.info(f"Column '{key}' has too many empty cells ({non_empty_cells}/{len(values)}) - rejecting table")
                        return False
            
            # All columns passed validation
            logger.info("All columns passed empty cell validation")
            
            # Check for empty rows (if table has row structure)
            if '_metadata' in table_data:
                metadata = table_data['_metadata']
                total_rows = metadata.get('total_rows', 0)
                total_cols = metadata.get('total_columns', 0)
                
                # NEW: Check if table has insufficient rows (only header + 1 data row)
                if total_rows <= 1:
                    logger.info(f"Table rejected: Only {total_rows} row(s) - insufficient data for meaningful table")
                    return False
                
                # If table has very few columns, it might be invalid
                if total_cols < 2:
                    return False
                
                # Check if most rows are empty
                empty_rows = 0
                for key in keys:
                    values = table_data.get(key, [])
                    if isinstance(values, list):
                        for row_idx in range(min(total_rows, len(values))):
                            if row_idx < len(values):
                                cell_value = values[row_idx]
                                if not cell_value or not str(cell_value).strip() or len(str(cell_value).strip()) <= 1:
                                    empty_rows += 1
                                    break
                
                # If more than 70% of rows are empty, table is invalid
                if total_rows > 0 and (empty_rows / total_rows) > 0.7:
                    return False
            
            
            
            # NEW: Check for tables that are just lists without meaningful structure
            meaningful_data_count = 0
            total_cells = 0
            
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    total_cells += len(values)
                    for value in values:
                        if value and str(value).strip() and len(str(value).strip()) > 2:
                            meaningful_data_count += 1
            
            # If less than 30% of cells have meaningful data, table is invalid
            if total_cells > 0 and (meaningful_data_count / total_cells) < 0.3:
                return False
            
            # NEW: Check for tables with suspicious patterns (all empty cells in certain rows)
            if '_metadata' in table_data:
                metadata = table_data['_metadata']
                total_rows = metadata.get('total_rows', 0)
                
                if total_rows > 0:
                    completely_empty_rows = 0
                    for row_idx in range(total_rows):
                        row_has_data = False
                        for key in keys:
                            values = table_data.get(key, [])
                            if isinstance(values, list) and row_idx < len(values):
                                cell_value = values[row_idx]
                                if cell_value and str(cell_value).strip() and len(str(cell_value).strip()) > 1:
                                    row_has_data = True
                                    break
                        
                        if not row_has_data:
                            completely_empty_rows += 1
                    
                    # If more than 60% of rows are completely empty, table is invalid
                    if (completely_empty_rows / total_rows) > 0.6:
                        return False
            
            # NEW: Check for tables that are just metadata or index without actual content
            if len(keys) <= 2 and total_cells < 4:
                # Check if it's just a simple index or reference table
                has_meaningful_content = False
                for key in keys:
                    values = table_data.get(key, [])
                    if isinstance(values, list):
                        for value in values:
                            if value and str(value).strip() and len(str(value).strip()) > 3:
                                has_meaningful_content = True
                                break
                
                if not has_meaningful_content:
                    return False
            
            # NEW: Fallback check for tables without metadata - ensure we have at least 2 rows of data
            if '_metadata' not in table_data:
                # Count actual data rows by checking the length of the first column
                first_key = keys[0] if keys else None
                if first_key:
                    first_column_values = table_data.get(first_key, [])
                    if isinstance(first_column_values, list):
                        actual_data_rows = len([v for v in first_column_values if v and str(v).strip() and len(str(v).strip()) > 1])
                        if actual_data_rows <= 1:
                            logger.info(f"Table rejected: Only {actual_data_rows} data row(s) - insufficient data for meaningful table")
                            return False
            
            
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating table: {e}")
            return False

    def _has_meaningful_table_content(self, table_json: str, query: str) -> bool:
        """Check if table has meaningful content relevant to the user's query."""
        try:
            import json
            
            # Parse table JSON with robust error handling
            if isinstance(table_json, str):
                try:
                    # First try direct parsing
                    table_data = json.loads(table_json)
                except json.JSONDecodeError:
                    # Try to extract from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\n(.*?)\n```', table_json, re.DOTALL)
                    if json_match:
                        table_data = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON object pattern
                        json_match = re.search(r'\{.*\}', table_json, re.DOTALL)
                        if json_match:
                            table_data = json.loads(json_match.group(0))
                        else:
                            logger.warning(f"No valid JSON found in table data: {table_json[:100]}...")
                            return False
            else:
                table_data = table_json
            
            if not isinstance(table_data, dict):
                return False
            
            # Get non-metadata keys
            keys = [k for k in table_data.keys() if not k.startswith('_')]
            if not keys:
                return False
            
            # NEW: Check if table is semantically relevant to the query
            if not self._is_table_semantically_relevant(table_data, query):
                logger.info(f"Table rejected due to low semantic relevance to query: '{query}'")
                return False
            
            # Check if table has substantial content
            total_cells = 0
            meaningful_cells = 0
            
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    total_cells += len(values)
                    
                    # Check if this column has any meaningful content at all
                    column_has_content = False
                    for value in values:
                        if value and str(value).strip() and len(str(value).strip()) > 2:
                            meaningful_cells += 1
                            column_has_content = True
                    
                    # If column has header but no meaningful content, log it
                    if not column_has_content and len(values) > 0:
                        logger.info(f"Column '{key}' has header but no meaningful content")
            
            # Table must have at least 50% meaningful cells (stricter threshold)
            if total_cells > 0 and (meaningful_cells / total_cells) < 0.5:
                logger.info(f"Table rejected due to low meaningful content: {meaningful_cells}/{total_cells} cells meaningful")
                return False
            
            # Check if table has enough data to be useful (at least 6 meaningful data points)
            if meaningful_cells < 6:
                logger.info(f"Table rejected due to insufficient meaningful data: {meaningful_cells} meaningful cells")
                return False
            
            # Check if table structure is meaningful (not just a simple list)
            if len(keys) < 2:
                logger.info(f"Table rejected due to insufficient columns: {len(keys)} columns")
                return False
            
            # Check if table has sufficient data rows (at least 2 rows of actual data)
            if total_cells > 0:
                # Calculate average rows per column to estimate total data rows
                estimated_rows = total_cells / len(keys)
                if estimated_rows <= 1:
                    logger.info(f"Table rejected due to insufficient rows: estimated {estimated_rows:.1f} rows - need at least 2 rows of data")
                    return False
            
            logger.info(f"Table passed content validation: {meaningful_cells}/{total_cells} meaningful cells, {len(keys)} columns")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking table content relevance: {e}")
            return False

    def _is_table_semantically_relevant(self, table_data: Dict, query: str) -> bool:
        """Check if table is semantically relevant to the user's query."""
        try:
            # Get non-metadata keys
            keys = [k for k in table_data.keys() if not k.startswith('_')]
            if not keys:
                return False
            
            # Extract all text content from the table for semantic analysis
            table_text_content = []
            
            # Add column headers
            table_text_content.extend(keys)
            
            # Add cell values
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    for value in values:
                        if value and str(value).strip():
                            table_text_content.append(str(value).strip())
            
            # Combine all text content
            combined_table_text = " ".join(table_text_content).lower()
            
            # Check if table is an exercise/worksheet (should be filtered out)
            exercise_indicators = [
                'exercise', 'question', 'answer', 'fill', 'blank', 'true', 'false',
                'match', 'matching', 'column', 'row', 'complete', 'worksheet',
                'activity', 'practice', 'test', 'quiz', 'homework'
            ]
            
            if any(indicator in combined_table_text for indicator in exercise_indicators):
                logger.info(f"Table identified as exercise/worksheet - filtering out for relevance")
                return False
            
            # Check if table is just a simple index or reference (low relevance)
            index_indicators = ['page', 'number', 'reference', 'index', 'list']
            if any(indicator in combined_table_text for indicator in index_indicators) and len(keys) <= 2:
                logger.info(f"Table identified as simple index/reference - low relevance")
                return False
            
            # Calculate semantic similarity between table content and query
            query_words = set(query.lower().split())
            table_words = set(combined_table_text.split())
            
            # Remove common stop words that don't add semantic value
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
            
            query_words = query_words - stop_words
            table_words = table_words - stop_words
            
            # Calculate word overlap
            if query_words:
                overlap = len(query_words & table_words)
                overlap_ratio = overlap / len(query_words)
                
                # Require at least 20% word overlap for relevance
                if overlap_ratio < 0.2:
                    logger.info(f"Table has low semantic overlap with query: {overlap}/{len(query_words)} words ({overlap_ratio:.2%})")
                    return False
                
                logger.info(f"Table has good semantic overlap with query: {overlap}/{len(query_words)} words ({overlap_ratio:.2%})")
                
                logger.info(f"Table has good semantic overlap with query: {overlap}/{len(query_words)} words ({overlap_ratio:.2%})")
            
            # Check for domain-specific relevance
            query_lower = query.lower()
            
            # If query is about nutrition, check if table contains nutrition-related terms
            if any(word in query_lower for word in ['nutrition', 'food', 'diet', 'eating']):
                nutrition_terms = ['food', 'nutrition', 'diet', 'eating', 'digestion', 'nutrients', 'proteins', 'carbohydrates', 'fats', 'vitamins']
                if not any(term in combined_table_text for term in nutrition_terms):
                    logger.info(f"Query about nutrition but table lacks nutrition-related terms")
                    return False
            
            # If query is about animals/plants, check if table contains biology terms
            if any(word in query_lower for word in ['animal', 'plant', 'biology', 'organism']):
                biology_terms = ['animal', 'plant', 'organism', 'species', 'biology', 'cell', 'tissue', 'organ', 'system']
                if not any(term in combined_table_text for term in biology_terms):
                    logger.info(f"Query about animals/plants but table lacks biology-related terms")
                    return False
            
            # Check if table appears to be meaningful data vs. just exercise content
            meaningful_data_indicators = ['data', 'result', 'analysis', 'comparison', 'measurement', 'statistic', 'information', 'detail']
            exercise_content_indicators = ['match', 'connect', 'draw', 'label', 'identify', 'choose', 'select', 'write']
            
            meaningful_score = sum(1 for indicator in meaningful_data_indicators if indicator in combined_table_text)
            exercise_score = sum(1 for indicator in exercise_content_indicators if indicator in combined_table_text)
            
            if exercise_score > meaningful_score:
                logger.info(f"Table appears to be exercise content rather than meaningful data")
                return False
            
            logger.info(f"Table passed semantic relevance check for query: '{query}'")
            return True
            
        except Exception as e:
            logger.warning(f"Error in semantic relevance check: {e}")
            # If we can't determine relevance, err on the side of caution and reject
            return False
    
    def _is_table_valid(self, table_json: str) -> bool:
        """Validate table content to filter out empty or distorted tables."""
        try:
            import json
            
            # Parse table JSON with robust error handling
            if isinstance(table_json, str):
                try:
                    # First try direct parsing
                    table_data = json.loads(table_json)
                except json.JSONDecodeError:
                    # Try to extract from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\n(.*?)\n```', table_json, re.DOTALL)
                    if json_match:
                        table_data = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON object pattern
                        json_match = re.search(r'\{.*\}', table_json, re.DOTALL)
                        if json_match:
                            table_data = json.loads(json_match.group(0))
                        else:
                            logger.warning(f"No valid JSON found in table data: {table_json[:100]}...")
                            return False
            else:
                table_data = table_json
            
            if not isinstance(table_data, dict):
                return False
            
            # Get non-metadata keys
            keys = [k for k in table_data.keys() if not k.startswith('_')]
            if not keys:
                return False
            
            # Check for empty columns - STRICTER: Even one empty column makes table invalid
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    # Count non-empty cells in this column
                    non_empty_cells = sum(1 for v in values if v and str(v).strip() and len(str(v).strip()) > 1)
                    
                    # NEW: If ANY column has empty cells, reject the entire table
                    if len(values) > 0 and non_empty_cells == 0:
                        logger.info(f"Column '{key}' has header but ALL cells are empty - rejecting table")
                        return False
                    # If more than 50% of cells in a column are empty, reject the table
                    elif len(values) > 0 and (non_empty_cells / len(values)) < 0.5:
                        logger.info(f"Column '{key}' has too many empty cells ({non_empty_cells}/{len(values)}) - rejecting table")
                        return False
            
            # All columns passed validation
            logger.info("All columns passed empty cell validation")
            
            # Check for empty rows (if table has row structure)
            if '_metadata' in table_data:
                metadata = table_data['_metadata']
                total_rows = metadata.get('total_rows', 0)
                total_cols = metadata.get('total_columns', 0)
                
                # NEW: Check if table has insufficient rows (only header + 1 data row)
                if total_rows <= 1:
                    logger.info(f"Table rejected: Only {total_rows} row(s) - insufficient data for meaningful table")
                    return False
                
                # If table has very few columns, it might be invalid
                if total_cols < 2:
                    return False
                
                # Check if most rows are empty
                empty_rows = 0
                for key in keys:
                    values = table_data.get(key, [])
                    if isinstance(values, list):
                        for row_idx in range(min(total_rows, len(values))):
                            if row_idx < len(values):
                                cell_value = values[row_idx]
                                if not cell_value or not str(cell_value).strip() or len(str(cell_value).strip()) <= 1:
                                    empty_rows += 1
                                    break
                
                # If more than 70% of rows are empty, table is invalid
                if total_rows > 0 and (empty_rows / total_rows) > 0.7:
                    return False
            
            
            
            # NEW: Check for tables that are just lists without meaningful structure
            meaningful_data_count = 0
            total_cells = 0
            
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    total_cells += len(values)
                    for value in values:
                        if value and str(value).strip() and len(str(value).strip()) > 2:
                            meaningful_data_count += 1
            
            # If less than 30% of cells have meaningful data, table is invalid
            if total_cells > 0 and (meaningful_data_count / total_cells) < 0.3:
                return False
            
            # NEW: Check for tables with suspicious patterns (all empty cells in certain rows)
            if '_metadata' in table_data:
                metadata = table_data['_metadata']
                total_rows = metadata.get('total_rows', 0)
                
                if total_rows > 0:
                    completely_empty_rows = 0
                    for row_idx in range(total_rows):
                        row_has_data = False
                        for key in keys:
                            values = table_data.get(key, [])
                            if isinstance(values, list) and row_idx < len(values):
                                cell_value = values[row_idx]
                                if cell_value and str(cell_value).strip() and len(str(cell_value).strip()) > 1:
                                    row_has_data = True
                                    break
                        
                        if not row_has_data:
                            completely_empty_rows += 1
                    
                    # If more than 60% of rows are completely empty, table is invalid
                    if (completely_empty_rows / total_rows) > 0.6:
                        return False
            
            # NEW: Check for tables that are just metadata or index without actual content
            if len(keys) <= 2 and total_cells < 4:
                # Check if it's just a simple index or reference table
                has_meaningful_content = False
                for key in keys:
                    values = table_data.get(key, [])
                    if isinstance(values, list):
                        for value in values:
                            if value and str(value).strip() and len(str(value).strip()) > 3:
                                has_meaningful_content = True
                                break
                
                if not has_meaningful_content:
                    return False
            
            # NEW: Fallback check for tables without metadata - ensure we have at least 2 rows of data
            if '_metadata' not in table_data:
                # Count actual data rows by checking the length of the first column
                first_key = keys[0] if keys else None
                if first_key:
                    first_column_values = table_data.get(first_key, [])
                    if isinstance(first_column_values, list):
                        actual_data_rows = len([v for v in first_column_values if v and str(v).strip() and len(str(v).strip()) > 1])
                        if actual_data_rows <= 1:
                            logger.info(f"Table rejected: Only {actual_data_rows} data row(s) - insufficient data for meaningful table")
                            return False
            
            
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating table: {e}")
            return False
    
    def _has_meaningful_table_content(self, table_json: str, query: str) -> bool:
        """Check if table has meaningful content relevant to the user's query."""
        try:
            import json
            
            # Parse table JSON with robust error handling
            if isinstance(table_json, str):
                try:
                    # First try direct parsing
                    table_data = json.loads(table_json)
                except json.JSONDecodeError:
                    # Try to extract from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\n(.*?)\n```', table_json, re.DOTALL)
                    if json_match:
                        table_data = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON object pattern
                        json_match = re.search(r'\{.*\}', table_json, re.DOTALL)
                        if json_match:
                            table_data = json.loads(json_match.group(0))
                        else:
                            logger.warning(f"No valid JSON found in table data: {table_json[:100]}...")
                            return False
            else:
                table_data = table_json
            
            if not isinstance(table_data, dict):
                return False
            
            # Get non-metadata keys
            keys = [k for k in table_data.keys() if not k.startswith('_')]
            if not keys:
                return False
            
            # NEW: Check if table is semantically relevant to the query
            if not self._is_table_semantically_relevant(table_data, query):
                logger.info(f"Table rejected due to low semantic relevance to query: '{query}'")
                return False
            
            # Check if table has substantial content
            total_cells = 0
            meaningful_cells = 0
            
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    total_cells += len(values)
                    
                    # Check if this column has any meaningful content at all
                    column_has_content = False
                    for value in values:
                        if value and str(value).strip() and len(str(value).strip()) > 2:
                            meaningful_cells += 1
                            column_has_content = True
                    
                    # If column has header but no meaningful content, log it
                    if not column_has_content and len(values) > 0:
                        logger.info(f"Column '{key}' has header but no meaningful content")
            
            # Table must have at least 50% meaningful cells (stricter threshold)
            if total_cells > 0 and (meaningful_cells / total_cells) < 0.5:
                logger.info(f"Table rejected due to low meaningful content: {meaningful_cells}/{total_cells} cells meaningful")
                return False
            
            # Check if table has enough data to be useful (at least 6 meaningful data points)
            if meaningful_cells < 6:
                logger.info(f"Table rejected due to insufficient meaningful data: {meaningful_cells} meaningful cells")
                return False
            
            # Check if table structure is meaningful (not just a simple list)
            if len(keys) < 2:
                logger.info(f"Table rejected due to insufficient columns: {len(keys)} columns")
                return False
            
            # Check if table has sufficient data rows (at least 2 rows of actual data)
            if total_cells > 0:
                # Calculate average rows per column to estimate total data rows
                estimated_rows = total_cells / len(keys)
                if estimated_rows <= 1:
                    logger.info(f"Table rejected due to insufficient rows: estimated {estimated_rows:.1f} rows - need at least 2 rows of data")
                    return False
            
            logger.info(f"Table passed content validation: {meaningful_cells}/{total_cells} meaningful cells, {len(keys)} columns")
            return True
            
        except Exception as e:
            logger.warning(f"Error checking table content relevance: {e}")
            return False
    
    def _is_table_semantically_relevant(self, table_data: Dict, query: str) -> bool:
        """Check if table is semantically relevant to the user's query."""
        try:
            # Get non-metadata keys
            keys = [k for k in table_data.keys() if not k.startswith('_')]
            if not keys:
                return False
            
            # Extract all text content from the table for semantic analysis
            table_text_content = []
            
            # Add column headers
            table_text_content.extend(keys)
            
            # Add cell values
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    for value in values:
                        if value and str(value).strip():
                            table_text_content.append(str(value).strip())
            
            # Combine all text content
            combined_table_text = " ".join(table_text_content).lower()
            
            # Check if table is an exercise/worksheet (should be filtered out)
            exercise_indicators = [
                'exercise', 'question', 'answer', 'fill', 'blank', 'true', 'false',
                'match', 'matching', 'column', 'row', 'complete', 'worksheet',
                'activity', 'practice', 'test', 'quiz', 'homework'
            ]
            
            if any(indicator in combined_table_text for indicator in exercise_indicators):
                logger.info(f"Table identified as exercise/worksheet - filtering out for relevance")
                return False
            
            # Check if table is just a simple index or reference (low relevance)
            index_indicators = ['page', 'number', 'reference', 'index', 'list']
            if any(indicator in combined_table_text for indicator in index_indicators) and len(keys) <= 2:
                logger.info(f"Table identified as simple index/reference - low relevance")
                return False
            
            # Calculate semantic similarity between table content and query
            query_words = set(query.lower().split())
            table_words = set(combined_table_text.split())
            
            # Remove common stop words that don't add semantic value
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
            }
            
            query_words = query_words - stop_words
            table_words = table_words - stop_words
            
            # Calculate word overlap
            if query_words:
                overlap = len(query_words & table_words)
                overlap_ratio = overlap / len(query_words)
                
                # Require at least 20% word overlap for relevance
                if overlap_ratio < 0.2:
                    logger.info(f"Table has low semantic overlap with query: {overlap}/{len(query_words)} words ({overlap_ratio:.2%})")
                    return False
                
                logger.info(f"Table has good semantic overlap with query: {overlap}/{len(query_words)} words ({overlap_ratio:.2%})")
                
                logger.info(f"Table has good semantic overlap with query: {overlap}/{len(query_words)} words ({overlap_ratio:.2%})")
            
            # Check for domain-specific relevance
            query_lower = query.lower()
            
            # If query is about nutrition, check if table contains nutrition-related terms
            if any(word in query_lower for word in ['nutrition', 'food', 'diet', 'eating']):
                nutrition_terms = ['food', 'nutrition', 'diet', 'eating', 'digestion', 'nutrients', 'proteins', 'carbohydrates', 'fats', 'vitamins']
                if not any(term in combined_table_text for term in nutrition_terms):
                    logger.info(f"Query about nutrition but table lacks nutrition-related terms")
                    return False
            
            # If query is about animals/plants, check if table contains biology terms
            if any(word in query_lower for word in ['animal', 'plant', 'biology', 'organism']):
                biology_terms = ['animal', 'plant', 'organism', 'species', 'biology', 'cell', 'tissue', 'organ', 'system']
                if not any(term in combined_table_text for term in biology_terms):
                    logger.info(f"Query about animals/plants but table lacks biology-related terms")
                    return False
            
            # Check if table appears to be meaningful data vs. just exercise content
            meaningful_data_indicators = ['data', 'result', 'analysis', 'comparison', 'measurement', 'statistic', 'information', 'detail']
            exercise_content_indicators = ['match', 'connect', 'draw', 'label', 'identify', 'choose', 'select', 'write']
            
            meaningful_score = sum(1 for indicator in meaningful_data_indicators if indicator in combined_table_text)
            exercise_score = sum(1 for indicator in exercise_content_indicators if indicator in combined_table_text)
            
            if exercise_score > meaningful_score:
                logger.info(f"Table appears to be exercise content rather than meaningful data")
                return False
            
            logger.info(f"Table passed semantic relevance check for query: '{query}'")
            return True
            
        except Exception as e:
            logger.warning(f"Error in semantic relevance check: {e}")
            # If we can't determine relevance, err on the side of caution and reject
            return False

    def _filter_and_validate_chunks(self, chunks: List[Dict], query: str) -> List[Dict]:
        """
        Filters chunks by validating their multimedia content, especially tables.
        Removes invalid tables from chunks and updates metadata accordingly.
        """
        validated_chunks = []
        for chunk in chunks:
            if chunk.get('contains_table'):
                valid_tables_json = []
                valid_table_ids = []
                valid_table_summaries = []
                
                table_jsons = chunk.get('table_content_jsons', [])
                table_ids = chunk.get('table_ids', [])
                table_summaries = chunk.get('table_summaries', [])

                for i, table_json_str in enumerate(table_jsons):
                    # Here we call the newly moved validation functions
                    if self._is_table_valid(table_json_str) and self._has_meaningful_table_content(table_json_str, query):
                        valid_tables_json.append(table_json_str)
                        if i < len(table_ids):
                            valid_table_ids.append(table_ids[i])
                        if i < len(table_summaries):
                            valid_table_summaries.append(table_summaries[i])
                
                if valid_tables_json:
                    chunk['table_content_jsons'] = valid_tables_json
                    chunk['table_ids'] = valid_table_ids
                    chunk['table_summaries'] = valid_table_summaries
                    chunk['table_count'] = len(valid_tables_json)
                else:
                    # If all tables in this chunk are invalid, update the metadata
                    chunk['contains_table'] = False
                    chunk['table_count'] = 0
                    chunk['table_content_jsons'] = []
                    chunk['table_ids'] = []
                    chunk['table_summaries'] = []
            
            validated_chunks.append(chunk)
            
        logger.info(f"Validated {len(chunks)} chunks, returning {len(validated_chunks)} chunks after filtering.")
        return validated_chunks
    
    def _extract_images_from_zip_response(self, zip_content: bytes, save_dir: Path, pdf_id: str) -> int:
        """Extract images from ZIP response."""
        try:
            extracted_count = 0
            
            with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:
                logger.info(f"ZIP contains {len(zip_file.filelist)} files")
                
                for file_info in zip_file.filelist:
                    logger.info(f"ZIP file: {file_info.filename}")
                    
                    # Check if it's an image file
                    if any(file_info.filename.lower().endswith(ext) 
                           for ext in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp']):
                        
                        # Extract image
                        image_data = zip_file.read(file_info.filename)
                        
                        # Create clean filename
                        clean_name = Path(file_info.filename).name
                        image_path = save_dir / f"{pdf_id}_{clean_name}"
                        
                        with open(image_path, 'wb') as f:
                            f.write(image_data)
                        
                        extracted_count += 1
                        logger.info(f"Extracted image: {image_path}")
            
            return extracted_count
            
        except Exception as e:
            logger.error(f"Error extracting images from ZIP: {e}")
            return 0
    
    def _process_json_image_response(self, json_data: dict, save_dir: Path, pdf_id: str, headers: dict) -> int:
        """Process JSON response containing image information."""
        try:
            import requests
            import base64
            downloaded_count = 0
            
            # Handle different JSON structures
            if isinstance(json_data, dict):
                # Look for image arrays or URLs
                for key in ['images', 'image_urls', 'files', 'results']:
                    if key in json_data:
                        items = json_data[key]
                        if isinstance(items, list):
                            for i, item in enumerate(items):
                                success = self._download_single_image_item(
                                    item, save_dir, pdf_id, i, headers
                                )
                                if success:
                                    downloaded_count += 1
                
                # Look for direct URLs in the response
                for key in ['url', 'download_url', 'image_url']:
                    if key in json_data and json_data[key]:
                        success = self._download_single_image_item(
                            json_data[key], save_dir, pdf_id, 0, headers
                        )
                        if success:
                            downloaded_count += 1
            
            elif isinstance(json_data, list):
                # Array of image items
                for i, item in enumerate(json_data):
                    success = self._download_single_image_item(
                        item, save_dir, pdf_id, i, headers
                    )
                    if success:
                        downloaded_count += 1
            
            return downloaded_count
            
        except Exception as e:
            logger.error(f"Error processing JSON image response: {e}")
            return 0
    
    def _download_single_image_item(self, item, save_dir: Path, pdf_id: str, index: int, headers: dict) -> bool:
        """Download a single image item."""
        try:
            import requests
            import base64
            
            url = None
            image_data = None
            
            if isinstance(item, str):
                if item.startswith('http'):
                    url = item
                elif item.startswith('data:image'):
                    # Base64 data URL
                    _, b64_data = item.split(',', 1)
                    image_data = base64.b64decode(b64_data)
                elif len(item) > 100:  # Likely base64
                    try:
                        image_data = base64.b64decode(item)
                    except:
                        pass
            elif isinstance(item, dict):
                # Look for URL in dict
                for url_key in ['url', 'download_url', 'image_url', 'src', 'href']:
                    if url_key in item and item[url_key]:
                        url = item[url_key]
                        break
                
                # Look for data in dict
                for data_key in ['data', 'image_data', 'base64', 'content']:
                    if data_key in item and item[data_key]:
                        try:
                            if isinstance(item[data_key], str):
                                if item[data_key].startswith('data:image'):
                                    _, b64_data = item[data_key].split(',', 1)
                                    image_data = base64.b64decode(b64_data)
                                else:
                                    image_data = base64.b64decode(item[data_key])
                                break
                        except:
                            continue
            
            # Download from URL
            if url:
                response = requests.get(url, headers=headers, timeout=30)
                if response.status_code == 200:
                    image_data = response.content
            
            # Save image data
            if image_data:
                image_path = save_dir / f"{pdf_id}_img_{index}.png"
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Downloaded image: {image_path}")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error downloading single image item: {e}")
            return False
    
    def _create_debug_info(self, job_id: str, save_dir: Path, pdf_id: str, headers: dict):
        """Log debug information for troubleshooting - no file creation."""
        try:
            logger.info(f"Debug info for job {job_id}, PDF {pdf_id} - check logs for details")
            
        except Exception as e:
            logger.error(f"Error logging debug info: {e}")
    
    def _filter_invalid_tables_from_chunks(self, chunks: List[Dict]) -> List[Dict]:
        """Filter out chunks with invalid tables before content selection."""
        try:
            filtered_chunks = []
            
            for chunk in chunks:
                # If chunk has tables, validate them
                if chunk.get('contains_table') and chunk.get('table_content_jsons'):
                    table_jsons = chunk.get('table_content_jsons', [])
                    valid_tables = []
                    
                    for table_json in table_jsons:
                        if self._is_table_valid_for_selection(table_json):
                            valid_tables.append(table_json)
                    
                    # Update chunk with only valid tables
                    if valid_tables:
                        chunk['table_content_jsons'] = valid_tables
                        chunk['table_count'] = len(valid_tables)
                        filtered_chunks.append(chunk)
                    else:
                        # Remove table flags if no valid tables
                        chunk['contains_table'] = False
                        chunk['table_count'] = 0
                        filtered_chunks.append(chunk)
                else:
                    filtered_chunks.append(chunk)
            
            logger.info(f"Table validation: {len(chunks)} chunks processed, {len(filtered_chunks)} chunks retained")
            return filtered_chunks
            
        except Exception as e:
            logger.error(f"Error filtering invalid tables: {e}")
            return chunks
    
    def _is_table_valid_for_selection(self, table_json: str) -> bool:
        """Validate table content for selection (lighter validation than display validation)."""
        try:
            import json
            
            # Parse table JSON with robust error handling
            if isinstance(table_json, str):
                try:
                    # First try direct parsing
                    table_data = json.loads(table_json)
                except json.JSONDecodeError:
                    # Try to extract from markdown code blocks
                    import re
                    json_match = re.search(r'```(?:json)?\n(.*?)\n```', table_json, re.DOTALL)
                    if json_match:
                        table_data = json.loads(json_match.group(1))
                    else:
                        # Try to find JSON object pattern
                        json_match = re.search(r'\{.*\}', table_json, re.DOTALL)
                        if json_match:
                            table_data = json.loads(json_match.group(0))
                        else:
                            logger.warning(f"No valid JSON found in table data: {table_json[:100]}...")
                            return False
            else:
                table_data = table_json
            
            if not isinstance(table_data, dict):
                return False
            
            # Get non-metadata keys
            keys = [k for k in table_data.keys() if not k.startswith('_')]
            if not keys:
                return False
            
            # Check for empty columns
            empty_columns = 0
            total_columns = len(keys)
            
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    # Count non-empty cells in this column
                    non_empty_cells = sum(1 for v in values if v and str(v).strip() and len(str(v).strip()) > 1)
                    
                    # If more than 80% of cells are empty, consider column empty (stricter threshold)
                    if len(values) > 0 and (non_empty_cells / len(values)) < 0.2:
                        empty_columns += 1
            
            # If more than 50% of columns are empty, table is invalid
            if empty_columns / total_columns > 0.5:
                return False
            
            # Check for very small tables (less than 2x2)
            if '_metadata' in table_data:
                metadata = table_data['_metadata']
                total_rows = metadata.get('total_rows', 0)
                total_cols = metadata.get('total_columns', 0)
                
                if total_rows < 2 or total_cols < 2:
                    return False
            
            # NEW: Check for tables that are just lists without meaningful structure
            meaningful_data_count = 0
            total_cells = 0
            
            for key in keys:
                values = table_data.get(key, [])
                if isinstance(values, list):
                    total_cells += len(values)
                    for value in values:
                        if value and str(value).strip() and len(str(value).strip()) > 2:
                            meaningful_data_count += 1
            
            # If less than 30% of cells have meaningful data, table is invalid
            if total_cells > 0 and (meaningful_data_count / total_cells) < 0.3:
                return False
            
            # NEW: Check for tables with suspicious patterns (all empty cells in certain rows)
            if '_metadata' in table_data:
                metadata = table_data['_metadata']
                total_rows = metadata.get('total_rows', 0)
                
                if total_rows > 0:
                    completely_empty_rows = 0
                    for row_idx in range(total_rows):
                        row_has_data = False
                        for key in keys:
                            values = table_data.get(key, [])
                            if isinstance(values, list) and row_idx < len(values):
                                cell_value = values[row_idx]
                                if cell_value and str(cell_value).strip() and len(str(cell_value).strip()) > 1:
                                    row_has_data = True
                                    break
                        
                        if not row_has_data:
                            completely_empty_rows += 1
                    
                    # If more than 60% of rows are completely empty, table is invalid
                    if (completely_empty_rows / total_rows) > 0.6:
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error validating table for selection: {e}")
            return False
    
