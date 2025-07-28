
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

# Third-party imports
import pinecone
from pinecone import Pinecone
import openai
from openai import OpenAI
import streamlit as st
from llama_cloud_services import LlamaParse

# Local imports
from config import AdvancedConfig
from .image_processor import ImageProcessor
from .table_processor import TableProcessor
from s3_handler import S3Handler
from .inline_multimodal_generator import InlineMultimodalGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalIntegrator:
    """Main integration system for processing and storing multimodal PDF content."""
    
    def __init__(self):
        """Initialize the multimodal integrator."""
        self.config = AdvancedConfig()
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
            
            self.pinecone_client = Pinecone(api_key=pinecone_api_key)
            
            index_name = os.getenv('PINECONE_INDEX_NAME_TEST')
            if not index_name:
                raise ValueError("PINECONE_INDEX_NAME not found")
            
            self.index = self.pinecone_client.Index(index_name)
            
            # Debug logging for index status
            logger.info(f"Using Pinecone index: {index_name}")
            try:
                index_stats = self.index.describe_index_stats()
                logger.info(f"Index stats: {index_stats}")
                if hasattr(index_stats, 'namespaces') and index_stats.namespaces:
                    logger.info(f"Available namespaces: {list(index_stats.namespaces.keys())}")
                else:
                    logger.warning("No namespaces found in index stats")
            except Exception as e:
                logger.error(f"Error getting index stats: {e}")
            
            # LlamaParse initialization
            llamaparse_api_key = os.getenv('LLAMA_PARSE_API_KEY')
            if not llamaparse_api_key:
                raise ValueError("LLAMA_PARSE_API_KEY not found")
            
            self.llamaparse_parser = LlamaParse(
                api_key=llamaparse_api_key,
                result_type="markdown",
                system_prompt="Extract all text content, images, and tables with complete metadata. Preserve document structure and formatting.",
                max_timeout=self.config.DEFAULT_TIMEOUT,
                split_by_page=True,
                verbose=True
            )
            
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
            pdf_name = original_filename if original_filename else Path(pdf_path).name
            
            # Generate PDF ID and namespace using the original filename
            pdf_id = self._generate_pdf_id_from_name(pdf_name)
            
            # Check if already processed
            if not force_reprocess and self._is_pdf_processed(pdf_id, pdf_name):
                st.info(f"📋 PDF already processed: {pdf_name}")
                return {
                    'pdf_id': pdf_id,
                    'cached': True,
                    'processing_time': 0
                }
            
            st.info(f"🚀 Starting complete multimodal processing: {pdf_name}")
            
            with st.spinner("Parsing PDF with LlamaParse..."):
                # Parse PDF with LlamaParse and capture job ID
                json_objs = self.llamaparse_parser.get_json_result(pdf_path)
                
                if not json_objs:
                    raise ValueError("No content extracted from PDF")
                
                # Extract images from LlamaParse results using proven methods
                logger.info(f"Extracting images from LlamaParse JSON results using proven extraction")
                extracted_count = self._extract_images_from_results(json_objs, pdf_id, pdf_name)
                logger.info(f"Enhanced image extraction completed: {extracted_count} images processed")
                
                logger.info(f"LlamaParse extraction complete for {pdf_name}")
            
            # Process all content types in parallel
            with st.spinner("Processing multimodal content..."):
                processing_results = self._process_multimodal_content_parallel(
                    json_objs, pdf_name, pdf_id
                )
            
            # Process text content
            with st.spinner("Processing text content..."):
                text_results = self._process_text_content(json_objs, pdf_name, pdf_id)
                processing_results['text'] = text_results
            
            # Display comprehensive results
            self._display_processing_results(processing_results, time.time() - start_time)
            
            return {
                'pdf_id': pdf_id,
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
        """Process images and tables in parallel for efficiency."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit processing tasks
            futures = {
                executor.submit(
                    self.image_processor.process_images_from_llamaparse, 
                    json_objs, pdf_name, pdf_id
                ): 'images',
                executor.submit(
                    self.table_processor.process_tables_from_llamaparse, 
                    json_objs, pdf_name, pdf_id
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
    
    def _process_text_content(self, json_objs: List[Dict], pdf_name: str, pdf_id: str) -> Dict[str, Any]:
        """Process text content and store in unified namespace."""
        try:
            if not json_objs:
                return {'processed_chunks': 0, 'pinecone_stored': 0}
            
            json_data = json_objs[0]
            pages = json_data.get('pages', [])
            
            text_chunks = []
            
            for page_data in pages:
                page_num = page_data.get('page', 0)
                
                # Extract text content
                text_content = page_data.get('text', '') or page_data.get('md', '')
                
                if text_content and len(text_content.strip()) > 50:
                    # Create text chunk with metadata
                    chunk = {
                        'chunk_id': f"{pdf_id}_text_p{page_num}",
                        'pdf_id': pdf_id,
                        'pdf_name': pdf_name,
                        'page_number': page_num,
                        'content': text_content,
                        'content_type': 'text',
                        'word_count': len(text_content.split()),
                        'char_count': len(text_content),
                        'processed_timestamp': datetime.now().isoformat()
                    }
                    
                    text_chunks.append(chunk)
            
            # Store text chunks in Pinecone (same namespace as images and tables)
            stored_count = self._store_text_chunks_in_pinecone(text_chunks, pdf_id)
            
            logger.info(f"Text processing complete: {len(text_chunks)} chunks, {stored_count} stored")
            
            return {
                'processed_chunks': len(text_chunks),
                'pinecone_stored': stored_count,
                'text_chunks': text_chunks
            }
            
        except Exception as e:
            logger.error(f"Error processing text content: {e}")
            return {'error': str(e)}
    
    def _store_text_chunks_in_pinecone(self, text_chunks: List[Dict], pdf_id: str) -> int:
        """Store text chunks in Pinecone with unified namespace."""
        if not text_chunks:
            return 0
        
        vectors = []
        
        for chunk in text_chunks:
            try:
                # Generate embedding
                embedding = self._get_embedding(chunk['content'])
                
                # Create metadata
                metadata = {
                    'content_type': 'text',
                    'chunk_id': chunk['chunk_id'],
                    'pdf_id': pdf_id,
                    'pdf_name': chunk['pdf_name'],
                    'page_number': chunk['page_number'],
                    'content': chunk['content'][:5000],  # Limit for Pinecone
                    'word_count': chunk['word_count'],
                    'char_count': chunk['char_count'],
                    'processed_timestamp': chunk['processed_timestamp']
                }
                
                vector = {
                    'id': f"txt_{chunk['chunk_id']}",
                    'values': embedding,
                    'metadata': metadata
                }
                
                vectors.append(vector)
                
            except Exception as e:
                logger.error(f"Error creating vector for chunk {chunk.get('chunk_id')}: {e}")
                continue
        
        # Batch upsert to unified namespace
        if vectors:
            try:
                self.index.upsert(vectors=vectors, namespace=pdf_id)
                logger.info(f"Stored {len(vectors)} text vectors in unified namespace: {pdf_id}")
                return len(vectors)
            except Exception as e:
                logger.error(f"Error upserting text vectors to Pinecone: {e}")
                return 0
        
        return 0
    
    def _generate_pdf_id(self, pdf_path: str) -> str:
        """Generate PDF ID based on filename for reuse detection."""
        pdf_name = Path(pdf_path).name
        return self.config.generate_pdf_id_from_name(pdf_name)
    
    def _generate_pdf_id_from_name(self, pdf_name: str) -> str:
        """Generate PDF ID directly from filename for reuse detection."""
        return self.config.generate_pdf_id_from_name(pdf_name)
    
    def _is_pdf_processed(self, pdf_id: str, pdf_name: str = None) -> bool:
        """Check if PDF has been fully processed by checking BOTH S3 storage AND Pinecone vectors."""
        try:
            # Check Pinecone namespace to see if any content exists
            namespace_stats = self.index.describe_index_stats()
            namespaces = namespace_stats.get('namespaces', {})
            
            if pdf_id in namespaces:
                vector_count = namespaces[pdf_id].get('vector_count', 0)
                if vector_count > 0:
                    logger.info(f"PDF {pdf_id} already processed ({vector_count} vectors in Pinecone)")
                    return True
            
            # If no vectors in Pinecone, PDF is not fully processed regardless of S3 files
            logger.info(f"PDF {pdf_id} not found in Pinecone namespaces - needs processing")
            return False
            
        except Exception as e:
            logger.error(f"Error checking if PDF processed: {e}")
            return False
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            if not text or len(text.strip()) < 3:
                logger.warning(f"Text too short for embedding: '{text}' (length: {len(text) if text else 0})")
                return [0.0] * 3072
            
            logger.debug(f"Generating embedding for text: '{text[:100]}...' (length: {len(text)})")
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8191]  # OpenAI token limit
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 3072
    
    def query_multimodal_content(self, query: str, pdf_id: str, 
                               max_images: int = 5, max_tables: int = 3, 
                               max_text_chunks: int = 8) -> Dict[str, Any]:
        """Query all content types with intelligent relevance scoring."""
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            # Debug: Log query parameters
            top_k_requested = max_images + max_tables + max_text_chunks + 10
            logger.info(f"Querying namespace '{pdf_id}' with top_k={top_k_requested}")
            logger.info(f"Query embedding shape: {len(query_embedding) if query_embedding else 'None'}")
            
            # Query unified namespace for all content types
            all_results = self.index.query(
                vector=query_embedding,
                top_k=top_k_requested,  # Extra for filtering
                namespace=pdf_id,
                include_metadata=True,
                include_values=False
            )
            
            logger.info(f"Pinecone query returned {len(all_results.matches)} matches for namespace '{pdf_id}'")
            
            # Debug: Log details about the results
            if len(all_results.matches) == 0:
                logger.warning(f"No matches found! Checking if namespace '{pdf_id}' exists...")
                try:
                    # Check current index stats to see available namespaces
                    current_stats = self.index.describe_index_stats()
                    if hasattr(current_stats, 'namespaces') and current_stats.namespaces:
                        available_namespaces = list(current_stats.namespaces.keys())
                        logger.info(f"Available namespaces in index: {available_namespaces}")
                        if pdf_id not in available_namespaces:
                            logger.error(f"Target namespace '{pdf_id}' not found in available namespaces!")
                    else:
                        logger.warning("No namespace information available in index stats")
                except Exception as e:
                    logger.error(f"Error checking namespaces: {e}")
            else:
                # Log sample match details
                sample_match = all_results.matches[0]
                logger.info(f"Sample match - Score: {sample_match.score}, Content type: {sample_match.metadata.get('content_type', 'unknown')}")
            
            # Separate results by content type
            text_results = []
            image_results = []
            table_results = []
            
            for match in all_results.matches:
                if match.score > 0.1:  # Minimum relevance threshold
                    content_type = match.metadata.get('content_type', 'unknown')
                    
                    if content_type == 'text':
                        text_results.append(match)
                    elif content_type == 'image':
                        image_results.append(match)
                    elif content_type == 'table':
                        table_results.append(match)
            
            logger.info(f"Content type breakdown: {len(text_results)} text, {len(image_results)} images, {len(table_results)} tables")
            
            # Apply content-specific filtering and limits
            filtered_content = self._apply_intelligent_filtering(
                query, text_results[:max_text_chunks], 
                image_results[:max_images], table_results[:max_tables]
            )
            
            logger.info(f"After filtering: {len(filtered_content['text'])} text, {len(filtered_content['images'])} images, {len(filtered_content['tables'])} tables")
            
            # Generate intelligent inline response
            response_data = self.inline_generator.generate_inline_response(
                query, filtered_content
            )
            
            response_data['performance_metrics'] = {
                'query_time': time.time() - start_time,
                'total_results': len(all_results.matches),
                'text_results': len(filtered_content['text']),
                'image_results': len(filtered_content['images']),
                'table_results': len(filtered_content['tables'])
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error in multimodal query: {e}")
            return {
                'response': f"Error processing query: {str(e)}",
                'content_elements': [],
                'error': str(e)
            }
    
    def _apply_intelligent_filtering(self, query: str, text_results: List, 
                                   image_results: List, table_results: List) -> Dict[str, List]:
        """Apply intelligent filtering based on query context and content relevance."""
        
        # Analyze query intent
        query_lower = query.lower()
        query_intent = self._analyze_query_intent(query_lower)
        
        filtered_content = {'text': [], 'images': [], 'tables': []}
        
        # Filter text results
        for match in text_results:
            if match.score > 0.2:  # Higher threshold for text
                text_data = {
                    'content': match.metadata.get('content', ''),
                    'page_number': match.metadata.get('page_number', 0),
                    'relevance_score': match.score,
                    'chunk_id': match.metadata.get('chunk_id', ''),
                    'word_count': match.metadata.get('word_count', 0)
                }
                filtered_content['text'].append(text_data)
        
        # Filter image results with context awareness
        for match in image_results:
            # Adjust threshold based on query intent - lowered for better recall
            threshold = 0.10 if query_intent.get('visual_focus', False) else 0.12
            
            if match.score > threshold:
                image_data = {
                    'image_id': match.metadata.get('image_id', ''),
                    'local_path': match.metadata.get('local_path', ''),
                    's3_url': match.metadata.get('s3_url', ''),
                    'display_url': match.metadata.get('s3_url', '') or match.metadata.get('local_path', ''),
                    'page_number': match.metadata.get('page_number', 0),
                    'relevance_score': match.score,
                    'ocr_text': match.metadata.get('ocr_text', ''),
                    'keywords': match.metadata.get('keywords', []),
                    'content_type': match.metadata.get('content_classification', 'figure'),
                    'confidence_score': match.metadata.get('confidence_score', 0.0)
                }
                filtered_content['images'].append(image_data)
        
        # Filter table results with structure awareness and duplicate prevention
        seen_table_keys = set()
        
        for match in table_results:
            # Adjust threshold based on query intent - lowered for better recall
            threshold = 0.10 if query_intent.get('data_focus', False) else 0.15
            
            if match.score > threshold:
                table_id = match.metadata.get('table_id', '')
                page_number = match.metadata.get('page_number', 0)
                
                # Create unique key for this table
                unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_idx_{len(filtered_content['tables'])}"
                
                # Only add if not already seen
                if unique_key not in seen_table_keys:
                    seen_table_keys.add(unique_key)
                    
                    table_data = {
                        'table_id': table_id,
                        'page_number': page_number,
                        'relevance_score': match.score,
                        'summary': match.metadata.get('summary', ''),
                        'markdown_content': match.metadata.get('markdown_content', ''),
                        'row_count': match.metadata.get('row_count', 0),
                        'column_count': match.metadata.get('column_count', 0),
                        'structure_type': match.metadata.get('structure_type', 'unknown'),
                        'quality_score': match.metadata.get('overall_quality', 0.0)
                    }
                    filtered_content['tables'].append(table_data)
        
        return filtered_content
    
    def _analyze_query_intent(self, query_lower: str) -> Dict[str, bool]:
        """Analyze query to understand user intent."""
        intent = {
            'visual_focus': False,
            'data_focus': False,
            'detail_focus': False,
            'summary_focus': False
        }
        
        # Visual indicators
        visual_keywords = ['image', 'figure', 'chart', 'graph', 'diagram', 'picture', 'visual', 'show']
        if any(keyword in query_lower for keyword in visual_keywords):
            intent['visual_focus'] = True
        
        # Data indicators
        data_keywords = ['table', 'data', 'number', 'statistic', 'value', 'result', 'metric']
        if any(keyword in query_lower for keyword in data_keywords):
            intent['data_focus'] = True
        
        # Detail indicators
        detail_keywords = ['detail', 'specific', 'exact', 'precise', 'complete']
        if any(keyword in query_lower for keyword in detail_keywords):
            intent['detail_focus'] = True
        
        # Summary indicators
        summary_keywords = ['summary', 'overview', 'general', 'main', 'key']
        if any(keyword in query_lower for keyword in summary_keywords):
            intent['summary_focus'] = True
        
        return intent
    
    def _generate_multimodal_response(self, query: str, filtered_content: Dict, pdf_id: str) -> Dict[str, Any]:
        """Generate comprehensive multimodal response."""
        try:
            # Build context from all content types
            context_parts = []
            content_elements = []
            
            # Add text context
            for text_data in filtered_content['text']:
                context_parts.append(f"\n--- PAGE {text_data['page_number']} TEXT ---\n{text_data['content'][:1000]}")
                content_elements.append({
                    'type': 'text',
                    'page_number': text_data['page_number'],
                    'content_preview': text_data['content'][:500],
                    'relevance_score': text_data['relevance_score']
                })
            
            # Add image context
            for img_data in filtered_content['images']:
                if img_data.get('ocr_text'):
                    context_parts.append(f"\n--- PAGE {img_data['page_number']} IMAGE ---\n{img_data['ocr_text']}")
                
                content_elements.append({
                    'type': 'image',
                    'image_id': img_data['image_id'],
                    's3_url': img_data.get('s3_url', ''),
                    'display_url': img_data.get('display_url', ''),
                    'page_number': img_data['page_number'],
                    'ocr_text': img_data['ocr_text'],
                    'content_type': img_data['content_type'],
                    'relevance_score': img_data['relevance_score']
                })
            
            # Add table context
            for table_data in filtered_content['tables']:
                context_parts.append(f"\n--- PAGE {table_data['page_number']} TABLE ---\n{table_data['summary']}\n{table_data['markdown_content']}")
                
                content_elements.append({
                    'type': 'table',
                    'table_id': table_data['table_id'],
                    'page_number': table_data['page_number'],
                    'summary': table_data['summary'],
                    'markdown_content': table_data['markdown_content'],
                    'row_count': table_data['row_count'],
                    'column_count': table_data['column_count'],
                    'relevance_score': table_data['relevance_score']
                })
            
            full_context = '\n'.join(context_parts[:15])  # Limit context length
            
            # Generate AI response if we have context
            if full_context.strip():
                response_text = self._generate_ai_response(query, full_context)
            else:
                response_text = "I couldn't find relevant information in the document for this query."
            
            return {
                'response': response_text,
                'content_elements': content_elements,
                'context_used': bool(full_context.strip())
            }
            
        except Exception as e:
            logger.error(f"Error generating multimodal response: {e}")
            return {
                'response': f"Error generating response: {str(e)}",
                'content_elements': [],
                'error': str(e)
            }
    
    def _generate_ai_response(self, query: str, context: str) -> str:
        """Generate AI response using OpenAI API."""
        try:
            system_prompt = """You are an expert AI assistant that provides comprehensive answers using multimodal document content.

You have access to text content, images with OCR data, and structured tables. When responding:
1. Integrate information from all content types naturally
2. Reference specific pages when mentioning content
3. Highlight key insights from images and tables
4. Provide structured, well-organized responses
5. Be concise but comprehensive
6. Use the multimodal content to provide rich, detailed answers"""
            
            user_prompt = f"""Document content:
{context}

Question: {query}

Please provide a comprehensive answer using the multimodal content above. Reference specific pages and content types (text, images, tables) when relevant."""
            
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=1500,
                temperature=0.3,
                top_p=0.9
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating AI response: {e}")
            return f"Error generating response: {str(e)}"
    
    def display_multimodal_results(self, response_data: Dict[str, Any]):
        """Display intelligent inline multimodal results in Streamlit UI."""
        try:
            # Check if we have inline elements (new format)
            if 'inline_elements' in response_data:
                # New inline format
                self.inline_generator.display_inline_response(response_data['inline_elements'])
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
                text_count = results.get('text', {}).get('processed_chunks', 0)
                st.metric("Text Chunks", text_count)
            
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
                import tempfile
                import base64
                import re
                import requests
                
                # Use S3 storage
                s3_prefix = self.config.get_pdf_s3_prefix(pdf_name)
                # Create temporary directory for processing
                temp_dir = Path(tempfile.mkdtemp())
                pdf_image_dir = temp_dir
                logger.info(f"Using S3 storage with prefix: {s3_prefix}")
            else:
                import base64
                import re
                import requests
                
                # Use local storage
                pdf_image_dir = self.config.get_pdf_image_dir(pdf_name)
                logger.info(f"Using local storage: {pdf_image_dir}")
            
            extracted_count = 0
            json_data = json_objs[0] if json_objs else {}
            
            logger.info(f"Analyzing LlamaParse JSON structure for image extraction")
            
            # Method 1: Look for base64 images in markdown/text content
            all_text_content = []
            
            # Collect all text/markdown content
            for page_data in json_data.get('pages', []):
                page_text = page_data.get('text', '') + page_data.get('md', '')
                all_text_content.append(page_text)
            
            full_content = '\n'.join(all_text_content)
            
            # More comprehensive base64 pattern matching
            base64_patterns = [
                r'data:image/([^;]+);base64,([A-Za-z0-9+/=]+)',  # Standard data URL
                r'!\[.*?\]\(data:image/([^;]+);base64,([A-Za-z0-9+/=]+)\)',  # Markdown image
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
            
            # Create PDF-specific image directory
            pdf_image_dir = self.config.IMAGES_DIR / pdf_id
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
    

print("MultimodalIntegrator class created successfully")