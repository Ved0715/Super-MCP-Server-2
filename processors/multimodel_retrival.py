#!/usr/bin/env python3
"""
Server-Ready Multimodal Retrieval Function
Comprehensive function that reuses existing project components for server deployment
Enhanced with Universal Lesson Plan Generation for ANY instruction on ANY PDF content
"""

import os
import json
import time
import logging
import re
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime

# Import your existing project components
from .multimodal_integrator import MultimodalIntegrator
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UniversalLessonPlanGenerator:
    """Handles ANY lesson plan instruction for ANY part of uploaded PDF - integrates with existing system."""
    
    def __init__(self, multimodal_integrator: MultimodalIntegrator):
        """Initialize with existing multimodal integrator for seamless integration."""
        self.multimodal_integrator = multimodal_integrator
        self.openai_client = multimodal_integrator.openai_client
        self.chat_model = multimodal_integrator.chat_model
        
        # Instruction parsing patterns
        self.instruction_patterns = {
            'lesson_plan_triggers': [
                'lesson plan', 'teaching plan', 'course plan', 'curriculum', 'syllabus',
                'create lesson', 'design lesson', 'make lesson', 'plan for teaching',
                'educational plan', 'learning plan', 'instruction plan', 'course outline',
                'semester plan', 'unit plan', 'workshop plan', 'class plan'
            ],
            'chapter_patterns': [
                r'chapter\s*(\d+)',
                r'ch\s*(\d+)', 
                r'chapter\s*(\d+)\s*[-:]?\s*(.+?)(?:\s|$)',
                r'chapters?\s*(\d+)\s*[-–to]+\s*(\d+)',
                r'chapters?\s*([\d,\s-]+)',
                r'part\s*(\d+)',
                r'unit\s*(\d+)'
            ],
            'section_patterns': [
                r'section\s*(\d+(?:\.\d+)?)',
                r'section\s*(\d+(?:\.\d+)?)\s*[-:]?\s*(.+?)(?:\s|$)',
                r'subsection\s*(\d+(?:\.\d+)?)',
                r'(\d+\.\d+)\s+(.+?)(?:\s|$)',
                r'topic\s*(.+?)(?:\s|$)',
                r'part\s*(.+?)(?:\s|$)'
            ],
            'duration_patterns': [
                r'(\d+)\s*[-]?\s*hour[s]?',
                r'(\d+)\s*[-]?\s*minute[s]?',
                r'(\d+)\s*[-]?\s*day[s]?',
                r'(\d+)\s*[-]?\s*week[s]?',
                r'semester',
                r'quarterly',
                r'monthly'
            ],
            'audience_patterns': [
                r'beginner[s]?', r'basic', r'introductory', r'fundamental',
                r'advanced', r'intermediate', r'expert', r'professional',
                r'undergraduate', r'graduate', r'student[s]?', r'learner[s]?'
            ],
            'format_patterns': [
                r'interactive', r'hands[-\s]?on', r'practical', r'lab[-\s]?based',
                r'lecture', r'workshop', r'seminar', r'tutorial',
                r'activity[-\s]?based', r'project[-\s]?based'
            ],
            'assessment_patterns': [
                r'quiz[zes]?', r'test[s]?', r'assignment[s]?', r'homework',
                r'project[s]?', r'exercise[s]?', r'assessment[s]?', r'evaluation'
            ]
        }
    
    def is_lesson_plan_request(self, query: str) -> bool:
        """Detect if query is requesting a lesson plan."""
        query_lower = query.lower()
        return any(trigger in query_lower for trigger in self.instruction_patterns['lesson_plan_triggers'])
    
    def parse_lesson_instruction(self, query: str) -> Dict[str, Any]:
        """Parse comprehensive lesson plan instruction and extract all requirements."""
        query_lower = query.lower()
        
        instruction_info = {
            'lesson_type': self._detect_lesson_type(query_lower),
            'content_scope': self._extract_content_scope(query_lower),
            'duration': self._extract_duration(query_lower),
            'audience': self._extract_audience(query_lower),
            'format_style': self._extract_format_style(query_lower),
            'assessment_needs': self._extract_assessment_needs(query_lower),
            'special_requirements': self._extract_special_requirements(query),
            'search_strategy': self._determine_search_strategy(query_lower)
        }
        
        logger.info(f"📚 Lesson plan instruction parsed: {instruction_info['lesson_type']} for {instruction_info['content_scope']}")
        return instruction_info
    
    def _detect_lesson_type(self, query: str) -> str:
        """Detect the type of lesson plan requested."""
        if any(word in query for word in ['semester', 'course', 'full', 'complete', 'entire']):
            return 'semester_course'
        elif any(word in query for word in ['week', 'weekly', 'unit']):
            return 'weekly_unit'
        elif re.search(r'chapter[s]?\s*\d+\s*[-–to]+\s*\d+', query):
            return 'multiple_chapters'
        elif 'chapter' in query:
            return 'single_chapter'
        elif 'section' in query or re.search(r'\d+\.\d+', query):
            return 'section_focused'
        elif any(word in query for word in ['topic', 'concept', 'about', 'on']):
            return 'concept_focused'
        else:
            return 'general_content'
    
    def _extract_content_scope(self, query: str) -> Dict[str, Any]:
        """Extract what content to focus on."""
        scope = {
            'chapters': [],
            'sections': [],
            'topics': [],
            'coverage': 'specific'
        }
        
        # Extract chapter references
        chapter_matches = []
        for pattern in self.instruction_patterns['chapter_patterns']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            chapter_matches.extend(matches)
        
        if chapter_matches:
            scope['chapters'] = self._process_chapter_matches(chapter_matches)
        
        # Extract section references
        section_matches = []
        for pattern in self.instruction_patterns['section_patterns']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            section_matches.extend(matches)
        
        if section_matches:
            scope['sections'] = [m[0] if isinstance(m, tuple) else m for m in section_matches[:5]]
        
        # Extract topic/concept references
        topic_indicators = ['about', 'on', 'focusing on', 'covering', 'topic', 'concept']
        for indicator in topic_indicators:
            if indicator in query:
                parts = query.split(indicator, 1)
                if len(parts) > 1:
                    topic_part = parts[1].strip()[:100]
                    scope['topics'].append(topic_part)
        
        # Determine coverage type
        if any(word in query for word in ['entire', 'complete', 'full', 'comprehensive', 'all']):
            scope['coverage'] = 'comprehensive'
        elif any(word in query for word in ['overview', 'introduction', 'brief', 'summary']):
            scope['coverage'] = 'broad'
        
        return scope
    
    def _process_chapter_matches(self, matches: List) -> List[str]:
        """Process chapter number matches into clean list."""
        chapters = []
        
        for match in matches:
            if isinstance(match, tuple):
                if len(match) == 1:
                    chapters.append(f"Chapter {match[0]}")
                elif len(match) == 2:
                    if match[1] and not match[1].isdigit():
                        chapters.append(f"Chapter {match[0]}: {match[1]}")
                    elif match[1] and match[1].isdigit():
                        start, end = int(match[0]), int(match[1])
                        chapters.extend([f"Chapter {i}" for i in range(start, end + 1)])
            elif str(match).isdigit():
                chapters.append(f"Chapter {match}")
            elif ',' in str(match):
                nums = re.findall(r'\d+', str(match))
                chapters.extend([f"Chapter {num}" for num in nums])
        
        return list(set(chapters))[:10]
    
    def _extract_duration(self, query: str) -> Dict[str, Any]:
        """Extract duration requirements."""
        duration_info = {
            'time_amount': None,
            'time_unit': None,
            'suggested_duration': '60-90 minutes'
        }
        
        for pattern in self.instruction_patterns['duration_patterns']:
            matches = re.findall(pattern, query)
            if matches:
                if pattern == 'semester':
                    duration_info['suggested_duration'] = '16 weeks (full semester)'
                elif pattern in ['quarterly', 'monthly']:
                    duration_info['suggested_duration'] = '4-12 weeks'
                else:
                    amount = matches[0]
                    if 'hour' in pattern:
                        duration_info['suggested_duration'] = f"{amount} hours"
                    elif 'minute' in pattern:
                        duration_info['suggested_duration'] = f"{amount} minutes"
                    elif 'day' in pattern:
                        duration_info['suggested_duration'] = f"{amount} days"
                    elif 'week' in pattern:
                        duration_info['suggested_duration'] = f"{amount} weeks"
                break
        
        return duration_info
    
    def _extract_audience(self, query: str) -> str:
        """Extract target audience level."""
        for pattern in self.instruction_patterns['audience_patterns']:
            if re.search(pattern, query):
                return pattern.replace('[s]?', 's').replace('[-\\s]?', ' ')
        return 'general'
    
    def _extract_format_style(self, query: str) -> List[str]:
        """Extract teaching format preferences."""
        formats = []
        for pattern in self.instruction_patterns['format_patterns']:
            if re.search(pattern, query):
                formats.append(pattern.replace('[-\\s]?', ' '))
        return formats if formats else ['interactive']
    
    def _extract_assessment_needs(self, query: str) -> List[str]:
        """Extract assessment requirements."""
        assessments = []
        for pattern in self.instruction_patterns['assessment_patterns']:
            if re.search(pattern, query):
                assessments.append(pattern.replace('[-\\s]?', ' '))
        return assessments
    
    def _extract_special_requirements(self, query: str) -> List[str]:
        """Extract any special requirements or constraints."""
        special = []
        requirement_indicators = [
            'with examples', 'include examples', 'practical examples',
            'step by step', 'detailed', 'comprehensive', 'brief', 'quick',
            'for beginners', 'advanced level', 'with activities', 'interactive'
        ]
        
        for indicator in requirement_indicators:
            if indicator in query.lower():
                special.append(indicator)
        
        return special
    
    def _determine_search_strategy(self, query: str) -> Dict[str, Any]:
        """Determine how to search the content based on the instruction."""
        return {
            'focus_chapters': 'chapter' in query,
            'focus_sections': 'section' in query or re.search(r'\d+\.\d+', query),
            'focus_topics': any(word in query for word in ['topic', 'concept', 'about', 'on']),
            'broad_search': any(word in query for word in ['entire', 'complete', 'comprehensive']),
            'max_chunks_needed': self._estimate_chunks_needed(query)
        }
    
    def _estimate_chunks_needed(self, query: str) -> int:
        """Estimate how many chunks we'll need based on scope."""
        if any(word in query for word in ['semester', 'entire', 'complete', 'all']):
            return 50
        elif 'chapters' in query and re.search(r'\d+\s*[-–to]+\s*\d+', query):
            return 30
        elif 'chapter' in query:
            return 20
        elif 'section' in query:
            return 15
        else:
            return 25


class ServerMultimodalRetrieval:
    """Server-ready class for multimodal content retrieval using existing project components."""
    
    def __init__(self):
        """Initialize the retrieval system using existing MultimodalIntegrator."""
        try:
            self.multimodal_integrator = MultimodalIntegrator()
            
            # Initialize Universal Lesson Plan Generator with existing integrator
            self.lesson_plan_generator = UniversalLessonPlanGenerator(self.multimodal_integrator)
            
            logger.info("✅ Server multimodal retrieval system initialized successfully")
            logger.info("✅ Universal lesson plan generator initialized")
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
            
            # 🎓 NEW: Check for lesson plan request first
            if self.lesson_plan_generator.is_lesson_plan_request(query):
                logger.info(f"📚 Lesson plan request detected, routing to universal lesson plan generator")
                return await self._generate_universal_lesson_plan(query, paper_id, max_text_chunks)
            
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
    
    # 🎓 UNIVERSAL LESSON PLAN GENERATION METHODS
    async def _generate_universal_lesson_plan(self, query: str, paper_id: str, max_text_chunks: int) -> Dict[str, Any]:
        """Generate lesson plan for ANY instruction with comprehensive edge case handling."""
        
        try:
            # Step 1: Parse the instruction comprehensively
            instruction_info = self.lesson_plan_generator.parse_lesson_instruction(query)
            
            # Step 2: Adjust search parameters based on instruction
            search_chunks = instruction_info['search_strategy']['max_chunks_needed']
            
            # Step 3: Simple similarity search for lesson plan content
            results = self._retrieve_text_only_for_lesson_plan(
                query=query,  # Use original query directly for similarity search
                document_uuid=paper_id,
                max_chunks=search_chunks
            )
            
            if not results or 'inline_elements' not in results or not results['inline_elements']:
                return {
                    "search_metadata": {
                        "query": query,
                        "paper_id": paper_id,
                        "lesson_plan_mode": True,
                        "error": "No text content retrieved for lesson plan",
                        "retrieval_type": "text_only"
                    },
                    "success": False,
                    "error": "No text content found for lesson plan generation"
                }
            
            # Step 4: Extract ALL text content for lesson plan (no filtering)
            logger.info(f"📝 Processing {len(results['inline_elements'])} TEXT-ONLY elements for lesson plan")
            
            filtered_content = {
                'text_content': "",
                'metadata': {
                    'total_elements': len(results['inline_elements']),
                    'content_types': [],
                    'retrieval_type': results.get('retrieval_type', 'text_only')
                }
            }
            
            # Extract all text content from elements
            for element in results['inline_elements']:
                if element.get('type') == 'text':
                    content = element.get('content', '')
                    if content.strip():
                        filtered_content['text_content'] += content + "\n\n"
                        filtered_content['metadata']['content_types'].append('text')
            
            logger.info(f"📊 Text content extracted: {len(filtered_content['text_content'])} characters from {len(filtered_content['metadata']['content_types'])} text elements")
            
            if not filtered_content['text_content'].strip():
                return {
                    "search_metadata": {
                        "query": query,
                        "paper_id": paper_id,
                        "lesson_plan_mode": True,
                        "error": "No text content found"
                    },
                    "success": False,
                    "error": "No text content found for lesson plan generation"
                }
            
            # Step 5: Generate simple lesson plan from retrieved content
            lesson_plan_content = self._create_simple_lesson_plan(
                filtered_content['text_content'], query
            )
            
            # Step 6: Return in MCP server expected format (inline_elements at top level)
            return {
                "search_metadata": {
                    "query": query,
                    "paper_id": paper_id,
                    "lesson_plan_mode": True,
                    "instruction_type": instruction_info['lesson_type'],
                    "content_scope": instruction_info['content_scope'],
                    "timestamp": time.time()
                },
                "inline_elements": [{
                    "type": "text",
                    "content": lesson_plan_content,
                    "page_number": 1,
                    "relevance_score": 1.0,
                    "source": "universal_lesson_plan_generator",
                    "instruction_analysis": instruction_info,
                    "content_coverage": {
                        "text_chunks_retrieved": len(filtered_content['metadata']['content_types']),
                        "total_characters": len(filtered_content['text_content']),
                        "retrieval_type": filtered_content['metadata']['retrieval_type']
                    }
                }],
                "performance_metrics": {
                    "query_time": 0,
                    "total_results": 1,
                    "relevant_chunks": 1,
                    "search_strategy": "lesson_plan_text_only"
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Universal lesson plan generation failed: {str(e)}")
            return {
                "search_metadata": {
                    "query": query,
                    "paper_id": paper_id,
                    "lesson_plan_mode": True,
                    "error": str(e)
                },
                "success": False,
                "error": f"Lesson plan generation failed: {str(e)}"
            }
    
    def _enhance_query_for_search(self, original_query: str, instruction_info: Dict) -> str:
        """Enhance search query based on parsed instruction for better retrieval."""
        
        enhanced_parts = [original_query]
        
        # Add chapter-specific terms
        if instruction_info['content_scope']['chapters']:
            chapter_terms = ' '.join(instruction_info['content_scope']['chapters'])
            enhanced_parts.append(chapter_terms)
        
        # Add section-specific terms  
        if instruction_info['content_scope']['sections']:
            section_terms = ' '.join(instruction_info['content_scope']['sections'])
            enhanced_parts.append(section_terms)
        
        # Add topic-specific terms
        if instruction_info['content_scope']['topics']:
            topic_terms = ' '.join(instruction_info['content_scope']['topics'])
            enhanced_parts.append(topic_terms)
        
        # Add educational context for better retrieval
        enhanced_parts.append("learning objectives teaching education curriculum concepts")
        
        enhanced_query = ' '.join(enhanced_parts)
        logger.info(f"🔍 Enhanced search query: {enhanced_query[:100]}...")
        return enhanced_query
    
    def _filter_content_by_instruction(self, elements: List[Dict], instruction_info: Dict) -> Dict[str, Any]:
        """Filter retrieved content to match specific instruction requirements."""
        
        filtered = {
            'text_content': "",
            'metadata': {
                'chapters_found': [],
                'sections_found': [],
                'topics_covered': []
            },
            'coverage_info': {
                'requested_chapters': instruction_info['content_scope']['chapters'],
                'found_chapters': [],
                'requested_sections': instruction_info['content_scope']['sections'],
                'found_sections': [],
                'coverage_percentage': 0
            }
        }
        
        content_scope = instruction_info['content_scope']
        relevant_elements = []
        
        for element in elements:
            if element.get('type') not in ['text', 'composite']:
                continue
            
            element_content = element.get('content', '')
            element_relevant = False
            
            # Check chapter relevance
            if content_scope['chapters']:
                for requested_chapter in content_scope['chapters']:
                    # Check if any chapter matches in the element content
                    for chapter in content_scope['chapters']:
                        if chapter.lower() in element_content.lower():
                            element_relevant = True
                            # Collect found chapters from metadata if available
                            if 'chapters_found' in element:
                                filtered['metadata']['chapters_found'].extend(element.get('chapters_found', []))
                            break
                    if element_relevant:
                        break
            
            # Check section relevance  
            elif content_scope['sections']:
                for requested_section in content_scope['sections']:
                    if requested_section.lower() in element_content.lower():
                        element_relevant = True
                        if 'sections_found' in element:
                            filtered['metadata']['sections_found'].extend(element.get('sections_found', []))
                        break
            
            # Check topic relevance
            elif content_scope['topics']:
                for topic in content_scope['topics']:
                    if topic.lower() in element_content.lower():
                        element_relevant = True
                        break
            
            # If no specific scope, include all text elements
            else:
                element_relevant = True
            
            if element_relevant:
                relevant_elements.append(element)
                filtered['text_content'] += element_content + "\n\n"
                
                # Collect metadata
                if 'chapters_found' in element:
                    filtered['metadata']['chapters_found'].extend(element.get('chapters_found', []))
                if 'sections_found' in element:
                    filtered['metadata']['sections_found'].extend(element.get('sections_found', []))
        
        # Calculate coverage percentage
        requested_chapters = set(content_scope['chapters'])
        found_chapters = set(filtered['metadata']['chapters_found'])
        
        if requested_chapters:
            coverage = len(requested_chapters & found_chapters) / len(requested_chapters)
            filtered['coverage_info']['coverage_percentage'] = int(coverage * 100)
            filtered['coverage_info']['found_chapters'] = list(found_chapters & requested_chapters)
        else:
            # If no specific chapters requested, coverage is 100% if we found any content
            filtered['coverage_info']['coverage_percentage'] = 100 if filtered['text_content'].strip() else 0
        
        logger.info(f"📊 Content filtered: {filtered['coverage_info']['coverage_percentage']}% coverage, {len(filtered['text_content'])} chars")
        return filtered
    
    def _create_adaptive_lesson_plan(self, filtered_content: Dict, instruction_info: Dict, original_query: str) -> str:
        """Create lesson plan with adaptive prompt based on instruction analysis."""
        
        # Build adaptive prompt based on instruction
        prompt_parts = []
        
        # Role and context
        prompt_parts.append(
            "You are a highly experienced university professor with 20+ years in curriculum development. "
            "Create a comprehensive lesson plan based on the specific instruction provided."
        )
        
        # Add instruction-specific context
        lesson_type = instruction_info['lesson_type']
        content_scope = instruction_info['content_scope']
        duration = instruction_info['duration']['suggested_duration']
        audience = instruction_info['audience']
        format_styles = instruction_info['format_style']
        assessments = instruction_info['assessment_needs']
        
        # Lesson type specific instructions
        if lesson_type == 'semester_course':
            prompt_parts.append(
                f"SCOPE: Full semester course plan covering extensive content\n"
                f"DURATION: {duration}\n"
                f"STRUCTURE: Weekly breakdown with modules, assessments, and progression"
            )
        elif lesson_type == 'multiple_chapters':
            chapters_str = ', '.join(content_scope['chapters'])
            prompt_parts.append(
                f"SCOPE: Multi-chapter lesson plan covering: {chapters_str}\n"
                f"DURATION: {duration}\n" 
                f"STRUCTURE: Integrated approach connecting concepts across chapters"
            )
        elif lesson_type == 'single_chapter':
            chapter_str = content_scope['chapters'][0] if content_scope['chapters'] else 'specified chapter'
            prompt_parts.append(
                f"SCOPE: Single chapter focused lesson: {chapter_str}\n"
                f"DURATION: {duration}\n"
                f"STRUCTURE: Deep dive into chapter concepts with comprehensive coverage"
            )
        elif lesson_type == 'section_focused':
            sections_str = ', '.join(content_scope['sections'])
            prompt_parts.append(
                f"SCOPE: Section-specific lesson covering: {sections_str}\n"
                f"DURATION: {duration}\n"
                f"STRUCTURE: Detailed focus on specific sections and subsections"
            )
        elif lesson_type == 'concept_focused':
            topics_str = ', '.join(content_scope['topics'])
            prompt_parts.append(
                f"SCOPE: Concept-focused lesson on: {topics_str}\n"
                f"DURATION: {duration}\n"
                f"STRUCTURE: Topic-centered approach with in-depth exploration"
            )
        else:
            prompt_parts.append(
                f"SCOPE: General content lesson plan\n"
                f"DURATION: {duration}\n"
                f"STRUCTURE: Comprehensive approach to provided content"
            )
        
        # Audience and format specifications
        if audience != 'general':
            prompt_parts.append(f"TARGET AUDIENCE: {audience.title()} level students")
        
        if format_styles:
            format_str = ', '.join(format_styles)
            prompt_parts.append(f"TEACHING FORMAT: {format_str} approach")
        
        if assessments:
            assessment_str = ', '.join(assessments)
            prompt_parts.append(f"ASSESSMENT REQUIREMENTS: Include {assessment_str}")
        
        # Coverage information - simplified for lesson plans
        total_elements = filtered_content['metadata']['total_elements']
        prompt_parts.append(f"CONTENT AVAILABLE: {total_elements} content sections retrieved for lesson planning")
        
        # Special requirements
        special_reqs = instruction_info['special_requirements']
        if special_reqs:
            special_str = ', '.join(special_reqs)
            prompt_parts.append(f"SPECIAL REQUIREMENTS: {special_str}")
        
        # Format requirements
        prompt_parts.append("""
REQUIRED FORMAT:
**LESSON PLAN: [Create appropriate title based on scope]**

**Course Information:**
- Scope: [Based on instruction analysis]
- Duration: [As specified or suggested]
- Level: [Based on audience]
- Format: [Based on format requirements]

**Learning Objectives:**
[3-5 specific, measurable objectives aligned with content scope]

**Prerequisites:**
[Based on content complexity and audience level]

**Content Outline:**
[Structured breakdown matching the requested scope - chapters/sections/topics]

**Teaching Methods & Activities:**
[Align with format style requirements - interactive, practical, etc.]

**Assessment & Evaluation:**
[Include specific assessment types if requested]

**Resources & Materials:**
[Content-specific resources and supplementary materials]

**Timeline & Pacing:**
[Break down the lesson/course timing based on duration]

**Homework & Follow-up:**
[Appropriate post-lesson activities]

**Edge Case Handling:**
[If content is limited or scope is unclear, provide adaptive suggestions]""")
        
        # Add the actual content - limit to 2000 chars to avoid token limits
        content_preview = filtered_content['text_content'][:2000]
        prompt_parts.append(f"""
ORIGINAL INSTRUCTION: {original_query}

CONTENT TO USE:
{content_preview}

Generate the complete lesson plan now:""")
        
        final_prompt = '\n\n'.join(prompt_parts)
        
        try:
            # Single LLM call with your existing OpenAI client
            response = self.multimodal_integrator.openai_client.chat.completions.create(
                model=self.multimodal_integrator.chat_model,  # Use your existing chat model
                messages=[{"role": "user", "content": final_prompt}],
                max_tokens=1500,  # Comprehensive lesson plan
                temperature=0.1,
                timeout=25
            )
            
            lesson_plan = response.choices[0].message.content
            logger.info("✅ Adaptive lesson plan generated successfully")
            return lesson_plan
            
        except Exception as e:
            logger.error(f"❌ Adaptive lesson plan generation failed: {e}")
            return self._create_emergency_fallback_lesson(instruction_info, filtered_content)
    
    def _retrieve_text_only_for_lesson_plan(self, query: str, document_uuid: str, max_chunks: int = 10) -> Dict[str, Any]:
        """
        Text-only retrieval specifically for lesson plans.
        Directly queries Pinecone for text metadata only - no images/tables.
        """
        logger.info(f"🔍 TEXT-ONLY retrieval for lesson plan: query='{query}', doc='{document_uuid}', max_chunks={max_chunks}")
        
        try:
            # Use the multimodal_integrator's OpenAI client but query only for text
            embedding_response = self.multimodal_integrator.openai_client.embeddings.create(
                input=query,
                model=self.multimodal_integrator.embedding_model  # Use same model as multimodal integrator
            )
            query_embedding = embedding_response.data[0].embedding
            
            # Query Pinecone directly for ALL chunks (no filter needed)
            namespace = document_uuid
            query_response = self.multimodal_integrator.index.query(
                vector=query_embedding,
                top_k=max_chunks,
                namespace=namespace,
                include_metadata=True
            )
            
            if not query_response.matches:
                logger.warning(f"No matches found for query: {query}")
                return {"inline_elements": [], "text_chunks_found": 0}
            
            # Convert to simple text elements format - extract text field from metadata
            text_elements = []
            for match in query_response.matches:
                metadata = match.metadata or {}
                text_content = metadata.get('text', '')
                
                if text_content.strip():
                    text_elements.append({
                        'type': 'text',
                        'content': text_content.strip(),
                        'metadata': {
                            'page': metadata.get('page_number', 'unknown'),
                            'score': float(match.score),
                            'chunk_id': match.id,
                            'has_text': metadata.get('has_text', False),
                            'contains_image': metadata.get('contains_image', False),
                            'contains_table': metadata.get('contains_table', False)
                        }
                    })
            
            logger.info(f"✅ TEXT-ONLY retrieval: {len(text_elements)} text chunks retrieved")
            
            return {
                'inline_elements': text_elements,
                'text_chunks_found': len(text_elements),
                'retrieval_type': 'text_only_for_lesson_plan'
            }
            
        except Exception as e:
            logger.error(f"❌ TEXT-ONLY retrieval failed: {e}")
            return {"inline_elements": [], "text_chunks_found": 0, "error": str(e)}

    def _create_simple_lesson_plan(self, relevant_content: str, query: str) -> str:
        """Create lesson plan directly from relevant retrieved content."""
        
        try:
            # Simple prompt that uses the retrieved content directly
            prompt = f"""You are an expert educator. Create a comprehensive lesson plan based on the content provided below.

USER REQUEST: {query}

RELEVANT CONTENT FROM DOCUMENT:
{relevant_content}

Create a detailed lesson plan that:
1. Uses ONLY the information provided in the content above
2. Structures the lesson based on what's actually covered in the content
3. Includes learning objectives based on the content
4. Provides a clear outline and timeline
5. Is appropriate for the subject matter found in the content

Format the lesson plan professionally with clear sections and structure."""

            # Generate lesson plan using OpenAI
            response = self.multimodal_integrator.openai_client.chat.completions.create(
                model=self.multimodal_integrator.chat_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000
            )
            
            lesson_plan = response.choices[0].message.content.strip()
            logger.info(f"✅ Simple lesson plan generated: {len(lesson_plan)} characters")
            return lesson_plan
            
        except Exception as e:
            logger.error(f"❌ Simple lesson plan generation failed: {e}")
            return f"# Lesson Plan\n\nBased on: {query}\n\nContent Summary:\n{relevant_content[:500]}...\n\nPlease refer to the above content for lesson details."

    def _create_emergency_fallback_lesson(self, instruction_info: Dict, content: Dict) -> str:
        """Emergency fallback when LLM fails - comprehensive template-based lesson."""
        
        lesson_type = instruction_info['lesson_type']
        duration = instruction_info['duration']['suggested_duration']
        scope_info = instruction_info['content_scope']
        
        # Determine title
        if scope_info['chapters']:
            title = f"Lesson Plan: {', '.join(scope_info['chapters'][:3])}"
        elif scope_info['sections']:
            title = f"Section-Focused Lesson: {', '.join(scope_info['sections'][:2])}"
        elif scope_info['topics']:
            title = f"Topic Lesson: {', '.join(scope_info['topics'][:2])}"
        else:
            title = "Academic Lesson Plan"
        
        fallback = f"""**{title}**

**Course Information:**
- Scope: {lesson_type.replace('_', ' ').title()}
- Duration: {duration}
- Level: {instruction_info['audience'].title()}
- Created: {datetime.now().strftime('%Y-%m-%d')}

**Learning Objectives:**
• Master core concepts from the specified content
• Apply theoretical knowledge through practical examples
• Demonstrate understanding through assessments
• Develop critical thinking and analytical skills

**Prerequisites:**
Completion of foundational coursework and basic understanding of related concepts.

**Content Outline:**"""
        
        # Add content-specific outline
        if scope_info['chapters']:
            fallback += f"\n• Chapter Coverage: {', '.join(scope_info['chapters'])}"
        if scope_info['sections']:
            fallback += f"\n• Section Focus: {', '.join(scope_info['sections'])}"
        if scope_info['topics']:
            fallback += f"\n• Topic Areas: {', '.join(scope_info['topics'])}"
        
        fallback += """
• Core theoretical foundations
• Practical applications and examples
• Problem-solving techniques and methods
• Real-world connections and case studies

**Teaching Methods & Activities:**"""
        
        # Add format-specific methods
        formats = instruction_info['format_style']
        if 'interactive' in formats:
            fallback += "\n• Interactive discussions and group activities"
        if any(f in formats for f in ['hands-on', 'practical']):
            fallback += "\n• Hands-on exercises and practical applications"
        if 'lab-based' in formats:
            fallback += "\n• Laboratory sessions and experiments"
        
        fallback += """
• Structured presentations and lectures
• Q&A sessions and clarifications
• Problem-solving workshops
• Peer learning and collaboration

**Assessment & Evaluation:**"""
        
        # Add assessment-specific methods
        assessments = instruction_info['assessment_needs']
        if any(a in assessments for a in ['quiz', 'test']):
            fallback += "\n• Quiz questions and knowledge checks"
        if 'assignment' in assessments:
            fallback += "\n• Written assignments and projects"
        if 'homework' in assessments:
            fallback += "\n• Homework exercises and practice problems"
        
        fallback += """
• Formative assessments during lessons
• Summative evaluations at completion
• Class participation and engagement
• Practical demonstrations of learning

**Resources & Materials:**
• Primary textbook and assigned readings
• Supplementary articles and references
• Online resources and digital materials
• Practice exercises and example problems

**Timeline & Pacing:**"""
        
        if 'semester' in duration:
            fallback += "\n• Weekly modules with progressive complexity"
            fallback += "\n• Mid-semester review and assessments"
            fallback += "\n• Final project and comprehensive evaluation"
        elif 'week' in duration:
            fallback += "\n• Daily sessions building on each other"
            fallback += "\n• Weekly review and consolidation"
        else:
            fallback += "\n• Introduction and context setting (20%)"
            fallback += "\n• Core content delivery (60%)"
            fallback += "\n• Review and assessment (20%)"
        
        fallback += """

**Homework & Follow-up:**
• Review and reinforce lesson concepts
• Complete assigned readings and exercises
• Prepare for upcoming assessments
• Research additional examples and applications

**Adaptive Notes:**
This lesson plan has been generated to match your specific instruction. The content has been adapted based on available materials. If you need modifications for different audience levels, duration changes, or additional assessment types, please specify your requirements."""

        logger.info("🛡️ Emergency fallback lesson plan created")
        return fallback

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