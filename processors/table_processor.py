#!/usr/bin/env python3
"""
Table Processor for LlamaParse Integration
Specialized module for detecting, extracting, and storing tables from JSON with comprehensive metadata
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import re
import time
import pandas as pd
import numpy as np

# Third-party imports
import pinecone
from pinecone import Pinecone
import openai
from openai import OpenAI
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TableProcessor:
    """Specialized processor for LlamaParse table detection and Pinecone storage."""
    
    def __init__(self, openai_client: OpenAI, pinecone_client: Pinecone, index):
        self.openai_client = openai_client
        self.pinecone_client = pinecone_client
        self.index = index
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-3-large')
        
        # Table processing settings
        self.min_table_rows = 2
        self.max_table_preview_length = 500
        self.table_formats = ['markdown', 'html', 'csv', 'json']
        
        # No local storage - tables stored only in Pinecone
        
    def process_tables_from_llamaparse(self, json_objs: List[Dict], pdf_name: str, pdf_id: str, skip_pinecone: bool = False) -> Dict[str, Any]:
        """Process all tables from LlamaParse JSON output."""
        try:
            start_time = time.time()
            
            # Extract tables from JSON using multiple detection methods
            all_tables = self._detect_all_tables_from_json(json_objs, pdf_id)
            
            if not all_tables:
                logger.warning(f"No tables found in {pdf_name}")
                return {
                    'processed_tables': [],
                    'saved_count': 0,
                    'pinecone_stored': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Process each table comprehensively
            processed_tables = []
            
            for i, table_data in enumerate(all_tables):
                try:
                    processed_table = self._process_single_table(
                        table_data, i, pdf_id, pdf_name
                    )
                    
                    if processed_table:
                        processed_tables.append(processed_table)
                        # No local saving - tables stored only in Pinecone
                        
                except Exception as e:
                    logger.error(f"Error processing table {i}: {e}")
                    continue
            
            # Store in Pinecone (skip if flag is set)
            pinecone_stored = 0
            if not skip_pinecone:
                pinecone_stored = self._store_tables_in_pinecone(processed_tables, pdf_id)
            else:
                logger.info("Skipping individual table storage in Pinecone (composite chunks will be stored instead)")
            
            processing_time = time.time() - start_time
            
            logger.info(f"Table processing complete: {len(processed_tables)} processed, "
                       f"{pinecone_stored} stored in Pinecone (no local storage)")
            
            return {
                'processed_tables': processed_tables,
                'saved_count': len(processed_tables),  # All tables "saved" to Pinecone
                'pinecone_stored': pinecone_stored,
                'processing_time': processing_time,
                'storage_type': 'pinecone_only'
            }
            
        except Exception as e:
            logger.error(f"Error in table processing: {e}")
            raise
    
    def _detect_all_tables_from_json(self, json_objs: List[Dict], pdf_id: str) -> List[Dict]:
        """Detect tables using multiple methods from LlamaParse JSON structure."""
        all_tables = []
        
        if not json_objs:
            return all_tables
        
        json_data = json_objs[0]
        
        # Primary method: Extract tables from pages -> items -> type:"table" structure
        try:
            primary_tables = self._extract_tables_from_items_json_structure(json_data, pdf_id)
            all_tables.extend(primary_tables)
            logger.info(f"Primary JSON structure extraction found {len(primary_tables)} tables")
        except Exception as primary_error:
            logger.error(f"Primary table extraction failed: {primary_error}")
        
        # Fallback method 1: Direct table extraction from pages (if primary failed)
        try:
            if len(all_tables) == 0:
                fallback_tables = self._extract_tables_from_pages(json_data, pdf_id)
                all_tables.extend(fallback_tables)
                logger.info(f"Fallback pages extraction found {len(fallback_tables)} tables")
        except Exception as fallback_error:
            logger.warning(f"Fallback pages extraction failed: {fallback_error}")
        
        # Fallback method 2: Pattern-based extraction (if both above failed)
        try:
            if len(all_tables) == 0:
                pattern_tables = self._extract_tables_from_text_patterns(json_data, pdf_id)
                all_tables.extend(pattern_tables)
                logger.info(f"Pattern-based extraction found {len(pattern_tables)} tables")
        except Exception as pattern_error:
            logger.warning(f"Pattern-based extraction failed: {pattern_error}")
        
        # Fallback method 3: Markdown extraction (last resort)
        try:
            if len(all_tables) == 0:
                markdown_tables = self._extract_tables_from_markdown(json_data, pdf_id)
                all_tables.extend(markdown_tables)
                logger.info(f"Markdown extraction found {len(markdown_tables)} tables")
        except Exception as markdown_error:
            logger.warning(f"Markdown extraction failed: {markdown_error}")
        
        # Remove duplicates based on position and content similarity
        unique_tables = self._deduplicate_tables(all_tables)
        
        logger.info(f"Detected {len(unique_tables)} unique tables from {len(all_tables)} candidates")
        return unique_tables
    
    def _extract_tables_from_pages(self, json_data: Dict, pdf_id: str) -> List[Dict]:
        """Extract tables directly from pages structure."""
        tables = []
        
        for page_data in json_data.get('pages', []):
            page_num = page_data.get('page', 0)
            page_tables = page_data.get('tables', [])
            
            for table_idx, table_data in enumerate(page_tables):
                enhanced_table = {
                    **table_data,
                    'pdf_id': pdf_id,
                    'page_number': page_num,
                    'table_index': table_idx,
                    'detection_method': 'direct_pages',
                    'table_id': f"{pdf_id}_p{page_num}_t{table_idx}"
                }
                tables.append(enhanced_table)
        
        return tables
    
    def _extract_tables_from_items_json_structure(self, json_data: Dict, pdf_id: str) -> List[Dict]:
        """Extract tables from pages->items->type:'table' JSON structure with comprehensive error handling."""
        tables = []
        
        try:
            logger.info("Extracting tables from LlamaParse JSON structure: pages->items->type:'table'")
            
            pages = json_data.get('pages', [])
            if not pages:
                logger.warning("No pages found in JSON data")
                return tables
            
            for page_idx, page_data in enumerate(pages):
                try:
                    page_num = page_data.get('page', page_idx + 1)  # Fallback to index+1
                    items = page_data.get('items', [])
                    
                    logger.debug(f"Processing page {page_num}: {len(items)} items found")
                    
                    table_count = 0
                    for item_idx, item in enumerate(items):
                        try:
                            item_type = item.get('type', '')
                            
                            if item_type == 'table':
                                table_count += 1
                                logger.info(f"Found table {table_count} on page {page_num} (item {item_idx})")
                                
                                # Process the table with comprehensive error handling
                                processed_table = self._process_table_from_llamaparse_item(
                                    item, page_num, table_count, pdf_id, item_idx
                                )
                                
                                if processed_table:
                                    tables.append(processed_table)
                                    logger.info(f"Successfully processed table {processed_table['table_id']}")
                                else:
                                    logger.warning(f"Failed to process table on page {page_num}, item {item_idx}")
                                    
                        except Exception as item_error:
                            logger.error(f"Error processing item {item_idx} on page {page_num}: {item_error}")
                            continue
                            
                except Exception as page_error:
                    logger.error(f"Error processing page {page_idx}: {page_error}")
                    continue
            
            logger.info(f"Total tables extracted from JSON structure: {len(tables)}")
            return tables
            
        except Exception as e:
            logger.error(f"Critical error in table extraction from JSON structure: {e}")
            return tables  # Return whatever we managed to extract
    
    def _process_table_from_llamaparse_item(self, item: Dict, page_num: int, table_count: int, 
                                           pdf_id: str, item_idx: int) -> Optional[Dict]:
        """Process a single table item from LlamaParse JSON with extensive error handling."""
        try:
            table_id = f"{pdf_id}_p{page_num}_table{table_count}"
            
            # Initialize table content dictionary
            table_content = {}
            
            # Method 1: Extract from 'rows' structure (primary method based on JSON analysis)
            if 'rows' in item and item['rows']:
                try:
                    rows = item['rows']
                    if isinstance(rows, list) and len(rows) >= self.min_table_rows:
                        table_content['raw_rows'] = rows
                        
                        # Convert to multiple formats with error handling
                        try:
                            formats = self._convert_table_to_formats(rows)
                            table_content.update(formats)
                            logger.debug(f"Converted table to {len(formats)} formats")
                        except Exception as format_error:
                            logger.warning(f"Error converting table formats: {format_error}")
                            # Continue with raw_rows at least
                        
                    else:
                        logger.warning(f"Table rows invalid or too few: {len(rows) if isinstance(rows, list) else 'not a list'}")
                        
                except Exception as rows_error:
                    logger.warning(f"Error processing rows structure: {rows_error}")
            
            # Method 2: Extract from 'md' (markdown) field
            if 'md' in item and item['md'] and not table_content:
                try:
                    markdown_content = item['md']
                    if self._looks_like_table_markdown(markdown_content):
                        table_content['markdown'] = markdown_content
                        table_content['raw_text'] = markdown_content
                        logger.debug("Extracted table from markdown field")
                except Exception as md_error:
                    logger.warning(f"Error processing markdown field: {md_error}")
            
            # Method 3: Extract from 'csv' field
            if 'csv' in item and item['csv'] and not table_content:
                try:
                    csv_content = item['csv']
                    table_content['csv'] = csv_content
                    table_content['raw_text'] = csv_content
                    logger.debug("Extracted table from CSV field")
                except Exception as csv_error:
                    logger.warning(f"Error processing CSV field: {csv_error}")
            
            # If no content found, skip this table
            if not table_content:
                logger.warning(f"No extractable content found in table item on page {page_num}")
                logger.debug(f"Available item keys: {list(item.keys())}")
                return None
            
            # Calculate dimensions with error handling
            row_count = self._calculate_row_count(table_content, item)
            column_count = self._calculate_column_count(table_content, item)
            
            # Create enhanced table structure
            enhanced_table = {
                'pdf_id': pdf_id,
                'page_number': page_num,
                'table_index': table_count,
                'detection_method': 'llamaparse_json_items',
                'table_id': table_id,
                
                # Content in various formats
                **table_content,
                
                # Metadata from JSON item (with safe extraction)
                'bounding_box': item.get('bBox', item.get('bbox', {})),
                'is_perfect_table': item.get('isPerfectTable', False),
                'confidence': item.get('confidence', 1.0),
                
                # Calculated dimensions
                'row_count': row_count,
                'column_count': column_count,
                
                # Store original item for debugging
                'raw_item_data': item,
                'extraction_metadata': {
                    'source': 'llamaparse_json',
                    'item_index': item_idx,
                    'available_fields': list(item.keys()),
                    'extraction_timestamp': datetime.now().isoformat()
                }
            }
            
            return enhanced_table
            
        except Exception as e:
            logger.error(f"Critical error processing table from JSON item on page {page_num}: {e}")
            return None
    
    def _looks_like_table_markdown(self, content: str) -> bool:
        """Check if markdown content looks like a table."""
        if not content or len(content.strip()) < 10:
            return False
        
        lines = content.split('\n')
        table_lines = [line for line in lines if '|' in line]
        return len(table_lines) >= 2  # At least header and one data row
    
    def _calculate_row_count(self, table_content: Dict, item: Dict) -> int:
        """Calculate row count with fallbacks."""
        try:
            if 'raw_rows' in table_content:
                return len(table_content['raw_rows'])
            elif 'markdown' in table_content:
                lines = [line for line in table_content['markdown'].split('\n') if '|' in line]
                return max(1, len(lines) - 1)  # Subtract separator line
            elif 'csv' in table_content:
                lines = table_content['csv'].split('\n')
                return len([line for line in lines if line.strip()])
            return 1
        except Exception as e:
            logger.warning(f"Error calculating row count: {e}")
            return 1
    
    def _calculate_column_count(self, table_content: Dict, item: Dict) -> int:
        """Calculate column count with fallbacks."""
        try:
            if 'raw_rows' in table_content and table_content['raw_rows']:
                first_row = table_content['raw_rows'][0]
                return len(first_row) if isinstance(first_row, list) else 1
            elif 'markdown' in table_content:
                first_line = table_content['markdown'].split('\n')[0]
                return max(1, first_line.count('|') - 1) if first_line.startswith('|') else first_line.count('|') + 1
            elif 'csv' in table_content:
                first_line = table_content['csv'].split('\n')[0]
                return len(first_line.split(','))
            return 1
        except Exception as e:
            logger.warning(f"Error calculating column count: {e}")
            return 1
    
    def _extract_tables_from_text_patterns(self, json_data: Dict, pdf_id: str) -> List[Dict]:
        """Extract tables by detecting patterns in text content."""
        tables = []
        
        for page_data in json_data.get('pages', []):
            page_num = page_data.get('page', 0)
            text = page_data.get('text', '') or page_data.get('md', '')
            
            if not text:
                continue
            
            # Look for table-like patterns
            table_patterns = self._find_table_patterns_in_text(text)
            
            for pattern_idx, pattern in enumerate(table_patterns):
                enhanced_table = {
                    'pdf_id': pdf_id,
                    'page_number': page_num,
                    'table_index': pattern_idx,
                    'detection_method': 'text_patterns',
                    'table_id': f"{pdf_id}_p{page_num}_tp{pattern_idx}",
                    
                    # Content
                    'raw_text': pattern['text'],
                    'markdown': pattern.get('markdown', ''),
                    'estimated_rows': pattern['rows'],
                    'estimated_columns': pattern['columns'],
                    
                    # Position in text
                    'text_position': pattern['position'],
                    'confidence_score': pattern['confidence']
                }
                tables.append(enhanced_table)
        
        return tables
    
    def _extract_tables_from_markdown(self, json_data: Dict, pdf_id: str) -> List[Dict]:
        """Extract tables from markdown content."""
        tables = []
        
        for page_data in json_data.get('pages', []):
            page_num = page_data.get('page', 0)
            markdown_content = page_data.get('md', '')
            
            if not markdown_content:
                continue
            
            # Find markdown tables
            markdown_tables = self._find_markdown_tables(markdown_content)
            
            for table_idx, table_data in enumerate(markdown_tables):
                enhanced_table = {
                    'pdf_id': pdf_id,
                    'page_number': page_num,
                    'table_index': table_idx,
                    'detection_method': 'markdown_extraction',
                    'table_id': f"{pdf_id}_p{page_num}_md{table_idx}",
                    
                    # Content
                    'markdown': table_data['markdown'],
                    'raw_text': table_data['raw_text'],
                    'row_count': table_data['rows'],
                    'column_count': table_data['columns'],
                    
                    # Quality metrics
                    'completeness_score': table_data['completeness'],
                    'structure_score': table_data['structure_quality']
                }
                tables.append(enhanced_table)
        
        return tables
    
    def _convert_table_to_formats(self, rows: List[List]) -> Dict[str, str]:
        """Convert table rows to multiple formats optimized for LlamaParse structure."""
        formats = {}
        
        try:
            # Convert to markdown
            markdown_rows = []
            for i, row in enumerate(rows):
                if isinstance(row, list):
                    # Clean and escape cells
                    cleaned_cells = []
                    for cell in row:
                        cell_str = str(cell).strip().replace('|', '\\|').replace('\n', ' ')
                        cleaned_cells.append(cell_str)
                    
                    markdown_row = '| ' + ' | '.join(cleaned_cells) + ' |'
                    markdown_rows.append(markdown_row)
                    
                    # Add header separator after first row
                    if i == 0 and len(rows) > 1:
                        separator = '|' + ''.join([' --- |' for _ in row])
                        markdown_rows.append(separator)
            
            formats['markdown'] = '\n'.join(markdown_rows)
            
            # Convert to CSV
            csv_rows = []
            for row in rows:
                if isinstance(row, list):
                    cleaned_row = [str(cell).replace(',', ';').replace('\n', ' ') for cell in row]
                    csv_rows.append(','.join(cleaned_row))
            formats['csv'] = '\n'.join(csv_rows)
            
            # Convert to HTML (basic)
            html_parts = ['<table>']
            for i, row in enumerate(rows):
                if isinstance(row, list):
                    tag = 'th' if i == 0 else 'td'
                    cells = [f'<{tag}>{str(cell)}</{tag}>' for cell in row]
                    html_parts.append(f'<tr>{"".join(cells)}</tr>')
            html_parts.append('</table>')
            formats['html'] = '\n'.join(html_parts)
            
            # Convert to structured JSON (enhanced for LlamaParse compatibility)
            if rows and len(rows) > 0:
                # Extract headers from first row
                headers = []
                data_rows = []
                
                if len(rows) > 1:
                    # Use first row as headers if it looks like headers
                    first_row = rows[0]
                    if all(isinstance(cell, str) and len(str(cell).strip()) > 0 for cell in first_row):
                        headers = [str(cell).strip() for cell in first_row]
                        data_rows = rows[1:]
                    else:
                        # Generate generic headers
                        headers = [f"Column_{i+1}" for i in range(len(first_row))]
                        data_rows = rows
                else:
                    # Single row - generate headers
                    headers = [f"Column_{i+1}" for i in range(len(rows[0]))]
                    data_rows = rows
                
                # Create structured table JSON (compatible with multimodal_integrator.py)
                table_dict = {}
                
                # Add each column as a key with its data
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
                    "source": "table_processor_enhanced"
                }
                
                formats['json'] = json.dumps(table_dict, indent=2)
            
        except Exception as e:
            logger.warning(f"Error converting table formats: {e}")
        
        return formats
    
    def _find_table_patterns_in_text(self, text: str) -> List[Dict]:
        """Find table-like patterns in plain text."""
        patterns = []
        
        # Split text into lines
        lines = text.split('\n')
        
        # Look for consecutive lines with table-like patterns
        current_table_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Check if line looks like a table row
            if self._is_table_like_line(line):
                current_table_lines.append((i, line))
            else:
                # Process accumulated table lines
                if len(current_table_lines) >= self.min_table_rows:
                    table_pattern = self._analyze_table_pattern(current_table_lines)
                    if table_pattern['confidence'] > 0.5:
                        patterns.append(table_pattern)
                
                current_table_lines = []
        
        # Handle table at end of text
        if len(current_table_lines) >= self.min_table_rows:
            table_pattern = self._analyze_table_pattern(current_table_lines)
            if table_pattern['confidence'] > 0.5:
                patterns.append(table_pattern)
        
        return patterns
    
    def _is_table_like_line(self, line: str) -> bool:
        """Check if a line looks like it could be part of a table."""
        if not line:
            return False
        
        # Common table indicators
        table_separators = ['|', '\t', '  ']  # Multiple spaces
        separator_count = sum(1 for sep in table_separators if sep in line)
        
        # Lines with multiple separators are likely table rows
        if separator_count > 0 and len(line.split()) >= 2:
            return True
        
        # Lines with consistent spacing patterns
        if re.search(r'\s{3,}', line):  # 3 or more spaces
            return True
        
        return False
    
    def _analyze_table_pattern(self, table_lines: List[Tuple[int, str]]) -> Dict:
        """Analyze a group of lines to determine if they form a valid table."""
        if not table_lines:
            return {'confidence': 0.0}
        
        lines_text = [line[1] for line in table_lines]
        
        # Estimate columns by analyzing separator patterns
        column_estimates = []
        
        for line in lines_text:
            # Count potential separators
            pipe_count = line.count('|')
            tab_count = line.count('\t')
            multi_space_count = len(re.findall(r'\s{2,}', line))
            
            estimated_cols = max(pipe_count + 1, tab_count + 1, multi_space_count + 1)
            column_estimates.append(estimated_cols)
        
        # Check consistency
        if column_estimates:
            avg_cols = sum(column_estimates) / len(column_estimates)
            col_consistency = 1.0 - (max(column_estimates) - min(column_estimates)) / max(max(column_estimates), 1)
        else:
            avg_cols = 0
            col_consistency = 0
        
        # Calculate confidence
        confidence = 0.0
        
        # More lines = higher confidence
        confidence += min(0.4, len(table_lines) / 10)
        
        # Consistent column count = higher confidence
        confidence += col_consistency * 0.3
        
        # Presence of separators = higher confidence
        has_separators = any('|' in line or '\t' in line for line in lines_text)
        if has_separators:
            confidence += 0.3
        
        # Create markdown format
        markdown_lines = []
        for line in lines_text:
            # Convert to markdown format
            if '|' not in line:
                # Add pipes if not present
                parts = re.split(r'\s{2,}|\t', line)
                markdown_line = '| ' + ' | '.join(parts) + ' |'
            else:
                markdown_line = line
            
            markdown_lines.append(markdown_line)
        
        # Add header separator
        if len(markdown_lines) > 1:
            first_line_cols = markdown_lines[0].count('|') - 1
            separator = '|' + ''.join([' --- |' for _ in range(max(first_line_cols, 1))])
            markdown_lines.insert(1, separator)
        
        return {
            'text': '\n'.join(lines_text),
            'markdown': '\n'.join(markdown_lines),
            'rows': len(table_lines),
            'columns': int(avg_cols),
            'confidence': min(1.0, confidence),
            'position': table_lines[0][0]  # Line number of first row
        }
    
    def _find_markdown_tables(self, markdown_content: str) -> List[Dict]:
        """Extract well-formed markdown tables."""
        tables = []
        
        # Split into sections
        sections = markdown_content.split('\n\n')
        
        for section in sections:
            lines = section.strip().split('\n')
            table_lines = [line for line in lines if '|' in line]
            
            if len(table_lines) >= 2:
                # Check for header separator
                has_separator = any('---' in line or '===' in line for line in table_lines[:3])
                
                if has_separator:
                    # Analyze table structure
                    header_line = table_lines[0]
                    separator_line = None
                    data_lines = []
                    
                    for line in table_lines[1:]:
                        if '---' in line or '===' in line:
                            separator_line = line
                        else:
                            data_lines.append(line)
                    
                    # Count columns
                    header_cols = header_line.count('|') - 1 if header_line.startswith('|') else header_line.count('|') + 1
                    
                    # Calculate completeness
                    total_cells = header_cols * (len(data_lines) + 1)  # +1 for header
                    filled_cells = sum(1 for line in [header_line] + data_lines 
                                     for cell in line.split('|') if cell.strip())
                    completeness = filled_cells / max(total_cells, 1)
                    
                    # Structure quality
                    consistent_cols = all(
                        line.count('|') == header_line.count('|') 
                        for line in data_lines
                    )
                    structure_quality = 1.0 if consistent_cols else 0.7
                    
                    table_data = {
                        'markdown': '\n'.join(table_lines),
                        'raw_text': section,
                        'rows': len(data_lines) + 1,  # Include header
                        'columns': header_cols,
                        'completeness': completeness,
                        'structure_quality': structure_quality
                    }
                    tables.append(table_data)
        
        return tables
    
    def _deduplicate_tables(self, tables: List[Dict]) -> List[Dict]:
        """Remove duplicate tables based on content similarity."""
        if not tables:
            return []
        
        unique_tables = []
        
        for table in tables:
            is_duplicate = False
            
            # Check against existing tables
            for existing_table in unique_tables:
                if self._are_tables_similar(table, existing_table):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_tables.append(table)
        
        return unique_tables
    
    def _are_tables_similar(self, table1: Dict, table2: Dict, threshold: float = 0.8) -> bool:
        """Check if two tables are similar enough to be considered duplicates."""
        # Same page and similar position
        if (table1.get('page_number') == table2.get('page_number') and
            abs(table1.get('table_index', 0) - table2.get('table_index', 0)) <= 1):
            
            # Similar content size
            content1 = table1.get('markdown', '') or table1.get('raw_text', '')
            content2 = table2.get('markdown', '') or table2.get('raw_text', '')
            
            if content1 and content2:
                # Simple similarity check based on length and common words
                len_ratio = min(len(content1), len(content2)) / max(len(content1), len(content2))
                
                if len_ratio > threshold:
                    words1 = set(content1.lower().split())
                    words2 = set(content2.lower().split())
                    
                    if words1 and words2:
                        word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                        return word_similarity > threshold
        
        return False
    
    def _process_single_table(self, table_data: Dict, index: int, pdf_id: str, pdf_name: str) -> Optional[Dict]:
        """Process a single table with comprehensive metadata extraction."""
        try:
            table_id = table_data.get('table_id', f"{pdf_id}_table_{index}")
            
            # Extract table content in multiple formats
            content_formats = self._extract_table_content_formats(table_data)
            
            # Generate table summary and keywords
            summary = self._generate_table_summary(content_formats)
            keywords = self._extract_table_keywords(content_formats)
            
            # Analyze table structure
            structure_analysis = self._analyze_table_structure(table_data, content_formats)
            
            # Calculate quality metrics
            quality_metrics = self._calculate_table_quality(table_data, content_formats)
            
            # Create comprehensive table metadata
            processed_table = {
                'table_id': table_id,
                'pdf_id': pdf_id,
                'pdf_name': pdf_name,
                'page_number': table_data.get('page_number', 0),
                'table_index': table_data.get('table_index', index),
                
                # Content in multiple formats
                'content_formats': content_formats,
                
                # Analysis results
                'summary': summary,
                'keywords': keywords,
                'structure_analysis': structure_analysis,
                'quality_metrics': quality_metrics,
                
                # Metadata
                'detection_method': table_data.get('detection_method', 'unknown'),
                'row_count': structure_analysis.get('estimated_rows', 0),
                'column_count': structure_analysis.get('estimated_columns', 0),
                
                # Processing metadata
                'processed_timestamp': datetime.now().isoformat(),
                
                # Vector embedding text
                'embedding_text': self._create_table_embedding_text(content_formats, summary, keywords)
            }
            
            logger.info(f"Processed table {table_id}: {processed_table['row_count']} rows, "
                       f"{processed_table['column_count']} columns")
            
            return processed_table
            
        except Exception as e:
            logger.error(f"Error processing single table {index}: {e}")
            return None
    
    def _extract_table_content_formats(self, table_data: Dict) -> Dict[str, str]:
        """Extract table content in all available formats."""
        formats = {}
        
        # Direct format extraction
        for format_type in self.table_formats:
            if format_type in table_data:
                formats[format_type] = table_data[format_type]
        
        # Raw content
        if 'raw_text' in table_data:
            formats['raw_text'] = table_data['raw_text']
        
        # If we have raw rows, convert to formats
        if 'raw_rows' in table_data and table_data['raw_rows']:
            converted_formats = self._convert_table_to_formats(table_data['raw_rows'])
            formats.update(converted_formats)
        
        return formats
    
    def _generate_table_summary(self, content_formats: Dict[str, str]) -> str:
        """Generate a descriptive summary of table content."""
        # Use the best available format for summary generation
        content = ""
        
        if 'markdown' in content_formats:
            content = content_formats['markdown']
        elif 'raw_text' in content_formats:
            content = content_formats['raw_text']
        elif 'csv' in content_formats:
            content = content_formats['csv']
        
        if not content:
            return "Table with structured data"
        
        # Extract key information
        lines = content.split('\n')[:10]  # First 10 lines
        
        # Look for patterns that suggest content type
        content_lower = content.lower()
        
        summary_parts = []
        
        # Identify data types
        if any(indicator in content_lower for indicator in ['total', 'sum', 'amount', 'cost', 'price']):
            summary_parts.append("financial data")
        if any(indicator in content_lower for indicator in ['name', 'person', 'contact']):
            summary_parts.append("contact information")
        if any(indicator in content_lower for indicator in ['date', 'time', 'year']):
            summary_parts.append("temporal data")
        if any(indicator in content_lower for indicator in ['result', 'score', 'rating']):
            summary_parts.append("performance metrics")
        
        # Build summary
        if summary_parts:
            summary = f"Table containing {', '.join(summary_parts)}"
        else:
            summary = "Structured data table"
        
        # Add size information
        row_count = len([line for line in lines if line.strip() and '|' in line])
        if row_count > 0:
            summary += f" with approximately {row_count} rows"
        
        return summary
    
    def _extract_table_keywords(self, content_formats: Dict[str, str]) -> List[str]:
        """Extract keywords from table content."""
        # Combine all text content
        all_text = ""
        for format_content in content_formats.values():
            if isinstance(format_content, str):
                all_text += " " + format_content
        
        if not all_text:
            return []
        
        # Extract meaningful words
        words = re.findall(r'\b[a-zA-Z][a-zA-Z0-9_-]*[a-zA-Z0-9]\b|\b[a-zA-Z]\b', all_text.lower())
        
        # Filter out common words and table markup
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'table', 'row', 'column', 'cell', 'data', 'value', 'item', 'list'
        }
        
        # Count word frequency
        word_counts = {}
        for word in words:
            if word not in stop_words and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:15]]
    
    def _analyze_table_structure(self, table_data: Dict, content_formats: Dict[str, str]) -> Dict[str, Any]:
        """Analyze table structure and organization."""
        analysis = {
            'estimated_rows': 0,
            'estimated_columns': 0,
            'has_header': False,
            'structure_type': 'unknown'
        }
        
        # Use the best format for analysis
        if 'markdown' in content_formats:
            content = content_formats['markdown']
            lines = [line for line in content.split('\n') if '|' in line]
            
            if lines:
                analysis['estimated_rows'] = len(lines)
                # Check for header separator
                analysis['has_header'] = any('---' in line for line in lines)
                
                # Estimate columns from first line
                first_line = lines[0]
                analysis['estimated_columns'] = first_line.count('|') - 1 if first_line.startswith('|') else first_line.count('|') + 1
        
        elif 'raw_rows' in table_data:
            rows = table_data['raw_rows']
            analysis['estimated_rows'] = len(rows)
            analysis['estimated_columns'] = len(rows[0]) if rows and isinstance(rows[0], list) else 0
            analysis['has_header'] = True  # Assume structured data has headers
        
        # Determine structure type
        if analysis['estimated_rows'] > 10:
            analysis['structure_type'] = 'large_dataset'
        elif analysis['has_header']:
            analysis['structure_type'] = 'structured_data'
        else:
            analysis['structure_type'] = 'simple_table'
        
        return analysis
    
    def _calculate_table_quality(self, table_data: Dict, content_formats: Dict[str, str]) -> Dict[str, float]:
        """Calculate quality metrics for the table."""
        metrics = {
            'completeness': 0.0,
            'structure_consistency': 0.0,
            'content_clarity': 0.0,
            'overall_quality': 0.0
        }
        
        try:
            # Completeness based on available formats
            format_score = len(content_formats) / len(self.table_formats)
            metrics['completeness'] = min(1.0, format_score)
            
            # Structure consistency
            if 'markdown' in content_formats:
                lines = content_formats['markdown'].split('\n')
                table_lines = [line for line in lines if '|' in line]
                
                if table_lines:
                    column_counts = [line.count('|') for line in table_lines]
                    if column_counts:
                        consistency = 1.0 - (max(column_counts) - min(column_counts)) / max(max(column_counts), 1)
                        metrics['structure_consistency'] = consistency
            
            # Content clarity based on text quality
            if content_formats:
                text = ' '.join(content_formats.values())
                if text:
                    # Simple heuristic: longer content with varied words = higher clarity
                    words = text.split()
                    unique_words = set(words)
                    clarity = min(1.0, len(unique_words) / max(len(words), 1) * 2)
                    metrics['content_clarity'] = clarity
            
            # Overall quality
            metrics['overall_quality'] = (
                metrics['completeness'] * 0.3 +
                metrics['structure_consistency'] * 0.4 +
                metrics['content_clarity'] * 0.3
            )
            
        except Exception as e:
            logger.warning(f"Error calculating table quality metrics: {e}")
        
        return metrics
    
    def _create_table_embedding_text(self, content_formats: Dict[str, str], 
                                   summary: str, keywords: List[str]) -> str:
        """Create text for vector embedding generation."""
        embedding_parts = []
        
        # Add summary
        embedding_parts.append(summary)
        
        # Add keywords
        if keywords:
            embedding_parts.append(f"Keywords: {' '.join(keywords)}")
        
        # Add full content (use markdown if available, otherwise best format)
        if 'markdown' in content_formats:
            full_content = content_formats['markdown']
        elif 'raw_text' in content_formats:
            full_content = content_formats['raw_text']
        elif content_formats:
            full_content = list(content_formats.values())[0]
        else:
            full_content = ""
        
        if full_content:
            embedding_parts.append(f"Table content: {full_content}")
        
        embedding_parts.append("Structured table data from document")
        
        return ' | '.join(embedding_parts)
    
    # Local table saving removed - tables stored only in Pinecone vectors
    

    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API."""
        try:
            if not text or len(text.strip()) < 3:
                return [0.0] * 3072  # Default embedding size
            
            response = self.openai_client.embeddings.create(
                model=self.embedding_model,
                input=text[:8191]  # OpenAI token limit
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return [0.0] * 3072
    


print("TableProcessor class created successfully")