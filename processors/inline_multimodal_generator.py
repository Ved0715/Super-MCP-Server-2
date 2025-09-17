#!/usr/bin/env python3
"""
Inline Multimodal Response Generator
Creates intelligent inline responses with contextual integration of text, images, and tables
"""

import re
import logging
from typing import List, Dict, Any, Tuple
import streamlit as st

logger = logging.getLogger(__name__)

class InlineMultimodalGenerator:
    """Generates intelligent inline multimodal responses with contextual content integration."""
    
    def __init__(self, openai_client, chat_model: str = 'gpt-4o-mini'):
        self.openai_client = openai_client
        self.chat_model = chat_model
    
    def generate_inline_response_from_chunks(self, query: str, chunks: List[Dict], conditional_instructions: Dict = None, response_scope: str = 'standard') -> Dict[str, Any]:
        """Generate inline response directly from raw chunks - simpler processing."""
        try:
            # Simple content extraction from chunks
            filtered_content = {
                'text': [],
                'images': [],
                'tables': []
            }
            
            # Extract content from chunks without complex filtering
            for chunk in chunks:
                # Add text content with smart extraction
                if chunk.get('text'):
                    # Apply smart text extraction to get only relevant sentences/paragraphs
                    original_text = chunk['text']
                    extracted_text = self._extract_relevant_text(original_text, query)
                    
                    # Use extracted text if it's substantially shorter, otherwise use original
                    extraction_applied = len(extracted_text) < len(original_text) * 0.7
                    final_text = extracted_text if extraction_applied else original_text
                    
                    # Log smart extraction usage
                    if extraction_applied:
                        logger.info(f"Smart extraction applied: Page {chunk.get('page_number', 0)} text reduced from {len(original_text)} to {len(extracted_text)} chars")
                    
                    filtered_content['text'].append({
                        'content': final_text,
                        'relevance_score': chunk.get('relevance_score', 0),
                        'page_number': chunk.get('page_number', 0),
                        'extraction_applied': extraction_applied
                    })
                
                # Add image content from arrays  
                if chunk.get('contains_image') and chunk.get('image_s3_urls'):
                    image_urls = chunk.get('image_s3_urls', [])
                    image_summaries = chunk.get('image_summaries', [])
                    image_ids = chunk.get('image_ids', [])
                    
                    # Process each image in the arrays
                    for i, image_url in enumerate(image_urls):
                        image_summary = image_summaries[i] if i < len(image_summaries) else ''
                        image_id = image_ids[i] if i < len(image_ids) else f'img_{i}'
                        
                        filtered_content['images'].append({
                            'image_summary': image_summary,
                            's3_url': image_url,
                            'image_id': image_id,
                            'page_number': chunk.get('page_number', 0),
                            'relevance_score': chunk.get('relevance_score', 0)
                        })
                
                # Add table content from arrays with validation
                if chunk.get('contains_table') and chunk.get('table_content_jsons'):
                    table_jsons = chunk.get('table_content_jsons', [])
                    table_summaries = chunk.get('table_summaries', [])
                    table_ids = chunk.get('table_ids', [])
                    
                    # Process each table in the arrays
                    for i, table_json in enumerate(table_jsons):
                        table_summary = table_summaries[i] if i < len(table_summaries) else ''
                        table_id = table_ids[i] if i < len(table_ids) else f'table_{i}'
                        
                        # Validate table before adding
                        if self._is_table_valid(table_json):
                            # Check if table has meaningful content and relevance
                            if self._has_meaningful_table_content(table_json, query):
                                filtered_content['tables'].append({
                                    'table_content_json': table_json,
                                    'table_id': table_id,
                                    'page_number': chunk.get('page_number', 0),
                                    'relevance_score': chunk.get('relevance_score', 0)
                                })
                                logger.info(f"Added valid table {table_id} from page {chunk.get('page_number', 0)}")
                            else:
                                logger.info(f"Filtered out low-relevance table {table_id} from page {chunk.get('page_number', 0)} - insufficient meaningful content")
                        else:
                            logger.info(f"Filtered out invalid table {table_id} from page {chunk.get('page_number', 0)} - failed validation checks")
            
            # Use the existing response generation logic
            return self.generate_inline_response(query, filtered_content, conditional_instructions, response_scope)
            
        except Exception as e:
            logger.error(f"Error in generate_inline_response_from_chunks: {e}")
            return {'inline_elements': [{'type': 'text', 'content': f"Error generating response: {str(e)}"}]}

    def generate_inline_response(self, query: str, filtered_content: Dict, conditional_instructions: Dict = None, response_scope: str = 'standard') -> Dict[str, Any]:
        """Generate intelligent inline multimodal response."""
        try:
            # Analyze content availability from filtered content
            input_has_text = len(filtered_content['text']) > 0
            input_has_images = len(filtered_content['images']) > 0  
            input_has_tables = len(filtered_content['tables']) > 0
        
            # If no content found, return helpful fallback response
            if not (input_has_text or input_has_images or input_has_tables):
                return self._create_fallback_response(query)

            # Create content map sorted by relevance
            content_map = self._create_content_map(filtered_content, query)

            # Generate response with natural multimedia placement using single regeneration prompt
            logger.info("Generating response with natural multimedia placement...")
            response_structure = self._generate_comprehensive_response(query, content_map, input_has_text, input_has_images, input_has_tables, response_scope)
            validation_message = "Single comprehensive generation with natural multimedia placement"
            logger.info(f"Response generated using natural placement strategy")

            # Create inline elements for display
            inline_elements = self._create_inline_elements(response_structure, content_map)
        
        # Analyze final inline elements for accurate content summary
            final_has_text = any(elem.get('type') == 'text' for elem in inline_elements)
            final_has_images = any(elem.get('type') == 'image' for elem in inline_elements)
            final_has_tables = any(elem.get('type') == 'table' for elem in inline_elements)
        
            return {
                'inline_elements': inline_elements,
                'content_summary': {
                    'has_text': final_has_text,
                    'has_images': final_has_images, 
                    'has_tables': final_has_tables,
                    'total_elements': len(inline_elements)
                },
                'inline_placement_quality': validation_message
            }
        except Exception as e:
            logger.error(f"Error generating inline response: {e}")
            return {
                'inline_elements': [{'type': 'text', 'content': f"Error generating response: {str(e)}"}],
                'content_summary': {'has_text': False, 'has_images': False, 'has_tables': False, 'total_elements': 1},
                'inline_placement_quality': 'Error occurred during generation'
            }
    
    def _create_fallback_response(self, query: str) -> Dict[str, Any]:
        """Create a fallback response when no content is found."""
        return {
            'inline_elements': [
                {
                    'type': 'text', 
                    'content': f"I couldn't find specific content matching '{query}' in the document. This might be because:\n\n• The content doesn't contain those specific terms\n• The similarity threshold is too high\n• The document hasn't been fully processed yet\n\nTry using broader search terms or check if the document was processed successfully."
                }
            ],
            'content_summary': {'has_text': False, 'has_images': False, 'has_tables': False, 'total_elements': 1}
        }
    
    def _create_content_map(self, filtered_content: Dict, query: str) -> Dict[str, List]:
        """Create a dynamic content map using chain-of-thought reasoning."""
        # Check for "show all" type queries
        query_lower = query.lower()
        show_all_keywords = ['show all', 'all the', 'every', 'complete', 'entire', 'full', 'comprehensive']
        is_show_all_query = any(keyword in query_lower for keyword in show_all_keywords)
        
        # Remove duplicate images by image_id and URL (Phase 2 enhancement)
        unique_images = []
        seen_image_ids = set()
        
        # Filter images by relevance score - only keep the most relevant ones
        sorted_images = sorted(filtered_content['images'], key=lambda x: x.get('relevance_score', 0), reverse=True)

        # Smart image selection: Only select the most relevant images that will be used
        max_images = 2  # Conservative limit - only select images that will definitely be used
        relevance_threshold = 0.2  # Higher threshold for more selective filtering

        # Process images in relevance order and select only the best ones
        for image in sorted_images:
            relevance_score = image.get('relevance_score', 0)

            # For show-all queries, be more permissive
            effective_threshold = 0.1 if is_show_all_query else relevance_threshold
            effective_max = 6 if is_show_all_query else max_images

            # Skip low-relevance images
            if relevance_score < effective_threshold:
                logger.debug(f"Filtered out low-relevance image: relevance={relevance_score:.3f} < threshold={effective_threshold}")
                continue

            # Stop if we've reached the maximum
            if len(unique_images) >= effective_max:
                logger.info(f"Reached maximum images limit ({effective_max}), selecting only top {len(unique_images)} most relevant images")
                break

            image_id = image.get('image_id', '')
            s3_url = image.get('s3_url', '')
            page_number = image.get('page_number', 0)

            # Create a unique identifier using multiple criteria for better deduplication
            unique_key = f"{image_id}_{page_number}" if image_id else f"{s3_url}_{page_number}" if s3_url else f"page_{page_number}_image_{len(unique_images)}"

            if unique_key not in seen_image_ids:
                unique_images.append(image)
                seen_image_ids.add(unique_key)
                logger.info(f"Selected image {len(unique_images)}: {unique_key} (relevance: {relevance_score:.3f})")
            else:
                logger.debug(f"Skipped duplicate image: {unique_key}")

        logger.info(f"Smart image selection complete: {len(unique_images)} high-quality images selected from {len(sorted_images)} available")

        # Remove duplicate tables by table_id
        unique_tables = []
        seen_table_ids = set()
        
        for table in sorted(filtered_content['tables'], key=lambda x: x['relevance_score'], reverse=True):
            table_id = table.get('table_id', '')
            page_number = table.get('page_number', 0)
            
            # Create a unique identifier using both table_id and page
            unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_table_{len(unique_tables)}"
            
            if unique_key not in seen_table_ids:
                unique_tables.append(table)
                seen_table_ids.add(unique_key)
                logger.debug(f"Added unique table: {unique_key}")
            else:
                logger.debug(f"Skipped duplicate table: {unique_key}")
        
        # Create unified content list for better inline integration
        unified_content = []
        
        # Add text content with type identifier
        for i, text_item in enumerate(filtered_content['text']):
            unified_content.append({
                'type': 'text',
                'content': text_item.get('content', ''),
                'page_number': text_item.get('page_number', 0),
                'relevance_score': text_item.get('relevance_score', 0),
                'section_id': f"TEXT_SECTION_{i+1}"
            })
        
        # Add image content with type identifier (using unique_images to prevent duplicates)
        for i, img_item in enumerate(unique_images):
            unified_content.append({
                'type': 'image',
                'data': img_item,
                'page_number': img_item.get('page_number', 0),
                'relevance_score': img_item.get('relevance_score', 0),
                'section_id': f"IMAGE_{i+1}"
            })
        
        # Add table content with type identifier
        for i, table_item in enumerate(unique_tables):
            unified_content.append({
                'type': 'table',
                'data': table_item,
                'page_number': table_item.get('page_number', 0),
                'relevance_score': table_item.get('relevance_score', 0),
                'section_id': f"TABLE_{i+1}"
            })
        
        # Sort unified content by relevance score for optimal integration
        unified_content.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Dynamic content limits based on query type and available content
        if is_show_all_query:
            # For "show all" queries, include all available content
            content_limit = len(unified_content)
        else:
            # For specific queries, limit to most relevant content
            content_limit = min(20, len(unified_content))
        
        # Take top content based on limit
        unified_content = unified_content[:content_limit]
        
        # Reconstruct type-specific maps for backward compatibility
        content_map = {
            'text': [item for item in unified_content if item['type'] == 'text'],
            'images': [item['data'] for item in unified_content if item['type'] == 'image'],
            'tables': [item['data'] for item in unified_content if item['type'] == 'table'],
            'unified': unified_content  # New unified content list
        }
        
        # Debug logging
        logger.info(f"Dynamic content map created with {len(content_map['tables'])} unique tables")
        logger.info(f"Query type: {'show_all' if is_show_all_query else 'specific'}")
        logger.info(f"Content map summary: {len(content_map['images'])} images, {len(content_map['tables'])} tables, {len(content_map['text'])} text sections")
        logger.info(f"Unified content list: {len(content_map['unified'])} items sorted by relevance")
        
        return content_map
    
    def _generate_comprehensive_response(self, query: str, content_map: Dict, has_text: bool, has_images: bool, has_tables: bool, response_scope: str = 'standard') -> str:
        """Generate comprehensive response with natural multimedia placement in a single call."""

        # Build comprehensive context
        context_parts = []
        if has_text:
            context_parts.append(f"Available text content from {len(content_map['text'])} sections")
        if has_images:
            context_parts.append(f"Available visual content: {len(content_map['images'])} images with descriptions")
        if has_tables:
            context_parts.append(f"Available structured data: {len(content_map['tables'])} tables")

        # Create detailed text content for AI to use
        text_content_sections = []
        for i, text_item in enumerate(content_map['text']):
            content = text_item.get('content', '')
            page_num = text_item.get('page_number', 0)
            relevance = text_item.get('relevance_score', 0)
            if content and len(content.strip()) > 10:
                text_content_sections.append(f"TEXT_SECTION_{i+1} (Page {page_num}, relevance: {relevance:.2f}):\n{content}")

        # Create multimedia content analysis
        content_analysis = []

        # Analyze images with context
        for i, img in enumerate(content_map['images']):
            img_desc = img.get('image_summary', img.get('ocr_text', 'Visual content'))
            page_num = int(img['page_number']) if isinstance(img['page_number'], (int, float)) else img['page_number']
            relevance = img.get('relevance_score', 0)
            contextual_desc = img_desc[:300] if img_desc and len(img_desc.strip()) > 10 else f"visual content from page {img['page_number']}"
            content_analysis.append(f"IMAGE_{i+1} (Page {page_num}, relevance: {relevance:.2f}): {contextual_desc}")

        # Analyze tables with structure
        for i, table in enumerate(content_map['tables']):
            table_summary = self._get_enhanced_table_summary(table)
            page_num = int(table['page_number']) if isinstance(table['page_number'], (int, float)) else table['page_number']
            relevance = table.get('relevance_score', 0)
            contextual_desc = table_summary if table_summary else "structured data"

            if table.get('table_content_json'):
                try:
                    import json
                    table_json = json.loads(table['table_content_json'])
                    if isinstance(table_json, dict) and '_metadata' in table_json:
                        metadata = table_json['_metadata']
                        contextual_desc += f" containing {metadata.get('total_rows', 0)} rows and {metadata.get('total_columns', 0)} columns"
                    headers = [k for k in table_json.keys() if k != '_metadata']
                    if headers:
                        contextual_desc += f" with columns including {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}"
                except:
                    pass
            content_analysis.append(f"TABLE_{i+1} (Page {page_num}, relevance: {relevance:.2f}): {contextual_desc}")

        # Build comprehensive single prompt
        text_content_prompt = ""
        if text_content_sections:
            text_content_prompt = f"\nDOCUMENT TEXT CONTENT:\n{chr(10).join(text_content_sections)}\n"

        system_prompt = f"""You are an expert AI assistant that creates comprehensive responses with natural multimedia integration.

AVAILABLE CONTENT:
{', '.join(context_parts)}
{text_content_prompt}

MULTIMEDIA CONTENT DETAILS:
{chr(10).join(content_analysis)}

MULTIMEDIA REFERENCES (use these exact placeholders in your response):
{self._generate_placeholder_list(content_map)}

USER'S QUESTION: "{query}"

INSTRUCTIONS:
Create a comprehensive, well-structured response that naturally integrates multimedia content. Follow these guidelines:

1. NATURAL PLACEMENT: Place images and tables naturally where they support your explanation. Don't force them into specific patterns - let the content flow guide placement.

2. CONTEXTUAL INTEGRATION: When referencing multimedia, use natural language like:
   - "As shown in the image below"
   - "The following table illustrates"
   - "Refer to the image for visual context"
   - "The data in the table demonstrates"
   - CRITICAL: DO NOT use markdown image syntax like ![...] or specific placeholders like IMAGE_1, IMAGE_2
   - CRITICAL: Only use general references like "the image below" or "the following table"
   - The system will automatically place multimedia content - you just write the text

3. CONTENT UTILIZATION: Use the full text content provided to create a comprehensive response. Build your explanation around the available text, supplemented by multimedia elements.

4. LOGICAL STRUCTURE: Organize your response logically with clear sections, headings, and smooth transitions between concepts.

5. MULTIMEDIA PLACEMENT STRATEGY:
   - Only use the most relevant images that directly support your explanation
   - Place images after relevant explanatory text where they add clear value
   - Insert tables when you need to present structured data
   - Don't cluster all multimedia at the end or beginning
   - Let the narrative flow determine optimal placement
   - CRITICAL: Only reference multimedia that you actually use in your response
   - IMPORTANT: Don't reference IMAGE_2, IMAGE_3, etc. unless you actually place them in your text
   - Better to use fewer, more relevant multimedia than to create broken references

6. RESPONSE SCOPE: Aim for a {response_scope} level response that thoroughly addresses the query.

Generate your complete response now, naturally incorporating the multimedia placeholders where they best support your explanation:"""

        try:
            # Single comprehensive LLM call
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=4000,  # Increased for comprehensive response
                temperature=0.3,
                timeout=60
            )

            generated_response = response.choices[0].message.content.strip()
            logger.info(f"Comprehensive response generated: {len(generated_response)} characters")
            return generated_response

        except Exception as e:
            logger.error(f"Error in comprehensive response generation: {e}")
            # Fallback to original method if needed
            return self._generate_response_structure(query, content_map, has_text, has_images, has_tables, response_scope)

    def _generate_response_structure(self, query: str, content_map: Dict, has_text: bool, has_images: bool, has_tables: bool, response_scope: str = 'standard') -> str:
        """Generate intelligent response structure with seamless multimodal integration."""
        
        # Build comprehensive context for AI with detailed content analysis
        context_parts = []
        
        # Add available content descriptions
        if has_text:
            context_parts.append(f"Available text content from {len(content_map['text'])} sections")
        if has_images:
            context_parts.append(f"Available visual content: {len(content_map['images'])} images with descriptions")
        if has_tables:
            context_parts.append(f"Available structured data: {len(content_map['tables'])} tables")
        
        # Create detailed content summary for AI reasoning with semantic analysis
        content_analysis = []
        
        # Add actual text content for AI to use
        text_content_sections = []
        for i, text_item in enumerate(content_map['text']):
            content = text_item.get('content', '')
            page_num = text_item.get('page_number', 0)
            relevance = text_item.get('relevance_score', 0)
            if content and len(content.strip()) > 10:  # Only include substantial text content
                text_content_sections.append(f"TEXT_SECTION_{i+1} (Page {page_num}, relevance: {relevance:.2f}):\n{content}")
        
        # Analyze images with context
        for i, img in enumerate(content_map['images']):
            img_desc = img.get('image_summary', img.get('ocr_text', 'Visual content'))
            page_num = img['page_number']
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            page_context = f"Page {page_num}"
            relevance = img.get('relevance_score', 0)
            
            # Create a more descriptive summary for contextual reference
            if img_desc and len(img_desc.strip()) > 10:
                # Use the full summary if available
                contextual_desc = img_desc[:300]  # Limit length but keep more detail
            else:
                contextual_desc = f"visual content from page {img['page_number']}"
            
            content_analysis.append(f"IMAGE_{i+1} ({page_context}, relevance: {relevance:.2f}): {contextual_desc}")
        
        # Analyze tables with structure and content preview
        for i, table in enumerate(content_map['tables']):
            table_summary = self._get_enhanced_table_summary(table)
            page_num = table['page_number']
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            page_context = f"Page {page_num}"
            relevance = table.get('relevance_score', 0)
            
            # Create a more descriptive summary for contextual reference
            contextual_desc = table_summary if table_summary else "structured data"
            
            # Add content preview if available
            if table.get('table_content_json'):
                try:
                    import json
                    table_json = json.loads(table['table_content_json'])
                    if isinstance(table_json, dict) and '_metadata' in table_json:
                        metadata = table_json['_metadata']
                        contextual_desc += f" containing {metadata.get('total_rows', 0)} rows and {metadata.get('total_columns', 0)} columns"
                    # Get column headers for context
                    headers = [k for k in table_json.keys() if k != '_metadata']
                    if headers:
                        contextual_desc += f" with columns including {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}"
                except:
                    pass
                    
            content_analysis.append(f"TABLE_{i+1} ({page_context}, relevance: {relevance:.2f}): {contextual_desc}")
        
        # Build comprehensive prompt with actual content
        text_content_prompt = ""
        if text_content_sections:
            text_content_prompt = f"\nDOCUMENT TEXT CONTENT (USE THIS FULL CONTENT IN YOUR RESPONSE):\n{chr(10).join(text_content_sections)}\n"
        
        # MERGED: Enhanced prompt combining original structure with regeneration focus
        system_prompt = f"""You are an expert AI assistant that creates perfectly integrated multimodal responses.

AVAILABLE CONTENT:
{', '.join(context_parts)}
{text_content_prompt}

CONTENT DETAILS:
{chr(10).join(content_analysis)}

MULTIMEDIA REFERENCES:
{self._generate_placeholder_list(content_map)}

USER'S QUESTION: "{query}"

CRITICAL REQUIREMENTS - FOLLOW THESE EXACTLY:

1. MULTIMEDIA INTEGRATION (HIGHEST PRIORITY):
You MUST create contextual references to multimedia throughout your response using these REQUIRED phrases:

FOR IMAGES - USE THESE EXACT PHRASES:
• "As illustrated in the image below"
• "The image shows" 
• "As shown in the figure"
• "The diagram illustrates"
• "As depicted in the image"
• "Referring to the image"

FOR TABLES - USE THESE EXACT PHRASES:
• "As shown in the table below"
• "The table shows"
• "According to the data"
• "The following table"
• "As presented in the table"
• "Referring to the table"

TABLE CONTEXT REQUIREMENTS (CRITICAL):
• NEVER generate fake tables - only use the retrieved tables from the document
• For each table you reference, EXPLAIN what it shows and WHY it's relevant to the user's question
• If you can't explain a table's relevance, DO NOT include it
• Always connect table data to the user's specific query
• Use phrases like "This table is relevant because..." or "This table shows..."
                • CRITICAL: If ANY column in a table has ANY empty cells, DO NOT display that table
                • CRITICAL: If ANY column has a header but ALL cells in that column are empty, DO NOT display that table
                • CRITICAL: If a table contains only one row of data after the header, DO NOT display it
                • CRITICAL: DO NOT display exercise/worksheet tables, matching exercises, or practice activities
                • CRITICAL: Only show tables with actual data, results, or meaningful information relevant to the query
                • Only show tables that have clear, structured data with meaningful content in ALL columns and at least 2 rows of data
                • Even one empty cell in any column makes the entire table invalid
• If a table appears distorted, has empty rows/columns, or lacks meaningful content, DO NOT display it
• Validate table quality before referencing - if data looks meaningless or unusable, skip it entirely

DISTRIBUTION REQUIREMENT (CRITICAL):
• Start your answer with text content first
• Integrate image and table references naturally in the MIDDLE sections of your response  
• Do NOT cluster all references at the beginning or end
• Spread references throughout different paragraphs
• Each multimedia element should be referenced EXACTLY ONCE

2. PROFESSIONAL FORMATTING:
• Use ## for main headers, ### for subsections
• Use **bold** for key terms and concepts
• Use > blockquotes for important findings
• Create logical paragraph flow with transitions

3. CONTENT SYNTHESIS:
• Use ONLY the provided text content - no external knowledge
• Create seamless narrative flow between sections
• Synthesize information from multiple sources into coherent themes
• Write complete, connected paragraphs

CORRECT FORMAT EXAMPLE (DO THIS):
## Research Methodology Overview
The study employed a three-phase experimental design to investigate the research question. Each phase built upon the previous findings to create a comprehensive framework.

### Phase 1: Experimental Setup  
The initial phase focused on establishing experimental parameters. As illustrated in the image below, the experimental setup demonstrates key components and their spatial arrangement. This configuration enabled precise control of variables.

### Results and Analysis
The collected data revealed significant improvements across all measured parameters. As shown in the table below, the species probability data and taxonomic classifications demonstrate a 15% improvement in accuracy.

> **Key Finding**: The integrated approach yielded consistently better results than traditional methods.

INCORRECT FORMAT (DO NOT DO THIS):
"Here are the images: {{IMAGE_1}} shows setup. {{TABLE_1}} shows results. The methodology involved..."

{self._get_scope_instructions(response_scope)}

FINAL INSTRUCTIONS:
Write your response using the exact required phrases for multimedia references. Distribute references throughout different paragraphs - NOT all at the beginning or end. Use the provided text content to create a comprehensive, well-structured answer.

SPECIAL TABLE HANDLING INSTRUCTIONS:
• NEVER generate fake tables - only use the retrieved tables from the document
• For each table you reference, EXPLAIN what it shows and WHY it's relevant to the user's question
• If you can't explain a table's relevance, DO NOT include it
• Always connect table data to the user's specific query
• Use phrases like "This table is relevant because..." or "This table shows..."
                • CRITICAL: Before referencing any table, validate its quality:
                  - Check if ALL columns contain meaningful, readable data
                  - If ANY column has ANY empty cells, DO NOT display that table
                  - If ANY column has a header but ALL cells in that column are empty, DO NOT display that table
                  - If a table contains only one row of data after the header, DO NOT display it
                  - DO NOT display exercise/worksheet tables, matching exercises, or practice activities
                  - Only reference tables with actual data, results, or meaningful information relevant to the query
                  - Only reference tables with clear, structured, and meaningful data and at least 2 rows of data
                  - Even one empty cell in any column makes the entire table invalid
• Quality over quantity - it's better to show fewer high-quality tables than many poor-quality ones

Your integrated multimodal response:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=2500,  # Optimal balance for comprehensive responses
                temperature=0.1  # Very low temperature for predictable multimedia placement (matches regeneration)
            )
            
            generated_response = response.choices[0].message.content
            
            # Clean and format the response for better presentation
            cleaned_response = self._clean_and_format_response(generated_response)
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error generating response structure: {e}")
            # Generate dynamic fallback with contextual references
            fallback_parts = ["Based on the document content provided, I can analyze the available information."]
            
            # Create inline fallback instead of separate sections
            image_count = len(content_map.get('images', []))
            table_count = len(content_map.get('tables', []))
            text_count = len(content_map.get('text', []))
            
            # Include key text content in fallback
            if text_count > 0:
                # Take the most relevant text sections
                top_text_sections = content_map['text'][:min(3, text_count)]
                for i, text_item in enumerate(top_text_sections):
                    content = text_item.get('content', '')[:300]  # Limit length for fallback
                    if content:
                        fallback_parts.append(f"Key finding {i+1}: {content}")
            
            if image_count > 0 or table_count > 0:
                fallback_parts.append("The document contains visual and structured data that supports the analysis.")
                
                # Integrate images with contextual references
                for i in range(min(image_count, 2)):  # Limit to 2 images for fallback
                    img_data = content_map['images'][i]
                    img_summary = img_data.get('image_summary', 'visual content')[:100]
                    fallback_parts.append(f"Refer to the following image which shows {img_summary}.")
                
                # Integrate tables with contextual references  
                for i in range(min(table_count, 2)):  # Limit to 2 tables for fallback
                    table_data = content_map['tables'][i]
                    table_summary = self._get_enhanced_table_summary(table_data)[:100]
                    fallback_parts.append(f"Refer to the following table which contains {table_summary}.")
            
            return " ".join(fallback_parts)
    
    def _get_scope_instructions(self, response_scope: str) -> str:
        """Get response scope-specific instructions to control response length and detail."""
        scope_instructions = {
            'minimal': """
**MINIMAL RESPONSE REQUIRED**:
- Provide a VERY SHORT direct answer (1-2 sentences maximum)
- Only include the most essential information that directly answers the question
- NO sections, headers, or detailed explanations
- NO images or tables unless absolutely critical to the answer
- Keep it extremely brief and to the point""",
            
            'concise': """
**CONCISE RESPONSE REQUIRED**:
- Provide a SHORT, focused answer (1-2 paragraphs maximum)
- Include only the key points that directly answer the question
- Use minimal formatting - at most one main header
- Include images/tables ONLY if they are directly relevant to the core answer
- Be brief but informative""",
            
            'standard': """
**STANDARD RESPONSE**:
- Provide a well-structured, informative response (3-4 paragraphs)
- Use appropriate formatting with main sections
- Include relevant images and tables that support the explanation
- Balance detail with readability""",
            
            'detailed': """
**DETAILED RESPONSE**:
- Provide a comprehensive, thorough response with multiple sections
- Use extensive formatting with headers, subheaders, and lists
- Include all available relevant images and tables
- Provide in-depth explanations and context""",
            
            'comprehensive': """
**COMPREHENSIVE RESPONSE**:
- Provide an exhaustive, complete coverage of the topic
- Use extensive formatting with multiple levels of headers
- Include all available multimedia content
- Provide detailed explanations, examples, and contextual information
- Cover all aspects related to the query"""
        }
        
        return scope_instructions.get(response_scope, scope_instructions['standard'])
    
    def _create_inline_elements(self, response_structure: str, content_map: Dict) -> List[Dict]:
        """LLM-driven inline placement - Let the AI handle the positioning naturally."""
        logger.info("Using LLM-driven inline placement - AI handles positioning")
        
        # Simple approach: Let the AI's response structure guide placement
        # Split the response into logical sections and place multimedia content naturally
        
        inline_elements = []
        used_table_ids = set()
        used_image_ids = set()
        
        # Get available multimedia content
        available_images = content_map.get('images', [])
        available_tables = content_map.get('tables', [])
        
        # Split response into paragraphs
        paragraphs = response_structure.split('\n\n')
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            # Add the text paragraph
            inline_elements.append({
                'type': 'text',
                'content': paragraph.strip()
            })
            
            # Check if this paragraph mentions tables and we have tables available
            if available_tables and any(word in paragraph.lower() for word in ['table', 'below', 'shown', 'following']):
                table_data = available_tables[0]
                table_id = table_data.get('table_id', '')
                page_number = table_data.get('page_number', 0)
                unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_table_1"
                
                if unique_key not in used_table_ids:
                    used_table_ids.add(unique_key)
                    inline_elements.append({
                        'type': 'table',
                        'data': table_data,
                        'context': self._extract_table_context(table_data, content_map)
                    })
                    logger.info(f"Added table after paragraph: {unique_key}")
                    available_tables = available_tables[1:]  # Remove used table
            
            # Check if this paragraph mentions images and we have images available
            elif available_images and any(word in paragraph.lower() for word in ['image', 'figure', 'shown', 'illustrated']):
                img_data = available_images[0]
                img_id = f"page_{img_data['page_number']}_img_1"
                if img_id not in used_image_ids:
                    used_image_ids.add(img_id)
                    inline_elements.append({
                        'type': 'image',
                        'data': img_data,
                        'context': self._extract_image_context(img_data, content_map)
                    })
                    logger.info(f"Added image after paragraph: {img_id}")
                    available_images = available_images[1:]  # Remove used image
        
        # Ensure ALL filtered images are used - add any remaining images at strategic points
        while available_images:
            img_data = available_images[0]
            img_id = f"page_{img_data['page_number']}_img_1"
            if img_id not in used_image_ids:
                used_image_ids.add(img_id)
                inline_elements.append({
                    'type': 'image',
                    'data': img_data,
                    'context': self._extract_image_context(img_data, content_map)
                })
                logger.info(f"Added remaining filtered image: {img_id}")
            available_images = available_images[1:]

        # Log any unused tables for debugging
        if available_tables:
            unused_tables = len(available_tables)
            logger.info(f"Skipped {unused_tables} unused tables to keep response focused")
        
        logger.info(f"LLM-driven inline placement complete: {len(inline_elements)} elements")
        return inline_elements
    
    def _extract_image_context(self, img_data: Dict, content_map: Dict) -> Dict:
        """Extract contextual information for image placement."""
        return {
            'page': img_data.get('page_number', 0),
            'relevance': img_data.get('relevance_score', 0),
            'type': 'visual_evidence'
        }
    
    def _extract_table_context(self, table_data: Dict, content_map: Dict) -> Dict:
        """Extract enhanced contextual information for table placement."""
        context = {
            'page': table_data.get('page_number', 0),
            'relevance': table_data.get('relevance_score', 0),
            'type': 'structured_data'
        }
        
        # Add structural information
        if table_data.get('table_content_json'):
            try:
                import json
                table_json = json.loads(table_data['table_content_json'])
                if isinstance(table_json, dict) and '_metadata' in table_json:
                    metadata = table_json['_metadata']
                    context['rows'] = metadata.get('total_rows', 0)
                    context['columns'] = metadata.get('total_columns', 0)
                    
                    # Enhanced context extraction
                    context['table_type'] = self._identify_table_type(table_json)
                    context['content_summary'] = self._generate_table_summary(table_json)
                    context['relevance_explanation'] = self._explain_table_relevance(table_data, content_map)
                    
            except Exception as e:
                logger.warning(f"Error extracting enhanced table context: {e}")
                
        return context
    
    def _identify_table_type(self, table_json: Dict) -> str:
        """Identify the type and purpose of the table."""
        try:
            # Check for common table patterns
            keys = [k for k in table_json.keys() if not k.startswith('_')]
            
            if not keys:
                return "unknown"
            
            # Extract all text content for better pattern detection
            all_text = " ".join([str(k) for k in keys]).lower()
            
            # Check for exercise/worksheet patterns (HIGH PRIORITY - should be filtered out)
            exercise_indicators = [
                'exercise', 'question', 'answer', 'fill', 'blank', 'true', 'false',
                'match', 'matching', 'connect', 'draw', 'label', 'identify', 'choose',
                'select', 'write', 'complete', 'worksheet', 'activity', 'practice',
                'test', 'quiz', 'homework', 'column i', 'column ii', 'a)', 'b)', 'c)',
                'i)', 'ii)', 'iii)', 'iv)', 'v)', 'vi)', 'vii)'
            ]
            
            if any(indicator in all_text for indicator in exercise_indicators):
                return "exercise_worksheet"
            
            # Check for simple matching/classification exercises
            if len(keys) == 2 and any('column' in k.lower() for k in keys):
                # Look for exercise-like content in the values
                for key in keys:
                    values = table_json.get(key, [])
                    if isinstance(values, list):
                        for value in values:
                            if value and str(value).strip():
                                value_str = str(value).strip().lower()
                                # Check for exercise patterns like "(a)", "(b)", "(i)", "(ii)"
                                if re.match(r'^\([a-z]\)|^\([ivx]+\)', value_str):
                                    return "exercise_worksheet"
            
            # Check for data/comparison patterns (GOOD - should be included)
            data_indicators = ['data', 'result', 'analysis', 'comparison', 'measurement', 'statistic', 'information', 'detail']
            if any(indicator in all_text for indicator in data_indicators):
                return "data_analysis"
            
            # Check for reference/index patterns (LOW PRIORITY)
            if len(keys) == 2 and any('page' in k.lower() or 'number' in k.lower() for k in keys):
                return "reference_index"
            
            # Check for classification/matching patterns (GOOD - if not exercise)
            if len(keys) == 2 and any('type' in k.lower() or 'category' in k.lower() for k in keys):
                return "classification_matching"
            
            # Check for content-based classification
            if len(keys) == 2:
                # Look at the actual content to determine if it's meaningful data
                has_meaningful_content = False
                for key in keys:
                    values = table_json.get(key, [])
                    if isinstance(values, list):
                        for value in values:
                            if value and str(value).strip() and len(str(value).strip()) > 3:
                                # Check if it's not just exercise markers
                                if not re.match(r'^\([a-z]\)|^\([ivx]+\)', str(value).strip()):
                                    has_meaningful_content = True
                                    break
                
                if has_meaningful_content:
                    return "classification_matching"
                else:
                    return "exercise_worksheet"
            
            return "general_data"
            
        except Exception as e:
            logger.warning(f"Error identifying table type: {e}")
            return "unknown"
    
    def _generate_table_summary(self, table_json: Dict) -> str:
        """Generate a meaningful summary of the table content."""
        try:
            keys = [k for k in table_json.keys() if not k.startswith('_')]
            if not keys:
                return "Empty table structure"
            
            # Count meaningful data points
            total_cells = 0
            non_empty_cells = 0
            
            for key in keys:
                values = table_json.get(key, [])
                if isinstance(values, list):
                    total_cells += len(values)
                    non_empty_cells += sum(1 for v in values if v and str(v).strip())
            
            if total_cells == 0:
                return "Empty table"
            
            # Generate descriptive summary
            if len(keys) == 2:
                return f"Comparison table with {len(keys)} columns and {non_empty_cells} data points"
            elif len(keys) > 2:
                return f"Multi-column data table with {len(keys)} categories and {non_empty_cells} data points"
            else:
                return f"Data table with {non_empty_cells} data points"
                
        except Exception as e:
            logger.warning(f"Error generating table summary: {e}")
            return "Structured data table"
    
    def _explain_table_relevance(self, table_data: Dict, content_map: Dict) -> str:
        """Explain why this table is relevant to the current context."""
        try:
            page_num = table_data.get('page_number', 0)
            relevance = table_data.get('relevance_score', 0)
            
            # Generate relevance explanation based on score and context
            if relevance > 0.8:
                relevance_level = "highly relevant"
            elif relevance > 0.6:
                relevance_level = "relevant"
            elif relevance > 0.4:
                relevance_level = "moderately relevant"
            else:
                relevance_level = "contextually related"
            
            return f"This table is {relevance_level} to your query and appears on page {page_num}. It provides structured information that supports the text content."
            
        except Exception as e:
            logger.warning(f"Error explaining table relevance: {e}")
            return "This table provides relevant structured data for your query."
    
    def _is_table_valid(self, table_json: str) -> bool:
        """Validate table content to filter out empty or distorted tables."""
        try:
            import json
            
            # Parse table JSON
            if isinstance(table_json, str):
                table_data = json.loads(table_json)
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
            
            # Parse table JSON
            if isinstance(table_json, str):
                table_data = json.loads(table_json)
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
    
    def _enhance_text_formatting(self, text: str) -> str:
        """Enhance text formatting for better presentation."""
        # Clean up extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper paragraph breaks
        text = re.sub(r'\. ([A-Z])', r'.\n\n\1', text)
        
        # Clean up any remaining formatting issues
        text = text.strip()
        
        return text
    
    def _optimize_element_flow(self, elements: List[Dict]) -> List[Dict]:
        """Optimize the flow of elements for better narrative coherence."""
        optimized = []
        
        for i, element in enumerate(elements):
            # Skip spacing elements - they cause fragmentation
            if element['type'] == 'spacing':
                continue
                
            # Add element
            optimized.append(element)
        
        return optimized
    
    def display_inline_response(self, inline_elements: List[Dict]):
        """Display seamlessly integrated multimodal response with perfect narrative flow."""
        try:
            logger.info(f"InlineMultimodalGenerator: Creating seamless narrative with {len(inline_elements)} elements")
            
            # Display elements in perfect narrative flow
            for i, element in enumerate(inline_elements):
                element_type = element.get('type', 'unknown')
                logger.info(f"Seamlessly integrating element {i+1}: {element_type}")
                
                if element_type == 'text':
                    if element['content'] and element['content'].strip() != "---":
                        # Enhanced text display with better formatting
                        self._display_text_seamlessly(element['content'])
                
                elif element_type == 'image':
                    # Display image with contextual integration
                    self._display_image_seamlessly(element['data'], element.get('context', {}))
                    
                    # Add small spacing after image for better flow
                    st.markdown("")  # Empty line for spacing
                
                elif element_type == 'table':
                    # Display table with contextual integration
                    self._display_table_seamlessly(element['data'], element.get('context', {}))
                    
                    # Add small spacing after table for better flow
                    st.markdown("")  # Empty line for spacing
                
                # Note: spacing elements are now filtered out in _optimize_element_flow
            
        except Exception as e:
            st.error(f"Error in seamless display: {e}")
            logger.error(f"Seamless display error: {e}")
    
    def _display_text_seamlessly(self, content: str):
        """Display text content with enhanced formatting for narrative flow."""
        # Split into paragraphs for better readability
        paragraphs = content.split('\n\n')
        
        for paragraph in paragraphs:
            if paragraph.strip():
                # Use markdown for better formatting
                st.markdown(paragraph.strip())
    
    def _display_image_seamlessly(self, img_data: Dict, context: Dict):
        """Display image with seamless integration and contextual information."""
        # Try multiple image sources in order of preference
        image_source = None
        for source_key in ['s3_url', 'display_url', 'local_path']:
            if img_data.get(source_key):
                image_source = img_data[source_key]
                break
        
        if image_source:
            try:
                # Display image without caption or description
                st.image(
                    image_source,
                    use_container_width=True
                )
                    
            except Exception as e:
                st.error(f"Unable to display image: {e}")
                logger.error(f"Image display error: {e}")
        else:
            page_num = img_data['page_number']
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            st.warning(f"Image from page {page_num} is not available")
    
    def _display_table_seamlessly(self, table_data: Dict, context: Dict):
        """Display table with seamless integration and enhanced presentation."""
        page_num = table_data.get('page_number', 'unknown')
        # Convert page number to integer if it's a number
        if isinstance(page_num, (int, float)):
            page_num = int(page_num)
        logger.info(f"Seamlessly displaying table from page {page_num}")
        
        # Display enhanced table context
        if context and context.get('table_type') and context.get('content_summary'):
            table_type = context.get('table_type', 'data')
            content_summary = context.get('content_summary', '')
            relevance_explanation = context.get('relevance_explanation', '')
            
            # Create contextual header for the table
            context_header = f"**{table_type.replace('_', ' ').title()}** - {content_summary}"
            if relevance_explanation:
                context_header += f"\n\n{relevance_explanation}"
            
            st.markdown(context_header)
        
        table_displayed = False
        
        # Enhanced table display with better formatting
        if table_data.get('table_content_json'):
            try:
                import json
                import pandas as pd
                
                json_str = table_data['table_content_json']
                
                # Handle incomplete JSON
                if json_str.count('{') != json_str.count('}') or json_str.count('[') != json_str.count(']'):
                    last_complete = json_str.rfind('}')
                    if last_complete > 0:
                        json_str = json_str[:last_complete + 1]
                
                table_json = json.loads(json_str)
                
                # Enhanced display logic for structured data
                if isinstance(table_json, dict):
                    has_list_values = any(isinstance(v, list) for v in table_json.values())
                    
                    if has_list_values:
                        # Structured format with headers as keys
                        try:
                            df_data = {}
                            max_rows = 0
                            
                            for key, values in table_json.items():
                                if isinstance(values, list) and key != '_metadata':
                                    df_data[key] = values
                                    max_rows = max(max_rows, len(values))
                            
                            if df_data:
                                # Normalize column lengths
                                for key in df_data:
                                    while len(df_data[key]) < max_rows:
                                        df_data[key].append("")
                                
                                df = pd.DataFrame(df_data)
                                
                                # Enhanced display with styling
                                st.dataframe(
                                    df, 
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                table_displayed = True
                                
                                # Add metadata information if available
                                if '_metadata' in table_json:
                                    metadata = table_json['_metadata']
                                    if metadata.get('total_rows') and metadata.get('total_columns'):
                                        st.caption(f"*Table contains {metadata['total_rows']} rows and {metadata['total_columns']} columns*")
                                
                                logger.info("Seamlessly displayed structured table")
                        except Exception as e:
                            logger.error(f"Error in structured table display: {e}")
                            # Fallback to JSON display
                            st.json(table_json)
                            table_displayed = True
                
                # Handle list format tables
                elif isinstance(table_json, list) and len(table_json) > 0:
                    try:
                        df = pd.DataFrame(table_json)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        table_displayed = True
                        logger.info("Seamlessly displayed list-format table")
                    except Exception as e:
                        logger.error(f"Error displaying list table: {e}")
                        st.json(table_json)
                        table_displayed = True
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error in seamless display: {e}")
                st.warning("Table data format issue - displaying raw content")
            except Exception as e:
                logger.error(f"Seamless table display error: {e}")
                st.warning(f"Unable to display table: {e}")
        
        if not table_displayed:
            st.info("Table data is being processed...")
    
    def _verify_response_grounding(self, response: str, content_map: Dict) -> bool:
        """Verify that the response is properly grounded in the document content."""
        response_lower = response.lower()
        
        # Check for multimedia placeholders (indicates document-based content)
        has_multimedia_placeholders = bool(re.search(r'\{\{(IMAGE|TABLE)_\d+\}\}', response))
        
        # Check for natural grounding indicators (more flexible than strict phrases)
        natural_grounding_indicators = [
            "shows", "reveals", "indicates", "demonstrates", "illustrates", "contains",
            "displays", "presents", "includes", "features", "depicts", "explains",
            "page", "figure", "table", "chart", "data", "image", "visual", "diagram"
        ]
        has_natural_grounding = any(indicator in response_lower for indicator in natural_grounding_indicators)
        
        # Check for technical/domain-specific content (indicates document analysis)
        has_technical_content = len(response) > 100  # Substantial response
        
        # Check for warning signs of external knowledge (avoid these)
        warning_phrases = [
            "it is well known that", "research generally shows", "studies typically indicate", 
            "experts commonly believe", "this is universally true", "according to general knowledge",
            "as everyone knows", "it is widely accepted", "scientific consensus shows",
            "in general", "typically", "usually", "commonly", "generally speaking"
        ]
        has_warning_phrases = any(phrase in response_lower for phrase in warning_phrases)
        
        # Response is grounded if it has multimedia placeholders OR natural indicators, is substantial, and no warning phrases
        is_grounded = (has_multimedia_placeholders or has_natural_grounding) and has_technical_content and not has_warning_phrases
        
        if not is_grounded:
            logger.info(f"Grounding check: multimedia={has_multimedia_placeholders}, natural={has_natural_grounding}, technical={has_technical_content}, warnings={has_warning_phrases}")
        
        return is_grounded
    
    def _create_fallback_grounded_response(self, query: str, content_map: Dict) -> str:
        """Create a fallback response that directly answers the query."""
        # Count available content
        text_count = len(content_map['text'])
        image_count = len(content_map['images'])
        table_count = len(content_map['tables'])
        
        # Create a direct response structure
        response_parts = []
        
        # Direct answer based on query type
        query_lower = query.lower()
        if any(word in query_lower for word in ['what', 'define', 'explain']):
            response_parts.append("Here's what I found about your question:")
        elif any(word in query_lower for word in ['summarize', 'summary', 'overview']):
            response_parts.append("This document covers several key areas:")
        elif any(word in query_lower for word in ['how', 'process', 'method']):
            response_parts.append("The process involves these key steps:")
        else:
            response_parts.append("I can answer your question using the available information:")
        
        if image_count > 0:
            response_parts.append("The visual content reveals important details:")
            for i in range(min(image_count, 3)):  # Include up to 3 images
                img_data = content_map['images'][i]
                img_desc = img_data.get('image_summary', img_data.get('ocr_text', 'visual content'))[:150]
                response_parts.append(f"{{{{IMAGE_{i+1}}}}} {img_desc}")
        
        if table_count > 0:
            response_parts.append("The data shows:")
            for i in range(min(table_count, 3)):  # Include up to 3 tables
                table_data = content_map['tables'][i]
                table_summary = self._get_enhanced_table_summary(table_data)[:150]
                response_parts.append(f"{{{{TABLE_{i+1}}}}} {table_summary}")
        
        if text_count > 0:
            response_parts.append("Additional detailed information is available throughout the content.")
        
        return " ".join(response_parts)
    
    def _generate_placeholder_list(self, content_map: Dict) -> str:
        """Generate a detailed list of available multimedia content with context for the AI to use."""
        placeholder_details = []
        
        # Add image placeholders with context
        for i, img in enumerate(content_map['images']):
            page_num = img.get('page_number', 'N/A')
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            summary = img.get('image_summary', 'Visual content')[:150]
            placeholder_details.append(f"IMAGE_{i+1} - Page {page_num}: {summary}")
        
        # Add table placeholders with context
        for i, table in enumerate(content_map['tables']):
            page_num = table.get('page_number', 'N/A')
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            summary = self._get_enhanced_table_summary(table)[:150]
            placeholder_details.append(f"TABLE_{i+1} - Page {page_num}: {summary}")
        
        if placeholder_details:
            return f"""Available multimedia content for contextual integration:

{chr(10).join(placeholder_details)}

IMPORTANT: Create proper contextual references that describe what each image and table shows, then place these references inline within your text flow. Use the summaries above to create meaningful descriptions."""
        else:
            return "No multimedia content available for contextual integration"
    
    
    def _display_image_inline(self, img_data: Dict):
        """Display image inline with minimal formatting."""
        # Try multiple image sources in order of preference
        image_source = None
        for source_key in ['s3_url', 'display_url', 'local_path']:
            if img_data.get(source_key):
                image_source = img_data[source_key]
                break
        
        if image_source:
            try:
                page_num = img_data['page_number']
                # Convert page number to integer if it's a number
                if isinstance(page_num, (int, float)):
                    page_num = int(page_num)
                st.image(
                    image_source,
                    caption=f"Page {page_num}",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.info(f"Image URL: {image_source}")
        else:
            page_num = img_data['page_number']
            # Convert page number to integer if it's a number
            if isinstance(page_num, (int, float)):
                page_num = int(page_num)
            st.warning(f"No image source found for page {page_num}")
    
    def _display_table_inline(self, table_data: Dict):
        """Display table inline with minimal formatting."""
        page_num = table_data.get('page_number', 'NO_PAGE')
        # Convert page number to integer if it's a number
        if isinstance(page_num, (int, float)):
            page_num = int(page_num)
        logger.info(f"Displaying table from page {page_num}")
        
        # Display table content with multiple formats
        table_displayed = False
        
        # Try to display JSON data as a formatted table
        if table_data.get('table_content_json'):
            try:
                import json
                import pandas as pd
                
                # Get the JSON string and check if it's complete
                json_str = table_data['table_content_json']
                
                # Check if JSON is complete (has matching braces)
                if json_str.count('{') != json_str.count('}') or json_str.count('[') != json_str.count(']'):
                    # Try to find the last complete object
                    last_complete = json_str.rfind('}')
                    if last_complete > 0:
                        json_str = json_str[:last_complete + 1]
                
                table_json = json.loads(json_str)
                
                # Handle the new structured table format (headers as keys)
                if isinstance(table_json, dict):
                    # Check if this is structured table format (has list values)
                    has_list_values = any(isinstance(v, list) for v in table_json.values())
                    
                    if has_list_values:
                        # This is the new structured format with headers as keys
                        try:
                            # Convert to DataFrame for better display
                            df_data = {}
                            max_rows = 0
                            
                            for key, values in table_json.items():
                                if isinstance(values, list) and key != '_metadata':
                                    df_data[key] = values
                                    max_rows = max(max_rows, len(values))
                            
                            if df_data:
                                # Pad shorter columns with empty strings
                                for key in df_data:
                                    while len(df_data[key]) < max_rows:
                                        df_data[key].append("")
                                
                                df = pd.DataFrame(df_data)
                                st.dataframe(df, use_container_width=True)
                                table_displayed = True
                                logger.info("Displayed table as DataFrame from structured JSON")
                            else:
                                st.warning("No valid table data found in JSON structure")
                                
                        except Exception as e:
                            logger.error(f"Failed to create DataFrame from structured JSON: {e}")
                            st.json(table_json)
                            table_displayed = True
                    else:
                        # Handle as key-value pairs (fallback)
                        display_data = {k: v for k, v in table_json.items() if k != '_metadata'}
                        if display_data:
                            for key, value in display_data.items():
                                if isinstance(value, list):
                                    st.write(f"**{key}:** {', '.join(map(str, value))}")
                                else:
                                    st.write(f"**{key}:** {value}")
                            table_displayed = True
                            logger.info("Displayed table as key-value pairs")
                        else:
                            st.warning("No displayable table data found")
                
                # Handle traditional list format
                elif isinstance(table_json, list) and len(table_json) > 0:
                    # If it's a list of dictionaries, convert to DataFrame
                    if isinstance(table_json[0], dict):
                        df = pd.DataFrame(table_json)
                        st.dataframe(df, use_container_width=True)
                        table_displayed = True
                        logger.info("Displayed table as DataFrame from list of dicts")
                    else:
                        # If it's a list of lists, try to create DataFrame
                        df = pd.DataFrame(table_json)
                        st.dataframe(df, use_container_width=True)
                        table_displayed = True
                        logger.info("Displayed table as DataFrame from list of lists")
                
                else:
                    # Fallback to JSON display
                    st.json(table_json)
                    table_displayed = True
                    logger.info("Displayed table as raw JSON")
                    
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                st.warning(f"Invalid JSON format: {e}")
            except Exception as e:
                logger.error(f"Failed to render table from JSON: {e}")
                st.warning(f"Failed to render table from JSON: {e}")
        
        # If no table content was displayed, show a message
        if not table_displayed:
            st.warning("No table content available for display")
    
    def _validate_inline_placement(self, response_structure: str, is_regenerated: bool = False) -> Tuple[bool, str]:
        """Validate that the response has proper inline placement of multimedia elements with contextual references."""
        # Check for contextual references to images and tables
        image_reference_patterns = [
            r'refer to the following image',
            r'as illustrated in the image',
            r'the following image',
            r'image below',
            r'image above',
            r'as shown in the image'
        ]
        
        table_reference_patterns = [
            r'refer to the following table',
            r'as shown in the table',
            r'the following table',
            r'table below',
            r'table above',
            r'as presented in the table'
        ]
        
        # Count contextual references
        image_references = 0
        table_references = 0
        
        for pattern in image_reference_patterns:
            image_references += len(re.findall(pattern, response_structure, re.IGNORECASE))
        
        for pattern in table_reference_patterns:
            table_references += len(re.findall(pattern, response_structure, re.IGNORECASE))
        
        total_references = image_references + table_references
        
        if total_references == 0:
            # Log what was actually found for debugging
            logger.debug(f"No contextual references found. Response text: {response_structure[:200]}...")
            # If this is a regenerated response, be more lenient and look for basic keywords
            if is_regenerated:
                # Look for basic image/table mentions as fallback
                basic_image_words = len(re.findall(r'\b(image|figure|diagram|illustration|visual|chart)\b', response_structure, re.IGNORECASE))
                basic_table_words = len(re.findall(r'\b(table|data|results|statistics|values)\b', response_structure, re.IGNORECASE))
                
                if basic_image_words > 0 or basic_table_words > 0:
                    return True, f"Fallback validation passed: Found {basic_image_words} image mentions and {basic_table_words} table mentions"
            
            return False, "No contextual references to multimedia content found"
        
        # Check if references are distributed throughout the text
        total_length = len(response_structure)
        if total_length == 0:
            return True, "Empty response"
        
        # Find positions of all references
        reference_positions = []
        
        for pattern in image_reference_patterns + table_reference_patterns:
            for match in re.finditer(pattern, response_structure, re.IGNORECASE):
                reference_positions.append(match.start())
        
        if not reference_positions:
            return False, "No contextual references found despite patterns"
        
        # Calculate distribution
        end_threshold = total_length * 0.7
        placeholders_in_end = sum(1 for pos in reference_positions if pos > end_threshold)
        
        start_threshold = total_length * 0.3
        placeholders_in_start = sum(1 for pos in reference_positions if pos < start_threshold)
        
        placeholders_in_middle = len(reference_positions) - placeholders_in_start - placeholders_in_end
        
        # Determine placement quality
        if placeholders_in_end > len(reference_positions) * 0.6:
            return False, f"Poor inline placement: {placeholders_in_end}/{len(reference_positions)} references clustered at the end"
        elif placeholders_in_start > len(reference_positions) * 0.6:
            return False, f"Poor inline placement: {placeholders_in_start}/{len(reference_positions)} references clustered at the beginning"
        elif placeholders_in_middle >= len(reference_positions) * 0.4:
            return True, f"Good inline placement: {placeholders_in_middle}/{len(reference_positions)} references in middle section"
        else:
            return True, f"Acceptable inline placement: {len(reference_positions)} contextual references distributed throughout"
    
    def _regenerate_with_inline_focus(self, query: str, content_map: Dict, has_text: bool, has_images: bool, has_tables: bool, response_scope: str = 'standard') -> str:
        """Regenerate response with stronger focus on contextual references."""
        
        # Prepare text content for the aggressive prompt
        text_content_parts = []
        for i, text_item in enumerate(content_map['text'][:5]):  # Limit to top 5 text sections
            content = text_item.get('content', '')
            if content and len(content.strip()) > 10:
                text_content_parts.append(f"Text Section {i+1}: {content[:500]}...")  # Limit each section
        
        text_content_summary = "\n".join(text_content_parts) if text_content_parts else "No detailed text content available."
        
        # Prepare multimedia content summaries
        image_summaries = []
        for i, img in enumerate(content_map['images']):
            summary = img.get('image_summary', 'Visual content')[:200]
            image_summaries.append(f"Image {i+1}: {summary}")
        
        table_summaries = []
        for i, table in enumerate(content_map['tables']):
            # Use enhanced table summary extraction with backend data
            summary = self._get_enhanced_table_summary(table)[:200]
            table_summaries.append(f"Table {i+1}: {summary}")
        
        # Create a more aggressive prompt for contextual references
        aggressive_prompt = f"""You are a helpful assistant that answers questions based on the document content provided.

AVAILABLE CONTENT:
- Text content from {len(content_map['text'])} sections
- Visual content: {len(content_map['images'])} images with descriptions  
- Structured data: {len(content_map['tables'])} tables

DETAILED TEXT CONTENT:
{text_content_summary}

IMAGE SUMMARIES:
{chr(10).join(image_summaries) if image_summaries else "No images available"}

TABLE SUMMARIES:
{chr(10).join(table_summaries) if table_summaries else "No tables available"}

USER'S QUESTION: "{query}"

CRITICAL INSTRUCTION - YOU MUST FOLLOW THIS EXACTLY:
Write your response by creating proper contextual references to images and tables throughout your answer, not just at the beginning. Use natural phrases like:

REQUIRED REFERENCE PHRASES (USE THESE):
For images: "the image shows", "as shown in the figure", "the diagram illustrates", "as depicted", "referring to the image", "the following image", "as seen in the figure"
For tables: "the table shows", "as shown in the table", "according to the data", "the table below", "referring to the table", "as presented in the table"

DISTRIBUTION REQUIREMENT:
- Start your answer with text content first
- Integrate image and table references naturally in the middle sections of your response
- Do NOT cluster all references at the beginning or end
- Spread references throughout different paragraphs

CORRECT FORMAT (DO THIS):
"The research methodology involves three phases of data collection and analysis. The initial phase focused on... [middle content] ...As shown in the figure, the experimental setup demonstrates the key components used in this validation process. Further analysis revealed... [more content] ...The table below presents the detailed results, showing species probability data and taxonomic classifications that support these findings."

INCORRECT FORMAT (DO NOT DO THIS):
"Here are the images and tables: {{IMAGE_1}} shows setup. {{TABLE_1}} shows results. The research methodology..."

{self._get_scope_instructions(response_scope)}

Use the image and table summaries provided above to create meaningful contextual descriptions. Integrate all multimedia content inline within your narrative and use the detailed text content provided. Your answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": aggressive_prompt}],
                max_tokens=2500,  # Increased for more comprehensive responses
                temperature=0.1  # Very low temperature for more predictable formatting
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error in aggressive inline regeneration: {e}")
            return "Error generating inline response"

    def _clean_redundant_references(self, text: str) -> str:
        """Remove redundant contextual references that appear after multimedia content."""
        # Common redundant patterns to remove
        redundant_patterns = [
            r'\s*\(refer to the following image[^)]*\)',
            r'\s*\(as shown in the image[^)]*\)',
            r'\s*\(as illustrated in the image[^)]*\)',
            r'\s*\(refer to the following table[^)]*\)',
            r'\s*\(as shown in the table[^)]*\)',
            r'\s*\(as presented in the table[^)]*\)',
            r'\s*following image which shows[^.]*\.',
            r'\s*following table which contains[^.]*\.',
            r'\s*as shown in the table below[^.]*\.',
            r'\s*as illustrated in the image below[^.]*\.',
            # Remove incomplete references
            r'\s*Refer to the$',
            r'\s*See the$',
            r'\s*As shown in the$',
            r'\s*As illustrated in the$',
            # Remove redundant text after multimedia
            r'\s*\(refer to the following image[^)]*\)[^.]*\.',
            r'\s*\(as shown in the table[^)]*\)[^.]*\.',
            r'\s*following image which shows[^.]*\.',
            r'\s*following table which contains[^.]*\.'
        ]
        
        cleaned_text = text
        for pattern in redundant_patterns:
            cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
        
        # Clean up extra whitespace and fix sentence endings
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
        cleaned_text = cleaned_text.strip()
        
        # Fix sentence endings that might be broken
        cleaned_text = re.sub(r'\s+\.', '.', cleaned_text)
        cleaned_text = re.sub(r'\s+,', ',', cleaned_text)
        
        return cleaned_text
    
    def _extract_relevant_text(self, text: str, query: str, max_sentences: int = 8) -> str:
        """Extract the most relevant sentences/paragraphs from page text based on query similarity."""
        import re
        
        if not text or not query:
            return text
        
        try:
            # Split text into sentences using simple regex (avoiding nltk dependency)
            sentences = re.split(r'(?<=[.!?])\s+', text.strip())
            
            # If text is already short, return as-is
            if len(sentences) <= max_sentences or len(text) <= 800:
                return text
            
            # Calculate simple similarity scores for each sentence
            query_words = set(query.lower().split())
            sentence_scores = []
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 20:  # Skip very short sentences
                    sentence_scores.append((0, i, sentence))
                    continue
                
                # Simple word overlap scoring
                sentence_words = set(sentence.lower().split())
                overlap = len(query_words & sentence_words)
                
                # Boost score for longer sentences (more context)
                length_bonus = min(len(sentence) / 100, 1.0)
                
                # Boost score for sentences with key terms
                key_terms_score = 0
                key_indicators = ['therefore', 'however', 'furthermore', 'moreover', 'consequently', 'additionally', 'in contrast', 'similarly']
                if any(term in sentence.lower() for term in key_indicators):
                    key_terms_score = 0.3
                
                final_score = overlap + length_bonus + key_terms_score
                sentence_scores.append((final_score, i, sentence))
            
            # Sort by score and keep original order for top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
            top_sentences = sorted(top_sentences, key=lambda x: x[1])  # Restore original order
            
            # Extract the sentences
            extracted_text = ' '.join([sentence for _, _, sentence in top_sentences])
            
            # Clean up the extracted text
            extracted_text = re.sub(r'\s+', ' ', extracted_text).strip()
            
            logger.debug(f"Smart extraction: {len(text)} -> {len(extracted_text)} chars ({len(sentences)} -> {len(top_sentences)} sentences)")
            
            return extracted_text
            
        except Exception as e:
            logger.warning(f"Error in smart text extraction: {e}. Using original text.")
            return text
    
    def _clean_and_format_response(self, response: str) -> str:
        """Clean and format the response text for better ChatGPT/Claude-style presentation."""
        import re
        
        if not response:
            return response
        
        try:
            # STEP 1: Remove ALL excessive newlines and spacing patterns
            # First remove the really problematic patterns like "\n \n\n" and "\n\n\n"
            cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', response)  # Replace 3+ newlines (with possible spaces) with exactly 2
            cleaned = re.sub(r'\n\s+\n', '\n\n', cleaned)  # Remove patterns like "\n \n"
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Ensure no more than 2 consecutive newlines
            
            # STEP 2: Fix header spacing (ChatGPT/Claude style)
            # Headers should have exactly one blank line before and after
            cleaned = re.sub(r'\n*(#+\s)', r'\n\n\1', cleaned)  # Ensure blank line before headers
            cleaned = re.sub(r'(#+\s[^\n]+)\n+', r'\1\n\n', cleaned)  # Ensure blank line after headers
            
            # STEP 3: Fix subheading and section spacing
            # Subheadings should be properly spaced
            cleaned = re.sub(r'\n*(###\s)', r'\n\n\1', cleaned)  # Subheadings
            cleaned = re.sub(r'(###\s[^\n]+)\n+', r'\1\n\n', cleaned)
            
            # STEP 4: Fix list formatting (clean bullet points)
            # Lists should have proper spacing without excessive gaps
            cleaned = re.sub(r'\n+(-\s)', r'\n\n\1', cleaned)  # Blank line before first list item
            cleaned = re.sub(r'(-\s[^\n]+)\n+(-\s)', r'\1\n\2', cleaned)  # No gap between list items
            cleaned = re.sub(r'(-\s[^\n]+)\n+([^-\n])', r'\1\n\n\2', cleaned)  # Blank line after list
            
            # STEP 5: Fix paragraph spacing
            # Paragraphs should be separated by exactly one blank line
            cleaned = re.sub(r'([.!?])\s*\n+([A-Z])', r'\1\n\n\2', cleaned)  # Proper paragraph separation
            
            # STEP 6: Clean bold formatting spacing
            cleaned = re.sub(r'\n+(\*\*[^*]+\*\*)', r'\n\n\1', cleaned)  # Bold headings
            
            # STEP 7: Remove excessive spaces and fix sentence spacing
            cleaned = re.sub(r' {2,}', ' ', cleaned)  # Multiple spaces to single space
            cleaned = re.sub(r'([.!?])([A-Z])', r'\1 \2', cleaned)  # Space after sentences
            
            # STEP 8: Clean beginning and end
            cleaned = cleaned.strip()
            
            # STEP 9: Final cleanup - ensure no weird trailing spaces
            lines = cleaned.split('\n')
            lines = [line.rstrip() for line in lines]  # Remove trailing spaces from each line
            cleaned = '\n'.join(lines)
            
            # STEP 10: Ensure proper structure - no more than 2 consecutive newlines anywhere
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            
            logger.debug(f"Response cleaning: {len(response)} -> {len(cleaned)} chars")
            if len(cleaned) != len(response):
                logger.info("Response formatting and spacing cleaned for better presentation")
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Error in response cleaning: {e}. Using original response.")
            return response
    
    def _get_enhanced_table_summary(self, table: dict) -> str:
        """Extract enhanced table summary using backend data and content analysis."""
        try:
            # Priority 1: Use backend-generated table summaries if available
            # These come directly from the table structure, not nested under 'data'
            for summary_field in ['table_summary', 'enhanced_summary', 'ai_summary', 'summary']:
                if summary_field in table and table[summary_field]:
                    summary = table[summary_field]
                    if isinstance(summary, str) and len(summary.strip()) > 10:
                        return summary.strip()
            
            # Priority 2: Analyze table content JSON for intelligent summary
            table_content_json = table.get('table_content_json', '')
            
            if table_content_json:
                try:
                    import json
                    table_content = json.loads(table_content_json)
                    
                    # Extract metadata for context
                    metadata = table_content.get('_metadata', {})
                    total_rows = metadata.get('total_rows', 0)
                    total_columns = metadata.get('total_columns', 0)
                    
                    # Analyze table content to create meaningful summary
                    table_keys = [k for k in table_content.keys() if not k.startswith('_')]
                    
                    # Create content-aware summary
                    if len(table_keys) >= 2 and total_rows >= 2:
                        # Try to identify the nature of the data
                        sample_columns = table_keys[:3]  # First 3 columns
                        column_descriptions = []
                        
                        for col in sample_columns:
                            column_data = table_content.get(col, [])
                            if isinstance(column_data, list) and len(column_data) > 0:
                                # Analyze column content type
                                sample_values = [str(val).strip() for val in column_data[:3] if str(val).strip()]
                                if sample_values:
                                    # Check if numeric data
                                    numeric_count = sum(1 for val in sample_values if val.replace('.', '').replace('-', '').replace('+', '').isdigit())
                                    if numeric_count >= len(sample_values) * 0.7:
                                        column_descriptions.append(f"{col} (numerical data)")
                                    else:
                                        column_descriptions.append(f"{col}")
                        
                        if column_descriptions:
                            content_desc = ", ".join(column_descriptions[:2])
                            if len(column_descriptions) > 2:
                                content_desc += f" and {len(column_descriptions) - 2} more columns"
                            
                            return f"Table with {total_rows} rows containing {content_desc}"
                    
                    # Fallback: Basic structure description
                    if total_rows > 0 and total_columns > 0:
                        return f"Data table with {total_rows} rows and {total_columns} columns"
                        
                except (json.JSONDecodeError, TypeError, KeyError) as e:
                    logger.debug(f"Error parsing table content for summary: {e}")
            
            # Priority 3: Use existing summary field
            if 'summary' in table and table['summary']:
                return str(table['summary']).strip()
            
            # Priority 4: Use context information
            if 'context' in table and table['context']:
                context = table['context']
                if isinstance(context, dict):
                    # Extract useful context information
                    page = context.get('page', '')
                    table_type = context.get('type', '')
                    if page and table_type:
                        return f"Structured data from page {page}"
                elif isinstance(context, str) and len(context.strip()) > 5:
                    return context.strip()[:100]
            
            # Priority 5: Final fallback
            page_number = table.get('page_number', '')
            if page_number:
                return f"Table from page {page_number} with structured data"
            
            return "Structured data table"
            
        except Exception as e:
            logger.warning(f"Error extracting enhanced table summary: {e}")
            return table.get('summary', 'Structured data table')


print("InlineMultimodalGenerator created successfully")