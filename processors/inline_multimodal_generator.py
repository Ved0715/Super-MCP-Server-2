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
    
    def generate_inline_response_from_chunks(self, query: str, chunks: List[Dict], conditional_instructions: Dict = None) -> Dict[str, Any]:
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
                
                # Add table content from arrays
                if chunk.get('contains_table') and chunk.get('table_content_jsons'):
                    table_jsons = chunk.get('table_content_jsons', [])
                    table_summaries = chunk.get('table_summaries', [])
                    table_ids = chunk.get('table_ids', [])
                    
                    # Process each table in the arrays
                    for i, table_json in enumerate(table_jsons):
                        table_summary = table_summaries[i] if i < len(table_summaries) else ''
                        table_id = table_ids[i] if i < len(table_ids) else f'table_{i}'
                        
                        filtered_content['tables'].append({
                            'table_content_json': table_json,
                            'table_id': table_id,
                            'page_number': chunk.get('page_number', 0),
                            'relevance_score': chunk.get('relevance_score', 0)
                        })
            
            # Use the existing response generation logic
            return self.generate_inline_response(query, filtered_content, conditional_instructions)
            
        except Exception as e:
            logger.error(f"Error in generate_inline_response_from_chunks: {e}")
            return {'inline_elements': [{'type': 'text', 'content': f"Error generating response: {str(e)}"}]}

    def generate_inline_response(self, query: str, filtered_content: Dict, conditional_instructions: Dict = None) -> Dict[str, Any]:
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

            # Generate contextual response structure
            response_structure = self._generate_response_structure(query, content_map, input_has_text, input_has_images, input_has_tables)
            
            # Validate inline placement
            is_inline_valid, validation_message = self._validate_inline_placement(response_structure)
            logger.info(f"Inline placement validation: {validation_message}")
            
            # If inline placement is poor, regenerate with stronger focus
            if not is_inline_valid and (input_has_images or input_has_tables):
                logger.info("Poor inline placement detected, regenerating with stronger focus...")
                response_structure = self._regenerate_with_inline_focus(query, content_map, input_has_text, input_has_images, input_has_tables)
                
                # Validate again with more lenient criteria
                is_inline_valid, validation_message = self._validate_inline_placement(response_structure, is_regenerated=True)
                logger.info(f"Regenerated inline placement validation: {validation_message}")
                
                # If second attempt also fails, accept it anyway but log the issue
                if not is_inline_valid:
                    logger.warning(f"Second validation attempt failed: {validation_message}. Proceeding with current response.")
        
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
        
        # Add image content with type identifier
        for i, img_item in enumerate(filtered_content['images']):
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
    
    def _generate_response_structure(self, query: str, content_map: Dict, has_text: bool, has_images: bool, has_tables: bool) -> str:
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
            table_summary = table.get('summary', 'Structured data')
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
        
        # Enhanced prompt with strong inline placement instructions
        system_prompt = f"""You are a helpful assistant that answers questions based on the document content provided. 

AVAILABLE CONTENT:
{', '.join(context_parts)}
{text_content_prompt}
CONTENT DETAILS:
{chr(10).join(content_analysis)}

MULTIMEDIA REFERENCES:
{self._generate_placeholder_list(content_map)}

USER'S QUESTION: "{query}"

CRITICAL FORMATTING REQUIREMENTS - YOU MUST FOLLOW THESE EXACTLY:

**STRUCTURED FORMATTING (ChatGPT/Claude Style):**
1. **USE MARKDOWN FORMATTING**: Structure your response with proper markdown:
   - Use ## for main section headers
   - Use ### for subsection headers
   - Use **bold text** for emphasis and key terms
   - Use *italic text* for subtle emphasis
   - Use `code formatting` for technical terms, formulas, or specific values
   - Use bullet points (- or *) for lists
   - Use numbered lists (1., 2., 3.) for sequences or steps
   - Use > blockquotes for important notes or definitions
   - Add proper line breaks between sections

2. **CONTENT ORGANIZATION**: Structure your response logically:
   - Start with a brief introductory paragraph
   - Use clear section headers to organize information
   - End with a summary or conclusion section when appropriate
   - Ensure smooth flow between sections

3. **USE ALL TEXT CONTENT**: Incorporate the full text content provided above into your response

4. **PROPER CONTEXTUAL REFERENCES**: Write complete contextual references that describe what the multimedia shows:
   - "Refer to the following image which shows [use the actual image summary from the content details]"
   - "As illustrated in the image below, [describe the visual content using the image summary]"
   - "The following image demonstrates [use key points from the image summary]"

5. **TABLE CONTEXTUAL REFERENCES**: Write complete contextual references that describe what the table contains:
   - "Refer to the following table which contains [use the actual table summary or describe the data structure]"
   - "As shown in the table below, [describe the data/statistics using table information]"
   - "The following table presents [use key information from the table summary]"

6. **COMPLETE SENTENCES**: Always write complete sentences for contextual references - never leave them incomplete

7. **INLINE PLACEMENT**: Place these contextual references DIRECTLY within your text flow where they are most relevant

8. **NATURAL INTEGRATION**: Integrate multimedia references naturally within sentences and paragraphs

9. **COMPREHENSIVE RESPONSE**: Use the detailed text content to provide a thorough answer

10. **NO REDUNDANT REFERENCES**: Do NOT repeat the same reference multiple times or add redundant text after the reference

EXAMPLE OF CORRECT STRUCTURED RESPONSE:

## Research Methodology Overview

The study employed a **three-phase experimental design** to investigate the research question. Each phase built upon the previous findings to create a comprehensive analysis framework.

### Phase 1: Experimental Setup
The initial phase focused on establishing the experimental parameters. Refer to the following image which shows the experimental setup used in Phase 1, demonstrating the key components and their arrangement. This configuration allowed for precise control of variables while maintaining `optimal measurement conditions`.

### Results Analysis
The collected data revealed significant improvements across all measured parameters. As shown in the table below which contains species probability data and taxonomic classifications, the results demonstrate a **15% improvement** in classification accuracy.

> **Key Finding**: The integrated approach yielded consistently better results than traditional methods.

## Conclusion
The three-phase methodology successfully addressed the research objectives and provided valuable insights for future investigations.

EXAMPLE OF INCORRECT REFERENCE (DO NOT DO THIS):
"The study methodology involved three main phases. Refer to the following image which shows the experimental setup (refer to the following image which shows the experimental setup). The results from the table below (as shown in the table below which contains data)."

**FINAL INSTRUCTIONS:**
Answer the question naturally and helpfully using the provided document text content. Structure your response with proper markdown formatting like ChatGPT/Claude, using headers, bold text, code formatting, lists, and blockquotes as shown in the example. Create proper contextual references that describe what each image and table shows using the summaries provided, then place these references inline throughout your response where they best support your explanation. Always write complete sentences for contextual references.

Your detailed, structured answer:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=3000,  # Further increased for more comprehensive responses
                temperature=0.2  # Lower for more focused, factual responses
            )
            
            generated_response = response.choices[0].message.content
            
            # Return the generated response directly - no grounding verification needed
            return generated_response
            
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
                    table_summary = table_data.get('summary', 'structured data')[:100]
                    fallback_parts.append(f"Refer to the following table which contains {table_summary}.")
            
            return " ".join(fallback_parts)
    
    def _create_inline_elements(self, response_structure: str, content_map: Dict) -> List[Dict]:
        """Convert response structure with contextual references into seamlessly integrated inline elements."""
        inline_elements = []
        used_table_ids = set()  # Track used tables to prevent duplicates
        used_image_ids = set()  # Track used images to prevent duplicates
        
        # First, identify where images and tables should be placed based on contextual references
        image_mentions = []
        table_mentions = []
        
        # Find ALL contextual references in the text first
        import re
        
        # Find all image references with their positions
        image_patterns = [
            r'(?:refer to the )?(?:following )?image(?:\s+below)?(?:\s+which shows [^.]*)?',
            r'as (?:illustrated|shown) in the image(?:\s+below)?',
            r'the image (?:shows?|illustrates?|demonstrates?)'
        ]
        
        # Collect all image references first
        all_image_matches = []
        for pattern in image_patterns:
            matches = list(re.finditer(pattern, response_structure, re.IGNORECASE))
            for match in matches:
                all_image_matches.append((match.start(), match.group()))
        
        # Sort by position and assign to images sequentially
        all_image_matches.sort(key=lambda x: x[0])
        for i, (pos, match_text) in enumerate(all_image_matches):
            if i < len(content_map['images']):
                img = content_map['images'][i]
                image_mentions.append((pos, i + 1, img))
                logger.info(f"Detected image reference: '{match_text}' at position {pos}")
        
        # Find all table references with their positions
        table_patterns = [
            r'(?:refer to the )?(?:following )?table(?:\s+below)?(?:\s+which (?:shows?|contains?|presents?) [^.]*)?',
            r'as (?:shown|presented) in the table(?:\s+below)?',
            r'the table (?:shows?|contains?|presents?)'
        ]
        
        # Collect all table references first
        all_table_matches = []
        for pattern in table_patterns:
            matches = list(re.finditer(pattern, response_structure, re.IGNORECASE))
            for match in matches:
                all_table_matches.append((match.start(), match.group()))
        
        # Sort by position and assign to tables sequentially
        all_table_matches.sort(key=lambda x: x[0])
        for i, (pos, match_text) in enumerate(all_table_matches):
            if i < len(content_map['tables']):
                table = content_map['tables'][i]
                table_mentions.append((pos, i + 1, table))
                logger.info(f"Detected table reference: '{match_text}' at position {pos}")
        
        # If no specific references found, try to place multimedia content at natural break points
        if not image_mentions and not table_mentions:
            logger.info("No specific contextual references detected, using fallback placement at paragraph breaks")
            
            # Check if there are any mentions of images or tables in the text
            has_image_mentions = any(word in response_structure.lower() for word in ['image', 'illustrated', 'shown', 'visual'])
            has_table_mentions = any(word in response_structure.lower() for word in ['table', 'data', 'results', 'statistics'])
            
            logger.info(f"General mentions detected - Images: {has_image_mentions}, Tables: {has_table_mentions}")
            
            # Place images and tables at paragraph breaks or sentence endings
            text_parts = response_structure.split('\n\n')
            current_pos = 0
            
            for i, text_part in enumerate(text_parts):
                if text_part.strip():
                    # Add text part
                    inline_elements.append({
                        'type': 'text',
                        'content': text_part.strip()
                    })
                    current_pos += len(text_part) + 2  # +2 for \n\n
                    
                    # Add multimedia content after text parts if available and if there are general mentions
                    if has_image_mentions and i < len(content_map['images']) and len(content_map['images']) > 0:
                        img_data = content_map['images'][i]
                        img_id = f"page_{img_data['page_number']}_img_{i+1}"
                        if img_id not in used_image_ids:
                            used_image_ids.add(img_id)
                            inline_elements.append({
                                'type': 'image',
                                'data': img_data,
                                'context': self._extract_image_context(img_data, content_map)
                            })
                            logger.info(f"Added image via fallback placement: {img_id}")
                    
                    if has_table_mentions and i < len(content_map['tables']) and len(content_map['tables']) > 0:
                        table_data = content_map['tables'][i]
                        table_id = table_data.get('table_id', '')
                        page_number = table_data.get('page_number', 0)
                        unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_table_{i+1}"
                        
                        if unique_key not in used_table_ids:
                            used_table_ids.add(unique_key)
                            inline_elements.append({
                                'type': 'table',
                                'data': table_data,
                                'context': self._extract_table_context(table_data, content_map)
                            })
                            logger.info(f"Added table via fallback placement: {unique_key}")
            
            # Post-process elements for better flow
            inline_elements = self._optimize_element_flow(inline_elements)
            
            # If still no multimedia content was added, add all available content at the end
            if not any(elem['type'] in ['image', 'table'] for elem in inline_elements):
                logger.info("No multimedia content added via fallback, adding all available content at the end")
                
                # Add all available images
                for i, img_data in enumerate(content_map['images']):
                    img_id = f"page_{img_data['page_number']}_img_{i+1}"
                    if img_id not in used_image_ids:
                        used_image_ids.add(img_id)
                        inline_elements.append({
                            'type': 'image',
                            'data': img_data,
                            'context': self._extract_image_context(img_data, content_map)
                        })
                        logger.info(f"Added image at end: {img_id}")
                
                # Add all available tables
                for i, table_data in enumerate(content_map['tables']):
                    table_id = table_data.get('table_id', '')
                    page_number = table_data.get('page_number', 0)
                    unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_table_{i+1}"
                    
                    if unique_key not in used_table_ids:
                        used_table_ids.add(unique_key)
                        inline_elements.append({
                            'type': 'table',
                            'data': table_data,
                            'context': self._extract_table_context(table_data, content_map)
                        })
                        logger.info(f"Added table at end: {unique_key}")
            
            return inline_elements
        else:
            logger.info(f"Detected {len(image_mentions)} image mentions and {len(table_mentions)} table mentions")
        
        # Sort mentions by position in the text
        image_mentions.sort(key=lambda x: x[0])
        table_mentions.sort(key=lambda x: x[0])
        
        # Combine and sort all mentions
        all_mentions = [(pos, 'image', num, data) for pos, num, data in image_mentions] + \
                      [(pos, 'table', num, data) for pos, num, data in table_mentions]
        all_mentions.sort(key=lambda x: x[0])
        
        # Split the text and insert multimedia elements at appropriate positions
        current_pos = 0
        current_text = ""
        
        for mention_pos, content_type, content_num, content_data in all_mentions:
            # Find a natural break point after the contextual reference (end of sentence)
            reference_start = mention_pos
            
            # Look for the next sentence ending after the reference
            reference_end = mention_pos + 50  # Start looking after the reference
            while reference_end < len(response_structure):
                char = response_structure[reference_end]
                # Look for end of sentence
                if char in ['.', '!', '?']:
                    # Look ahead to see if this is truly end of sentence
                    next_pos = reference_end + 1
                    while next_pos < len(response_structure) and response_structure[next_pos] in [' ', '\n', '\t']:
                        next_pos += 1
                    
                    # If next character is uppercase or end of text, this is sentence end
                    if (next_pos >= len(response_structure) or 
                        response_structure[next_pos].isupper() or 
                        response_structure[next_pos] in ['\n']):
                        reference_end = next_pos
                        break
                reference_end += 1
            
            # Include text up to the reference point plus the complete sentence
            if reference_end > current_pos:
                text_chunk = response_structure[current_pos:reference_end].strip()
                if text_chunk:
                    current_text += " " + text_chunk if current_text else text_chunk
            
            # Add the multimedia element if not already used
            if content_type == 'image':
                img_id = f"page_{content_data['page_number']}_img_{content_num}"
                if img_id not in used_image_ids:
                    used_image_ids.add(img_id)
                    
                    # Add accumulated text as a text element (complete sentence with reference)
                    if current_text.strip():
                        inline_elements.append({
                            'type': 'text',
                            'content': current_text.strip()
                        })
                        current_text = ""
                    
                    # Add the image
                    inline_elements.append({
                        'type': 'image',
                        'data': content_data,
                        'context': self._extract_image_context(content_data, content_map)
                    })
                    logger.info(f"Added image with contextual reference: {img_id}")
            
            elif content_type == 'table':
                table_id = content_data.get('table_id', '')
                page_number = content_data.get('page_number', 0)
                unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_table_{content_num}"
                
                if unique_key not in used_table_ids:
                    used_table_ids.add(unique_key)
                    
                    # Add accumulated text as a text element (complete sentence with reference)
                    if current_text.strip():
                        inline_elements.append({
                            'type': 'text',
                            'content': current_text.strip()
                        })
                        current_text = ""
                    
                    # Add the table
                    inline_elements.append({
                        'type': 'table',
                        'data': content_data,
                        'context': self._extract_table_context(content_data, content_map)
                    })
                    logger.info(f"Added table with contextual reference: {unique_key}")
            
            # Move to the end of the processed text
            current_pos = reference_end
        
        # Add any remaining text
        if current_pos < len(response_structure):
            remaining_text = response_structure[current_pos:].strip()
            if remaining_text:
                current_text += " " + remaining_text if current_text else remaining_text
        
        if current_text.strip():
            inline_elements.append({
                'type': 'text',
                'content': current_text.strip()
            })
        
        # Post-process elements for better flow
        inline_elements = self._optimize_element_flow(inline_elements)
        
        return inline_elements
    
    def _merge_consecutive_text_parts(self, parts: List[str]) -> List[str]:
        """Merge consecutive text parts to avoid fragmentation while preserving placeholders."""
        merged_parts = []
        current_text = ""
    
        for part in parts:
            if part.startswith('{IMAGE_') or part.startswith('{TABLE_') or part.startswith('{TEXT_SECTION_'):
                # This is a placeholder
                if current_text.strip():
                    merged_parts.append(current_text.strip())
                    current_text = ""
                merged_parts.append(part)
            else:
                # This is text content - accumulate it
                if current_text:
                    # Add a space between text parts to avoid word concatenation
                    current_text += " " + part
                else:
                    current_text = part
    
        # Add any remaining text
        if current_text.strip():
            merged_parts.append(current_text.strip())
    
        return merged_parts
    
    def _extract_image_context(self, img_data: Dict, content_map: Dict) -> Dict:
        """Extract contextual information for image placement."""
        return {
            'page': img_data.get('page_number', 0),
            'relevance': img_data.get('relevance_score', 0),
            'type': 'visual_evidence'
        }
    
    def _extract_table_context(self, table_data: Dict, content_map: Dict) -> Dict:
        """Extract contextual information for table placement."""
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
            except:
                pass
                
        return context
    
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
                    if element['content']:
                        # Enhanced text display with better formatting
                        self._display_text_seamlessly(element['content'])
                
                elif element_type == 'image':
                    # Display image with contextual integration
                    self._display_image_seamlessly(element['data'], element.get('context', {}))
                
                elif element_type == 'table':
                    # Display table with contextual integration
                    self._display_table_seamlessly(element['data'], element.get('context', {}))
                
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
        
        # Remove redundant table summary - let data speak for itself
        
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
                table_summary = table_data.get('summary', 'structured information')[:150]
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
            summary = table.get('summary', 'Data table')[:150]
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
    
    def _regenerate_with_inline_focus(self, query: str, content_map: Dict, has_text: bool, has_images: bool, has_tables: bool) -> str:
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
            summary = table.get('summary', 'Structured data')[:200]
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


print("InlineMultimodalGenerator created successfully")