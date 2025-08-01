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
    
    def generate_inline_response(self, query: str, filtered_content: Dict, conditional_instructions: Dict = None) -> Dict[str, Any]:
        """Generate intelligent inline multimodal response."""
        try:
            # Analyze content availability
            has_text = len(filtered_content['text']) > 0
            has_images = len(filtered_content['images']) > 0  
            has_tables = len(filtered_content['tables']) > 0
            
            # If no content found, return helpful fallback response
            if not (has_text or has_images or has_tables):
                return self._create_fallback_response(query)
            
            # Create content map sorted by relevance
            content_map = self._create_content_map(filtered_content, query)
            
            # Generate contextual response structure
            response_structure = self._generate_response_structure(query, content_map, has_text, has_images, has_tables)
            
            # Create inline elements for Streamlit display
            inline_elements = self._create_inline_elements(response_structure, content_map)
            
            return {
                'inline_elements': inline_elements,
                'content_summary': {
                    'has_text': has_text,
                    'has_images': has_images, 
                    'has_tables': has_tables,
                    'total_elements': len(inline_elements)
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating inline response: {e}")
            return {
                'inline_elements': [{'type': 'text', 'content': f"Error generating response: {str(e)}"}],
                'content_summary': {'has_text': False, 'has_images': False, 'has_tables': False, 'total_elements': 1}
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
        
        # Dynamic content limits based on query type and available content
        if is_show_all_query:
            # For "show all" queries, include all available content
            content_limit = len(filtered_content['text']) + len(filtered_content['images']) + len(unique_tables)
        else:
            # For specific queries, let the AI decide what's relevant
            content_limit = min(20, len(filtered_content['text']) + len(filtered_content['images']) + len(unique_tables))
        
        content_map = {
            'text': sorted(filtered_content['text'], key=lambda x: x['relevance_score'], reverse=True),
            'images': sorted(filtered_content['images'], key=lambda x: x['relevance_score'], reverse=True),
            'tables': unique_tables
        }
        
        # Debug logging
        logger.info(f"Dynamic content map created with {len(content_map['tables'])} unique tables")
        logger.info(f"Query type: {'show_all' if is_show_all_query else 'specific'}")
        logger.info(f"Content map summary: {len(content_map['images'])} images, {len(content_map['tables'])} tables, {len(content_map['text'])} text sections")
        
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
        
        # Analyze images with context
        for i, img in enumerate(content_map['images']):
            img_desc = img.get('image_summary', img.get('ocr_text', 'Visual content'))[:200]
            page_context = f"Page {img['page_number']}"
            relevance = img.get('relevance_score', 0)
            content_analysis.append(f"IMAGE_{i+1} ({page_context}, relevance: {relevance:.2f}): {img_desc}")
        
        # Analyze tables with structure and content preview
        for i, table in enumerate(content_map['tables']):
            table_summary = table.get('summary', 'Structured data')
            page_context = f"Page {table['page_number']}"
            relevance = table.get('relevance_score', 0)
            
            # Add content preview if available
            if table.get('table_content_json'):
                try:
                    import json
                    table_json = json.loads(table['table_content_json'])
                    if isinstance(table_json, dict) and '_metadata' in table_json:
                        metadata = table_json['_metadata']
                        table_summary += f" ({metadata.get('total_rows', 0)} rows, {metadata.get('total_columns', 0)} cols)"
                    # Get column headers for context
                    headers = [k for k in table_json.keys() if k != '_metadata']
                    if headers:
                        table_summary += f" - Columns: {', '.join(headers[:5])}{'...' if len(headers) > 5 else ''}"
                except:
                    pass
                    
            content_analysis.append(f"TABLE_{i+1} ({page_context}, relevance: {relevance:.2f}): {table_summary}")
        
        # Enhanced system prompt for comprehensive document-based responses
        system_prompt = f"""You are a document analysis assistant that creates comprehensive, informative responses STRICTLY based on the provided document content.

CRITICAL CONSTRAINTS:
- You MUST only use information explicitly provided in the document content below
- Do NOT add any external knowledge, assumptions, or general information
- Base your response ONLY on the text, images, and tables listed below
- Provide detailed explanations and analysis of the document content

AVAILABLE DOCUMENT CONTENT:
{', '.join(context_parts)}

DETAILED CONTENT FROM DOCUMENT:
{chr(10).join(content_analysis)}

RESPONSE REQUIREMENTS:
1. Create a comprehensive, well-structured response that explains the document content
2. Use descriptive text to introduce and explain each multimedia element
3. Place {{IMAGE_X}} placeholders for ALL relevant images with explanatory context
4. Place {{TABLE_X}} placeholders for ALL relevant tables with explanatory context
5. Connect multimedia elements to the overall narrative
6. Use transitional phrases: "The document shows...", "As illustrated in...", "The data reveals..."
7. Provide detailed analysis of what each visual element demonstrates
8. Create flowing narrative that weaves together text explanations and multimedia

AVAILABLE MULTIMEDIA PLACEHOLDERS:
{self._generate_placeholder_list(content_map)}

QUERY TO ANSWER:
"{query}"

RESPONSE STRUCTURE EXAMPLE:
"Based on the document content, [explanation of topic]. The document shows [detailed description]. {{IMAGE_1}} illustrates [specific explanation of what the image shows]. 

Furthermore, the document reveals [additional analysis]. {{TABLE_1}} presents [detailed explanation of table data and its significance].

The content also demonstrates [more analysis]. {{IMAGE_2}} provides [specific context about second image]..."

CRITICAL REQUIREMENTS:
- Include explanatory text BEFORE and AFTER each multimedia element
- Explain what each {{IMAGE_X}} and {{TABLE_X}} demonstrates or contains
- Create a narrative flow that connects all elements
- Use ALL available multimedia placeholders with proper context
- Provide substantive analysis, not just placeholder placement

Generate a comprehensive, explanatory response that thoroughly analyzes the document content."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=800,
                temperature=0.1  # Lower temperature for more deterministic, grounded responses
            )
            
            generated_response = response.choices[0].message.content
            
            # Verify response is properly grounded
            if self._verify_response_grounding(generated_response, content_map):
                return generated_response
            else:
                logger.warning("Generated response failed grounding verification, using fallback")
                return self._create_fallback_grounded_response(query, content_map)
            
        except Exception as e:
            logger.error(f"Error generating response structure: {e}")
            # Generate dynamic fallback with all available content
            fallback_parts = ["Based on the document content provided, I can analyze the available information."]
            
            # Add all available images
            for i in range(len(content_map.get('images', []))):
                fallback_parts.append(f"{{{{IMAGE_{i+1}}}}}")
            
            # Add all available tables
            for i in range(len(content_map.get('tables', []))):
                fallback_parts.append(f"{{{{TABLE_{i+1}}}}}")
            
            return " ".join(fallback_parts)
    
    def _create_inline_elements(self, response_structure: str, content_map: Dict) -> List[Dict]:
        """Convert response structure with placeholders into seamlessly integrated inline elements."""
        inline_elements = []
        used_table_ids = set()  # Track used tables to prevent duplicates
        used_image_ids = set()  # Track used images to prevent duplicates
        
        # Enhanced splitting that preserves context around placeholders
        parts = re.split(r'(\{(?:IMAGE|TABLE|TEXT_SECTION)_\d+\})', response_structure)
        
        for i, part in enumerate(parts):
            if not part.strip():
                continue
                
            if part.startswith('{IMAGE_'):
                # Extract image number and add with enhanced context
                try:
                    img_num = int(re.search(r'IMAGE_(\d+)', part).group(1))
                    if img_num <= len(content_map['images']):
                        img_data = content_map['images'][img_num - 1]
                        img_id = f"page_{img_data['page_number']}_img_{img_num}"
                        
                        # Only add if not already used
                        if img_id not in used_image_ids:
                            used_image_ids.add(img_id)
                            
                            # Add contextual transition if needed
                            prev_part = parts[i-1] if i > 0 else ""
                            if prev_part.strip() and not prev_part.strip().endswith((':','.','!','?')):
                                # Add smooth transition
                                inline_elements.append({
                                    'type': 'text',
                                    'content': " As illustrated below:"
                                })
                            
                            inline_elements.append({
                                'type': 'image',
                                'data': img_data,
                                'context': self._extract_image_context(img_data, content_map)
                            })
                            logger.info(f"Added image with context: {img_id}")
                        else:
                            logger.info(f"Skipped duplicate image: {img_id}")
                except Exception as e:
                    logger.error(f"Error processing image placeholder: {e}")
                    
            elif part.startswith('{TABLE_'):
                # Extract table number and add with enhanced context
                try:
                    table_num = int(re.search(r'TABLE_(\d+)', part).group(1))
                    if table_num <= len(content_map['tables']):
                        table_data = content_map['tables'][table_num - 1]
                        table_id = table_data.get('table_id', '')
                        page_number = table_data.get('page_number', 0)
                        
                        # Create unique identifier for this table
                        unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_table_{table_num}"
                        
                        # Only add if not already used
                        if unique_key not in used_table_ids:
                            used_table_ids.add(unique_key)
                            
                            # Add contextual transition if needed
                            prev_part = parts[i-1] if i > 0 else ""
                            if prev_part.strip() and not prev_part.strip().endswith((':','.','!','?')):
                                # Add smooth transition
                                inline_elements.append({
                                    'type': 'text',
                                    'content': " The data is presented below:"
                                })
                            
                            inline_elements.append({
                                'type': 'table',
                                'data': table_data,
                                'context': self._extract_table_context(table_data, content_map)
                            })
                            logger.info(f"Added table with context: {unique_key}")
                        else:
                            logger.info(f"Skipped duplicate table: {unique_key}")
                except Exception as e:
                    logger.error(f"Error processing table placeholder: {e}")
                    
            elif part.startswith('{TEXT_SECTION_'):
                # Text sections removed from inline elements
                pass
                    
            else:
                # Regular text content with enhanced formatting
                text_content = part.strip()
                if text_content:
                    # Clean up and enhance text presentation
                    text_content = self._enhance_text_formatting(text_content)
                    inline_elements.append({
                        'type': 'text',
                        'content': text_content
                    })
        
        # Post-process elements for better flow
        inline_elements = self._optimize_element_flow(inline_elements)
        
        return inline_elements
    
    def _extract_image_context(self, img_data: Dict, content_map: Dict) -> Dict:
        """Extract contextual information for image placement."""
        return {
            'page': img_data.get('page_number', 0),
            'relevance': img_data.get('relevance_score', 0),
            'description': img_data.get('image_summary', img_data.get('ocr_text', ''))[:100],
            'type': 'visual_evidence'
        }
    
    def _extract_table_context(self, table_data: Dict, content_map: Dict) -> Dict:
        """Extract contextual information for table placement."""
        context = {
            'page': table_data.get('page_number', 0),
            'relevance': table_data.get('relevance_score', 0),
            'summary': table_data.get('summary', ''),
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
            # Add element
            optimized.append(element)
            
            # Add spacing after multimedia elements if next element is text
            if (element['type'] in ['image', 'table'] and 
                i + 1 < len(elements) and 
                elements[i + 1]['type'] == 'text'):
                
                # Add a small spacing element
                optimized.append({
                    'type': 'spacing',
                    'content': ''
                })
        
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
                
                elif element_type == 'spacing':
                    # Add natural spacing between elements
                    st.write("")
            
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
                # Create a natural caption
                caption = self._create_natural_caption(img_data, context)
                
                # Display with optimal sizing
                st.image(
                    image_source,
                    caption=caption,
                    use_container_width=True
                )
                
                # Remove redundant image summary - let image speak for itself
                    
            except Exception as e:
                st.error(f"Unable to display image: {e}")
                logger.error(f"Image display error: {e}")
        else:
            st.warning(f"Image from page {img_data['page_number']} is not available")
    
    def _display_table_seamlessly(self, table_data: Dict, context: Dict):
        """Display table with seamless integration and enhanced presentation."""
        logger.info(f"Seamlessly displaying table from page {table_data.get('page_number', 'unknown')}")
        
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
        # Check for document-grounding phrases (more flexible)
        grounding_phrases = [
            "according to the document", "the document states", "the document shows",
            "the text states", "on page", "the provided content", "the document explains",
            "as mentioned in the document", "the document indicates", "based on the document",
            "the document contains", "as shown in", "the content shows", "from the document",
            "the data shows", "the information shows", "according to the content",
            "the visual content", "the structured data", "the available information"
        ]
        
        response_lower = response.lower()
        has_grounding_phrases = any(phrase in response_lower for phrase in grounding_phrases)
        
        # Check for multimedia placeholders (indicates document-based content)
        has_multimedia_placeholders = bool(re.search(r'\{\{(IMAGE|TABLE)_\d+\}\}', response))
        
        # Check for warning signs of external knowledge (more targeted)
        warning_phrases = [
            "it is well known that", "research generally shows", "studies typically indicate", 
            "experts commonly believe", "this is universally true", "according to general knowledge",
            "as everyone knows", "it is widely accepted", "scientific consensus shows"
        ]
        
        has_warning_phrases = any(phrase in response_lower for phrase in warning_phrases)
        
        # Response is grounded if it has grounding phrases OR multimedia placeholders, and no strong warning phrases
        is_grounded = (has_grounding_phrases or has_multimedia_placeholders) and not has_warning_phrases
        
        if not is_grounded:
            logger.info(f"Grounding check: phrases={has_grounding_phrases}, multimedia={has_multimedia_placeholders}, warnings={has_warning_phrases}")
        
        return is_grounded
    
    def _create_fallback_grounded_response(self, query: str, content_map: Dict) -> str:
        """Create a fallback response that is strictly grounded in document content."""
        # Count available content
        text_count = len(content_map['text'])
        image_count = len(content_map['images'])
        table_count = len(content_map['tables'])
        
        # Create a comprehensive grounded response structure
        response_parts = []
        response_parts.append(f"Based on the document content related to '{query}', I can provide the following analysis:")
        
        if text_count > 0:
            response_parts.append("The document contains detailed textual information that directly addresses your query.")
        
        if image_count > 0:
            response_parts.append("The document includes visual content that illustrates key concepts:")
            for i in range(min(image_count, 5)):  # Include up to 5 images
                img_data = content_map['images'][i]
                img_desc = img_data.get('image_summary', img_data.get('ocr_text', 'visual content'))[:100]
                response_parts.append(f"{{{{IMAGE_{i+1}}}}} - This visual element from page {img_data['page_number']} shows {img_desc}")
        
        if table_count > 0:
            response_parts.append("The document provides structured data that offers detailed insights:")
            for i in range(min(table_count, 5)):  # Include up to 5 tables
                table_data = content_map['tables'][i]
                table_summary = table_data.get('summary', 'structured information')[:100]
                response_parts.append(f"{{{{TABLE_{i+1}}}}} - This data table from page {table_data['page_number']} contains {table_summary}")
        
        response_parts.append("The above multimedia elements provide comprehensive coverage of the topic as documented in the source material.")
        
        return " ".join(response_parts)
    
    def _generate_placeholder_list(self, content_map: Dict) -> str:
        """Generate a list of available placeholders for the AI to use."""
        placeholders = []
        
        # Add image placeholders
        for i in range(len(content_map['images'])):
            placeholders.append(f"{{IMAGE_{i+1}}}")
        
        # Add table placeholders  
        for i in range(len(content_map['tables'])):
            placeholders.append(f"{{TABLE_{i+1}}}")
        
        if placeholders:
            return f"Available: {', '.join(placeholders)}"
        else:
            return "No multimedia content available"
    
    def _create_natural_caption(self, img_data: Dict, context: Dict) -> str:
        """Create a natural, contextual caption for images."""
        page_num = img_data.get('page_number', 0)
        
        # Create contextual caption based on content
        if context.get('description'):
            return f"Figure from page {page_num}"
        else:
            return f"Visual content from page {page_num}"
    
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
                st.image(
                    image_source,
                    caption=f"Page {img_data['page_number']}",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.info(f"Image URL: {image_source}")
        else:
            st.warning(f"No image source found for page {img_data['page_number']}")
    
    def _display_table_inline(self, table_data: Dict):
        """Display table inline with minimal formatting."""
        logger.info(f"Displaying table from page {table_data.get('page_number', 'NO_PAGE')}")
        
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
    


print("InlineMultimodalGenerator created successfully")