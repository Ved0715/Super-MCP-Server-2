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
    
    def generate_inline_response(self, query: str, filtered_content: Dict) -> Dict[str, Any]:
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
            content_map = self._create_content_map(filtered_content)
            
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
    
    def _create_content_map(self, filtered_content: Dict) -> Dict[str, List]:
        """Create a sorted map of all content by relevance with duplicate prevention."""
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
                
            # Stop after getting enough unique tables
            if len(unique_tables) >= 3:
                break
        
        content_map = {
            'text': sorted(filtered_content['text'], key=lambda x: x['relevance_score'], reverse=True)[:3],
            'images': sorted(filtered_content['images'], key=lambda x: x['relevance_score'], reverse=True)[:3],
            'tables': unique_tables[:3]  # Take top 3 unique tables
        }
        
        # Debug logging
        logger.info(f"Content map created with {len(content_map['tables'])} unique tables")
        for i, table in enumerate(content_map['tables']):
            table_id = table.get('table_id', 'NO_ID')
            page = table.get('page_number', 'NO_PAGE')
            logger.info(f"  Table {i+1}: ID={table_id}, Page={page}")
        
        return content_map
    
    def _generate_response_structure(self, query: str, content_map: Dict, has_text: bool, has_images: bool, has_tables: bool) -> str:
        """Generate intelligent response structure with placeholders for inline content."""
        
        # Build context for AI
        context_parts = []
        
        # Add available content descriptions
        if has_text:
            context_parts.append(f"Available text content from {len(content_map['text'])} sections")
        if has_images:
            context_parts.append(f"Available visual content: {len(content_map['images'])} images with descriptions")
        if has_tables:
            context_parts.append(f"Available structured data: {len(content_map['tables'])} tables")
        
        # Create content summary for AI
        content_summary = []
        for img in content_map['images']:
            content_summary.append(f"Image from page {img['page_number']}: {img.get('ocr_text', 'Visual content')[:100]}")
        for table in content_map['tables']:
            content_summary.append(f"Table from page {table['page_number']}: {table.get('summary', 'Structured data')}")
        
        system_prompt = f"""You are an expert AI that creates engaging, contextual responses using multimodal content.

Available content: {', '.join(context_parts)}

Content details:
{chr(10).join(content_summary)}

Create a response structure using these special placeholders:
- {{IMAGE_1}}, {{IMAGE_2}}, {{IMAGE_3}} - to insert relevant images inline
- {{TABLE_1}} - to insert the most relevant table inline (use only one table placeholder)  
- Avoid using text section references

Guidelines:
1. Create a natural, flowing response that uses inline content contextually
2. Place images where they provide visual support for your explanation  
3. Place tables where data analysis is needed
4. Only use placeholders for content that adds value to the answer
5. Write 2-4 paragraphs with inline content naturally integrated
6. Start with a direct answer, then provide supporting evidence via inline content
7. Be concise but comprehensive

Query: {query}

Generate a response structure with appropriate inline placeholders:"""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": system_prompt}],
                max_tokens=800,
                temperature=0.3
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating response structure: {e}")
            return f"Based on the document content, I can provide information about your query using the available text, images, and tables. {{IMAGE_1}} {{TABLE_1}}"
    
    def _create_inline_elements(self, response_structure: str, content_map: Dict) -> List[Dict]:
        """Convert response structure with placeholders into inline elements."""
        inline_elements = []
        used_table_ids = set()  # Track used tables to prevent duplicates
        
        # Split response by placeholders
        parts = re.split(r'(\{(?:IMAGE|TABLE|TEXT_SECTION)_\d+\})', response_structure)
        
        for part in parts:
            if not part.strip():
                continue
                
            if part.startswith('{IMAGE_'):
                # Extract image number
                img_num = int(re.search(r'IMAGE_(\d+)', part).group(1))
                if img_num <= len(content_map['images']):
                    img_data = content_map['images'][img_num - 1]
                    inline_elements.append({
                        'type': 'image',
                        'data': img_data
                    })
                    
            elif part.startswith('{TABLE_'):
                # Extract table number
                table_num = int(re.search(r'TABLE_(\d+)', part).group(1))
                if table_num <= len(content_map['tables']):
                    table_data = content_map['tables'][table_num - 1]
                    table_id = table_data.get('table_id', '')
                    page_number = table_data.get('page_number', 0)
                    
                    # Create unique identifier for this table
                    unique_key = f"{table_id}_{page_number}" if table_id else f"page_{page_number}_unnamed"
                    
                    # Only add if not already used
                    if unique_key not in used_table_ids:
                        used_table_ids.add(unique_key)
                        inline_elements.append({
                            'type': 'table',
                            'data': table_data
                        })
                        logger.info(f"Added table to inline elements: {unique_key}")
                    else:
                        logger.info(f"Skipped duplicate table: {unique_key}")
                    
            elif part.startswith('{TEXT_SECTION_'):
                # Text sections removed from inline elements
                pass
                    
            else:
                # Regular text content
                inline_elements.append({
                    'type': 'text',
                    'content': part.strip()
                })
        
        return inline_elements
    
    def display_inline_response(self, inline_elements: List[Dict]):
        """Display inline multimodal response in Streamlit."""
        try:
            st.subheader("🤖 AI Response")
            
            for i, element in enumerate(inline_elements):
                if element['type'] == 'text':
                    if element['content']:
                        st.write(element['content'])
                        
                elif element['type'] == 'image':
                    img_data = element['data']
                    image_source = img_data.get('display_url', '') or img_data.get('s3_url', '') or img_data.get('local_path', '')
                    
                    if image_source:
                        st.image(
                            image_source,
                            caption=f"📊 Page {img_data['page_number']} | Relevance: {img_data['relevance_score']:.3f}",
                            use_container_width=True
                        )
                        
                        # OCR text removed from display
                
                elif element['type'] == 'table':
                    table_data = element['data']
                    
                    # Display table directly without expander
                    st.subheader(f"📊 Table from Page {table_data['page_number']}")
                    
                    if table_data.get('summary'):
                        st.write(f"**Summary:** {table_data['summary']}")
                    
                    if table_data.get('markdown_content'):
                        try:
                            st.markdown(table_data['markdown_content'])
                        except:
                            st.text(table_data['markdown_content'])
                
                elif element['type'] == 'text_highlight':
                    # Text highlights removed from display
                    pass
            
        except Exception as e:
            st.error(f"Error displaying inline response: {e}")
            logger.error(f"Display error: {e}")

print("InlineMultimodalGenerator created successfully")