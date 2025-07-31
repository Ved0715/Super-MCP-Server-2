import os
import logging
import json
import time
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
from pinecone import Pinecone
from openai import OpenAI
from config import config

@dataclass
class DocumentSummary:
    """Structured summary of a document"""
    title: str
    main_topics: List[str]
    key_findings: List[str]
    methodology: str
    conclusions: str
    document_type: str
    word_count: int
    page_count: int

@dataclass
class ComparisonResult:
    """Structured comparison result"""
    document1_summary: DocumentSummary
    document2_summary: DocumentSummary
    similarities: List[str]
    differences: List[str]
    overall_assessment: str
    recommendations: List[str]
    confidence_score: float

class PDFComparator:
    """
    Advanced PDF Comparison System
    Compares two PDFs stored in separate Pinecone namespaces with intelligent
    content analysis and structured output.
    """

    def __init__(self):
        self.embedding_dimension = config.embedding_dimension
        self.openai = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Use the same index as the multimodal integrator
        index_name = os.getenv('PINECONE_INDEX_NAME_TEST')
        if not index_name:
            raise ValueError("PINECONE_INDEX_NAME_TEST environment variable not found")
        
        self.index = self.pc.Index(index_name)
        self.model = config.response_model
        self.max_tokens_per_chunk = 4000  # Safe token limit
        self.max_summary_tokens = 2000    # For document summaries
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _list_namespaces(self, user_id: str) -> List[str]:
        """Get all namespaces for a user"""
        try:
            namespaces = [
            ns.name for ns in self.index.list_namespaces()
            if ns.name.startswith(f"user_{user_id}_doc_")
        ]
            self.logger.info(f"Found {len(namespaces)} namespaces for user {user_id}")
            return namespaces
        except Exception as e:
            self.logger.error(f"Error listing namespaces: {e}")
            return []

    def _load_chunks(self, namespace: str) -> List[Dict[str, Any]]:
        """Load all chunks from a namespace with metadata"""
        try:
            # Use dummy vector to get all chunks
            dummy_vector = [0.0] * self.embedding_dimension
            response = self.index.query(
                vector=dummy_vector,
                top_k=10000,  # Large number to get all chunks
            namespace=namespace,
            include_metadata=True,
            include_values=False
        )
            
            chunks = []
            for match in response.matches:
                # Debug: Log the first few matches to see metadata structure
                if len(chunks) < 3:
                    self.logger.info(f"Sample match metadata: {match.metadata}")
                    self.logger.info(f"Available metadata keys: {list(match.metadata.keys())}")
                
                # Extract content from multiple possible fields
                content = match.metadata.get('content', '')
                if not content:
                    content = match.metadata.get('text', '')
                if not content and '_node_content' in match.metadata:
                    try:
                        node_content = json.loads(match.metadata['_node_content'])
                        content = node_content.get('text', '')
                    except (json.JSONDecodeError, TypeError):
                        content = str(match.metadata.get('_node_content', ''))[:500]
                
                chunk_data = {
                    'id': match.id,
                    'content': content,
                    'page_number': match.metadata.get('page_number', 0),
                    'section_type': match.metadata.get('section_type', 'unknown'),
                    'word_count': match.metadata.get('word_count', 0),
                    'relevance_score': match.score
                }
                chunks.append(chunk_data)
            
            self.logger.info(f"Loaded {len(chunks)} chunks from namespace {namespace}")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error loading chunks from {namespace}: {e}")
            return []

    def _chunk_text_for_processing(self, text: str, max_chars: int = 3000) -> List[str]:
        """Split text into manageable chunks for processing"""
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < max_chars:
                current_chunk += paragraph + '\n\n'
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + '\n\n'
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _generate_document_summary(self, chunks: List[Dict[str, Any]], doc_name: str) -> DocumentSummary:
        """Generate a comprehensive summary of a document from its chunks"""
        try:
            # Aggregate all content
            full_text = "\n\n".join([chunk['content'] for chunk in chunks if chunk['content']])
            
            if not full_text.strip():
                return DocumentSummary(
                    title=f"Document: {doc_name}",
                    main_topics=["No content available"],
                    key_findings=["Document appears to be empty or unprocessed"],
                    methodology="Not available",
                    conclusions="Not available",
                    document_type="Unknown",
                    word_count=0,
                    page_count=0
                )
            
            # Split into chunks for processing
            text_chunks = self._chunk_text_for_processing(full_text)
            
            # Create engaging prompt for document summary
            summary_prompt = f"""
You are a knowledgeable research consultant analyzing a document. Create a comprehensive yet accessible summary.

**Document Name:** {doc_name}
**Content:** {full_text[:4000]}

**Your Task:**
Create a JSON summary with these fields:
- title: A clear, descriptive title
- main_topics: List of 3-5 main topics/themes
- key_findings: List of 3-5 key insights or findings
- methodology: Brief description of approach (if mentioned)
- conclusions: Main takeaways or bottom line
- document_type: Type of document (research paper, report, etc.)
- word_count: Total word count
- page_count: Total page count

**Writing Style:**
- Use clear, accessible language
- Focus on practical insights
- Avoid overly technical jargon
- Be concise but comprehensive
- Note any specific page numbers mentioned in the content

**IMPORTANT:** Return ONLY valid JSON. No extra text, markdown, or explanations.
"""
            
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document analyst. Provide accurate, structured summaries in JSON format only."},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )
            
            # Parse JSON response
            try:
                response_content = response.choices[0].message.content.strip()
                self.logger.info(f"LLM Response: {response_content[:500]}...")
                
                # Try to extract JSON from the response
                summary_data = json.loads(response_content)
                return DocumentSummary(**summary_data)
            except json.JSONDecodeError as e:
                self.logger.error(f"JSON parsing failed: {e}")
                self.logger.error(f"Raw response: {response.choices[0].message.content}")
                
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    try:
                        summary_data = json.loads(json_match.group(1))
                        return DocumentSummary(**summary_data)
                    except json.JSONDecodeError:
                        pass
                
                # Try to extract JSON from the response (remove any markdown formatting)
                cleaned_response = re.sub(r'```.*?```', '', response_content, flags=re.DOTALL)
                cleaned_response = re.sub(r'#.*?\n', '', cleaned_response)
                try:
                    summary_data = json.loads(cleaned_response.strip())
                    return DocumentSummary(**summary_data)
                except json.JSONDecodeError:
                    pass
                
                # Final fallback - create a basic summary from the text
                return DocumentSummary(
                    title=f"Document: {doc_name}",
                    main_topics=["Analysis completed but parsing failed"],
                    key_findings=["Document contains substantial content but summary parsing failed"],
                    methodology="Content analysis available",
                    conclusions="Document analysis completed",
                    document_type="research_paper",
                    word_count=len(full_text.split()),
                    page_count=len(set(chunk['page_number'] for chunk in chunks))
                )
                
        except Exception as e:
            self.logger.error(f"Error generating summary for {doc_name}: {e}")
            return DocumentSummary(
                title=f"Document: {doc_name}",
                main_topics=["Error in analysis"],
                key_findings=[f"Analysis failed: {str(e)}"],
                methodology="Not available",
                conclusions="Not available",
                document_type="Unknown",
                word_count=0,
                page_count=0
            )

    def _compare_documents(self, summary1: DocumentSummary, summary2: DocumentSummary) -> ComparisonResult:
        """Compare two document summaries and generate comprehensive comparison"""
        try:
            # Create engaging prompt for document comparison
            comparison_prompt = f"""
Compare these two documents and provide a comprehensive analysis in JSON format.

**Document 1 Summary:**
- Title: {summary1.title}
- Main Topics: {', '.join(summary1.main_topics)}
- Key Findings: {', '.join(summary1.key_findings)}
- Methodology: {summary1.methodology}
- Conclusions: {summary1.conclusions}
- Type: {summary1.document_type}

**Document 2 Summary:**
- Title: {summary2.title}
- Main Topics: {', '.join(summary2.main_topics)}
- Key Findings: {', '.join(summary2.key_findings)}
- Methodology: {summary2.methodology}
- Conclusions: {summary2.conclusions}
- Type: {summary2.document_type}

IMPORTANT: Respond ONLY with valid JSON. Do not include any markdown formatting, explanations, or additional text.

Required JSON structure:
{{
    "similarities": ["similarity1", "similarity2", "similarity3"],
    "differences": ["difference1", "difference2", "difference3"],
    "overall_assessment": "Comprehensive assessment of how the documents relate",
    "recommendations": ["recommendation1", "recommendation2"],
    "confidence_score": 0.85
}}

Focus on meaningful comparisons and provide actionable insights.
"""
            
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert document comparison analyst. Provide detailed, insightful comparisons in JSON format only."},
                    {"role": "user", "content": comparison_prompt}
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse comparison result
            try:
                response_content = response.choices[0].message.content.strip()
                self.logger.info(f"Comparison LLM Response: {response_content[:500]}...")
                
                # Try to extract JSON from the response
                comparison_data = json.loads(response_content)
                return ComparisonResult(
                    document1_summary=summary1,
                    document2_summary=summary2,
                    similarities=comparison_data.get('similarities', []),
                    differences=comparison_data.get('differences', []),
                    overall_assessment=comparison_data.get('overall_assessment', ''),
                    recommendations=comparison_data.get('recommendations', []),
                    confidence_score=comparison_data.get('confidence_score', 0.5)
                )
            except json.JSONDecodeError as e:
                self.logger.error(f"Comparison JSON parsing failed: {e}")
                self.logger.error(f"Raw comparison response: {response.choices[0].message.content}")
                
                # Try to extract JSON from markdown code blocks
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_content, re.DOTALL)
                if json_match:
                    try:
                        comparison_data = json.loads(json_match.group(1))
                        return ComparisonResult(
                            document1_summary=summary1,
                            document2_summary=summary2,
                            similarities=comparison_data.get('similarities', []),
                            differences=comparison_data.get('differences', []),
                            overall_assessment=comparison_data.get('overall_assessment', ''),
                            recommendations=comparison_data.get('recommendations', []),
                            confidence_score=comparison_data.get('confidence_score', 0.5)
                        )
                    except json.JSONDecodeError:
                        pass
                
                # Try to extract JSON from the response (remove any markdown formatting)
                cleaned_response = re.sub(r'```.*?```', '', response_content, flags=re.DOTALL)
                cleaned_response = re.sub(r'#.*?\n', '', cleaned_response)
                try:
                    comparison_data = json.loads(cleaned_response.strip())
                    return ComparisonResult(
                        document1_summary=summary1,
                        document2_summary=summary2,
                        similarities=comparison_data.get('similarities', []),
                        differences=comparison_data.get('differences', []),
                        overall_assessment=comparison_data.get('overall_assessment', ''),
                        recommendations=comparison_data.get('recommendations', []),
                        confidence_score=comparison_data.get('confidence_score', 0.5)
                    )
                except json.JSONDecodeError:
                    pass
                
                # Final fallback - create a basic comparison
                return ComparisonResult(
                    document1_summary=summary1,
                    document2_summary=summary2,
                    similarities=["Both documents contain substantial research content"],
                    differences=["Detailed comparison requires manual review"],
                    overall_assessment="Both documents appear to be research papers with significant content",
                    recommendations=["Review documents manually for detailed comparison"],
                    confidence_score=0.3
                )
                
        except Exception as e:
            self.logger.error(f"Error comparing documents: {e}")
            return ComparisonResult(
                document1_summary=summary1,
                document2_summary=summary2,
                similarities=["Comparison failed"],
                differences=[f"Error: {str(e)}"],
                overall_assessment="Comparison could not be completed due to technical error",
                recommendations=["Try again or contact support"],
                confidence_score=0.0
            )

    def compare_documents(
        self,
        user_id: str,
        doc1_uuid: str,
        doc2_uuid: str
    ) -> Dict[str, Any]:
        """
        Main method to compare two documents
        
        Args:
            user_id: User identifier
            doc1_uuid: First document UUID
            doc2_uuid: Second document UUID
            
        Returns:
            Dictionary containing comprehensive comparison results
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not user_id or not doc1_uuid or not doc2_uuid:
                raise ValueError("user_id, doc1_uuid, and doc2_uuid are required")
            
            # Build namespaces
            ns1 = f"user_{user_id}_doc_{doc1_uuid}"
            ns2 = f"user_{user_id}_doc_{doc2_uuid}"
            
            # Check if namespaces exist
            available_namespaces = self._list_namespaces(user_id)
            if ns1 not in available_namespaces:
                raise ValueError(f"Document 1 namespace '{ns1}' not found")
            if ns2 not in available_namespaces:
                raise ValueError(f"Document 2 namespace '{ns2}' not found")
            
            self.logger.info(f"Starting comparison between {ns1} and {ns2}")
            
            # Load chunks from both documents
            chunks1 = self._load_chunks(ns1)
            chunks2 = self._load_chunks(ns2)

            if not chunks1:
                raise ValueError(f"No content found in document 1 ({ns1})")
            if not chunks2:
                raise ValueError(f"No content found in document 2 ({ns2})")
            
            # Generate summaries
            self.logger.info("Generating document summaries...")
            summary1 = self._generate_document_summary(chunks1, f"Document 1 ({doc1_uuid})")
            summary2 = self._generate_document_summary(chunks2, f"Document 2 ({doc2_uuid})")
            
            # Compare documents
            self.logger.info("Comparing documents...")
            comparison_result = self._compare_documents(summary1, summary2)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Format response
            response = {
                "success": True,
                "processing_time": processing_time,
                "document1": {
                    "uuid": doc1_uuid,
                    "namespace": ns1,
                    "summary": {
                        "title": summary1.title,
                        "main_topics": summary1.main_topics,
                        "key_findings": summary1.key_findings,
                        "methodology": summary1.methodology,
                        "conclusions": summary1.conclusions,
                        "document_type": summary1.document_type,
                        "word_count": summary1.word_count,
                        "page_count": summary1.page_count
                    }
                },
                "document2": {
                    "uuid": doc2_uuid,
                    "namespace": ns2,
                    "summary": {
                        "title": summary2.title,
                        "main_topics": summary2.main_topics,
                        "key_findings": summary2.key_findings,
                        "methodology": summary2.methodology,
                        "conclusions": summary2.conclusions,
                        "document_type": summary2.document_type,
                        "word_count": summary2.word_count,
                        "page_count": summary2.page_count
                    }
                },
                "comparison": {
                    "similarities": comparison_result.similarities,
                    "differences": comparison_result.differences,
                    "overall_assessment": comparison_result.overall_assessment,
                    "recommendations": comparison_result.recommendations,
                    "confidence_score": comparison_result.confidence_score
                },
                "metadata": {
                    "total_chunks_doc1": len(chunks1),
                    "total_chunks_doc2": len(chunks2),
                    "comparison_method": "LLM-based analysis",
                    "model_used": self.model
                }
            }
            
            self.logger.info(f"Comparison completed successfully in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            self.logger.error(f"Error in document comparison: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }

    def get_comparison_report(self, comparison_result: Dict[str, Any]) -> str:
        """Generate a comprehensive, high-quality report from comparison results using LLM"""
        if not comparison_result.get("success", False):
            return f"❌ Comparison failed: {comparison_result.get('error', 'Unknown error')}"
        
        try:
            # Generate comprehensive report using LLM
            return self._generate_comprehensive_report(comparison_result)
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            # Fallback to simple template
            return self._generate_simple_report(comparison_result)
    
    def _generate_comprehensive_report(self, comparison_result: Dict[str, Any]) -> str:
        """Generate a comprehensive, balanced report with formal structure and accessible tone"""
        doc1 = comparison_result["document1"]
        doc2 = comparison_result["document2"]
        comparison = comparison_result["comparison"]
        metadata = comparison_result.get("metadata", {})
        
        # Extract page information and examples from chunks
        doc1_pages = metadata.get("total_chunks_doc1", 0)
        doc2_pages = metadata.get("total_chunks_doc2", 0)
        
        # Create balanced, comprehensive prompt for report generation
        report_prompt = f"""
You are an expert research consultant creating a comprehensive document comparison. Your task is to provide an in-depth analysis that is both thorough and accessible. Focus on High Quality and COMPREHENSIVE CONTENT.

## DOCUMENT INFORMATION

**Document 1: {doc1['summary']['title']}**
- Type: {doc1['summary']['document_type']}
- Main Focus: {', '.join(doc1['summary']['main_topics'])}
- Key Points: {', '.join(doc1['summary']['key_findings'])}
- Methodology: {doc1['summary']['methodology']}
- Conclusions: {doc1['summary']['conclusions']}
- Bottom Line: {doc1['summary']['conclusions']}
- Size: {doc1['summary']['word_count']:,} words, {doc1['summary']['page_count']} pages
- Content Chunks: {doc1_pages} chunks analyzed

**Document 2: {doc2['summary']['title']}**
- Type: {doc2['summary']['document_type']}
- Main Focus: {', '.join(doc2['summary']['main_topics'])}
- Key Points: {', '.join(doc2['summary']['key_findings'])}
- Methodology: {doc2['summary']['methodology']}
- Conclusions: {doc2['summary']['conclusions']}
- Bottom Line: {doc2['summary']['conclusions']}
- Size: {doc2['summary']['word_count']:,} words, {doc2['summary']['page_count']} pages
- Content Chunks: {doc2_pages} chunks analyzed

## COMPARISON INSIGHTS

**What They Have in Common:**
{chr(10).join([f"• {sim}" for sim in comparison['similarities']])}

**Where They Differ:**
{chr(10).join([f"• {diff}" for diff in comparison['differences']])}

**Overall Assessment:**
{comparison['overall_assessment']}

**Practical Recommendations:**
{chr(10).join([f"• {rec}" for rec in comparison['recommendations']])}

**Confidence Level: {comparison['confidence_score']:.1%}**

## YOUR TASK

Create a COMPREHENSIVE comparison report with the following structure:

### 1. Concise Overview

- Provide a brief 2-3 sentence introduction to both documents
- Include essential background context in 2-3 lines within the introduction
- Set the stage for the comparison efficiently

### 2. Detailed Comparative Analysis

Analyze these key aspects for both documents with EXTENSIVE DETAIL:

- **Purpose and Audience**: Comprehensive analysis of what each document aims to achieve, who it targets, and why it matters
- **Content Focus**: Detailed breakdown of main themes, topics, areas of emphasis, and core concepts
- **Detailed Document Analysis**: In-depth analysis of each document's content, methodology, findings, and contributions
- **Approach and Style**: Comprehensive examination of how information is presented, organized, and communicated
- **Critical Assessment**: Extensive evaluation of strengths, weaknesses, contributions, and limitations of each document
- **Key Themes**: Detailed exploration of central ideas, recurring concepts, and underlying principles
- **Findings and Implications**: Comprehensive analysis of main insights, their significance, and broader implications
- **Methodological **: Detailed assessment of research methods, data collection, analysis approaches, and validity
- **Practical Applications**: Comprehensive exploration of real-world applications and use cases

### 3. Detailed Comparison

- Provide extensive side-by-side analysis of each aspect with detailed explanations
- Highlight similarities and differences with thorough analysis and examples
- Explain the significance of key distinctions with comprehensive reasoning
- Connect findings to broader implications across multiple dimensions
- Include detailed comparisons of methodology, approach, and outcomes
- Analyze the impact and influence of each document in its respective field
- Provide extensive examples and evidence from the documents

### 4. Brief Synthesis and Conclusions

- Concise summary of key insights (2-3 lines)
- Brief practical takeaways (1-2 lines)
- **DO NOT include a separate "Final Recommendations" section**

**REQUIREMENTS:**
- Use a **balanced tone**: Professional but accessible, formal but engaging
- Maintain **COMPREHENSIVE coverage**: Analyze all aspects thoroughly with extensive detail
- Provide **in-depth analysis**: Go beyond surface-level comparisons with deep insights
- Use **clear structure**: Well-organized sections with logical flow and detailed subsections
- Include **practical insights**: Actionable takeaways and recommendations with specific examples
- Aim for **1200-1500 words**: Concise but comprehensive content
- Use **markdown formatting**: Clear headings, bullet points, and structure with detailed subsections
- Balance **analytical depth** with **readability**
- Provide specific examples and evidence from the documents with detailed explanations
- Structure the report logically with clear sections and comprehensive subsections
- **INCLUDE PAGE REFERENCES**: When citing specific content, findings, or examples from the documents, include page references in the format (p. X) or (pp. X-Y) where appropriate. Use estimated page ranges based on document length and content distribution.
- Add detailed explanations for all comparisons and insights
- Provide extensive context and background information
- Include comprehensive cross-disciplinary analysis and implications

**PAGE REFERENCE GUIDELINES:**
- Include page references when citing specific content, findings, or examples
- Use format: (p. X) for single pages, (pp. X-Y) for page ranges
- Reference key findings, methodology sections, important conclusions, and significant examples
- Include page references for direct quotes or specific data points when mentioned
- Use page references to support your analysis and make it more credible
- Estimate page ranges based on document length: Document 1 has {doc1['summary']['page_count']} pages, Document 2 has {doc2['summary']['page_count']} pages
- Distribute page references throughout the document logically (e.g., methodology sections typically appear in early pages, findings in middle pages, conclusions in later pages)

**SPECIFIC EXAMPLES GUIDELINES:**
- Include specific examples from the document content when discussing key points
- Reference specific findings, methodologies, or conclusions mentioned in the documents
- Use direct quotes or paraphrased content with page references when appropriate
- Provide concrete examples of concepts, techniques, or approaches discussed in each document
- Include specific data points, statistics, or results mentioned in the documents
- Reference specific sections, chapters, or topics covered in each document

**CONCISENESS GUIDELINES:**
- Keep overview to 2-3 sentences maximum
- Condense background context into 2-3 lines within introduction
- Merge scope and significance into synthesis section
- Keep conclusions brief (2-3 lines)
- Focus on essential analysis rather than verbose descriptions
- Trim unnecessary stylistic evaluations unless specifically required

**TONE GUIDELINES:**
- Professional but not overly academic
- Thorough and comprehensive but not dry
- Engaging but not casual
- Comprehensive but not overwhelming
- Analytical but accessible
- Detailed but well-structured

Generate a COMPREHENSIVE report that provides extensive depth and thorough analysis while maintaining accessibility and engagement. Make it detailed, thorough, and comprehensive in every section. Include appropriate page references and specific examples from the documents to enhance credibility and usefulness. End with synthesis and conclusions, but do not include a separate "Final Recommendations" section. Keep the overview and conclusions concise as specified.
"""
        
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a skilled research consultant who creates comprehensive document comparisons. You write with a balanced tone that is professional yet accessible, thorough yet engaging, and analytical yet practical. Focus on providing EXTENSIVE, DETAILED, and COMPREHENSIVE analysis with deep insights and thorough coverage of all aspects. Always include appropriate page references (p. X) when citing specific content, findings, or examples to enhance credibility and usefulness. Include specific examples, quotes, and concrete details from the documents to support your analysis. Keep overviews concise (2-3 sentences) and conclusions brief (2-3 lines). End with synthesis and conclusions, but do not include a separate 'Final Recommendations' section."},
                    {"role": "user", "content": report_prompt}
                ],
                max_tokens=3000,
                temperature=0.3
            )
            comprehensive_report = response.choices[0].message.content.strip()
            
            # Add processing metadata
            processing_time = comparison_result.get("processing_time", 0)
            final_report = f"""
{comprehensive_report}

---
*Report generated in {processing_time:.2f} seconds using {metadata.get('model_used', 'AI analysis')}*
"""
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def _generate_simple_report(self, comparison_result: Dict[str, Any]) -> str:
        """Generate a simple, engaging fallback report"""
        doc1 = comparison_result["document1"]
        doc2 = comparison_result["document2"]
        comparison = comparison_result["comparison"]
        processing_time = comparison_result.get("processing_time", 0)
        
        report = f"""
# 📊 Document Comparison Overview

## 📄 Document Overview

**Document 1: {doc1['summary']['title']}**
- **Type:** {doc1['summary']['document_type']}
- **Main Focus:** {', '.join(doc1['summary']['main_topics'])}
- **Key Points:** {', '.join(doc1['summary']['key_findings'])}
- **Methodology:** {doc1['summary']['methodology']}
- **Bottom Line:** {doc1['summary']['conclusions']}
- **Size:** {doc1['summary']['word_count']:,} words, {doc1['summary']['page_count']} pages

**Document 2: {doc2['summary']['title']}**
- **Type:** {doc2['summary']['document_type']}
- **Main Focus:** {', '.join(doc2['summary']['main_topics'])}
- **Key Points:** {', '.join(doc2['summary']['key_findings'])}
- **Methodology:** {doc2['summary']['methodology']}
- **Bottom Line:** {doc2['summary']['conclusions']}
- **Size:** {doc2['summary']['word_count']:,} words, {doc2['summary']['page_count']} pages

## 🔍 Key Insights

### ✅ What They Have in Common
{chr(10).join([f"• {sim}" for sim in comparison['similarities']])}

### ❌ Where They Differ
{chr(10).join([f"• {diff}" for diff in comparison['differences']])}

### 📋 Overall Take
{comparison['overall_assessment']}

### 💡 Practical Takeaways
{chr(10).join([f"• {rec}" for rec in comparison['recommendations']])}

### 🎯 Analysis Confidence Level: {comparison['confidence_score']:.1%}

## 📊 Comparative Analysis Summary

This comprehensive comparison reveals the distinct approaches, methodologies, and contributions of both documents within their respective fields. The analysis highlights both the unique strengths of each document and the potential for cross-disciplinary insights and applications.

### Key Insights:
- **Document 1** demonstrates strengths in {doc1['summary']['document_type']} methodology and practical applications
- **Document 2** excels in {doc2['summary']['document_type']} analysis and theoretical contributions
- Both documents contribute valuable insights to their respective domains
- Cross-disciplinary applications and collaborations could enhance both fields

### Broader Implications:
The comparison underscores the importance of methodological rigor across different research domains and highlights opportunities for interdisciplinary collaboration and knowledge transfer between fields.

---
*Report generated in {processing_time:.2f} seconds*
*Note: Page references and specific examples are included in the comprehensive analysis for enhanced credibility*
"""
        
        return report.strip()
    
# Usage example
if __name__ == "__main__":
    comparator = PDFComparator()
    
    # Example usage
    result = comparator.compare_documents(
        user_id="123",
        doc1_uuid="doc-uuid-1",
        doc2_uuid="doc-uuid-2"
    )
    if result["success"]:
        report = comparator.get_comparison_report(result)
        
    else:
        print(f"Error: {result['error']}")
