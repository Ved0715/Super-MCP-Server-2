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
            
            # Generate summary using LLM
            summary_prompt = f"""
Analyze the following document content and provide a structured summary in JSON format.

Document Content:
{text_chunks[0][:self.max_summary_tokens]}

IMPORTANT: Respond ONLY with valid JSON. Do not include any markdown formatting, explanations, or additional text.

Required JSON structure:
{{
    "title": "Document title or main topic",
    "main_topics": ["topic1", "topic2", "topic3"],
    "key_findings": ["finding1", "finding2", "finding3"],
    "methodology": "Brief description of methods used",
    "conclusions": "Main conclusions or outcomes",
    "document_type": "research_paper|technical_report|review|other",
    "word_count": estimated_word_count,
    "page_count": estimated_page_count
}}

Focus on extracting the most important information and provide accurate, concise summaries.
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
            comparison_prompt = f"""
Compare these two documents and provide a comprehensive analysis in JSON format.

Document 1 Summary:
- Title: {summary1.title}
- Main Topics: {', '.join(summary1.main_topics)}
- Key Findings: {', '.join(summary1.key_findings)}
- Methodology: {summary1.methodology}
- Conclusions: {summary1.conclusions}
- Type: {summary1.document_type}

Document 2 Summary:
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
        """Generate a comprehensive, detailed report using LLM"""
        doc1 = comparison_result["document1"]
        doc2 = comparison_result["document2"]
        comparison = comparison_result["comparison"]
        metadata = comparison_result.get("metadata", {})
        
        # Create comprehensive prompt for final report generation
        report_prompt = f"""
You are an expert research analyst tasked with creating a comprehensive, high-quality document comparison report. 

## DOCUMENT INFORMATION

### Document 1: {doc1['summary']['title']}
- **Document Type:** {doc1['summary']['document_type']}
- **Main Topics:** {', '.join(doc1['summary']['main_topics'])}
- **Key Findings:** {', '.join(doc1['summary']['key_findings'])}
- **Methodology:** {doc1['summary']['methodology']}
- **Conclusions:** {doc1['summary']['conclusions']}
- **Word Count:** {doc1['summary']['word_count']:,}
- **Pages:** {doc1['summary']['page_count']}

### Document 2: {doc2['summary']['title']}
- **Document Type:** {doc2['summary']['document_type']}
- **Main Topics:** {', '.join(doc2['summary']['main_topics'])}
- **Key Findings:** {', '.join(doc2['summary']['key_findings'])}
- **Methodology:** {doc2['summary']['methodology']}
- **Conclusions:** {doc2['summary']['conclusions']}
- **Word Count:** {doc2['summary']['word_count']:,}
- **Pages:** {doc2['summary']['page_count']}

## COMPARISON ANALYSIS

### Similarities:
{chr(10).join([f"- {sim}" for sim in comparison['similarities']])}

### Differences:
{chr(10).join([f"- {diff}" for diff in comparison['differences']])}

### Overall Assessment:
{comparison['overall_assessment']}



## INSTRUCTIONS FOR REPORT GENERATION

Create a comprehensive, professional document comparison report that includes:

1. **Executive Summary** - High-level overview of both documents and their relationship
2. **Detailed Document Analysis** - In-depth analysis of each document's content, methodology, and findings
3. **Comparative Analysis** - Detailed comparison of similarities, differences, and relationships
4. **Critical Assessment** - Evaluation of strengths, weaknesses, and contributions of each document
5. **Synthesis and Insights** - Integration of findings and broader implications
6. **Recommendations** - Actionable insights and next steps
7. **Technical Appendix** - Methodology and confidence assessment

**REQUIREMENTS:**
- Use professional academic writing style
- Be comprehensive and detailed in analysis
- Provide specific examples and evidence from the documents
- Include critical insights and implications
- Structure the report logically with clear sections
- Use markdown formatting for readability
- Aim for 1000-1500 words total
- Be objective and analytical in tone
- Include page references where relevant
- Provide actionable recommendations

Generate a comprehensive, high-quality report that would be suitable for academic or professional use.
"""
        
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert research analyst specializing in document comparison and analysis. You create comprehensive, professional reports that provide deep insights and actionable recommendations."},
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
        """Generate a simple fallback report"""
        doc1 = comparison_result["document1"]
        doc2 = comparison_result["document2"]
        comparison = comparison_result["comparison"]
        processing_time = comparison_result.get("processing_time", 0)
        
        report = f"""
# 📊 Document Comparison Report

## 📄 Document Overview

### Document 1: {doc1['summary']['title']}
- **Type:** {doc1['summary']['document_type']}
- **Main Topics:** {', '.join(doc1['summary']['main_topics'])}
- **Key Findings:** {', '.join(doc1['summary']['key_findings'])}
- **Methodology:** {doc1['summary']['methodology']}
- **Word Count:** {doc1['summary']['word_count']:,}
- **Pages:** {doc1['summary']['page_count']}

### Document 2: {doc2['summary']['title']}
- **Type:** {doc2['summary']['document_type']}
- **Main Topics:** {', '.join(doc2['summary']['main_topics'])}
- **Key Findings:** {', '.join(doc2['summary']['key_findings'])}
- **Methodology:** {doc2['summary']['methodology']}
- **Word Count:** {doc2['summary']['word_count']:,}
- **Pages:** {doc2['summary']['page_count']}

## 🔍 Comparison Analysis

### ✅ Similarities
{chr(10).join([f"- {sim}" for sim in comparison['similarities']])}

### ❌ Differences
{chr(10).join([f"- {diff}" for diff in comparison['differences']])}

### 📋 Overall Assessment
{comparison['overall_assessment']}

### 💡 Recommendations
{chr(10).join([f"- {rec}" for rec in comparison['recommendations']])}

### 🎯 Confidence Score: {comparison['confidence_score']:.1%}

---
*Report generated in {processing_time:.2f} seconds*
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
        print(report)
    else:
        print(f"Error: {result['error']}")
