import os
import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import pinecone
from openai import OpenAI

# Configure logging - only show essential information
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QAResponse:
    """Structured Q&A response with comprehensive information"""
    answer: str
    confidence_score: float
    sources: List[Dict[str, Any]]
    supporting_evidence: List[str]
    document_breakdown: Dict[str, str]
    question_type: str
    processing_time: float

class DocumentQA:
    """
    Advanced Q&A system for document comparison using single API call approach
    Leverages existing infrastructure from paper_compare.py for optimal quality
    """
    
    def __init__(self):
        """Initialize the Q&A system with Pinecone and OpenAI"""
        self.pc = pinecone.Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
        self.index = self.pc.Index(os.getenv('PINECONE_INDEX_NAME_TEST'))
        self.openai = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.model = "gpt-4o-mini"  # Use gpt-4o for complex questions
        
        logger.info("Document Q&A system initialized successfully")
    
    def ask_question(self, user_id: str, doc1_uuid: str, doc2_uuid: str, question: str) -> Dict[str, Any]:
        """
        Ask a question about two documents using single API call approach
        
        Args:
            user_id: User identifier
            doc1_uuid: First document UUID
            doc2_uuid: Second document UUID
            question: User's question
            
        Returns:
            Dictionary containing comprehensive Q&A response
        """
        import time
        start_time = time.time()
        
        try:
            # Create namespace names
            namespace1 = f"user_{user_id}_doc_{doc1_uuid}"
            namespace2 = f"user_{user_id}_doc_{doc2_uuid}"
            
            # Load document chunks
            doc1_chunks = self._load_chunks(namespace1)
            doc2_chunks = self._load_chunks(namespace2)
            
            if not doc1_chunks or not doc2_chunks:
                return {
                    "error": "One or both documents appear to be empty or unprocessed",
                    "success": False
                }
            
            # Analyze question type for better prompting
            question_type = self._analyze_question_type(question)
            
            # Generate comprehensive answer using single API call
            qa_response = self._generate_comprehensive_answer(
                doc1_chunks, doc2_chunks, question, question_type
            )
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "question": question,
                "question_type": question_type,
                "answer": qa_response.answer,
                "confidence_score": qa_response.confidence_score,
                "sources": qa_response.sources,
                "supporting_evidence": qa_response.supporting_evidence,
                "document_breakdown": qa_response.document_breakdown,
                "processing_time": processing_time,
                "doc1_chunks": len(doc1_chunks),
                "doc2_chunks": len(doc2_chunks)
            }
            
        except Exception as e:
            logger.error(f"Error during Q&A processing: {str(e)}")
            return {
                "error": f"Q&A failed: {str(e)}",
                "success": False
            }
    
    def _load_chunks(self, namespace: str) -> List[Dict[str, Any]]:
        """Load document chunks from Pinecone namespace"""
        try:
            # Query all chunks in the namespace
            results = self.index.query(
                vector=[0] * 3072,  # Dummy vector for metadata-only query
                namespace=namespace,
                top_k=1000,
                include_metadata=True
            )
            
            if not results.matches:
                return []
            
            chunks = []
            for match in results.matches:
                content = self._extract_content_from_metadata(match.metadata)
                if content:
                    chunks.append({
                        "content": content,
                        "page": match.metadata.get('page_number', 0),
                        "metadata": match.metadata
                    })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error loading chunks from namespace {namespace}: {str(e)}")
            return []
    
    def _extract_content_from_metadata(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract content from various metadata fields"""
        content_fields = ['content', 'text', '_node_content']
        
        for field in content_fields:
            if field in metadata:
                content = metadata[field]
                if field == '_node_content':
                    try:
                        if isinstance(content, str):
                            parsed = json.loads(content)
                            return parsed.get('text', '')
                        return content
                    except json.JSONDecodeError:
                        continue
                return content
        
        return None
    
    def _analyze_question_type(self, question: str) -> str:
        """Analyze question type for better prompting"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['compare', 'difference', 'similar', 'versus', 'vs']):
            return "comparative"
        elif any(word in question_lower for word in ['how', 'method', 'approach', 'process']):
            return "methodological"
        elif any(word in question_lower for word in ['what', 'define', 'explain', 'describe']):
            return "factual"
        elif any(word in question_lower for word in ['why', 'reason', 'cause', 'impact']):
            return "analytical"
        else:
            return "general"
    
    def _generate_comprehensive_answer(self, doc1_chunks: List[Dict[str, Any]], 
                                     doc2_chunks: List[Dict[str, Any]], 
                                     question: str, question_type: str) -> QAResponse:
        """Generate comprehensive answer using single API call"""
        try:
            # Prepare document content
            doc1_content = self._prepare_document_content(doc1_chunks, "Document 1")
            doc2_content = self._prepare_document_content(doc2_chunks, "Document 2")
            
            # Create comprehensive prompt based on question type
            prompt = self._create_comprehensive_prompt(doc1_content, doc2_content, question, question_type)
            
            # Make single API call for comprehensive answer
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert Q&A system specializing in document analysis and comparison. You provide comprehensive, evidence-based answers that thoroughly utilize both documents. Write in a natural, flowing style without section headers. Include specific quotes, page references, and clear source attribution within the narrative flow."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=3000,
                temperature=0.3
            )
            
            # Parse the response
            answer_content = response.choices[0].message.content.strip()
            
            # Extract structured information from the response
            qa_response = self._parse_qa_response(answer_content, question_type)
            
            return qa_response
            
        except Exception as e:
            logger.error(f"Error generating comprehensive answer: {str(e)}")
            return QAResponse(
                answer=f"Error generating answer: {str(e)}",
                confidence_score=0.0,
                sources=[],
                supporting_evidence=[],
                document_breakdown={},
                question_type=question_type,
                processing_time=0.0
            )
    
    def _prepare_document_content(self, chunks: List[Dict[str, Any]], doc_name: str) -> str:
        """Prepare document content for the prompt"""
        if not chunks:
            return f"{doc_name}: No content available"
        
        # Combine chunks with page information
        content_parts = []
        for chunk in chunks:
            page_info = f"[Page {chunk['page']}]" if chunk['page'] else ""
            content_parts.append(f"{page_info} {chunk['content']}")
        
        return f"{doc_name}:\n" + "\n\n".join(content_parts)
    
    def _create_comprehensive_prompt(self, doc1_content: str, doc2_content: str, 
                                   question: str, question_type: str) -> str:
        """Create comprehensive prompt for single API call"""
        
        # Base prompt template
        base_prompt = f"""
You are analyzing two documents to answer a user's question. Provide a comprehensive, evidence-based answer that thoroughly utilizes BOTH documents.

DOCUMENT CONTENT:

{doc1_content}

{doc2_content}

USER QUESTION: {question}

QUESTION TYPE: {question_type}

INSTRUCTIONS:
1. **Answer the question comprehensively** using BOTH documents
2. **Include specific quotes** with page references when possible
3. **Provide evidence-based insights** from both documents
4. **Show similarities and differences** when relevant to the question
5. **Structure your response clearly** with logical sections
6. **Attribute information** to specific documents clearly
7. **Provide actionable insights** and key takeaways

RESPONSE GUIDELINES:
- Write in a natural, flowing narrative style
- Provide a comprehensive answer that directly addresses the question
- Include specific quotes and examples from both documents
- Show similarities/differences when relevant
- Provide key insights and implications
- Attribute information clearly to specific documents
- Do not use section headers or bullet points in the response

QUALITY REQUIREMENTS:
- Use BOTH documents thoroughly and equally
- Include specific page references when possible
- Provide concrete examples and quotes
- Ensure balanced coverage of both documents
- Maintain professional, academic tone
- Be comprehensive yet concise
"""

        # Add question-type specific instructions
        if question_type == "comparative":
            base_prompt += """

COMPARATIVE ANALYSIS FOCUS:
- Highlight key differences between the documents
- Identify areas of similarity or overlap
- Provide balanced comparison of approaches
- Include specific examples from both documents
"""
        elif question_type == "methodological":
            base_prompt += """

METHODOLOGICAL ANALYSIS FOCUS:
- Explain the approaches used in each document
- Compare methodologies and techniques
- Highlight strengths and limitations
- Provide practical insights about methods
"""
        elif question_type == "analytical":
            base_prompt += """

ANALYTICAL ANALYSIS FOCUS:
- Provide deep insights and reasoning
- Explain causes, effects, and implications
- Connect findings across both documents
- Offer critical analysis and evaluation
"""

        return base_prompt
    
    def _parse_qa_response(self, answer_content: str, question_type: str) -> QAResponse:
        """Parse the LLM response into structured QAResponse"""
        try:
            # For now, return the full answer as-is
            # In a more advanced implementation, you could parse structured sections
            
            # Estimate confidence based on answer quality indicators
            confidence_score = self._estimate_confidence(answer_content)
            
            # Extract sources and evidence (simplified approach)
            sources = self._extract_sources(answer_content)
            supporting_evidence = self._extract_evidence(answer_content)
            document_breakdown = self._analyze_document_usage(answer_content)
            
            return QAResponse(
                answer=answer_content,
                confidence_score=confidence_score,
                sources=sources,
                supporting_evidence=supporting_evidence,
                document_breakdown=document_breakdown,
                question_type=question_type,
                processing_time=0.0  # Will be set by caller
            )
            
        except Exception as e:
            logger.error(f"Error parsing QA response: {str(e)}")
            return QAResponse(
                answer=answer_content,
                confidence_score=0.5,
                sources=[],
                supporting_evidence=[],
                document_breakdown={},
                question_type=question_type,
                processing_time=0.0
            )
    
    def _estimate_confidence(self, answer_content: str) -> float:
        """Estimate confidence score based on answer quality indicators"""
        confidence = 0.7  # Base confidence
        
        # Positive indicators
        if "page" in answer_content.lower():
            confidence += 0.1
        if "document 1" in answer_content.lower() and "document 2" in answer_content.lower():
            confidence += 0.1
        if "quote" in answer_content.lower() or '"' in answer_content:
            confidence += 0.1
        if len(answer_content) > 500:  # Comprehensive answer
            confidence += 0.1
        
        # Negative indicators
        if "error" in answer_content.lower() or "failed" in answer_content.lower():
            confidence -= 0.3
        if len(answer_content) < 100:  # Very short answer
            confidence -= 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    def _extract_sources(self, answer_content: str) -> List[Dict[str, Any]]:
        """Extract source information from answer"""
        sources = []
        
        # Look for page references
        import re
        page_matches = re.findall(r'page (\d+)', answer_content, re.IGNORECASE)
        for page in page_matches:
            sources.append({"type": "page_reference", "value": page})
        
        # Look for document references
        if "document 1" in answer_content.lower():
            sources.append({"type": "document", "name": "Document 1"})
        if "document 2" in answer_content.lower():
            sources.append({"type": "document", "name": "Document 2"})
        
        return sources
    
    def _extract_evidence(self, answer_content: str) -> List[str]:
        """Extract supporting evidence from answer"""
        evidence = []
        
        # Look for quoted text
        import re
        quotes = re.findall(r'"([^"]*)"', answer_content)
        evidence.extend(quotes[:3])  # Limit to first 3 quotes
        
        return evidence
    
    def _analyze_document_usage(self, answer_content: str) -> Dict[str, str]:
        """Analyze how each document was used in the answer"""
        doc1_mentions = answer_content.lower().count("document 1")
        doc2_mentions = answer_content.lower().count("document 2")
        
        return {
            "Document 1": f"Referenced {doc1_mentions} times",
            "Document 2": f"Referenced {doc2_mentions} times"
        }
    
    def get_qa_report(self, result: Dict[str, Any], include_metadata: bool = False) -> str:
        """Format the Q&A result into a human-readable report"""
        if result.get("success"):
            qa_data = result
            
            # Base report - just the answer
            report = f"""{qa_data.get('answer', 'No answer generated')}"""
            
            # Add metadata if requested (for debugging/logging)
            if include_metadata:
                report += f"""

## 📊 Analysis Details

### 📝 Original Question
**{qa_data.get('question', 'N/A')}**

### 🎯 Confidence Score
**{qa_data.get('confidence_score', 0):.1%}**

### 📚 Sources Used
{self._format_sources(qa_data.get('sources', []))}

### 🔍 Supporting Evidence
{self._format_evidence(qa_data.get('supporting_evidence', []))}

### 📄 Document Usage
{self._format_document_breakdown(qa_data.get('document_breakdown', {}))}

### ⏱️ Processing Information
- **Processing Time:** {qa_data.get('processing_time', 0):.2f} seconds
- **Document 1 Chunks:** {qa_data.get('doc1_chunks', 0)}
- **Document 2 Chunks:** {qa_data.get('doc2_chunks', 0)}"""
            
            report += "\n\n---\n*Generated using comprehensive document analysis*"
            return report
        else:
            error_msg = result.get("error", "Unknown error")
            return f"# ❌ Q&A Failed\n\n**Error:** {error_msg}\n\nPlease check that both documents are properly processed and try again."
    
    def _format_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Format sources for display"""
        if not sources:
            return "No specific sources identified"
        
        formatted = []
        for source in sources:
            if source.get("type") == "page_reference":
                formatted.append(f"- Page {source.get('value')}")
            elif source.get("type") == "document":
                formatted.append(f"- {source.get('name')}")
        
        return "\n".join(formatted) if formatted else "No specific sources identified"
    
    def _format_evidence(self, evidence: List[str]) -> str:
        """Format evidence for display"""
        if not evidence:
            return "No specific quotes identified"
        
        formatted = []
        for i, quote in enumerate(evidence, 1):
            formatted.append(f"{i}. \"{quote}\"")
        
        return "\n".join(formatted)
    
    def _format_document_breakdown(self, breakdown: Dict[str, str]) -> str:
        """Format document breakdown for display"""
        if not breakdown:
            return "No document usage analysis available"
        
        formatted = []
        for doc, usage in breakdown.items():
            formatted.append(f"- **{doc}:** {usage}")
        
        return "\n".join(formatted)

if __name__ == "__main__":
    # Test the Q&A system
    qa_system = DocumentQA()
    result = qa_system.ask_question(
        user_id="44",
        doc1_uuid="test-doc-1",
        doc2_uuid="test-doc-2",
        question="What are the main differences between these two documents?"
    )
    print(qa_system.get_qa_report(result))
