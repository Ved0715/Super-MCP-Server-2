import os
import re
import asyncio
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
import json
import time
from tqdm import tqdm

# Core imports
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import CrossEncoder
import tiktoken

from config import config

@dataclass
class CoTStep:
    """Chain of Thought step tracking"""
    step_name: str
    reasoning: str
    action: str
    result: str
    confidence: float
    timestamp: float
    tokens_used: int = 0

@dataclass
class PaperSearchResult:
    """Research paper search result with academic metadata"""
    chunk_id: str
    content: str
    meta: Dict
    relevance_score: float
    page_number: Optional[int] = None
    section_type: Optional[str] = None
    paper_title: Optional[str] = None
    authors: Optional[List[str]] = None
    confidence: float = 0.0
    reasoning: Optional[str] = None  # CoT reasoning for this result

@dataclass
class ResearchQuery:
    """Structured research query with academic focus"""
    query: str
    query_type: str
    user_id: str
    document_uuid: str
    max_results: int = 10
    similarity_threshold: float = 0.7
    focus_sections: Optional[List[str]] = None
    enable_cot: bool = True  # Enable Chain of Thought

class PaperRetrieverWithCoT:
    """
    Advanced Research Paper Retrieval System with Chain of Thought
    Specialized for academic document analysis with step-by-step reasoning
    """
    
    GENERAL_PDF_ANALYSIS_PROMPT = """
You are an expert document analyst specializing in extracting accurate and structured information from general-purpose PDF documents. 
IMPORTANT: Use ALL information explicitly provided in the document context. If the document contains relevant information that answers the query, provide a comprehensive response using that information.
Only respond with "I could not find this information in the document" if NO relevant information exists in the provided context.

**DOCUMENT ANALYSIS GUIDELINES**
1. PAGE REFERENCES: Always include page numbers when referencing specific content, using the format [Page X].
2. STRUCTURE & CONTENT SECTIONS: Break down content into logical sections
3. INFORMATION EXTRACTION: Extract and summarize key information
4. VERIFIABLE ANSWERS ONLY: Never guess, always tie answers to content with clear references
5. INSIGHT EXTRACTION: Highlight document purpose, gaps, and actionable insights
6. REPRODUCIBILITY: Rephrase processes step-by-step for clarity

Use clear formatting, logical breakdowns, and structured summaries.
"""
    
    def __init__(self):
        """Initialize the research paper retriever with CoT capabilities"""
        print("📚 Initializing Research Paper Retrieval System with Chain of Thought...")
        
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Research-specific configuration
        self.index_name = "all-pdfs-index"
        self.embedding_model = config.embedding_model
        self.response_model = config.response_model
        self.cost_model = "gpt-4o" # For CoT reasoning steps
        self.embedding_dimension = config.embedding_dimension
        
        # CoT tracking
        self.cot_steps: List[CoTStep] = []
        self.total_tokens_used = 0
        self.max_reasoning_tokens = 1000
        
        # Academic analysis settings
        self.cross_encoder = None
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        self.setup_models()
        self.setup_pinecone()
        
        print("✅ Research Paper Retrieval System with CoT initialized successfully")
    
    def _get_research_system_prompt(self) -> str:
        """Get the specialized research paper analysis system prompt"""
        return (
            "You are an expert document analyst specializing in academic document analysis. "
            "Use ALL information explicitly provided in the document context. "
            "If the document contains relevant information, provide a comprehensive response. "
            "Only say you cannot find information if truly no relevant content exists. "
            "Always include page references [Page X] and provide step-by-step reasoning."
        )
    
    def setup_models(self):
        """Initialize cross-encoder for research paper reranking"""
        try:
            print("🔧 Loading cross-encoder for research paper reranking...")
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Cross-encoder loaded successfully")
        except Exception as e:
            print(f"⚠️ Cross-encoder loading failed: {e}")
    
    def setup_pinecone(self):
        """Setup connection to research papers Pinecone index"""
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Connected to research papers index: {self.index_name}")
        except Exception as e:
            print(f"❌ Pinecone setup failed: {e}")
            self.index = None
    
    async def add_cot_step(self, step_name: str, reasoning: str, action: str, result: str, confidence: float, tokens_used: int = 0):
        """Add a Chain of Thought step"""
        step = CoTStep(
            step_name=step_name,
            reasoning=reasoning,
            action=action,
            result=result,
            confidence=confidence,
            timestamp=time.time(),
            tokens_used=tokens_used
        )
        self.cot_steps.append(step)
        print(f"🧠 CoT Step: {step_name} - {result[:100]}...")
    
    def reset_cot(self):
        """Reset Chain of Thought tracking for new query"""
        self.cot_steps = []
        self.total_tokens_used = 0
    
    async def comprehensive_thinking_and_analysis_cot(self, query: str, query_type: str) -> Dict[str, Any]:
        """
        PHASE 1-3: Enhanced thinking phase with question generation and query analysis
        Combines thinking, question generation, and analysis in single API call
        """
        reasoning = f"I need to think step by step about this document query: '{query}'"
        
        comprehensive_prompt = f"""
        **COMPREHENSIVE CHAIN OF THOUGHT ANALYSIS FOR DOCUMENT RETRIEVAL**
        
        Query: "{query}"
        Query Type: "{query_type}"
        
        **PHASE 1: THINKING STEP BY STEP**
        Think about this document analysis query:
        - What is the user really asking for from the document?
        - What type of information should I look for in the document?
        - What sections or pages might contain relevant information?
        
        **PHASE 2: QUESTION GENERATION**
        Generate 4-6 specific sub-questions about the document that would help answer this query comprehensively.
        
        **PHASE 3: RETRIEVAL STRATEGY**
        Based on the query, what's the best approach to find relevant information in the document?
        
        **RESPOND IN VALID JSON ONLY:**
        {{
            "thinking_phase": {{
                "initial_thoughts": ["thought 1", "thought 2", "thought 3"],
                "document_focus": ["what to look for", "where to look", "how to analyze"],
                "reasoning_approach": "description of analysis approach"
            }},
            "question_generation": {{
                "sub_questions": [
                    "What is the core information being requested?",
                    "What specific details should I extract?",
                    "What context is needed for complete understanding?",
                    "How should I structure the response?"
                ],
                "question_rationale": "why these questions ensure comprehensive document analysis"
            }},
            "retrieval_strategy": {{
                "search_approach": "semantic_search",
                "key_terms": ["optimized", "search", "terms"],
                "expected_sections": ["methodology", "results", "discussion"],
                "similarity_threshold": 0.7,
                "max_results": 10
            }},
            "confidence": 0.85
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.cost_model,
                messages=[
                    {"role": "system", "content": "You are a JSON response generator for document analysis planning. Always respond with valid JSON only."},
                    {"role": "user", "content": comprehensive_prompt}
                ],
                max_tokens=self.max_reasoning_tokens,
                temperature=0.2
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0]
            elif "```" in response_content:
                response_content = response_content.split("```")[1].split("```")[0]
            
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                response_content = response_content[start_idx:end_idx]
            
            result = json.loads(response_content)
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            await self.add_cot_step(
                step_name="Phase 1-3: Thinking, Questions & Strategy",
                reasoning=reasoning,
                action="Completed comprehensive analysis planning",
                result=f"Strategy: {result['retrieval_strategy']['search_approach']}, Generated {len(result['question_generation']['sub_questions'])} sub-questions",
                confidence=result.get('confidence', 0.8),
                tokens_used=tokens_used
            )
            
            return result
            
        except Exception as e:
            print(f"❌ CoT Analysis failed: {e}")
            # Enhanced fallback
            fallback_result = {
                "thinking_phase": {
                    "initial_thoughts": [
                        "User wants information from a specific document",
                        "Need to search document content semantically",
                        "Should provide comprehensive answer with page references"
                    ],
                    "document_focus": [
                        "Extract relevant text chunks",
                        "Look across all document sections", 
                        "Provide structured analysis"
                    ],
                    "reasoning_approach": "Semantic search with comprehensive analysis"
                },
                "question_generation": {
                    "sub_questions": [
                        "What specific information does the query request?",
                        "What document sections are most relevant?",
                        "What level of detail is appropriate?",
                        "How should I structure the comprehensive response?"
                    ],
                    "question_rationale": "These questions ensure thorough document analysis"
                },
                "retrieval_strategy": {
                    "search_approach": "semantic_search",
                    "key_terms": query.lower().split()[:5],
                    "expected_sections": ["all"],
                    "similarity_threshold": 0.7,
                    "max_results": 10
                },
                "confidence": 0.6
            }
            
            await self.add_cot_step(
                step_name="CoT Analysis (Fallback)",
                reasoning="API failed, using rule-based fallback",
                action="Applied comprehensive fallback analysis",
                result="Generated fallback strategy and questions",
                confidence=0.6
            )
            
            return fallback_result
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of research query for specialized handling"""
        query_lower = query.lower()
        
        # Methodology queries
        if any(term in query_lower for term in ['method', 'algorithm', 'approach', 'technique', 'procedure', 'experimental setup']):
            return 'methodology'
        elif any(term in query_lower for term in ['results', 'findings', 'performance', 'accuracy', 'evaluation', 'metrics']):
            return 'results'
        elif any(term in query_lower for term in ['p-value', 'statistical', 'significance', 'confidence interval', 'correlation']):
            return 'statistical'
        elif any(term in query_lower for term in ['cited', 'references', 'bibliography', 'related work', 'previous studies']):
            return 'citations'
        elif any(term in query_lower for term in ['discussion', 'analysis', 'interpretation', 'implications', 'limitations']):
            return 'discussion'
        elif any(term in query_lower for term in ['conclusion', 'summary', 'future work', 'contributions', 'key findings']):
            return 'conclusion'
        else:
            return 'general'
    
    def _build_namespace(self, user_id: str, document_uuid: str) -> str:
        """Build the namespace for user-specific document search"""
        return f"user_{user_id}_doc_{document_uuid}"
    
    async def search_research_paper_cot(self, research_query: ResearchQuery, cot_analysis: Dict) -> List[PaperSearchResult]:
        """
        PHASE 4: Advanced research paper search with CoT guidance
        """
        namespace = self._build_namespace(research_query.user_id, research_query.document_uuid)
        reasoning = f"Executing semantic search in namespace: {namespace}"
        
        print(f"🔍 CoT-Guided Search in namespace: {namespace}")
        print(f"📝 Query: '{research_query.query}'")
        print(f"🎯 Query Type: {research_query.query_type}")
        
        if not self.index:
            await self.add_cot_step(
                step_name="Phase 4: Document Search (Failed)",
                reasoning="No Pinecone index available",
                action="Search aborted",
                result="No search performed - missing index",
                confidence=0.0
            )
            return []
        
        try:
            # Use CoT analysis to optimize search
            retrieval_strategy = cot_analysis.get('retrieval_strategy', {})
            search_terms = retrieval_strategy.get('key_terms', [research_query.query])
            
            # Generate embedding for the research query
            embedding_response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=research_query.query,
                model=self.embedding_model,
                dimensions=self.embedding_dimension
            )
            
            query_embedding = embedding_response.data.embedding
            
            # Search in the specific user document namespace
            search_results = self.index.query(
                vector=query_embedding,
                top_k=research_query.max_results,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            
            print(f"✅ Found {len(search_results.matches)} results from document")
            
            # Convert to PaperSearchResult objects with CoT reasoning
            paper_results = []
            for i, match in enumerate(search_results.matches):
                if match.score >= research_query.similarity_threshold:
                    # Extract text content
                    content = match.metadata.get('text', '')
                    if not content and '_node_content' in match.meta
                        try:
                            node_content = json.loads(match.metadata['_node_content'])
                            content = node_content.get('text', '')
                        except (json.JSONDecodeError, TypeError):
                            content = str(match.metadata.get('_node_content', ''))[:500]
                    
                    # Add CoT reasoning for this result
                    cot_reasoning = f"Ranked #{i+1} with similarity {match.score:.3f} - {'High' if match.score > 0.8 else 'Medium' if match.score > 0.7 else 'Low'} relevance"
                    
                    result = PaperSearchResult(
                        chunk_id=match.id,
                        content=content,
                        metadata=match.metadata,
                        relevance_score=match.score,
                        page_number=match.metadata.get('page_number'),
                        section_type=match.metadata.get('section_type'),
                        paper_title=match.metadata.get('title'),
                        authors=match.metadata.get('authors', []),
                        confidence=min(match.score * 1.1, 1.0),
                        reasoning=cot_reasoning
                    )
                    paper_results.append(result)
            
            # Rerank results if cross-encoder is available
            if self.cross_encoder and len(paper_results) > 1:
                paper_results = await self._rerank_results_cot(research_query.query, paper_results)
            
            await self.add_cot_step(
                step_name="Phase 4: Document Search Execution",
                reasoning=reasoning,
                action=f"Performed semantic search with {len(search_terms)} key terms",
                result=f"Retrieved {len(paper_results)} relevant chunks above threshold {research_query.similarity_threshold}",
                confidence=0.85 if paper_results else 0.3
            )
            
            return paper_results
            
        except Exception as e:
            print(f"❌ CoT-guided search failed: {e}")
            await self.add_cot_step(
                step_name="Phase 4: Document Search (Error)",
                reasoning="Search execution failed",
                action=f"Error during search: {str(e)[:100]}",
                result="No results due to error",
                confidence=0.0
            )
            return []
    
    async def _rerank_results_cot(self, query: str, results: List[PaperSearchResult]) -> List[PaperSearchResult]:
        """Rerank results using cross-encoder with CoT tracking"""
        reasoning = "Applying cross-encoder reranking for better relevance"
        
        try:
            # Prepare query-result pairs for reranking
            pairs = [(query, result.content[:512]) for result in results]
            
            # Get reranking scores
            rerank_scores = await asyncio.to_thread(
                self.cross_encoder.predict,
                pairs
            )
            
            # Update results with rerank scores and enhanced reasoning
            for i, result in enumerate(results):
                original_score = result.relevance_score
                result.relevance_score = float(rerank_scores[i])
                result.confidence = min(result.relevance_score * 1.1, 1.0)
                result.reasoning += f" -> Reranked: {original_score:.3f} → {result.relevance_score:.3f}"
            
            # Sort by rerank score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            await self.add_cot_step(
                step_name="Phase 4.1: Results Reranking",
                reasoning=reasoning,
                action="Applied cross-encoder reranking",
                result=f"Reranked {len(results)} results, top score: {results.relevance_score:.3f}",
                confidence=0.9
            )
            
        except Exception as e:
            print(f"⚠️ CoT Reranking failed: {e}")
            await self.add_cot_step(
                step_name="Phase 4.1: Reranking (Failed)",
                reasoning="Cross-encoder reranking failed",
                action=f"Reranking error: {str(e)[:50]}",
                result="Using original semantic similarity scores",
                confidence=0.7
            )
        
        return results
    
    async def analyze_research_paper_with_cot(self, research_query: ResearchQuery) -> Dict[str, Any]:
        """
        MAIN METHOD: Comprehensive research paper analysis with full Chain of Thought
        """
        print(f"🧠 Starting CoT-Enhanced Document Analysis...")
        start_time = time.time()
        
        # Reset CoT tracking for new query
        self.reset_cot()
        
        try:
            # PHASE 1-3: Comprehensive thinking and analysis
            cot_analysis = await self.comprehensive_thinking_and_analysis_cot(
                research_query.query, 
                research_query.query_type
            )
            
            # PHASE 4: Execute CoT-guided search
            search_results = await self.search_research_paper_cot(research_query, cot_analysis)
            
            if not search_results:
                await self.add_cot_step(
                    step_name="Analysis Termination",
                    reasoning="No relevant content found to analyze",
                    action="Terminating analysis due to no results",
                    result="Analysis incomplete - no relevant document content",
                    confidence=0.0
                )
                
                return {
                    "success": False,
                    "query": research_query.query,
                    "error": "No relevant content found in the document",
                    "namespace": self._build_namespace(research_query.user_id, research_query.document_uuid),
                    "cot_steps": [step.__dict__ for step in self.cot_steps],
                    "total_tokens": self.total_tokens_used
                }
            
            # PHASE 5: Generate comprehensive analysis
            analysis_response = await self._generate_cot_analysis(
                research_query, search_results, cot_analysis
            )
            
            execution_time = time.time() - start_time
            
            # Final CoT step
            await self.add_cot_step(
                step_name="Phase 5: Analysis Completion",
                reasoning="Completed comprehensive document analysis",
                action="Generated final response with CoT insights",
                result=f"Analysis complete in {execution_time:.2f}s with {len(search_results)} sources",
                confidence=0.9
            )
            
            return {
                "success": True,
                "query": research_query.query,
                "query_type": research_query.query_type,
                "response": analysis_response,
                "metadata": {
                    "execution_time": execution_time,
                    "results_count": len(search_results),
                    "namespace": self._build_namespace(research_query.user_id, research_query.document_uuid),
                    "paper_title": search_results.paper_title if search_results else None,
                    "authors": search_results.authors if search_results else None,
                    "pages_referenced": list(set([r.page_number for r in search_results if r.page_number])),
                    "sections_analyzed": list(set([r.section_type for r in search_results if r.section_type])),
                    "cot_enabled": research_query.enable_cot,
                    "total_tokens_used": self.total_tokens_used
                },
                "search_results": [
                    {
                        "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                        "page": result.page_number,
                        "section": result.section_type,
                        "relevance": result.relevance_score,
                        "reasoning": result.reasoning
                    }
                    for result in search_results[:5]
                ],
                "cot_steps": [step.__dict__ for step in self.cot_steps],
                "cot_summary": {
                    "total_steps": len(self.cot_steps),
                    "avg_confidence": sum(step.confidence for step in self.cot_steps) / len(self.cot_steps) if self.cot_steps else 0,
                    "reasoning_path": [step.step_name for step in self.cot_steps]
                }
            }
            
        except Exception as e:
            print(f"❌ CoT Analysis failed: {e}")
            await self.add_cot_step(
                step_name="Critical Error",
                reasoning="Analysis pipeline encountered critical error",
                action=f"Error handling: {str(e)[:100]}",
                result="Analysis failed with error",
                confidence=0.0
            )
            
            return {
                "success": False,
                "query": research_query.query,
                "error": f"Analysis failed: {str(e)}",
                "cot_steps": [step.__dict__ for step in self.cot_steps],
                "total_tokens": self.total_tokens_used
            }
    
    async def _generate_cot_analysis(self, research_query: ResearchQuery, results: List[PaperSearchResult], cot_analysis: Dict) -> str:
        """
        PHASE 5: Generate comprehensive analysis with CoT insights
        """
        reasoning = "Generating comprehensive analysis using CoT insights and retrieved content"
        
        # Prepare enhanced context with CoT reasoning
        context_parts = []
        for i, result in enumerate(results[:8]):  # Top 8 results
            page_ref = f"[Page {result.page_number}]" if result.page_number else "[Source]"
            context_parts.append(f"""
{page_ref}
{result.content}
CoT Reasoning: {result.reasoning}
""")
        
        combined_context = "\n---\n".join(context_parts)
        
        # Use CoT insights to enhance the prompt
        sub_questions = cot_analysis.get('question_generation', {}).get('sub_questions', [])
        thinking_approach = cot_analysis.get('thinking_phase', {}).get('reasoning_approach', '')
        
        # Choose system prompt
        system_prompt = (self.GENERAL_PDF_ANALYSIS_PROMPT 
                        if research_query.query_type == 'general' 
                        else self._get_research_system_prompt())
        
        analysis_prompt = f"""
**CoT-ENHANCED DOCUMENT ANALYSIS**

Query: "{research_query.query}"
Query Type: {research_query.query_type}

**CHAIN OF THOUGHT GUIDANCE:**
Reasoning Approach: {thinking_approach}

Sub-questions to address:
{chr(10).join([f"- {q}" for q in sub_questions])}

**DOCUMENT CONTENT:**
{combined_context}

**INSTRUCTIONS:**
- Use the CoT guidance above to structure your comprehensive response
- Address each sub-question systematically
- Include page references for all content cited
- Provide step-by-step reasoning where appropriate
- Structure response: Overview → Detailed Analysis → Conclusions
- Use point-wise descriptions when beneficial for clarity

**COMPREHENSIVE ANALYSIS:**
"""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.response_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2500,  # More tokens for comprehensive CoT response
                temperature=0.1   # Low temperature for consistency
            )
            
            analysis = response.choices.message.content.strip()
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            await self.add_cot_step(
                step_name="Phase 5: Response Generation",
                reasoning=reasoning,
                action="Generated comprehensive analysis using CoT guidance",
                result=f"Created {len(analysis)} character response addressing {len(sub_questions)} sub-questions",
                confidence=0.9,
                tokens_used=tokens_used
            )
            
            return analysis
            
        except Exception as e:
            print(f"❌ CoT Analysis generation failed: {e}")
            await self.add_cot_step(
                step_name="Phase 5: Response Generation (Failed)",
                reasoning="Failed to generate comprehensive analysis",
                action=f"Error: {str(e)[:100]}",
                result="Providing fallback response",
                confidence=0.3
            )
            
            # Fallback response with available content
            if results:
                top_result = results
                page_ref = f"[Page {top_result.page_number}]" if top_result.page_number else "[Source]"
                return f"Based on the document analysis, here is the most relevant content found:\n\n{page_ref}\n{top_result.content}\n\nCoT Reasoning: {top_result.reasoning}"
            else:
                return "I apologize, but I encountered an error while generating the comprehensive analysis and no relevant content was found in the document."

# Usage example
async def example_cot_usage():
    """Example usage of the CoT-enhanced PaperRetriever"""
    retriever = PaperRetrieverWithCoT()
    
    # Create research query with CoT enabled
    query = ResearchQuery(
        query="What is PGP and how does it work?",
        query_type="general",
        user_id="5",
        document_uuid="7346b737-9b41-4d9a-a652-4c7b2757bb06",
        max_results=10,
        similarity_threshold=0.7,
        enable_cot=True
    )
    
    # Perform CoT-enhanced analysis
    result = await retriever.analyze_research_paper_with_cot(query)
    
    print("🧠 CoT-Enhanced Analysis Result:")
    print(f"Success: {result['success']}")
    print(f"Response: {result.get('response', 'No response')[:500]}...")
    print(f"CoT Steps: {result.get('cot_summary', {}).get('total_steps', 0)}")
    print(f"Average Confidence: {result.get('cot_summary', {}).get('avg_confidence', 0):.2f}")
    
    return result
