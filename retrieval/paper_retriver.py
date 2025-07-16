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
class PaperSearchResult:
    """Research paper search result with academic metadata"""
    chunk_id: str
    content: str
    metadata: Dict
    relevance_score: float
    page_number: Optional[int] = None
    section_type: Optional[str] = None  # abstract, introduction, methodology, results, discussion, conclusion
    paper_title: Optional[str] = None
    authors: Optional[List[str]] = None
    confidence: float = 0.0

@dataclass
class ResearchQuery:
    """Structured research query with academic focus"""
    query: str
    query_type: str  # methodology, results, discussion, conclusion, general, citations, statistical
    user_id: str
    document_uuid: str
    max_results: int = 10
    similarity_threshold: float = 0.7
    focus_sections: Optional[List[str]] = None  # Filter by specific paper sections

class PaperRetriever:
    """
    Advanced Research Paper Retrieval System
    Specialized for academic document analysis with research-focused capabilities
    """
    
    def __init__(self):
        """Initialize the research paper retriever"""
        print("📚 Initializing Research Paper Retrieval System...")
        
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Research-specific configuration
        self.index_name = "all-pdfs-index"  # Research papers index
        self.embedding_model = config.embedding_model
        self.response_model = config.response_model
        self.embedding_dimension = config.embedding_dimension
        
        # Academic analysis settings
        self.cross_encoder = None
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Research paper specific prompts
        self.research_system_prompt = self._get_research_system_prompt()
        
        self.setup_models()
        self.setup_pinecone()
        
        print("✅ Research Paper Retrieval System initialized successfully")
    
    def _get_research_system_prompt(self) -> str:
        """Get the specialized research paper analysis system prompt"""
        return (
            "You are an expert research paper analyst specializing in academic document analysis. Only answer using information found in the provided research paper. "
            "If you cannot find the answer, say 'I could not find this information in the research paper.' Never make up facts or speculate. "
            "When answering, first explain your reasoning step by step, citing the paper sections or pages you used. Then provide your final answer. "
            "Your responses must follow these research-focused guidelines:\n\n"
            "1. **PAGE REFERENCES**: Always include page references in the format [Page X] for citations, methodology, results, and conclusions.\n"
            "2. **RESEARCH METHODOLOGY**: When discussing methods, extract exact algorithms, datasets, evaluation metrics, and experimental setups.\n"
            "3. **ACADEMIC FORMATTING**: Use proper academic formatting with clear sections for Abstract, Methods, Results, Discussion.\n"
            "4. **CITATION ANALYSIS**: Identify and extract in-text citations and reference patterns when relevant.\n"
            "5. **RESEARCH CONTRIBUTIONS**: Highlight novel contributions, research gaps addressed, and key findings.\n"
            "6. **STATISTICAL SIGNIFICANCE**: Extract p-values, confidence intervals, and statistical measures when present.\n"
            "7. **REPRODUCIBILITY**: Note details about code availability, data access, and experimental reproducibility.\n"
            "8. **COMPARATIVE ANALYSIS**: When comparing papers, focus on methodological differences and result variations.\n"
            "9. **FUTURE WORK**: Identify limitations and suggested future research directions.\n"
            "10. **TECHNICAL DEPTH**: Provide detailed technical explanations suitable for researchers in the field.\n"
            "NOTE: Focus on research quality indicators like methodology rigor, result significance, and contribution novelty."
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
    
    def _detect_query_type(self, query: str) -> str:
        """Detect the type of research query for specialized handling"""
        query_lower = query.lower()
        
        # Methodology queries
        if any(term in query_lower for term in ['method', 'algorithm', 'approach', 'technique', 'procedure', 'experimental setup']):
            return 'methodology'
        
        # Results queries
        elif any(term in query_lower for term in ['results', 'findings', 'performance', 'accuracy', 'evaluation', 'metrics']):
            return 'results'
        
        # Statistical queries
        elif any(term in query_lower for term in ['p-value', 'statistical', 'significance', 'confidence interval', 'correlation']):
            return 'statistical'
        
        # Citation queries
        elif any(term in query_lower for term in ['cited', 'references', 'bibliography', 'related work', 'previous studies']):
            return 'citations'
        
        # Discussion/Analysis queries
        elif any(term in query_lower for term in ['discussion', 'analysis', 'interpretation', 'implications', 'limitations']):
            return 'discussion'
        
        # Conclusion queries
        elif any(term in query_lower for term in ['conclusion', 'summary', 'future work', 'contributions', 'key findings']):
            return 'conclusion'
        
        else:
            return 'general'
    
    def _build_namespace(self, user_id: str, document_uuid: str) -> str:
        """Build the namespace for user-specific document search"""
        return f"user_{user_id}_doc_{document_uuid}"
    
    async def search_research_paper(self, research_query: ResearchQuery) -> List[PaperSearchResult]:
        """
        Advanced research paper search with academic focus
        """
        namespace = self._build_namespace(research_query.user_id, research_query.document_uuid)
        
        print(f"🔍 Searching research paper in namespace: {namespace}")
        print(f"📝 Query: '{research_query.query}'")
        print(f"🎯 Query Type: {research_query.query_type}")
        
        if not self.index:
            print("❌ No Pinecone index available")
            return []
        
        try:
            # Generate embedding for the research query
            embedding_response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=research_query.query,
                model=self.embedding_model,
                dimensions=self.embedding_dimension
            )
            
            query_embedding = embedding_response.data[0].embedding
            
            # Search in the specific user document namespace
            search_results = self.index.query(
                vector=query_embedding,
                top_k=research_query.max_results,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=self._build_search_filter(research_query)
            )
            
            print(f"✅ Found {len(search_results.matches)} results from research paper")
            
            # Convert to PaperSearchResult objects
            paper_results = []
            for match in search_results.matches:
                if match.score >= research_query.similarity_threshold:
                    # Extract text content - handle both direct text and JSON node content
                    content = match.metadata.get('text', '')
                    if not content and '_node_content' in match.metadata:
                        try:
                            node_content = json.loads(match.metadata['_node_content'])
                            content = node_content.get('text', '')
                        except (json.JSONDecodeError, TypeError):
                            content = str(match.metadata.get('_node_content', ''))[:500]
                    
                    result = PaperSearchResult(
                        chunk_id=match.id,
                        content=content,
                        metadata=match.metadata,
                        relevance_score=match.score,
                        page_number=match.metadata.get('page_number'),
                        section_type=match.metadata.get('section_type'),
                        paper_title=match.metadata.get('title'),
                        authors=match.metadata.get('authors', []),
                        confidence=min(match.score * 1.1, 1.0)  # Slight confidence boost
                    )
                    paper_results.append(result)
            
            # Rerank results if cross-encoder is available
            if self.cross_encoder and len(paper_results) > 1:
                paper_results = await self._rerank_results(research_query.query, paper_results)
            
            print(f"📊 Returning {len(paper_results)} high-quality results")
            return paper_results
            
        except Exception as e:
            print(f"❌ Research paper search failed: {e}")
            return []
    
    def _build_search_filter(self, research_query: ResearchQuery) -> Optional[Dict]:
        """
        Build Pinecone filter based on research query requirements
        Note: Section type filtering is disabled as most documents lack structured metadata
        """
        filters = {}
        
        # Skip section-based filtering - rely on semantic search instead
        # Most documents don't have section_type metadata, and semantic search
        # with cross-encoder reranking provides better relevance than rigid filtering
        
        # Future: Could add other metadata filters here (date, author, etc.)
        # but avoid section_type filtering which excludes too much content
        
        return None  # No filters - let semantic search handle relevance
    
    async def _rerank_results(self, query: str, results: List[PaperSearchResult]) -> List[PaperSearchResult]:
        """Rerank results using cross-encoder for better academic relevance"""
        try:
            # Prepare query-result pairs for reranking
            pairs = [(query, result.content[:512]) for result in results]  # Limit content for efficiency
            
            # Get reranking scores
            rerank_scores = await asyncio.to_thread(
                self.cross_encoder.predict,
                pairs
            )
            
            # Update results with rerank scores
            for i, result in enumerate(results):
                result.relevance_score = float(rerank_scores[i])
                result.confidence = min(result.relevance_score * 1.1, 1.0)
            
            # Sort by rerank score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            print("✅ Results reranked using cross-encoder")
            
        except Exception as e:
            print(f"⚠️ Reranking failed: {e}")
        
        return results
    
    async def analyze_research_paper(self, research_query: ResearchQuery) -> Dict[str, Any]:
        """
        Comprehensive research paper analysis with academic focus
        """
        print(f"📊 Starting comprehensive research paper analysis...")
        start_time = time.time()
        
        # Search for relevant content
        search_results = await self.search_research_paper(research_query)
        
        if not search_results:
            return {
                "success": False,
                "query": research_query.query,
                "error": "No relevant content found in the research paper",
                "namespace": self._build_namespace(research_query.user_id, research_query.document_uuid)
            }
        
        # Prepare context for analysis
        context_parts = []
        for i, result in enumerate(search_results[:10]):  # Limit to top 10 for context
            page_ref = f"[Page {result.page_number}]" if result.page_number else "[Page Unknown]"
            section_ref = f"({result.section_type})" if result.section_type else ""
            
            context_parts.append(f"""
{page_ref} {section_ref}
{result.content}
(Relevance Score: {result.relevance_score:.3f})
""")
        
        combined_context = "\n".join(context_parts)
        
        # Generate research-focused analysis
        analysis_response = await self._generate_research_analysis(
            research_query, combined_context, search_results
        )
        
        execution_time = time.time() - start_time
        
        return {
            "success": True,
            "query": research_query.query,
            "query_type": research_query.query_type,
            "response": analysis_response,
            "metadata": {
                "execution_time": execution_time,
                "results_count": len(search_results),
                "namespace": self._build_namespace(research_query.user_id, research_query.document_uuid),
                "paper_title": search_results[0].paper_title if search_results else None,
                "authors": search_results[0].authors if search_results else None,
                "pages_referenced": list(set([r.page_number for r in search_results if r.page_number])),
                "sections_analyzed": list(set([r.section_type for r in search_results if r.section_type]))
            },
            "search_results": [
                {
                    "content": result.content[:200] + "..." if len(result.content) > 200 else result.content,
                    "page": result.page_number,
                    "section": result.section_type,
                    "relevance": result.relevance_score
                }
                for result in search_results[:5]  # Top 5 results summary
            ]
        }
    
    async def _generate_research_analysis(self, research_query: ResearchQuery, context: str, results: List[PaperSearchResult]) -> str:
        """Generate comprehensive research analysis using specialized academic prompt"""
        
        # Build query-specific analysis prompt
        analysis_prompt = f"""
**RESEARCH PAPER ANALYSIS REQUEST**

Query: "{research_query.query}"
Query Type: {research_query.query_type}
Document Namespace: user_{research_query.user_id}_doc_{research_query.document_uuid}

**RESEARCH PAPER CONTEXT:**
{context}

**ANALYSIS REQUIREMENTS:**
Based on the provided research paper content, provide a comprehensive academic analysis that:

1. **STEP-BY-STEP REASONING**: Explain your analytical process citing specific sections and pages
2. **ACADEMIC RIGOR**: Focus on methodological details, statistical significance, and research contributions
3. **PAGE CITATIONS**: Include [Page X] references for all claims and findings
4. **TECHNICAL DEPTH**: Provide detailed explanations suitable for researchers
5. **RESEARCH QUALITY**: Assess methodology rigor, result significance, and contribution novelty

**QUERY-SPECIFIC FOCUS:**
{self._get_query_specific_instructions(research_query.query_type)}

**CRITICAL CONSTRAINTS:**
- Only use information explicitly stated in the provided research paper context
- Include specific page references for all claims
- If information is not available, clearly state: "I could not find this information in the research paper"
- Maintain academic formatting and terminology
- Focus on research quality indicators

Provide your comprehensive research analysis:
"""
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.response_model,
                messages=[
                    {"role": "system", "content": self.research_system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2000,  # Allow comprehensive responses
                temperature=0.1  # Low temperature for academic precision
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ Research analysis generation failed: {e}")
            return f"Error generating research analysis: {str(e)}"
    
    def _get_query_specific_instructions(self, query_type: str) -> str:
        """Get specific instructions based on query type"""
        instructions = {
            'methodology': "Focus on experimental design, algorithms, datasets, evaluation metrics, and methodological rigor. Extract exact technical details.",
            'results': "Analyze findings, performance metrics, statistical significance, and result interpretation. Include all numerical data and comparisons.",
            'statistical': "Extract p-values, confidence intervals, effect sizes, and statistical tests. Assess statistical rigor and significance.",
            'citations': "Identify related work, citation patterns, and comparative studies. Analyze how this work builds on previous research.",
            'discussion': "Focus on result interpretation, implications, limitations, and research impact. Analyze the authors' analytical insights.",
            'conclusion': "Extract key contributions, future work suggestions, and research impact. Summarize novel findings and their significance.",
            'general': "Provide comprehensive overview covering methodology, results, and contributions with balanced academic analysis."
        }
        return instructions.get(query_type, instructions['general'])

# Example usage and integration guide
def create_research_query_example():
    """Example of how to create a research query"""
    return ResearchQuery(
        query="What machine learning algorithm was used and what was its accuracy?",
        query_type="methodology",
        user_id="user123",
        document_uuid="doc-uuid-456",
        max_results=10,
        similarity_threshold=0.7,
        focus_sections=["methodology", "results"]
    )

async def example_usage():
    """Example usage of the PaperRetriever"""
    retriever = PaperRetriever()
    
    # Create research query
    query = ResearchQuery(
        query="What are the main research contributions of this paper?",
        query_type="conclusion",
        user_id="user123",
        document_uuid="paper-uuid-789",
        max_results=15,
        similarity_threshold=0.75
    )
    
    # Perform research analysis
    result = await retriever.analyze_research_paper(query)
    
    print("📊 Research Analysis Result:")
    print(f"Success: {result['success']}")
    print(f"Response: {result.get('response', 'No response')}")
    print(f"Pages Referenced: {result.get('metadata', {}).get('pages_referenced', [])}")
    
    return result
