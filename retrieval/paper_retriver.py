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
    query_type: List[str]  # methodology, results, discussion, conclusion, general, citations, statistical
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


    OVERVIEW_PDF_ANALYSIS_PROMPT = """
**OVERVIEW FOCUS**
You are an expert at summarizing academic and technical documents. Provide a clear and concise **overview** of the content provided based on the Query asked, highlighting its purpose, scope, and main themes. Focus on giving a high-level understanding without diving into detailed sections.
Use [Page X] format for references and maintain structured organization.
"""

    
    GENERAL_PDF_ANALYSIS_PROMPT = """
You are a knowledgeable assistant helping someone understand document 
  content. Answer the question naturally and conversationally, as if explaining
   to an intelligent colleague.

  **GUIDELINES:**
  1. Answer the question directly first, then add supporting context
  2. Include page references naturally: "as mentioned on [Page X]" or "according to [Page 4.0]"
  3. Use clear, accessible language while maintaining accuracy
  4. Organize information in the most logical way for this specific question
  5. Be informative but not overwhelming - focus on what was actually asked
  6. Only use information explicitly provided in the document

  Provide a helpful, natural response that directly addresses what the user 
  wants to know.
"""
    
# Section-specific prompts (extendable)
    METHODOLOGY_PDF_ANALYSIS_PROMPT = """
**METHODOLOGY FOCUS**
Carefully analyze the methodology design section. Extract and rephrase:
- Study design and experimental setup
- Data sources, datasets, and tools used
- Step-by-step process or workflow
- Validation methods or evaluation protocols
- Control variables or assumptions
Use [Page X] format for references and bullet points where appropriate.
"""

    DISCUSSION_PDF_ANALYSIS_PROMPT = """
**DISCUSSION FOCUS**
Focus on the Discussion section and extract:
- Interpretations of results
- Limitations or implications highlighted
- Theoretical impact or future directions proposed
Use [Page X] format and group logically.
"""

    RESULTS_PDF_ANALYSIS_PROMPT = """
**RESULTS FOCUS**
Summarize the Results section:
- Key metrics and data trends
- Visual aids (charts, graphs, tables)
- Any performance benchmarks
Back all facts with [Page X] references.
"""

    CONCLUSION_PDF_ANALYSIS_PROMPT = """
**CONCLUSION FOCUS**
Summarize the final section:
- Main takeaways and research contributions
- Recommendations or closing thoughts
- Future work (if any)
Use [Page X] format and bullet points where necessary.
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
        
        # Research paper specific prompts - now handled dynamically
        
        self.setup_models()
        self.setup_pinecone()
        
        print("✅ Research Paper Retrieval System initialized successfully")
    
    
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
        
        # Prepare context for analysis with logical ordering
        # Sort results by page number first, then by relevance for logical flow
        sorted_results = sorted(search_results[:10], key=lambda x: (x.page_number or 999, -x.relevance_score))
        
        context_parts = []
        for i, result in enumerate(sorted_results):
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
        """Generate comprehensive research analysis using specialized academic prompt or general prompt"""
        # Choose the system prompt based on query_type
        system_prompt = self._get_query_specific_instructions(research_query.query_type)


        # Build query-specific analysis prompt with structured flow
        if 'general' in research_query.query_type and len(research_query.query_type) == 1:
            analysis_prompt = f"""
  **QUESTION:** "{research_query.query}"

  **DOCUMENT CONTENT:**
  {context}

  **INSTRUCTIONS:**
  Answer this question naturally and directly using the document content above.
   
  - Start with the most direct answer to what was asked
  - Add relevant context and details that help explain the answer
  - Use headings only if they genuinely improve clarity
  - Include page references naturally within your explanation
  - Be conversational yet informative
  - Use the document's natural flow when it makes sense
  - Be conversational yet accurate
  - Include relevant details without forcing completeness
  - Generate a Structure response. Not just a point wise response.
  - Focus on what the user actually wants to know
  - More comprensive about answering the question, more human like responce.

  ANSWER:

  """
        else:
      # Keep existing structured approach for multi-type queries
            analysis_prompt = f"""
  **DOCUMENT ANALYSIS REQUEST**
  Question: "{research_query.query}"
  Query Type: {', '.join(research_query.query_type)}

  **DOCUMENT CONTENT:**
  {context}

  **RESPONSE STRUCTURE REQUIREMENTS:**
  1. Start with a clear, direct answer to the Query
  2. Use smooth transitions between sections
  3. Include page references naturally within sentences
  4. End with a brief synthesis (if needed)

  **INSTRUCTIONS:**
  - Analyse the Query Type and be focus on Query Type while answering the question. [“general”, “overview”, “methodology”, “results”, “discussion”, “conclusion”] this are the quary types. only focus on what is asked and. like if i asked overview dont include conclusion. if i asked about methodology and overview dont include results. general Quary Type is always included. 
  - Answer the question directly and naturally
  - Begin with "**[Topic Name]**" as the main heading
  - Use subheadings for each major section
  - Add transitional phrases between sections ("Building on this concept...", 
  "This approach enables...", "Furthermore...")
  - Integrate page references smoothly: "The system operates by... [Page X]" 
  not "According to Page X..."
  - Organize content logically rather than jumping between topics
  - Use point-wise description only when structurally needed
  - Maintain academic flow while being accessible

  ANSWER:

  """
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.response_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=2000,  # Allow comprehensive responses
                temperature=0.3  # Low temperature for academic precision
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Research analysis generation failed: {e}")
            return f"Error generating research analysis: {str(e)}"
    
    def _get_query_specific_instructions(self, query_type: List[str]) -> str:
        """Get dynamic instructions based on query type"""
        
        # Structure guidance that applies to all response types
        structure_guidance = """
  **RESPONSE ORGANIZATION:**
  - Answer the question directly and naturally
  - Organize information in the most logical way for this specific query
  - Use the document's natural flow when it makes sense
  - Be conversational yet accurate
  - Include relevant details without forcing completeness
  """
        
        instructions = {
            'general': self.GENERAL_PDF_ANALYSIS_PROMPT,
            'overview': self.OVERVIEW_PDF_ANALYSIS_PROMPT,
            'methodology': self.METHODOLOGY_PDF_ANALYSIS_PROMPT,
            'results': self.RESULTS_PDF_ANALYSIS_PROMPT,
            'discussion': self.DISCUSSION_PDF_ANALYSIS_PROMPT,
            'conclusion': self.CONCLUSION_PDF_ANALYSIS_PROMPT
        }
        
        # If only 'general' is requested, return general prompt with structure guidance
        if query_type == ['general']:
            return instructions['general'] + structure_guidance
        
        # For multiple types, combine them
        combined_prompts = []
        
        # Always start with general as base
        combined_prompts.append(instructions['general'])
        
        # Add specific focus areas
        for qtype in query_type:
            if qtype != 'general' and qtype in instructions:
                combined_prompts.append(f"\n{instructions[qtype]}")
        
        # Add enhanced structure guidance for multiple sections
        if len(query_type) > 1:
            combined_prompts.append(
                "\n**MULTI-SECTION STRUCTURE:** Organize your response with clear sections for each focus area requested. Use transitional phrases between sections to maintain flow."
            )
        
        # Add general structure guidance
        combined_prompts.append(structure_guidance)
        
        return "\n".join(combined_prompts)

# Example usage and integration guide
def create_research_query_example():
      """Example of how to create a research query"""
      return ResearchQuery(
          query="What machine learning algorithm was used and what was its accuracy?",
          query_type=["general", "methodology", "results"],
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
        query_type=["general", "conclusion", "overview"],
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
