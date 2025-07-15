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
from rank_bm25 import BM25Okapi
import tiktoken

from config import AdvancedConfig
from vector_storage import AdvancedVectorStorage

# Use the existing SearchResult from vector_storage
from vector_storage import SearchResult as BaseSearchResult

@dataclass
class EnhancedSearchResult(BaseSearchResult):
    """Enhanced search result with additional scoring fields"""
    sparse_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float = 0.0
    confidence: float = 0.0

class QueryProcessor:
    """Advanced query understanding and expansion"""
    
    def __init__(self, openai_client: OpenAI):
        self.openai_client = openai_client
        self.encoding = tiktoken.encoding_for_model("gpt-4")
    
    def understand_query(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and type"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Use mini for efficiency
                messages=[
                    {
                        "role": "system", 
                        "content": """Analyze the user query and classify it:
                        
                        Query Types:
                        - factual: asking for specific information
                        - conceptual: asking about concepts/definitions
                        - procedural: asking how to do something
                        - mathematical: involving formulas/calculations
                        - comparative: comparing concepts/methods
                        - study_plan: requesting learning structure
                        - book_analysis: asking about books/chapters
                        
                        Difficulty Levels:
                        - basic: introductory concepts
                        - intermediate: requires some background
                        - advanced: complex technical content
                        
                        Return JSON with: {"type": "...", "difficulty": "...", "key_concepts": [...], "mathematical": true/false}"""
                    },
                    {"role": "user", "content": f"Query: {query}"}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content.strip())
            return result
            
        except Exception as e:
            print(f"⚠️ Query analysis failed: {e}")
            return {
                "type": "factual", 
                "difficulty": "intermediate", 
                "key_concepts": [], 
                "mathematical": False
            }
    
    def expand_query(self, query: str, query_analysis: Dict) -> List[str]:
        """Generate expanded query variations for better retrieval"""
        try:
            expanded_queries = [query]  # Always include original
            
            # Add synonyms and related terms
            synonym_prompt = f"""Generate 3 alternative phrasings for this query: "{query}"
            Focus on synonyms, related terms, and different ways to express the same concept.
            Return only the alternative queries, one per line."""
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": synonym_prompt}],
                max_tokens=150,
                temperature=0.3
            )
            
            alternatives = response.choices[0].message.content.strip().split('\n')
            expanded_queries.extend([alt.strip() for alt in alternatives if alt.strip()])
            
            # Add mathematical variations if relevant
            if query_analysis.get("mathematical"):
                math_prompt = f"""Generate mathematical query variations for: "{query}"
                Include formal mathematical terms, equation names, theorem names.
                Return only the variations, one per line."""
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": math_prompt}],
                    max_tokens=100,
                    temperature=0.3
                )
                
                math_variations = response.choices[0].message.content.strip().split('\n')
                expanded_queries.extend([var.strip() for var in math_variations if var.strip()])
            
            return expanded_queries[:5]  # Limit to prevent noise
            
        except Exception as e:
            print(f"⚠️ Query expansion failed: {e}")
            return [query]

class AdvancedKnowledgeBaseRetriever:
    """Advanced Knowledge Base Retriever with hybrid search and OpenAI-powered responses"""
    
    def __init__(self):
        """Initialize the advanced retriever with all components"""
        print("🚀 Initializing Advanced Knowledge Base Retriever...")
        
        # Initialize configuration
        self.config = AdvancedConfig()
        print(f"⚙️  Configuration loaded with model: {self.config.LLM_MODEL}")
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=self.config.OPENAI_API_KEY)
        print(f"🤖 OpenAI client initialized")
        
        # Initialize query processor
        self.query_processor = QueryProcessor(self.openai_client)
        print(f"🧠 Query processor initialized")
        
        # Initialize vector storage
        self.vector_storage = AdvancedVectorStorage(self.config)
        print(f"📄 Research Papers index: {self.config.PINECONE_INDEX_NAME}")
        print(f"📚 Knowledge Base index: {self.config.PINECONE_KB_INDEX_NAME}")
        
        # Initialize reranker
        self.reranker = None
        self._initialize_reranker()
        
        # Sparse retrieval preparation
        self.bm25_index = None
        self.content_chunks = []
        
        # Response generation settings
        self.response_model = self.config.LLM_MODEL
        self.max_response_tokens = 1500
        self.temperature = 0.3
        
        print("✅ Advanced Knowledge Base Retriever initialized with OpenAI-powered responses")
    
    def _initialize_reranker(self):
        """Initialize cross-encoder for reranking"""
        try:
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("✅ Cross-encoder reranker initialized")
        except Exception as e:
            print(f"⚠️ Reranker initialization failed: {e}")
    
    async def _prepare_sparse_retrieval(self, namespace: str = "knowledge-base", index_name: str = None):
        """Prepare BM25 sparse retrieval corpus"""
        try:
            if self.bm25_index is not None:
                return  # Already prepared
            
            # Use configured KB index if not specified
            if index_name is None:
                index_name = self.config.PINECONE_KB_INDEX_NAME
            
            print("🔄 Preparing sparse retrieval corpus...")
            
            # Get all documents from knowledge base
            inventory = await self.vector_storage.get_knowledge_base_inventory(namespace, index_name)
            
            # Build corpus for BM25
            self.content_chunks = []
            
            # Sample documents for BM25 (due to memory constraints)
            sample_results = await self._sample_knowledge_base_content(namespace, index_name)
            
            for result in sample_results:
                self.content_chunks.append(result['content'])
            
            # Tokenize corpus
            tokenized_corpus = [doc.lower().split() for doc in self.content_chunks]
            self.bm25_index = BM25Okapi(tokenized_corpus)
            
            print(f"✅ Sparse retrieval prepared with {len(self.content_chunks)} documents")
            
        except Exception as e:
            print(f"⚠️ Sparse retrieval preparation failed: {e}")
    
    async def _sample_knowledge_base_content(self, namespace: str, index_name: str, sample_size: int = 500) -> List[Dict]:
        """Sample content from knowledge base for BM25 corpus"""
        try:
            # Use a broad query to sample diverse content
            results = await self.vector_storage.enhanced_knowledge_base_search(
                query="machine learning algorithms data science mathematics",
                namespace=namespace,
                top_k=sample_size,
                index_name=index_name
            )
            return results
        except Exception as e:
            print(f"⚠️ Knowledge base sampling failed: {e}")
            return []
    
    async def intelligent_search(self, query: str, top_k: int = 10, namespace: str = "knowledge-base", index_name: str = None) -> Dict[str, Any]:
        """Main intelligent search that determines query type and responds appropriately"""
        try:
            # Use configured KB index if not specified
            if index_name is None:
                index_name = self.config.PINECONE_KB_INDEX_NAME
                
            print(f"🧠 Starting intelligent search for: '{query}'")
            
            # Analyze query intent
            query_analysis = self.query_processor.understand_query(query)
            print(f"📊 Query analysis: {query_analysis}")
            
            # Route to appropriate handler based on query type
            if query_analysis["type"] == "study_plan":
                return await self._generate_study_plan(query, query_analysis)
            elif query_analysis["type"] == "book_analysis":
                return await self._analyze_books(query, query_analysis)
            elif "chapters" in query.lower() or "table of contents" in query.lower():
                return await self._extract_chapters(query, query_analysis)
            else:
                # Default to hybrid search
                return await self._hybrid_search(query, query_analysis, top_k, namespace, index_name)
            
        except Exception as e:
            print(f"❌ Intelligent search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _hybrid_search(self, query: str, query_analysis: Dict, top_k: int, namespace: str, index_name: str) -> Dict[str, Any]:
        """Hybrid search combining dense, sparse, and reranking with OpenAI-powered response generation"""
        try:
            print(f"🔍 Performing hybrid search...")
            
            # Prepare sparse retrieval if needed
            await self._prepare_sparse_retrieval(namespace, index_name)
            
            # 1. Dense retrieval (semantic search)
            dense_results = await self._dense_search(query, query_analysis, top_k * 2, namespace, index_name)
            
            # 2. Sparse retrieval (BM25)
            sparse_results = await self._sparse_search(query, top_k, namespace, index_name)
            
            # 3. Combine and rerank
            final_results = await self._combine_and_rerank(query, dense_results, sparse_results, top_k)
            
            # 4. Generate OpenAI-powered response based on search results
            ai_response = await self._generate_intelligent_response(query, final_results, query_analysis)
            
            # 5. Format comprehensive response
            return {
                "success": True,
                "query": query,
                "query_analysis": query_analysis,
                "search_type": "enhanced",
                "ai_response": ai_response,
                "search_results": final_results,
                "total_results": len(final_results),
                "namespace": namespace,
                "index_name": index_name
            }
            
        except Exception as e:
            print(f"❌ Hybrid search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }

    async def _generate_intelligent_response(self, query: str, search_results: List[Dict], query_analysis: Dict) -> str:
        """Generate intelligent response using OpenAI based on search results and query analysis"""
        try:
            print(f"🧠 Generating AI response for query type: {query_analysis.get('type', 'general')}")
            
            # Check if we have valid search results
            if not search_results or len(search_results) == 0:
                return "I couldn't find relevant information to answer your question. Please try rephrasing your query or ask about a different topic."
            
            # Format context from search results
            context = self._format_search_context(search_results)
            
            # Detect query type and get specialized prompt
            query_type = query_analysis.get("type", "general")
            system_prompt = self._get_specialized_system_prompt(query_type, context, query)
            
            # Generate response using OpenAI
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=self.response_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": f"Based on the context provided, please answer: {query}"
                        }
                    ],
                    max_tokens=self.max_response_tokens,
                    temperature=self.temperature
                )
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ AI response generation failed: {e}")
            # Fallback to formatted search results
            return self._format_fallback_response(search_results)

    def _format_search_context(self, search_results: List[Dict]) -> str:
        """Format search results into context for OpenAI"""
        try:
            if not search_results:
                return "No relevant information found."
            
            context_parts = []
            for i, result in enumerate(search_results[:5], 1):  # Use top 5 results
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                book_name = metadata.get('book_name', 'Unknown Source')
                confidence = result.get('confidence', 0.0)
                
                context_parts.append(f"[Source {i} - {book_name} (Confidence: {confidence:.2f})]:\n{content[:800]}...")
            
            return "\n\n".join(context_parts)
            
        except Exception as e:
            print(f"❌ Context formatting failed: {e}")
            return "Error formatting search context."

    def _get_specialized_system_prompt(self, query_type: str, context: str, query: str) -> str:
        """Get specialized system prompt based on query type"""
        
        base_prompt = f"""You are an expert AI assistant with access to a comprehensive knowledge base containing books on machine learning, data science, algorithms, and related topics.

Context from knowledge base:
{context}

Your task is to provide a helpful, accurate, and well-structured answer based on the provided context."""

        if query_type == "definition":
            return base_prompt + """

Focus on:
- Clear, concise definitions
- Key characteristics and properties
- Simple examples when helpful
- Avoid overly technical jargon unless necessary"""

        elif query_type == "explanation":
            return base_prompt + """

Focus on:
- Step-by-step explanations
- Breaking down complex concepts
- Using analogies when helpful
- Connecting related concepts"""

        elif query_type == "comparison":
            return base_prompt + """

Focus on:
- Clear comparisons and contrasts
- Advantages and disadvantages
- Use cases for each approach
- Structured comparison format"""

        elif query_type == "implementation":
            return base_prompt + """

Focus on:
- Practical implementation details
- Step-by-step procedures
- Best practices and common pitfalls
- Concrete examples"""

        elif query_type == "mathematical":
            return base_prompt + """

Focus on:
- Mathematical concepts and formulations
- Clear explanations of formulas
- When and how to apply mathematical principles
- Examples with calculations when relevant"""

        else:  # general
            return base_prompt + """

Provide a comprehensive, well-structured answer that:
- Directly addresses the question
- Uses information from the provided context
- Is clear and easy to understand
- Includes relevant examples when helpful"""

    def _format_fallback_response(self, search_results: List[Dict]) -> str:
        """Format fallback response when OpenAI generation fails"""
        try:
            if not search_results:
                return "No relevant information found in the knowledge base."
            
            response_parts = ["Based on the available information:\n"]
            
            for i, result in enumerate(search_results[:3], 1):
                content = result.get('content', '')
                metadata = result.get('metadata', {})
                book_name = metadata.get('book_name', 'Unknown Source')
                
                response_parts.append(f"{i}. From {book_name}:")
                response_parts.append(f"   {content[:300]}...")
                response_parts.append("")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            print(f"❌ Fallback response formatting failed: {e}")
            return "Error generating response from search results."

    async def _dense_search(self, query: str, query_analysis: Dict, top_k: int, namespace: str, index_name: str) -> List[EnhancedSearchResult]:
        """Dense semantic search with query expansion using existing vector storage"""
        try:
            # Expand query for better retrieval
            expanded_queries = self.query_processor.expand_query(query, query_analysis)
            
            all_results = []
            
            for expanded_query in expanded_queries:
                # Use existing vector storage method
                results = await self.vector_storage.enhanced_knowledge_base_search(
                    query=expanded_query,
                    namespace=namespace,
                    top_k=top_k,
                    index_name=index_name
                )
                
                # Convert to EnhancedSearchResult objects
                for result in results:
                    search_result = EnhancedSearchResult(
                        chunk_id=result['chunk_id'],
                        content=result['content'],
                        score=result['score'],  # Use 'score' for dense_score
                        metadata=result['metadata'],
                        confidence=result['confidence']
                    )
                    all_results.append(search_result)
            
            # Remove duplicates and sort by score
            unique_results = {}
            for result in all_results:
                if result.chunk_id not in unique_results or result.score > unique_results[result.chunk_id].score:
                    unique_results[result.chunk_id] = result
            
            return sorted(unique_results.values(), key=lambda x: x.score, reverse=True)[:top_k]
            
        except Exception as e:
            print(f"❌ Dense search failed: {e}")
            return []
    
    async def _sparse_search(self, query: str, top_k: int, namespace: str, index_name: str) -> List[EnhancedSearchResult]:
        """Sparse BM25 search"""
        try:
            if self.bm25_index is None:
                return []
            
            # Tokenize query
            tokenized_query = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top results
            top_indices = np.argsort(bm25_scores)[-top_k:][::-1]
            
            sparse_results = []
            for idx in top_indices:
                if idx < len(self.content_chunks) and bm25_scores[idx] > 0:
                    search_result = EnhancedSearchResult(
                        chunk_id=f"sparse_{idx}",
                        content=self.content_chunks[idx],
                        score=0.0,  # No dense score for sparse results
                        metadata={}, # No metadata for sparse results
                        sparse_score=bm25_scores[idx]
                    )
                    sparse_results.append(search_result)
            
            return sparse_results
            
        except Exception as e:
            print(f"❌ Sparse search failed: {e}")
            return []
    
    async def _combine_and_rerank(self, query: str, dense_results: List[EnhancedSearchResult], sparse_results: List[EnhancedSearchResult], top_k: int) -> List[Dict[str, Any]]:
        """Combine dense and sparse results and rerank"""
        try:
            # Combine results
            all_results = {}
            
            # Add dense results
            for result in dense_results:
                all_results[result.chunk_id] = result
            
            # Add sparse results (merge if duplicate)
            for result in sparse_results:
                if result.chunk_id in all_results:
                    all_results[result.chunk_id].sparse_score = result.sparse_score
                else:
                    all_results[result.chunk_id] = result
            
            # Rerank if reranker available
            if self.reranker:
                reranked_results = await self._rerank_results(query, list(all_results.values()))
            else:
                reranked_results = list(all_results.values())
            
            # Calculate final scores
            final_results = []
            for result in reranked_results[:top_k]:
                # Combine scores (weighted average)
                dense_weight = 0.6
                sparse_weight = 0.3
                rerank_weight = 0.1
                
                final_score = 0.0
                if result.score > 0:  # Use score instead of dense_score
                    final_score += dense_weight * result.score
                if result.sparse_score and result.sparse_score > 0:
                    final_score += sparse_weight * (result.sparse_score / 10)  # Normalize BM25 score
                if result.rerank_score and result.rerank_score > 0:
                    final_score += rerank_weight * result.rerank_score
                
                result.final_score = final_score
                result.confidence = min(final_score * 1.2, 1.0)  # Cap at 1.0
                
                final_results.append({
                    "chunk_id": result.chunk_id,
                    "content": result.content,
                    "metadata": result.metadata,
                    "scores": {
                        "dense": result.score,  # Use score instead of dense_score
                        "sparse": result.sparse_score,
                        "rerank": result.rerank_score,
                        "final": result.final_score
                    },
                    "confidence": result.confidence
                })
            
            return final_results
            
        except Exception as e:
            print(f"❌ Combine and rerank failed: {e}")
            return []
    
    async def _rerank_results(self, query: str, results: List[EnhancedSearchResult]) -> List[EnhancedSearchResult]:
        """Rerank results using cross-encoder"""
        try:
            if not self.reranker or len(results) <= 1:
                return results
            
            # Prepare pairs for reranking
            pairs = [(query, result.content) for result in results]
            
            # Get rerank scores
            rerank_scores = self.reranker.predict(pairs)
            
            # Update results with rerank scores
            for i, result in enumerate(results):
                result.rerank_score = float(rerank_scores[i])
            
            # Sort by rerank score
            return sorted(results, key=lambda x: x.rerank_score, reverse=True)
            
        except Exception as e:
            print(f"❌ Reranking failed: {e}")
            return results
    
    async def _generate_study_plan(self, query: str, query_analysis: Dict) -> Dict[str, Any]:
        """Generate intelligent study plan based on query"""
        try:
            print(f"📚 Generating study plan for: '{query}'")
            
            # Extract topic from query
            topic = self._extract_topic_from_query(query)
            
            # Get book inventory using existing vector storage method
            inventory = await self.vector_storage.get_knowledge_base_inventory()
            relevant_books = []
            
            # Find relevant books from inventory
            for book_name in inventory.get("books", []):
                book_info = inventory.get("books_structure", {}).get(book_name, {})
                # Simple relevance check based on key concepts
                if any(concept.lower() in book_name.lower() for concept in query_analysis.get("key_concepts", [])):
                    relevant_books.append((book_name, book_info))
            
            # Generate study plan
            study_plan = await self._create_study_plan(topic, relevant_books, query_analysis)
            
            return {
                "success": True,
                "query": query,
                "response_type": "study_plan",
                "study_plan": study_plan,
                "topic": topic,
                "difficulty": query_analysis.get("difficulty", "intermediate")
            }
            
        except Exception as e:
            print(f"❌ Study plan generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _analyze_books(self, query: str, query_analysis: Dict) -> Dict[str, Any]:
        """Analyze books and provide recommendations using existing vector storage inventory"""
        try:
            print(f"📖 Analyzing books for: '{query}'")
            
            # Use existing vector storage method for book inventory
            inventory = await self.vector_storage.get_knowledge_base_inventory()
            
            # Analyze available books
            book_analysis = {}
            for book in inventory.get("books", []):
                analysis = await self._analyze_single_book(book, query_analysis)
                book_analysis[book] = analysis
            
            return {
                "success": True,
                "query": query,
                "response_type": "book_analysis",
                "book_analysis": book_analysis,
                "total_books": len(book_analysis)
            }
            
        except Exception as e:
            print(f"❌ Book analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    async def _extract_chapters(self, query: str, query_analysis: Dict) -> Dict[str, Any]:
        """Extract and analyze chapters from books using existing vector storage methods"""
        try:
            print(f"📑 Extracting chapters for: '{query}'")
            
            # Extract book name from query if specified
            book_name = self._extract_book_from_query(query)
            
            if book_name:
                chapters = await self._get_book_chapters(book_name)
            else:
                # Get chapters from all books using existing inventory method
                chapters = {}
                inventory = await self.vector_storage.get_knowledge_base_inventory()
                for book in inventory.get("books", []):
                    book_chapters = await self._get_book_chapters(book)
                    chapters[book] = book_chapters
            
            return {
                "success": True,
                "query": query,
                "response_type": "chapters",
                "chapters": chapters,
                "book_name": book_name
            }
            
        except Exception as e:
            print(f"❌ Chapter extraction failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    # Helper methods
    def _extract_topic_from_query(self, query: str) -> str:
        """Extract main topic from query"""
        # Simple keyword extraction
        keywords = ["machine learning", "data science", "algorithms", "mathematics", "statistics", "deep learning", "neural networks"]
        for keyword in keywords:
            if keyword in query.lower():
                return keyword
        return "general"
    
    def _extract_book_from_query(self, query: str) -> Optional[str]:
        """Extract book name from query"""
        # Simple synchronous check - will use inventory in async context elsewhere
        keywords = ["machine learning", "data science", "algorithms", "foundations", "theories", "models"]
        for keyword in keywords:
            if keyword in query.lower():
                return f"book containing {keyword}"
        return None
    
    async def _create_study_plan(self, topic: str, relevant_books: List[Tuple[str, Dict]], query_analysis: Dict) -> Dict[str, Any]:
        """Create structured study plan"""
        try:
            difficulty = query_analysis.get("difficulty", "intermediate")
            
            # Generate study plan using AI
            study_plan_prompt = f"""Create a comprehensive study plan for "{topic}" at {difficulty} level.
            
            Available books: {[book for book, _ in relevant_books]}
            
            Structure the plan with:
            1. Prerequisites
            2. Learning objectives
            3. Recommended sequence
            4. Time estimates
            5. Practice exercises
            
            Return as structured JSON."""
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(
                    model=self.response_model,
                    messages=[{"role": "user", "content": study_plan_prompt}],
                    max_tokens=1000,
                    temperature=self.temperature
                )
            )
            
            study_plan = json.loads(response.choices[0].message.content.strip())
            
            return {
                "topic": topic,
                "difficulty": difficulty,
                "plan": study_plan,
                "relevant_books": relevant_books,
                "estimated_duration": study_plan.get("estimated_duration", "4-6 weeks")
            }
            
        except Exception as e:
            print(f"❌ Study plan creation failed: {e}")
            return {
                "topic": topic,
                "difficulty": difficulty,
                "plan": "Study plan generation failed",
                "error": str(e)
            }
    
    async def _analyze_single_book(self, book_name: str, query_analysis: Dict) -> Dict[str, Any]:
        """Analyze a single book for relevance and content using existing vector storage"""
        try:
            # Use existing vector storage method for book content search
            book_content = await self.vector_storage.enhanced_knowledge_base_search(
                query=f"book {book_name}",
                namespace="knowledge-base",
                top_k=10
            )
            
            # Analyze content using the results from existing method
            total_chunks = len(book_content)
            has_math = any(result.get('has_formulas', False) for result in book_content)
            topics = set()
            
            for result in book_content:
                topics.update(result.get('metadata', {}).get('topics', []))
            
            return {
                "total_chunks": total_chunks,
                "has_mathematical_content": has_math,
                "topics": list(topics),
                "relevance_score": self._calculate_relevance_score(book_content, query_analysis),
                "difficulty": self._estimate_difficulty(book_content),
                "chapters": self._extract_chapters_from_content(book_content)
            }
            
        except Exception as e:
            print(f"❌ Book analysis failed for {book_name}: {e}")
            return {
                "error": str(e),
                "total_chunks": 0,
                "has_mathematical_content": False,
                "topics": [],
                "relevance_score": 0.0
            }
    
    async def _get_book_chapters(self, book_name: str) -> List[str]:
        """Get chapters for a specific book using existing methods and hardcoded structure"""
        try:
            # First try hardcoded structure
            inventory = await self.vector_storage.get_knowledge_base_inventory()
            if book_name in inventory.get("books_structure", {}):
                return inventory["books_structure"][book_name]["chapters"]
            
            # Fallback to existing vector storage content analysis
            book_content = await self.vector_storage.enhanced_knowledge_base_search(
                query=f"chapters table of contents {book_name}",
                namespace="knowledge-base",
                top_k=20
            )
            
            chapters = set()
            for result in book_content:
                chapters.update(result.get('chapters_found', []))
            
            return sorted(list(chapters))
            
        except Exception as e:
            print(f"❌ Chapter extraction failed for {book_name}: {e}")
            return []
    
    def _calculate_relevance_score(self, book_content: List[Dict], query_analysis: Dict) -> float:
        """Calculate relevance score for a book"""
        try:
            key_concepts = query_analysis.get("key_concepts", [])
            if not key_concepts:
                return 0.5
            
            total_score = 0.0
            for result in book_content:
                content = result.get('content', '').lower()
                for concept in key_concepts:
                    if concept.lower() in content:
                        total_score += result.get('score', 0.0)
            
            return min(total_score / len(book_content), 1.0) if book_content else 0.0
            
        except Exception as e:
            print(f"❌ Relevance score calculation failed: {e}")
            return 0.0
    
    def _estimate_difficulty(self, book_content: List[Dict]) -> str:
        """Estimate difficulty level of book content"""
        try:
            math_count = sum(1 for result in book_content if result.get('has_formulas', False))
            avg_semantic_density = np.mean([result.get('semantic_density', 0.0) for result in book_content])
            
            if math_count > len(book_content) * 0.5 or avg_semantic_density > 0.8:
                return "advanced"
            elif math_count > len(book_content) * 0.2 or avg_semantic_density > 0.5:
                return "intermediate"
            else:
                return "basic"
                
        except Exception as e:
            print(f"❌ Difficulty estimation failed: {e}")
            return "intermediate"
    
    def _extract_chapters_from_content(self, book_content: List[Dict]) -> List[str]:
        """Extract chapters from book content"""
        try:
            chapters = set()
            for result in book_content:
                chapters.update(result.get('chapters_found', []))
                chapters.update(result.get('sections_found', []))
            
            return sorted(list(chapters))[:20]  # Limit to prevent noise
            
        except Exception as e:
            print(f"❌ Chapter extraction from content failed: {e}")
            return [] 