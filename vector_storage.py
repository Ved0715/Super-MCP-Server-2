"""
🔍 Advanced Vector Storage Module
Pinecone integration with OpenAI embeddings for semantic search
"""

import logging
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
from datetime import datetime

import openai
import pinecone
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Enhanced document chunk with metadata"""
    id: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = None
    page_number: Optional[int] = None
    section: Optional[str] = None
    chunk_index: int = 0

@dataclass
class SearchResult:
    """Enhanced search result with relevance scoring"""
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    section: Optional[str] = None

class AdvancedVectorStorage:
    """Advanced vector storage with semantic search capabilities"""
    
    def __init__(self, config):
        self.config = config
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=config.PINECONE_API_KEY)
        self.index_name = config.PINECONE_INDEX_NAME
        self.dimensions = config.EMBEDDING_DIMENSIONS
        
        # Create index if it doesn't exist
        self._initialize_index()
        self.index = self.pc.Index(self.index_name)

    def _initialize_index(self):
        """Initialize Pinecone index"""
        try:
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {self.index_name}")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimensions,
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region=self.config.PINECONE_ENVIRONMENT
                    )
                )
                # Wait for index to be ready
                import time
                time.sleep(10)
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
                
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            raise

    async def process_and_store_document(self, 
                                       paper_content: Dict[str, Any],
                                       paper_id: str,
                                       namespace: Optional[str] = None) -> Dict[str, Any]:
        """Process document and store in vector database"""
        try:
            # Create intelligent chunks
            chunks = await self._create_intelligent_chunks(paper_content, paper_id)
            
            # Generate embeddings in batches
            embedded_chunks = await self._generate_embeddings_batch(chunks)
            
            # Store in Pinecone
            success = await self._store_chunks_in_pinecone(embedded_chunks, namespace or paper_id)
            
            return {
                "success": success,
                "chunks_created": len(chunks),
                "chunks_stored": len(embedded_chunks) if success else 0,
                "namespace": namespace or paper_id,
                "paper_id": paper_id
            }
            
        except Exception as e:
            logger.error(f"Error processing document {paper_id}: {e}")
            return {"success": False, "error": str(e)}

    async def semantic_search(self, 
                            query: str,
                            namespace: str,
                            top_k: int = 10,
                            filter_metadata: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Perform semantic search in the vector database"""
        try:
            # Generate query embedding
            query_embedding = await self._generate_single_embedding(query)
            
            # Search in Pinecone
            search_response = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False,
                filter=filter_metadata
            )
            
            # Parse results
            results = []
            for match in search_response.matches:
                results.append(SearchResult(
                    chunk_id=match.id,
                    content=match.metadata.get('content', ''),
                    score=match.score,
                    metadata=match.metadata,
                    page_number=match.metadata.get('page_number'),
                    section=match.metadata.get('section')
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return []

    async def contextual_search(self,
                              user_prompt: str,
                              namespace: str,
                              context_type: str = "general") -> List[SearchResult]:
        """Advanced contextual search based on user prompts"""
        try:
            # Enhance query based on context type
            enhanced_query = self._enhance_query_for_context(user_prompt, context_type)
            
            # Multi-level search
            results = []
            
            # 1. Direct semantic search
            direct_results = await self.semantic_search(enhanced_query, namespace, top_k=15)
            results.extend(direct_results)
            
            # 2. Section-specific search if needed
            if context_type in ["methodology", "results", "discussion"]:
                section_results = await self.semantic_search(
                    enhanced_query, 
                    namespace, 
                    top_k=10,
                    filter_metadata={"section": context_type}
                )
                results.extend(section_results)
            
            # 3. Remove duplicates and re-rank
            unique_results = self._deduplicate_and_rerank(results, enhanced_query)
            
            return unique_results[:20]  # Return top 20 results
            
        except Exception as e:
            logger.error(f"Error in contextual search: {e}")
            return []

    async def _create_intelligent_chunks(self, 
                                       paper_content: Dict[str, Any],
                                       paper_id: str) -> List[DocumentChunk]:
        """Create intelligent chunks with research paper awareness"""
        chunks = []
        sections = paper_content.get("sections", {})
        pages = paper_content.get("pages", [])
        
        chunk_id_counter = 0
        
        # Process by sections first (better semantic coherence)
        for section_name, section_content in sections.items():
            if not section_content or len(section_content.strip()) < 50:
                continue
                
            section_chunks = self._chunk_section_intelligently(
                section_content, 
                section_name,
                paper_id,
                chunk_id_counter
            )
            chunks.extend(section_chunks)
            chunk_id_counter += len(section_chunks)
        
        # Process remaining content by pages if sections are incomplete
        if len(chunks) < 3:  # If very few chunks from sections
            for page in pages:
                page_chunks = self._chunk_page_content(
                    page.get("text", ""),
                    page.get("page_number", 1),
                    paper_id,
                    chunk_id_counter
                )
                chunks.extend(page_chunks)
                chunk_id_counter += len(page_chunks)
        
        return chunks

    def _chunk_section_intelligently(self, 
                                   content: str, 
                                   section_name: str,
                                   paper_id: str,
                                   start_index: int) -> List[DocumentChunk]:
        """Chunk section content with academic awareness"""
        chunks = []
        
        # Split by sentences for better semantic coherence
        sentences = self._split_into_sentences(content)
        
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > self.config.CHUNK_SIZE and current_chunk:
                # Create chunk
                chunk = self._create_chunk(
                    current_chunk.strip(),
                    paper_id,
                    start_index + len(chunks),
                    section=section_name,
                    sentences=current_sentences
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_sentences = current_sentences[-2:] if len(current_sentences) > 2 else current_sentences
                current_chunk = " ".join(overlap_sentences) + " " + sentence
                current_sentences = overlap_sentences + [sentence]
            else:
                current_chunk += " " + sentence
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk.strip():
            chunk = self._create_chunk(
                current_chunk.strip(),
                paper_id,
                start_index + len(chunks),
                section=section_name,
                sentences=current_sentences
            )
            chunks.append(chunk)
        
        return chunks

    def _create_chunk(self, 
                     content: str, 
                     paper_id: str, 
                     chunk_index: int,
                     section: str = None,
                     page_number: int = None,
                     sentences: List[str] = None) -> DocumentChunk:
        """Create a document chunk with rich metadata"""
        
        # Generate unique ID
        chunk_id = f"{paper_id}_{chunk_index}_{hashlib.md5(content.encode()).hexdigest()[:8]}"
        
        # Create metadata
        metadata = {
            "paper_id": paper_id,
            "chunk_index": chunk_index,
            "content": content,
            "content_length": len(content),
            "word_count": len(content.split()),
            "created_at": datetime.now().isoformat()
        }
        
        if section:
            metadata["section"] = section
            metadata["section_type"] = self._classify_section_type(section)
        
        if page_number:
            metadata["page_number"] = page_number
            
        if sentences:
            metadata["sentence_count"] = len(sentences)
            metadata["contains_statistics"] = self._contains_statistical_content(content)
            metadata["contains_citations"] = self._contains_citations(content)
        
        return DocumentChunk(
            id=chunk_id,
            content=content,
            metadata=metadata,
            page_number=page_number,
            section=section,
            chunk_index=chunk_index
        )

    async def _generate_embeddings_batch(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for chunks in batches"""
        try:
            batch_size = 50  # OpenAI batch limit
            embedded_chunks = []
            
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk.content for chunk in batch]
                
                # Generate embeddings
                response = self.openai_client.embeddings.create(
                    model=self.config.EMBEDDING_MODEL,
                    input=texts
                )
                
                # Add embeddings to chunks
                for j, chunk in enumerate(batch):
                    chunk.embedding = response.data[j].embedding
                    embedded_chunks.append(chunk)
                
                # Rate limiting
                if i + batch_size < len(chunks):
                    await asyncio.sleep(0.1)
            
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []

    async def _generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating single embedding: {e}")
            return []

    async def _store_chunks_in_pinecone(self, chunks: List[DocumentChunk], namespace: str) -> bool:
        """Store chunks in Pinecone"""
        try:
            # Prepare vectors for upsert
            vectors = []
            for chunk in chunks:
                if chunk.embedding:
                    vectors.append({
                        "id": chunk.id,
                        "values": chunk.embedding,
                        "metadata": chunk.metadata
                    })
            
            # Upsert in batches
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch, namespace=namespace)
                
                # Rate limiting
                if i + batch_size < len(vectors):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Stored {len(vectors)} chunks in namespace {namespace}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing chunks in Pinecone: {e}")
            return False

    def _enhance_query_for_context(self, query: str, context_type: str) -> str:
        """Enhance query based on context type"""
        enhancements = {
            "methodology": f"{query} methodology methods approach experimental design",
            "results": f"{query} results findings data analysis statistical significance",
            "discussion": f"{query} discussion interpretation implications analysis",
            "conclusion": f"{query} conclusion summary findings implications future work",
            "introduction": f"{query} introduction background literature review context"
        }
        
        return enhancements.get(context_type, query)

    def _deduplicate_and_rerank(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Remove duplicates and rerank results"""
        seen_content = set()
        unique_results = []
        
        for result in results:
            content_hash = hashlib.md5(result.content.encode()).hexdigest()
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        # Sort by score (higher is better)
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results

    # Helper methods
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import nltk
        try:
            return nltk.sent_tokenize(text)
        except:
            # Fallback to simple splitting
            return [s.strip() for s in text.split('.') if s.strip()]

    def _classify_section_type(self, section_name: str) -> str:
        """Classify section type for better metadata"""
        section_lower = section_name.lower()
        
        if any(word in section_lower for word in ["abstract", "summary"]):
            return "summary"
        elif any(word in section_lower for word in ["introduction", "background"]):
            return "background"
        elif any(word in section_lower for word in ["method", "approach", "procedure"]):
            return "methodology"
        elif any(word in section_lower for word in ["result", "finding", "experiment"]):
            return "results"
        elif any(word in section_lower for word in ["discussion", "analysis", "interpretation"]):
            return "discussion"
        elif any(word in section_lower for word in ["conclusion", "summary", "future"]):
            return "conclusion"
        else:
            return "other"

    def _contains_statistical_content(self, content: str) -> bool:
        """Check if content contains statistical information"""
        stat_indicators = ["p <", "p=", "correlation", "significance", "confidence interval", "std", "mean"]
        return any(indicator in content.lower() for indicator in stat_indicators)

    def _contains_citations(self, content: str) -> bool:
        """Check if content contains citations"""
        import re
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[\d+[,\s\-\d]*\]',    # [1, 2, 3]
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _chunk_page_content(self, content: str, page_number: int, paper_id: str, start_index: int) -> List[DocumentChunk]:
        """Chunk page content as fallback"""
        chunks = []
        words = content.split()
        
        chunk_size_words = self.config.CHUNK_SIZE // 6  # Approximate words per chunk
        overlap_words = self.config.CHUNK_OVERLAP // 6
        
        for i in range(0, len(words), chunk_size_words - overlap_words):
            chunk_words = words[i:i + chunk_size_words]
            chunk_content = " ".join(chunk_words)
            
            if len(chunk_content.strip()) > 50:  # Only create substantial chunks
                chunk = self._create_chunk(
                    chunk_content,
                    paper_id,
                    start_index + len(chunks),
                    page_number=page_number
                )
                chunks.append(chunk)
        
        return chunks

    def get_namespace_stats(self, namespace: str) -> Dict[str, Any]:
        """Get statistics for a namespace"""
        try:
            stats = self.index.describe_index_stats()
            namespace_stats = stats.namespaces.get(namespace, {})
            
            return {
                "vector_count": namespace_stats.get("vector_count", 0),
                "namespace": namespace,
                "index_fullness": stats.index_fullness,
                "total_vector_count": stats.total_vector_count
            }
        except Exception as e:
            logger.error(f"Error getting namespace stats: {e}")
            return {}

    def delete_namespace(self, namespace: str) -> bool:
        """Delete all vectors in a namespace"""
        try:
            self.index.delete(delete_all=True, namespace=namespace)
            logger.info(f"Deleted namespace: {namespace}")
            return True
        except Exception as e:
            logger.error(f"Error deleting namespace {namespace}: {e}")
            return False 

    async def search_in_namespace(
        self,
        namespace: str,
        query: str,
        max_results: int = 20,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for content in specific namespace
        
        Args:
            namespace: Vector database namespace (user_{user_id}_doc_{doc_id})
            query: Search query
            max_results: Maximum results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of matching content chunks
        """
        import time
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Starting vector search in namespace: {namespace}")
            logger.info(f"📝 Query: '{query[:100]}{'...' if len(query) > 100 else ''}'")
            logger.info(f"⚙️  Parameters: max_results={max_results}, threshold={similarity_threshold}")
            
            if not self.index:
                logger.error("❌ Pinecone index not initialized")
                return []
            
            # Generate query embedding
            logger.info(f"🤖 Generating embedding for query...")
            embedding_start_time = time.time()
            
            query_embedding = await self._generate_single_embedding(query)
            
            embedding_duration = time.time() - embedding_start_time
            logger.info(f"✅ Query embedding generated in {embedding_duration:.2f}s")
            
            if not query_embedding:
                logger.error("❌ Failed to generate query embedding")
                return []
            
            # Search in Pinecone with namespace filter
            logger.info(f"🔎 Querying Pinecone index with namespace filter...")
            pinecone_start_time = time.time()
            
            search_results = self.index.query(
                vector=query_embedding,
                top_k=max_results,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            
            pinecone_duration = time.time() - pinecone_start_time
            logger.info(f"✅ Pinecone query completed in {pinecone_duration:.2f}s")
            logger.info(f"📊 Raw results from Pinecone: {len(search_results.matches)} matches")
            
            # Filter by similarity threshold
            logger.info(f"🔧 Filtering results by similarity threshold {similarity_threshold}...")
            filtered_results = []
            scores = []
            
            for match in search_results.matches:
                scores.append(match.score)
                if match.score >= similarity_threshold:
                    filtered_results.append({
                        'content': match.metadata.get('content', ''),
                        'metadata': match.metadata,
                        'score': match.score,
                        'chunk_id': match.id
                    })
            
            total_duration = time.time() - start_time
            
            # Log detailed results
            if scores:
                avg_score = sum(scores) / len(scores)
                max_score = max(scores)
                min_score = min(scores)
                logger.info(f"📈 Score statistics: avg={avg_score:.3f}, max={max_score:.3f}, min={min_score:.3f}")
            
            logger.info(f"✅ Vector search completed in {total_duration:.2f}s")
            logger.info(f"📊 Results: {len(filtered_results)}/{len(search_results.matches)} chunks passed threshold")
            
            if filtered_results:
                total_content_chars = sum(len(r['content']) for r in filtered_results)
                logger.info(f"📝 Total content retrieved: {total_content_chars} characters")
                
                # Log top 3 results summary
                logger.info(f"🏆 Top results preview:")
                for i, result in enumerate(filtered_results[:3]):
                    content_preview = result['content'][:100] + "..." if len(result['content']) > 100 else result['content']
                    logger.info(f"   {i+1}. Score: {result['score']:.3f} | Content: {content_preview}")
            else:
                logger.warning(f"⚠️  No results found above threshold {similarity_threshold}")
                if scores:
                    logger.info(f"💡 Suggestion: Consider lowering threshold (best score was {max(scores):.3f})")
            
            # Performance breakdown
            logger.info(f"⏱️  Search performance breakdown:")
            logger.info(f"   - Embedding generation: {embedding_duration:.2f}s ({embedding_duration/total_duration*100:.1f}%)")
            logger.info(f"   - Pinecone query: {pinecone_duration:.2f}s ({pinecone_duration/total_duration*100:.1f}%)")
            logger.info(f"   - Result processing: {(total_duration-embedding_duration-pinecone_duration):.2f}s")
            
            return filtered_results
            
        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(f"❌ Error searching namespace {namespace} after {total_duration:.2f}s: {e}")
            logger.error(f"🔍 Error details: {str(e)}")
            import traceback
            logger.error(f"📋 Full traceback: {traceback.format_exc()}")
            return [] 