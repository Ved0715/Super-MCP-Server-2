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

from config import config
from processors.universal_document_processor import ProcessedChunk
from .prompt_templates import detect_query_type, format_system_prompt, get_prompt_template

@dataclass
class CoTStep:
    """Represents a single step in the Chain of Thought process"""
    step_number: int
    step_name: str
    reasoning: str
    action: str
    result: str
    confidence: float
    tokens_used: int

@dataclass
class CoTSearchResult:
    """Enhanced search result with Chain of Thought reasoning"""
    chunk_id: str
    content: str
    metadata: Dict
    relevance_score: float
    reasoning: str
    confidence: float

class ChainOfThoughtKBRetriever:
    """
    Advanced Knowledge Base Retriever with comprehensive Chain of Thought reasoning.
    Optimized for cost-efficiency while maintaining thorough analysis.
    """
    
    def __init__(self):
        """Initialize the CoT KB retriever with cost optimization"""
        print("🧠 Initializing Chain of Thought Knowledge Base Retriever...")
        
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Cost optimization settings
        self.cost_model = "gpt-4o-mini"  # Use mini for reasoning steps to save cost
        self.final_model = config.response_model  # Use full model only for final response
        self.max_reasoning_tokens = 300  # Limit reasoning tokens
        self.max_final_tokens = 800  # Limit final response tokens
        
        # CoT tracking
        self.cot_steps = []
        self.total_tokens_used = 0
        self.step_counter = 0
        
        # Initialize components
        self.cross_encoder = None
        self.bm25_index = None
        self.chunks_data = {}
        
        self.setup_models()
        self.setup_pinecone()
        
        print("✅ Chain of Thought KB Retriever initialized with cost optimization")
    
    def setup_models(self):
        """Initialize cross-encoder for reranking"""
        if config.cross_encoder_reranking:
            try:
                print("🔧 Loading cross-encoder for CoT reranking...")
                self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
                print("✅ Cross-encoder loaded successfully")
            except Exception as e:
                print(f"⚠️ Cross-encoder loading failed: {e}")
    
    def setup_pinecone(self):
        """Setup or connect to Pinecone index"""
        try:
            self.index = self.pc.Index(config.index_name)
            print(f"✅ Connected to Pinecone index: {config.index_name}")
        except Exception as e:
            print(f"❌ Pinecone setup failed: {e}")
            self.index = None
    
    def reset_cot_tracking(self):
        """Reset Chain of Thought tracking for new query"""
        self.cot_steps = []
        self.total_tokens_used = 0
        self.step_counter = 0
    
    async def add_cot_step(self, step_name: str, reasoning: str, action: str, result: str, confidence: float = 0.8) -> CoTStep:
        """Add a step to the Chain of Thought with cost tracking"""
        self.step_counter += 1
        
        # Estimate tokens used (rough estimation to avoid API calls)
        estimated_tokens = len(reasoning.split()) + len(action.split()) + len(result.split())
        self.total_tokens_used += estimated_tokens
        
        step = CoTStep(
            step_number=self.step_counter,
            step_name=step_name,
            reasoning=reasoning,
            action=action,
            result=result,
            confidence=confidence,
            tokens_used=estimated_tokens
        )
        
        self.cot_steps.append(step)
        print(f"🧠 CoT Step {self.step_counter}: {step_name} (Confidence: {confidence:.2f})")
        
        return step
    
    async def comprehensive_thinking_and_analysis_cot(self, query: str) -> Dict[str, Any]:
        """
        COMPREHENSIVE Step 1: Enhanced thinking phase with question generation
        Combines thinking, question generation, and analysis in single API call for cost efficiency
        """
        reasoning = f"I need to think step by step and ask myself questions before analyzing: '{query}'"
        
        comprehensive_prompt = f"""
        **COMPREHENSIVE CHAIN OF THOUGHT ANALYSIS**
        
        Query: "{query}"
        
        **PHASE 1: THINKING STEP BY STEP**
        Think step by step. Ask yourself questions before answering.
        
        Initial Thinking:
        - What is the user really asking for?
        - What type of response would be most helpful?
        - What context do I need to understand this properly?
        
        Deep Analysis:
        - What are the key concepts involved?
        - What level of complexity is appropriate?
        - What knowledge domains are relevant?
        
        **PHASE 2: QUESTION GENERATION**
        What questions should I ask myself to answer this comprehensively?
        
        Generate 4-6 specific sub-questions that, when answered completely, would provide a comprehensive response.
        Focus on: core concepts, context/background, applications, relationships, practical implications.
        
        **PHASE 3: QUERY ANALYSIS**
        Based on your thinking and questions, analyze the query systematically.
        
        Available books: Data Science - 40 Algorithms, Data Science - Foundations, Data Science - Kelleher & Tierney, Data Science - Theories Models Analytics
        
        **RESPOND IN VALID JSON ONLY (no extra text):**
        {{
            "thinking_phase": {{
                "initial_thoughts": ["thought 1", "thought 2", "thought 3"],
                "deep_analysis": ["analysis 1", "analysis 2", "analysis 3"],
                "reasoning_approach": "description of how I'll approach this"
            }},
            "question_generation": {{
                "sub_questions": [
                    "What is the core concept the user wants to understand?",
                    "What practical applications should I cover?",
                    "What examples would be most helpful?",
                    "How does this relate to other concepts?",
                    "What level of detail is appropriate?"
                ],
                "question_rationale": "why these questions will lead to comprehensive coverage"
            }},
            "query_analysis": {{
                "query_type": "concept_explanation",
                "key_concepts": ["list", "of", "concepts"],
                "target_books": ["specific book names if relevant"],
                "search_terms": ["optimized", "search", "terms"],
                "complexity": "intermediate",
                "expected_depth": "moderate"
            }},
            "confidence": 0.85
        }}
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.cost_model,
                messages=[
                    {"role": "system", "content": "You are a JSON response generator. Always respond with valid JSON only. No extra text or explanation."},
                    {"role": "user", "content": comprehensive_prompt}
                ],
                max_tokens=self.max_reasoning_tokens * 2,  # Allow more tokens for comprehensive analysis
                temperature=0.2
            )
            
            response_content = response.choices[0].message.content.strip()
            
            # Clean up response to extract JSON
            if "```json" in response_content:
                response_content = response_content.split("```json")[1].split("```")[0].strip()
            elif "```" in response_content:
                response_content = response_content.split("```")[1].strip()
            
            # Remove any extra text before/after JSON
            start_idx = response_content.find('{')
            end_idx = response_content.rfind('}') + 1
            if start_idx >= 0 and end_idx > start_idx:
                response_content = response_content[start_idx:end_idx]
            
            result = json.loads(response_content)
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Add CoT step
            await self.add_cot_step(
                step_name="Phase 1-3: Thinking, Questions & Analysis",
                reasoning=reasoning,
                action="Completed thinking phase, question generation, and query analysis",
                result=f"Query type: {result['query_analysis']['query_type']}, Generated {len(result['question_generation']['sub_questions'])} sub-questions",
                confidence=result.get('confidence', 0.8)
            )
            
            return result
            
        except Exception as e:
            print(f"❌ Query analysis failed: {e}")
            # Enhanced fallback analysis with better default questions
            fallback_result = {
                "thinking_phase": {
                    "initial_thoughts": [
                        "User is asking about a fundamental concept",
                        "Need to provide comprehensive explanation", 
                        "Should include practical examples"
                    ],
                    "deep_analysis": [
                        "Query involves core data science concepts",
                        "Requires balanced technical depth",
                        "Should connect theory to practice"
                    ],
                    "reasoning_approach": "Systematic explanation with examples and applications"
                },
                "question_generation": {
                    "sub_questions": [
                        "What is the fundamental definition and core concept?",
                        "What are the key types and categories involved?",
                        "How does this work in practice with real examples?",
                        "What are the main applications and use cases?",
                        "How does this relate to other important concepts?",
                        "What should someone know to get started?"
                    ],
                    "question_rationale": "These questions ensure comprehensive coverage from basics to applications"
                },
                "query_analysis": {
                    "query_type": "concept_explanation",
                    "key_concepts": query.lower().split()[:5],
                    "target_books": [],
                    "search_terms": [query, *query.split()[:3]],
                    "complexity": "intermediate",
                    "expected_depth": "moderate"
                },
                "confidence": 0.5
            }
            
            await self.add_cot_step(
                step_name="Query Intent Analysis (Fallback)",
                reasoning="API analysis failed, using enhanced rule-based fallback",
                action="Applied enhanced fallback query analysis with comprehensive questions",
                result=f"Generated {len(fallback_result['question_generation']['sub_questions'])} sub-questions for comprehensive analysis",
                confidence=0.5
            )
            
            return fallback_result
    
    async def plan_retrieval_strategy_cot(self, query_analysis: Dict) -> Dict[str, Any]:
        """
        Step 2: Plan the retrieval strategy based on query analysis
        No API call - rule-based to save cost
        """
        reasoning = "Based on the query analysis, I need to determine the optimal retrieval strategy."
        
        query_type = query_analysis.get('query_type', 'concept_explanation')
        complexity = query_analysis.get('complexity', 'intermediate')
        key_concepts = query_analysis.get('key_concepts', [])
        
        # Rule-based strategy planning (no API cost)
        if query_type == 'meta_query':
            strategy = {
                "primary_method": "hardcoded_inventory",
                "secondary_method": "none",
                "search_depth": "shallow",
                "expected_chunks": 0,
                "reasoning": "Meta query about knowledge base - use hardcoded responses"
            }
        elif query_type == 'study_plan':
            strategy = {
                "primary_method": "structured_curriculum",
                "secondary_method": "book_analysis",
                "search_depth": "comprehensive",
                "expected_chunks": 10,
                "reasoning": "Study plan requires structured approach with multiple books"
            }
        elif query_type == 'topic_location':
            strategy = {
                "primary_method": "targeted_search",
                "secondary_method": "chapter_analysis",
                "search_depth": "focused",
                "expected_chunks": 5,
                "reasoning": "Topic location needs precise search in specific books"
            }
        else:  # concept_explanation, comparison
            if complexity == 'basic':
                strategy = {
                    "primary_method": "semantic_search",
                    "secondary_method": "keyword_search",
                    "search_depth": "moderate",
                    "expected_chunks": 3,
                    "reasoning": "Basic concept needs clear, focused explanation"
                }
            elif complexity == 'advanced':
                strategy = {
                    "primary_method": "hybrid_search",
                    "secondary_method": "multi_book_synthesis",
                    "search_depth": "deep",
                    "expected_chunks": 8,
                    "reasoning": "Advanced concept requires comprehensive coverage"
                }
            else:  # intermediate
                strategy = {
                    "primary_method": "semantic_search",
                    "secondary_method": "related_concept_expansion",
                    "search_depth": "moderate",
                    "expected_chunks": 5,
                    "reasoning": "Intermediate concept needs balanced explanation"
                }
        
        await self.add_cot_step(
            step_name="Phase 4: Retrieval Strategy Planning",
            reasoning=reasoning,
            action=f"Selected {strategy['primary_method']} with {strategy['search_depth']} depth",
            result=f"Strategy: {strategy['reasoning']}",
            confidence=0.9
        )
        
        return strategy
    
    async def execute_retrieval_cot(self, query: str, query_analysis: Dict, strategy: Dict) -> List[CoTSearchResult]:
        """
        Step 3: Execute the planned retrieval strategy
        Optimized to minimize API calls
        """
        reasoning = f"Executing {strategy['primary_method']} retrieval strategy"
        
        search_results = []
        
        try:
            if strategy['primary_method'] == 'hardcoded_inventory':
                # No search needed - use hardcoded responses
                result_content = self.get_hardcoded_inventory_response(query)
                search_results = [CoTSearchResult(
                    chunk_id="hardcoded-001",
                    content=result_content,
                    metadata={"source": "hardcoded", "type": "inventory"},
                    relevance_score=1.0,
                    reasoning="Used hardcoded inventory for meta query",
                    confidence=1.0
                )]
                
            elif strategy['primary_method'] == 'structured_curriculum':
                # Use existing study plan generation
                result_content = await self.generate_study_plan_response(query_analysis)
                search_results = [CoTSearchResult(
                    chunk_id="curriculum-001",
                    content=result_content,
                    metadata={"source": "structured", "type": "study_plan"},
                    relevance_score=1.0,
                    reasoning="Generated structured study plan",
                    confidence=0.9
                )]
                
            else:
                # Perform actual vector search
                search_terms = query_analysis.get('search_terms', [query])
                expected_chunks = strategy.get('expected_chunks', 5)
                
                # Primary search
                primary_results = await self.vector_search(search_terms[0], expected_chunks)
                search_results.extend(primary_results)
                
                # Secondary search if needed and strategy allows
                if len(search_results) < expected_chunks and len(search_terms) > 1:
                    secondary_results = await self.vector_search(search_terms[1], 2)
                    search_results.extend(secondary_results)
            
            await self.add_cot_step(
                step_name="Phase 5: Retrieval Execution",
                reasoning=reasoning,
                action=f"Executed {strategy['primary_method']} search",
                result=f"Retrieved {len(search_results)} relevant chunks",
                confidence=0.85
            )
            
            return search_results[:strategy.get('expected_chunks', 10)]  # Limit results
            
        except Exception as e:
            print(f"❌ Retrieval execution failed: {e}")
            
            await self.add_cot_step(
                step_name="Retrieval Execution (Failed)",
                reasoning="Retrieval failed, using fallback",
                action="Applied fallback retrieval",
                result="Limited results obtained",
                confidence=0.3
            )
            
            return []
    
    async def vector_search(self, query: str, top_k: int) -> List[CoTSearchResult]:
        """Perform vector search with CoT reasoning"""
        if not self.index:
            return []
        
        try:
            # Generate embedding
            embedding_response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=query,
                model=config.embedding_model,
                dimensions=config.embedding_dimension
            )
            
            query_embedding = embedding_response.data[0].embedding
            self.total_tokens_used += embedding_response.usage.total_tokens
            
            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=config.namespace,
                include_metadata=True,
                include_values=False
            )
            
            # Convert to CoT search results
            cot_results = []
            for i, match in enumerate(results.matches):
                reasoning = f"Chunk ranked #{i+1} with similarity {match.score:.3f} - contains relevant information about the query"
                
                cot_result = CoTSearchResult(
                    chunk_id=match.id,
                    content=match.metadata.get('text', ''),
                    metadata=match.metadata,
                    relevance_score=match.score,
                    reasoning=reasoning,
                    confidence=min(match.score * 1.2, 1.0)  # Boost confidence slightly
                )
                cot_results.append(cot_result)
            
            return cot_results
            
        except Exception as e:
            print(f"❌ Vector search failed: {e}")
            return []
    
    async def enhanced_synthesis_with_quality_check_cot(self, search_results: List[CoTSearchResult], query: str, comprehensive_result: Dict) -> str:
        """
        COMPREHENSIVE Step 4: Enhanced synthesis with sequential question answering and quality check
        Combines multiple phases in single API call for cost efficiency
        """
        reasoning = "Now I need to answer sub-questions sequentially, synthesize comprehensively, and perform quality check"
        
        if not search_results:
            await self.add_cot_step(
                step_name="Enhanced Synthesis & Quality Check",
                reasoning="No search results to analyze",
                action="Generated fallback response with quality validation",
                result="Informed user of limited information",
                confidence=0.8
            )
            return "Based on the available knowledge base content: I have limited information about this topic in the current knowledge base. Please try rephrasing your question or asking about specific data science concepts that might be covered in our available books: Data Science - 40 Algorithms, Data Science - Foundations, Data Science - Kelleher & Tierney, or Data Science - Theories Models Analytics."
        
        # Extract sub-questions from comprehensive analysis
        sub_questions = comprehensive_result.get('question_generation', {}).get('sub_questions', [])
        query_analysis = comprehensive_result.get('query_analysis', {})
        
        # Prepare comprehensive context from search results
        context_chunks = []
        source_books = set()
        
        for i, result in enumerate(search_results[:8]):  # Use more results for comprehensive response
            book_name = result.metadata.get('book_name', 'Unknown Book')
            chapter = result.metadata.get('section_title', 'Unknown Chapter')
            page_number = result.metadata.get('page_references')
            source_books.add(book_name)
            
            context_chunks.append(f"""
[ {book_name}, {chapter}]:
{result.content}
(Relevance Score: {result.relevance_score:.3f})
""")
        
        combined_context = "\n".join(context_chunks)
        
        # Create comprehensive synthesis prompt with all phases
        comprehensive_synthesis_prompt = f"""
        **COMPREHENSIVE CHAIN OF THOUGHT SYNTHESIS WITH QUALITY CHECK**
        
        Original Query: "{query}"
        Query Type: {query_analysis.get('query_type', 'concept_explanation')}
        Complexity Level: {query_analysis.get('complexity', 'intermediate')}
        
        **AVAILABLE KNOWLEDGE BASE SOURCES:**
        From books: {', '.join(source_books)}
        Total relevant chunks found: {len(search_results)}
        
        **PHASE 1: SEQUENTIAL QUESTION ANSWERING**
        Answer each question using available knowledge, then proceed to next.
        
        Sub-Questions to Answer Comprehensively:
        {chr(10).join([f"{i+1}. {q}" for i, q in enumerate(sub_questions)])}
        
        **KNOWLEDGE BASE CONTEXT:**
        {combined_context}
        
        **INSTRUCTIONS FOR EACH QUESTION:**
        - Use ONLY information from the knowledge base context above
        - Be detailed and specific - aim for 2-3 paragraphs per question
        - Cite specific sources with book names
        - Include examples, algorithms, or practical details when available
        - If context is insufficient for a question, state this clearly
        - Build upon previous answers where relevant
        - Focus on practical understanding and real-world applications
        - Take care of the structure of the responce.
        - If needed the responce should be mix of paras, poinsts, concluson, description ...
        - Do not add generic summaries or meta-commentary.
        - Do not use phrases like 'This comprehensive overview...' or similar.
        - Be create about the structure of the content, the way it Apperance and Readability. You can change the little wording if necessary.
        - It should look like you are answering the quary.
        
        **PHASE 2: SYNTHESIS PHASE**
        Combine all answers into a coherent, comprehensive response for the user.
        
        Requirements:
        1. Create a detailed, well-structured response (minimum 4-5 paragraphs)
        2. Start with fundamental concepts and build complexity
        3. Include practical examples and applications from the knowledge base
        4. Connect different aspects and show relationships
        5. Use specific terminology and technical details from the sources
        6. Maintain logical flow and smooth transitions
        7. Ensure comprehensive coverage addressing all aspects of the query
        
        **PHASE 3: QUALITY CHECK**
        Review the synthesized answer for accuracy, completeness, and clarity.
        
        Quality Criteria:
        - ACCURACY: Only uses information from knowledge base context
        - COMPLETENESS: Comprehensively addresses the user's query
        - DEPTH: Provides sufficient technical detail and examples
        - CLARITY: Well-organized, logical flow, easy to understand
        - RELEVANCE: Stays focused on what was asked
        - CITATIONS: Properly references knowledge base sources
        - LENGTH: Substantial response with good coverage
        
        **CRITICAL CONSTRAINTS:**
        - ONLY use information from the knowledge base context above
        - DO NOT add external knowledge or general assumptions
        - Must start with: "Based on the available knowledge base content:"
        - Include specific book citations and examples
        - Aim for comprehensive 4-6 paragraph response minimum
        - If information is incomplete, state this clearly but provide all available details
        
        **OUTPUT FORMAT:**
        
        Sequential Question Analysis:
        Q1: [Detailed answer with citations - 2-3 paragraphs]
        Q2: [Detailed answer with citations - 2-3 paragraphs]
        [Continue for all questions...]
        
        Comprehensive Synthesis:
        [Detailed integration of all answers - 4-6 paragraphs minimum]
        
        Quality Assessment:
        - Accuracy: [Verified knowledge base only]
        - Completeness: [How well query is addressed]
        - Technical Depth: [Level of detail provided]
        - Practical Value: [Real-world applicability]
        - Final Quality Score: [1-10]
        
        **FINAL COMPREHENSIVE RESPONSE:**
        Based on the available knowledge base content: [Your detailed, comprehensive final response that synthesizes all the above - minimum 4-6 substantial paragraphs with specific citations, examples, and technical details from the knowledge base]
        
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.final_model,  # Use full model for comprehensive synthesis
                messages=[
                    {"role": "system", "content": "You are an expert knowledge synthesizer. Create comprehensive, detailed responses using only the provided knowledge base context. Always include specific citations and examples."},
                    {"role": "user", "content": comprehensive_synthesis_prompt}
                ],
                max_tokens=2000,  # Increase token limit for comprehensive responses
                temperature=0.3
            )
            
            full_response = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Extract the final comprehensive response
            final_response = ""
            if "**FINAL COMPREHENSIVE RESPONSE:**" in full_response:
                final_response = full_response.split("**FINAL COMPREHENSIVE RESPONSE:**")[1].strip()
            elif "FINAL COMPREHENSIVE RESPONSE:" in full_response:
                final_response = full_response.split("FINAL COMPREHENSIVE RESPONSE:")[1].strip()
            elif "Based on the available knowledge base content:" in full_response:
                # Find the last occurrence which should be the final response
                parts = full_response.split("Based on the available knowledge base content:")
                final_response = "Based on the available knowledge base content:" + parts[-1].strip()
            else:
                # If no clear final response marker, use the whole response but add prefix
                final_response = "Based on the available knowledge base content:\n" + full_response
            
            # Ensure minimum quality and length
            if len(final_response) < 500:  # If response too short, enhance it
                enhanced_response = await self.enhance_short_response(final_response, search_results, query)
                final_response = enhanced_response
            
            await self.add_cot_step(
                step_name="Phase 6: Sequential Q&A, Synthesis & Quality Check",
                reasoning=reasoning,
                action="Completed comprehensive sequential Q&A, synthesis, and quality validation",
                result=f"Generated detailed response ({len(final_response)} chars) with citations from {len(source_books)} books",
                confidence=0.9
            )
            
            return final_response
            
        except Exception as e:
            print(f"❌ Synthesis failed: {e}")
            
            # Enhanced fallback synthesis using actual search results
            fallback_response = "Based on the available knowledge base content:\n\n"
            
            if search_results:
                top_sources = search_results[:3]  # Use top 3 results
                
                for i, result in enumerate(top_sources):
                    book_name = result.metadata.get('book_name', 'the knowledge base')
                    chapter = result.metadata.get('chapter', '')
                    chapter_text = f" from {chapter}" if chapter else ""
                    
                    fallback_response += f"According to {book_name}{chapter_text}:\n"
                    fallback_response += f"{result.content}\n\n"
                
                fallback_response += f"This information comes from {len(set(r.metadata.get('book_name', '') for r in search_results))} different books in our knowledge base, providing comprehensive coverage of your question."
            else:
                fallback_response += "I have limited information about this specific topic in the current knowledge base. The available books cover various data science concepts including algorithms, foundations, practical applications, and advanced analytics."
            
            await self.add_cot_step(
                step_name="Enhanced Synthesis & Quality Check (Fallback)",
                reasoning="AI synthesis failed, using enhanced fallback with actual search results",
                action="Generated comprehensive response from search results directly",
                result=f"Fallback response using {len(search_results)} search results",
                confidence=0.7
            )
            
            return fallback_response
    
    async def enhance_short_response(self, short_response: str, search_results: List[CoTSearchResult], query: str) -> str:
        """Enhance a short response with additional context from search results"""
        additional_context = []
        
        for result in search_results[:5]:
            book_name = result.metadata.get('section_title', 'Unknown Book')
            page = result.metadata.get('page_references', 'Unknown Page')
            additional_context.append(f"[From {book_name}, Page{page}]\n: {result.content}...")
        
        enhanced_response = short_response + "\n\nAdditional relevant information from the knowledge base:\n\n"
        enhanced_response += "\n\n".join(additional_context)
        
        return enhanced_response
    
    async def answer_question_with_cot(self, question: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Main method: Answer question using comprehensive Chain of Thought reasoning
        Optimized for cost while maintaining thoroughness
        """
        print(f"\n🧠 Starting Chain of Thought Knowledge Base Retrieval for: '{question}'")
        print(f"💰 Cost optimization: Using {self.cost_model} for reasoning, {self.final_model} for final response")
        
        start_time = time.time()
        self.reset_cot_tracking()
        
        try:
            # Step 1-3: Comprehensive thinking, questions, and analysis with CoT (Combined for efficiency)
            comprehensive_result = await self.comprehensive_thinking_and_analysis_cot(question)
            query_analysis = comprehensive_result.get('query_analysis', {})
            
            # Step 4: Plan retrieval strategy (no API cost)
            strategy = await self.plan_retrieval_strategy_cot(query_analysis)
            
            # Step 5: Execute retrieval with CoT
            search_results = await self.execute_retrieval_cot(question, query_analysis, strategy)
            
            # Step 6: Enhanced synthesis with quality check using CoT
            synthesis_response = await self.enhanced_synthesis_with_quality_check_cot(search_results, question, comprehensive_result)
            
            # Step 7: Generate final enhanced response based on all CoT analysis
            final_response = await self.generate_final_enhanced_response_cot(synthesis_response, comprehensive_result, search_results, question)
            
            execution_time = time.time() - start_time
            
            # Generate Chain of Thought summary
            cot_summary = self.generate_cot_summary()
            
            print(f"✅ Chain of Thought retrieval completed in {execution_time:.2f}s")
            print(f"💰 Total tokens used: {self.total_tokens_used}")
            print(f"🧠 Chain of Thought steps: {len(self.cot_steps)}")
            
            return {
                "success": True,
                "query": question,
                "response": final_response,
                "chain_of_thought": {
                    "comprehensive_analysis": {
                        "thinking_phase": comprehensive_result.get("thinking_phase", {}),
                        "question_generation": comprehensive_result.get("question_generation", {}),
                        "query_analysis": comprehensive_result.get("query_analysis", {})
                    },
                    "steps": [
                        {
                            "step": step.step_number,
                            "name": step.step_name,
                            "reasoning": step.reasoning,
                            "action": step.action,
                            "result": step.result,
                            "confidence": step.confidence
                        }
                        for step in self.cot_steps
                    ],
                    "summary": cot_summary,
                    "total_steps": len(self.cot_steps),
                    "average_confidence": sum(step.confidence for step in self.cot_steps) / len(self.cot_steps) if self.cot_steps else 0
                },
                "metadata": {
                    "execution_time": execution_time,
                    "total_tokens_used": self.total_tokens_used,
                    "search_results_count": len(search_results),
                    "query_analysis": query_analysis,
                    "retrieval_strategy": strategy
                }
            }
            
        except Exception as e:
            print(f"❌ Chain of Thought retrieval failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": question,
                "chain_of_thought": {
                    "steps": [
                        {
                            "step": step.step_number,
                            "name": step.step_name,
                            "reasoning": step.reasoning,
                            "action": step.action,
                            "result": step.result,
                            "confidence": step.confidence
                        }
                        for step in self.cot_steps
                    ],
                    "summary": "Process failed during execution",
                    "total_steps": len(self.cot_steps)
                }
            }
    
    async def generate_final_enhanced_response_cot(self, synthesis_response: str, comprehensive_result: Dict, search_results: List[CoTSearchResult], query: str) -> str:
        """
        FINAL Step 7: Generate the ultimate enhanced response considering all CoT analysis
        """
        reasoning = "Creating the final enhanced response by considering all Chain of Thought analysis and synthesis"
        
        # Extract key insights from the comprehensive analysis
        thinking_phase = comprehensive_result.get('thinking_phase', {})
        question_generation = comprehensive_result.get('question_generation', {})
        query_analysis = comprehensive_result.get('query_analysis', {})
        
        # Prepare context about the analysis process
        analysis_context = f"""
        **CoT Analysis Summary:**
        - Query Type: {query_analysis.get('query_type', 'concept_explanation')}
        - Complexity: {query_analysis.get('complexity', 'intermediate')}
        - Key Concepts: {', '.join(query_analysis.get('key_concepts', []))}
        - Sub-questions Generated: {len(question_generation.get('sub_questions', []))}
        - Search Results: {len(search_results)} chunks from knowledge base
        - Books Referenced: {len(set(r.metadata.get('book_name', '') for r in search_results))}
        
        **Reasoning Approach:** {thinking_phase.get('reasoning_approach', 'Systematic analysis')}
        """
        
        final_enhancement_prompt = f"""
        **FINAL CHAIN OF THOUGHT RESPONSE ENHANCEMENT**
        
        Original Query: "{query}"
        
        **CONTEXT FROM COMPREHENSIVE ANALYSIS:**
        {analysis_context}
        
        **SYNTHESIZED RESPONSE TO ENHANCE:**
        {synthesis_response}
        
        **YOUR TASK:**
        Create the ultimate final response that takes into account ALL the Chain of Thought analysis. This should be:
        
        1. **More Comprehensive**: Build upon the synthesis but add even more depth and insight
        2. **Better Structured**: Organize information in the most logical, flowing manner
        3. **More Engaging**: Make it more readable and compelling while maintaining technical accuracy
        4. **Complete Coverage**: Ensure all aspects of the original query are thoroughly addressed
        5. **Enhanced Examples**: Add more specific details and examples from the knowledge base
        6. **Stronger Connections**: Better link concepts and show relationships between ideas
        
        **CRITICAL REQUIREMENTS:**
        - Must start with: "Based on the available knowledge base content:"
        - Use ONLY information from the knowledge base (no external knowledge)
        - Include specific book citations and references
        - Maintain all technical accuracy from the synthesis
        - Enhance readability and flow significantly
        - Aim for comprehensive coverage (minimum 5-6 substantial paragraphs)
        - Include practical implications and real-world relevance
        
        **ENHANCEMENT GOALS:**
        - Transform synthesis into the definitive answer on this topic
        - Make complex concepts accessible and engaging
        - Provide a response that fully satisfies the user's query
        - Create a response worthy of the comprehensive CoT analysis performed
        
        Generate the final enhanced response now:
        """
        
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.final_model,
                messages=[
                    {"role": "system", "content": "You are an expert knowledge synthesizer creating the ultimate final response. Enhance the provided synthesis to create the best possible answer using only knowledge base content."},
                    {"role": "user", "content": final_enhancement_prompt}
                ],
                max_tokens=2500,  # Allow more tokens for the final enhanced response
                temperature=0.1
            )
            
            final_enhanced_response = response.choices[0].message.content.strip()
            tokens_used = response.usage.total_tokens
            self.total_tokens_used += tokens_used
            
            # Ensure proper prefix
            if not final_enhanced_response.startswith("Based on the available knowledge base content:"):
                final_enhanced_response = "Based on the available knowledge base content:\n\n" + final_enhanced_response
            
            await self.add_cot_step(
                step_name="Phase 7: Final Enhanced Response Generation",
                reasoning=reasoning,
                action="Generated ultimate final response considering all CoT analysis",
                result=f"Created enhanced final response ({len(final_enhanced_response)} chars) incorporating all insights",
                confidence=0.95
            )
            
            return final_enhanced_response
            
        except Exception as e:
            print(f"❌ Final enhancement failed: {e}")
            
            # Fall back to synthesis response with enhancement note
            fallback_response = synthesis_response
            if not fallback_response.startswith("Based on the available knowledge base content:"):
                fallback_response = "Based on the available knowledge base content:\n\n" + fallback_response
            
            await self.add_cot_step(
                step_name="Final Enhanced Response Generation (Fallback)",
                reasoning="Enhancement failed, using synthesis response",
                action="Applied fallback final response",
                result="Using synthesis response as final output",
                confidence=0.8
            )
            
            return fallback_response
    
    def generate_cot_summary(self) -> str:
        """Generate a summary of the Chain of Thought process"""
        if not self.cot_steps:
            return "No Chain of Thought steps recorded"
        
        high_confidence_steps = [step for step in self.cot_steps if step.confidence > 0.8]
        avg_confidence = sum(step.confidence for step in self.cot_steps) / len(self.cot_steps)
        
        summary = f"""
        Chain of Thought Summary:
        - Completed {len(self.cot_steps)} reasoning steps
        - Average confidence: {avg_confidence:.2f}
        - High confidence steps: {len(high_confidence_steps)}/{len(self.cot_steps)}
        - Total tokens used: {self.total_tokens_used}
        - Process flow: {' → '.join(step.step_name for step in self.cot_steps)}
        """
        
        return summary.strip()
    
    def get_hardcoded_inventory_response(self, query: str) -> str:
        """Get hardcoded inventory response for meta queries"""
        return """
        📚 **Knowledge Base Inventory:**
        
        **Available Books:**
        1. **Data Science - 40 Algorithms Every Data Scientist Should Know**
           - 15 chapters covering supervised, unsupervised, reinforcement learning
           - Focus: Practical algorithms and implementations
        
        2. **Data Science - Foundations of Data Science**
           - 12 chapters on mathematical foundations
           - Focus: Statistical learning, linear algebra, optimization
        
        3. **Data Science - John D Kelleher And Brendan Tierney**
           - 10 chapters on data science fundamentals
           - Focus: Practical introduction and real-world applications
        
        4. **Data Science - Theories Models Algorithms And Analytics**
           - 15 chapters on advanced analytics
           - Focus: Financial applications and advanced modeling
        
        **Content Types Available:**
        - Algorithmic explanations and implementations
        - Mathematical foundations and theory
        - Practical applications and case studies
        - Advanced modeling techniques
        """
    
    async def generate_study_plan_response(self, query_analysis: Dict) -> str:
        """Generate study plan response based on query analysis"""
        concepts = query_analysis.get('key_concepts', [])
        complexity = query_analysis.get('complexity', 'intermediate')
        
        return f"""
        📚 **Structured Study Plan Based on Available Knowledge Base:**
        
        **Learning Path for {', '.join(concepts[:3])}:**
        
        **Phase 1: Foundations (Weeks 1-2)**
        - Start with "Data Science - Kelleher & Tierney" for basic concepts
        - Focus on understanding core principles
        
        **Phase 2: Mathematical Foundations (Weeks 3-4)**
        - Study "Data Science - Foundations" for theoretical understanding
        - Cover statistical learning and optimization
        
        **Phase 3: Practical Algorithms (Weeks 5-6)**
        - Work through "Data Science - 40 Algorithms"
        - Implement key algorithms relevant to your concepts
        
        **Phase 4: Advanced Applications (Weeks 7-8)**
        - Explore "Data Science - Theories Models Analytics"
        - Focus on advanced techniques and real-world applications
        
        **Difficulty Level:** {complexity.title()}
        **Estimated Duration:** 8 weeks
        """
