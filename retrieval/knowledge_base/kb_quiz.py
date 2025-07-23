import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import random
from openai import OpenAI
from pinecone import Pinecone

# Add config import - adjust path as needed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import config

@dataclass
class KBQuizQuestion:
    """Single MCQ question with options and answer for Knowledge Base quiz"""
    question: str
    options: Dict[str, str]  # {a: option1, b: option2, c: option3, d: option4}
    answer: str  # correct option key (a, b, c, or d)
    topic: str   # topic/section this question covers
    difficulty: str  # easy, medium, hard
    source_info: str  # brief info about source content
    book_name: str = "Unknown"
    page_reference: str = "Unknown"
    reasoning: str = ""

@dataclass
class KBQuizRequest:
    """Knowledge Base quiz generation request parameters"""
    topic_description: str  # User's topic or description
    search_query: str  # Specific query to search knowledge base
    max_chunks: int = 50  # Maximum chunks to analyze from KB
    num_questions: int = 10  # Number of questions to generate
    difficulty_mix: Optional[Dict[str, int]] = None  # {easy: 4, medium: 4, hard: 2}
    namespace: str = "knowledge-base"  # Specific namespace to search (optional)

class KnowledgeBaseQuizGenerator:
    """
    Advanced Knowledge Base Quiz Generation System
    
    Takes a topic/query, retrieves relevant content from the knowledge base,
    and generates comprehensive MCQ quizzes based on the retrieved information.
    Optimized for single API call efficiency.
    """
    
    def __init__(self):
        """Initialize the knowledge base quiz generator"""
        print("📚 Initializing Knowledge Base Quiz Generator...")
        
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Configuration
        self.index_name ="optimized-kb-index"
        self.embedding_model = config.embedding_model
        self.response_model = config.response_model
        
        # Quiz generation settings
        self.min_questions = 3
        self.max_questions = 25
        self.min_chunks = 5
        self.max_chunks = 100
        self.option_keys = ['a', 'b', 'c', 'd']
        
        # Connect to index
        self.setup_pinecone()
        
        print("✅ Knowledge Base Quiz Generator initialized successfully")
    
    def setup_pinecone(self):
        """Setup Pinecone index connection"""
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Connected to knowledge base index: {self.index_name}")
        except Exception as e:
            print(f"❌ Failed to connect to Pinecone index: {e}")
            self.index = None
    
    async def generate_kb_quiz(self, quiz_request: KBQuizRequest) -> Dict[str, Any]:
        """
        Generate a comprehensive quiz from knowledge base content
        
        Args:
            quiz_request: Quiz generation parameters
            
        Returns:
            Formatted quiz dictionary as specified
        """
        try:
            # Validate parameters
            if not self._validate_request(quiz_request):
                return self._error_response("Invalid quiz request parameters")
            
            print(f"🔍 Generating quiz for topic: {quiz_request.topic_description}")
            print(f"🔎 Search query: {quiz_request.search_query}")
            print(f"📊 Max chunks: {quiz_request.max_chunks}")
            print(f"❓ Questions requested: {quiz_request.num_questions}")
            
            # Step 1: Retrieve relevant chunks from knowledge base
            chunks = await self._retrieve_relevant_chunks(quiz_request)
            if not chunks:
                return self._error_response("No relevant content found in the knowledge base")
            
            print(f"📄 Retrieved {len(chunks)} relevant chunks for analysis")
            
            # Step 2: Generate ALL quiz questions in one API call
            questions = await self._generate_all_questions_single_call(
                chunks, 
                quiz_request.topic_description,
                quiz_request.search_query,
                quiz_request.num_questions,
                quiz_request.difficulty_mix or self._default_difficulty_mix(quiz_request.num_questions)
            )
            
            # Step 3: Format output as requested
            formatted_quiz = self._format_quiz_output(questions)
            
            print(f"✅ Successfully generated {len(questions)} quiz questions in one API call")
            
            return {
                "success": True,
                "quiz": formatted_quiz,
                "metadata": {
                    "topic_description": quiz_request.topic_description,
                    "search_query": quiz_request.search_query,
                    "total_questions": len(questions),
                    "chunks_analyzed": len(chunks),
                    "topics_covered": list(set(q.topic for q in questions)),
                    "difficulty_distribution": {
                        "easy": len([q for q in questions if q.difficulty == "easy"]),
                        "medium": len([q for q in questions if q.difficulty == "medium"]),
                        "hard": len([q for q in questions if q.difficulty == "hard"])
                    }
                }
            }
            
        except Exception as e:
            print(f"❌ KB Quiz generation failed: {e}")
            return self._error_response(f"Quiz generation error: {str(e)}")
    
    def _validate_request(self, request: KBQuizRequest) -> bool:
        """Validate quiz request parameters"""
        if not (self.min_questions <= request.num_questions <= self.max_questions):
            print(f"❌ Invalid number of questions: {request.num_questions} (must be {self.min_questions}-{self.max_questions})")
            return False
        
        if not (self.min_chunks <= request.max_chunks <= self.max_chunks):
            print(f"❌ Invalid max chunks: {request.max_chunks} (must be {self.min_chunks}-{self.max_chunks})")
            return False
        
        if not request.topic_description or not request.search_query:
            print("❌ Missing topic_description or search_query")
            return False
            
        return True
    
    async def _retrieve_relevant_chunks(self, quiz_request: KBQuizRequest) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks from knowledge base using semantic search with strict validation"""
        
        if not self.index:
            print("❌ No Pinecone index available")
            return []
        
        try:
            # Generate embedding for the search query
            print(f"🔍 Generating embedding for query: {quiz_request.search_query}")
            
            embedding_response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                input=quiz_request.search_query,
                model=self.embedding_model
            )
            
            query_vector = embedding_response.data[0].embedding
            # print(f"🔍 Query vector: {query_vector}")
            # Search knowledge base

            # print(f" index: {self.index}")
            result = self.index.query(
                vector=query_vector,
                top_k=quiz_request.max_chunks,
                namespace="knowledge-base",
                include_metadata=True,
                include_values=False
            )
            # print(f"🔍 Result: {result}")
            chunks = []
            if result and hasattr(result, 'matches'):
                matches = getattr(result, 'matches', [])
                for match in matches:
                    content = self._extract_text_content(match.metadata)
                    # print(f"🔍 Content: {content}")
                    if content:
                        chunks.append({
                            "id": match.id,
                            "content": content,
                            "source": match.metadata.get('source', 'Unknown'),
                            "page": match.metadata.get('page_references', 'Unknown'),
                            "book_name": match.metadata.get('section_title', 'Unknown'),
                            "metadata": match.metadata
                        })
            print(f"✅ Returning {len(chunks)} raw chunks without any filtering")
            return chunks[:quiz_request.max_chunks]

            
        except Exception as e:
            print(f"❌ Failed to retrieve chunks: {e}")
            return []

    def _extract_text_content(self, metadata: Dict[str, Any]) -> str:
        """Extract text content from metadata (handles _node_content JSON)"""
        # First try direct text field
        content = metadata.get('text', '')
        
        # If no direct text, try _node_content JSON
        if not content and '_node_content' in metadata:
            try:
                node_content = json.loads(metadata['_node_content'])
                content = node_content.get('text', '')
            except (json.JSONDecodeError, TypeError):
                content = str(metadata.get('_node_content', ''))[:1000]
        
        return content.strip() if content else ""
    
    def _validate_content_relevance(self, content: str, search_query: str, topic_description: str) -> float:
        """Validate that content is actually relevant to the search query and topic"""
        
        # Extract key terms from search query and topic
        search_terms = search_query.lower().split()
        topic_terms = topic_description.lower().split()
        content_lower = content.lower()
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        search_keywords = [term for term in search_terms if len(term) > 2 and term not in stop_words]
        topic_keywords = [term for term in topic_terms if len(term) > 2 and term not in stop_words]
        
        # Count matches
        search_matches = sum(1 for keyword in search_keywords if keyword in content_lower)
        topic_matches = sum(1 for keyword in topic_keywords if keyword in content_lower)
        
        # Calculate relevance score
        search_relevance = search_matches / max(len(search_keywords), 1)
        topic_relevance = topic_matches / max(len(topic_keywords), 1)
        
        # Combined relevance (weighted towards search query)
        combined_relevance = (search_relevance * 0.7) + (topic_relevance * 0.3)
        
        return combined_relevance

    def _validate_question_content_reference(self, question_text: str, chunks: List[Dict[str, Any]]) -> bool:
        """Validate that a question actually references content from the provided chunks"""
        
        if not question_text or not chunks:
            return False
        
        question_lower = question_text.lower()
        
        # Check if question contains phrases that suggest content reference
        content_indicators = [
            "according to",
            "based on the content",
            "the document states",
            "the text mentions",
            "as described in",
            "the content indicates",
            "mentioned in the content"
        ]
        
        has_content_indicator = any(indicator in question_lower for indicator in content_indicators)
        
        # Extract key terms from all chunks
        all_content = " ".join([chunk["content"] for chunk in chunks])
        content_words = set(all_content.lower().split())
        
        # Remove common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'what', 'when', 'where', 'who', 'why', 'how'}
        meaningful_content_words = {word for word in content_words if len(word) > 3 and word not in stop_words}
        
        # Count how many content-specific words appear in the question
        question_words = set(question_lower.split())
        content_word_matches = len(question_words.intersection(meaningful_content_words))
        
        # Question is valid if it has content indicators OR references specific content words
        return has_content_indicator or content_word_matches >= 2
    
    def _default_difficulty_mix(self, num_questions: int) -> Dict[str, int]:
        """Generate default difficulty distribution"""
        if num_questions <= 5:
            return {"easy": 2, "medium": 2, "hard": 1}
        elif num_questions <= 10:
            return {"easy": 4, "medium": 4, "hard": 2}
        elif num_questions <= 15:
            return {"easy": 5, "medium": 6, "hard": 4}
        elif num_questions <= 20:
            return {"easy": 6, "medium": 8, "hard": 6}
        else:  # 21-25
            return {"easy": 8, "medium": 10, "hard": 7}

    async def _generate_all_questions_single_call(
        self, 
        chunks: List[Dict[str, Any]], 
        topic_description: str,
        search_query: str,
        num_questions: int, 
        difficulty_mix: Dict[str, int]
    ) -> List[KBQuizQuestion]:
        """
        Generate ALL quiz questions in a single API call using a comprehensive prompt
        """
        print(f"🔄 Generating {num_questions} questions in single API call...")
        
        # Prepare content with source information
        content_blocks = []
        for i, chunk in enumerate(chunks):
            source_info = f"Source: {chunk['source']}, Page: {chunk['page']}"
            content_blocks.append(f"[Content Block {i+1}] ({source_info})\n{chunk['content']}")
        
        full_content = "\n\n".join(content_blocks)
        
        # Limit content size for API call (OpenAI has token limits)
        max_chars = 25000  # Larger limit for knowledge base content
        if len(full_content) > max_chars:
            full_content = full_content[:max_chars] + "\n\n[Content truncated for length...]"
        
        # Create detailed difficulty breakdown string
        difficulty_breakdown = []
        for difficulty, count in difficulty_mix.items():
            difficulty_breakdown.append(f"- {count} {difficulty} questions")
        difficulty_str = "\n".join(difficulty_breakdown)
        
        # Enhanced quiz generation prompt with strict content validation
        quiz_prompt = f"""
You are an expert knowledge base quiz generator. Generate {num_questions} multiple choice questions STRICTLY based on the knowledge base content provided below.

CRITICAL INSTRUCTION: You must ONLY use information from the provided content blocks. Do NOT use your general knowledge or training data. If the provided content is insufficient or irrelevant, respond with an error message instead of generating questions.

TOPIC CONTEXT:
- Main Topic: {topic_description}
- Search Query: {search_query}

KNOWLEDGE BASE CONTENT:
{full_content}

CONTENT VALIDATION CHECK:
Before generating questions, verify that the provided content blocks contain relevant information about "{topic_description}" and "{search_query}". If the content is mostly irrelevant or off-topic, respond with:
{{"error": "Insufficient relevant content found in knowledge base for the requested topic"}}

QUIZ REQUIREMENTS (only if content is relevant):
1. Total Questions: {num_questions}
2. Difficulty Distribution:
{difficulty_str}

3. Question Categories to Cover:
   - Factual Knowledge: Direct facts, definitions, specific information FROM THE PROVIDED CONTENT
   - Conceptual Understanding: Core concepts, principles, relationships FROM THE PROVIDED CONTENT
   - Application Knowledge: How concepts are applied, used, or implemented FROM THE PROVIDED CONTENT
   - Analytical Thinking: Analysis, comparison, evaluation of information FROM THE PROVIDED CONTENT
   - Synthesis: Combining multiple pieces of information FROM THE PROVIDED CONTENT

STRICT CONTENT REQUIREMENTS:
- Questions must be answerable ONLY from the provided content blocks
- Each question must reference specific information found in the content
- Do NOT create questions about general knowledge of the topic
- Do NOT use information not present in the provided content
- If a content block is irrelevant, ignore it completely
- Questions should quote or paraphrase specific details from the content

DIFFICULTY GUIDELINES:
- Easy: Basic facts, definitions, direct recall from provided content
- Medium: Understanding concepts, relationships from provided content
- Hard: Analysis, synthesis of multiple content blocks

QUESTION REQUIREMENTS:
- DO NOT use phrases like "as per the content", "according to the provided content", "as mentioned above", "based on the content", or similar. 
- Each question must have exactly 4 options (a, b, c, d)
- Only ONE option should be correct
- Incorrect options should be plausible but based on content variations
- Include variety in question types (what/who/when/where/why/how)
- Write questions in a natural, academic style, as if for a university exam or textbook. DO NOT use phrases like "according to the content", "based on the provided content", or similar meta-references.
- Instead, provide a direct explanation for why the answer is correct, using facts or logic from the content, but do not reference the content itself.
- Each question should stand alone and make sense without referencing the content source.
- For each question, provide a direct, fact-based explanation for why the answer is correct. Do NOT reference the content or say 'as per the content'. Example:
        Q: What is the main function of supervised learning?
        A: It learns a function that maps input attributes to a target attribute using labeled data....

Reasoning: Supervised learning uses labeled data to train models, allowing them to predict outcomes based on input features.- For each question, include the book name, section title, page reference, and a brief reasoning for the answer, all based on the provided content.

RESONING:
BAD: "The correct answer is supported by the content."
BAD: "As per the provided content, the answer is..."
GOOD: "Machine learning is defined as a subset of AI that allows computers to learn from data and make decisions without explicit programming...."

OUTPUT FORMAT:
Return a valid JSON object with the exact structure below. Do not include any text outside the JSON.

{{
    "Q1": {{
        "question": "According to the provided content, [specific question based on content]?",
        "options": {{
            "a": "Option A from content",
            "b": "Option B from content",
            "c": "Option C from content", 
            "d": "Option D from content"
        }},
        "answer": {{
            "a": "Option A from content"
        }},
        "difficulty": "easy",
        "topic": "factual_knowledge",
        "source_info": "Based on content from [specific source/block reference]",
        "content_reference": "Quote or reference from the actual content"
        "book_name": "title from content",
        "page_reference": "Page number(s) from content",
        "reasoning": "Reason why the answer is correct out of all."
    }},
    "Q2": {{
        "question": "According to the provided content, [specific question based on content]?",
        "options": {{
            "a": "Option A from content",
            "b": "Option B from content",
            "c": "Option C from content", 
            "d": "Option D from content"
        }},
        "answer": {{
            "a": "Option A from content"
        }},
        "difficulty": "easy",
        "topic": "factual_knowledge",
        "source_info": "Based on content from [specific source/block reference]",
        "content_reference": "Quote or reference from the actual content"
        "book_name": "title from content",
        "page_reference": "Page number(s) from content",
        "reasoning": "This is why the answer is correct"
    }},
    ... continue for all {num_questions} questions
}}

CRITICAL VALIDATION REQUIREMENTS:
1. Verify content relevance before generating ANY questions
2. Only generate questions if content is directly relevant to the topic
3. Include "content_reference" field with actual quotes/references
4. Ensure answer options are based on content variations, not general knowledge
5. If content is insufficient or irrelevant, return error JSON instead
6. Questions must pass the test: "Can this be answered only from the provided content?"

FINAL CHECK: Before returning, verify each question can ONLY be answered using the provided content blocks. If any question relies on general knowledge, regenerate it or return an error.
"""

        try:
            print("🤖 Making single API call to generate all KB quiz questions...")
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.response_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert knowledge base quiz generator. Create comprehensive MCQ quizzes that test understanding of provided content. Always respond with valid JSON only."
                    },
                    {
                        "role": "user", 
                        "content": quiz_prompt
                    }
                ],
                temperature=0.7,  # Balance between consistency and variety
                max_tokens=8000  # Enough tokens for all questions
            )
            
            response_text = response.choices[0].message.content
            if response_text is None:
                response_text = ""
            response_text = response_text.strip()
            print(f"📝 Received response with {len(response_text)} characters")
            
            # Parse the JSON response
            try:
                # Clean up response - remove markdown code blocks if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()
                
                # Find JSON object
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    quiz_json = json.loads(response_text[json_start:json_end])
                    
                    # Check if LLM detected insufficient content
                    if "error" in quiz_json:
                        error_msg = quiz_json["error"]
                        print(f"🚫 LLM detected insufficient relevant content: {error_msg}")
                        return []  # Return empty list to trigger error in main function
                    
                    # Validate that we have actual questions, not just error response
                    question_keys = [key for key in quiz_json.keys() if key.startswith('Q')]
                    if not question_keys:
                        print("🚫 No questions found in LLM response")
                        return []
                    
                    # Convert to KBQuizQuestion objects
                    questions = []
                    topics_cycle = ["factual_knowledge", "conceptual_understanding", "application_knowledge", "analytical_thinking", "synthesis"]
                    
                    for i, (q_key, q_data) in enumerate(quiz_json.items()):
                        if not q_key.startswith('Q'):
                            continue  # Skip non-question keys
                            
                        # Validate question has required fields
                        if not all(key in q_data for key in ['question', 'options', 'answer']):
                            print(f"⚠️ Question {q_key} missing required fields, skipping")
                            continue
                        
                        # Extract the correct answer key and ensure it's valid
                        answer_dict = q_data.get("answer", {})
                        if answer_dict:
                            answer_key = list(answer_dict.keys())[0]
                        else:
                            # Fallback - find correct answer by looking for it in the response
                            answer_key = "a"  # Default fallback
                        
                        # Ensure answer key is valid
                        if answer_key not in ['a', 'b', 'c', 'd']:
                            answer_key = "a"
                        
                        # Validate that the question references the content
                        question_text = q_data.get("question", "")
                        if not self._validate_question_content_reference(question_text, chunks):
                            print(f"⚠️ Question {q_key} doesn't reference provided content, skipping")
                            continue
                        
                        # Get topic from response or cycle through defaults
                        topic = q_data.get("topic", topics_cycle[i % len(topics_cycle)])
                        difficulty = q_data.get("difficulty", "medium")
                        source_info = q_data.get("source_info", "Knowledge base content")
                        
                        # Find the most relevant chunk for fallback
                        chunk_fallback = chunks[0] if chunks else {}
                        question = KBQuizQuestion(
                            question=question_text,
                            options=q_data.get("options", {
                                "a": "Option A", "b": "Option B", 
                                "c": "Option C", "d": "Option D"
                            }),
                            answer=answer_key,
                            topic=topic,
                            difficulty=difficulty,
                            source_info=source_info,
                            book_name=q_data.get("section_title", chunk_fallback.get("section_title", "Unknown")),
                            page_reference=q_data.get("page_reference", chunk_fallback.get("page", "Unknown")),
                            reasoning=q_data.get("reasoning", "")
                        )
                        questions.append(question)
                    
                    if not questions:
                        print("❌ No valid questions generated after content validation")
                        return []
                    
                    print(f"✅ Successfully parsed {len(questions)} validated questions from single API response")
                    return questions[:num_questions]  # Ensure we don't exceed requested count
                
                else:
                    raise json.JSONDecodeError("No valid JSON found", response_text, 0)
                    
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse quiz JSON: {e}")
                print(f"Response preview: {response_text[:500]}...")
                return self._generate_fallback_questions(chunks, topic_description, num_questions, difficulty_mix)
                
        except Exception as e:
            print(f"❌ Quiz generation API call failed: {e}")
            return self._generate_fallback_questions(chunks, topic_description, num_questions, difficulty_mix)

    def _generate_fallback_questions(
        self, 
        chunks: List[Dict[str, Any]], 
        topic_description: str,
        num_questions: int, 
        difficulty_mix: Dict[str, int]
    ) -> List[KBQuizQuestion]:
        """Generate basic fallback questions when API fails"""
        print("🔄 Generating fallback questions...")
        
        questions = []
        content_sample = " ".join([chunk["content"][:100] for chunk in chunks[:3]])
        topics = ["factual_knowledge", "conceptual_understanding", "application_knowledge", "analytical_thinking", "synthesis"]
        
        for i in range(min(num_questions, 5)):  # Generate up to 5 fallback questions
            difficulty = "easy" if i < 2 else "medium"
            topic = topics[i % len(topics)]
            
            question = KBQuizQuestion(
                question=f"Based on the knowledge base content about {topic_description}, which of the following is most relevant to {topic.replace('_', ' ')}?",
                options={
                    "a": f"Information related to {topic.replace('_', ' ')} aspect A",
                    "b": f"Information related to {topic.replace('_', ' ')} aspect B", 
                    "c": f"Information related to {topic.replace('_', ' ')} aspect C",
                    "d": f"Information related to {topic.replace('_', ' ')} aspect D"
                },
                answer="a",
                topic=topic,
                difficulty=difficulty,
                source_info="Fallback content"
            )
            questions.append(question)
        
        return questions
    
    def _format_quiz_output(self, questions: List[KBQuizQuestion]) -> Dict[str, Any]:
        """Format quiz questions in the requested output format"""
        
        formatted_quiz = {}
        
        for i, question in enumerate(questions, 1):
            question_key = f"Q{i}"
            
            formatted_quiz[question_key] = {
                "question": question.question,
                "options": question.options,
                "answer": {
                    question.answer: question.options[question.answer]
                },
                "book_name": question.book_name,
                "page_reference": question.page_reference,
                "reasoning": question.reasoning
            }
        
        return formatted_quiz
    
    def _error_response(self, error_message: str) -> Dict[str, Any]:
        """Generate error response"""
        return {
            "success": False,
            "error": error_message,
            "quiz": {}
        }

# Convenience functions for easy usage

async def generate_knowledge_base_quiz(
    topic_description: str,
    search_query: str,
    max_chunks: int = 30,
    num_questions: int = 10,
    difficulty_mix: Optional[Dict[str, int]] = None,
    namespace: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to generate a quiz from knowledge base
    
    Args:
        topic_description: Description of the topic to quiz about
        search_query: Specific query to search the knowledge base
        max_chunks: Maximum number of chunks to analyze (default: 30)
        num_questions: Number of questions to generate (3-25, default: 10)
        difficulty_mix: Custom difficulty distribution (optional)
        namespace: Specific namespace to search (optional)
    
    Returns:
        Formatted quiz dictionary
        
    Example:
        result = await generate_knowledge_base_quiz(
            topic_description="Machine Learning Algorithms",
            search_query="supervised learning classification algorithms",
            max_chunks=25,
            num_questions=8
        )
    """
    generator = KnowledgeBaseQuizGenerator()
    
    request = KBQuizRequest(
        topic_description=topic_description,
        search_query=search_query,
        max_chunks=max_chunks,
        num_questions=num_questions,
        difficulty_mix=difficulty_mix,
        namespace=namespace
    )
    
    return await generator.generate_kb_quiz(request)

# Test function for development
async def test_kb_quiz_generation():
    """Test function for knowledge base quiz generation"""
    
    # Test 1: General AI/ML topic
    print("🧪 Test 1: AI/ML Quiz")
    result1 = await generate_knowledge_base_quiz(
        topic_description="Artificial Intelligence and Machine Learning",
        search_query="machine learning algorithms neural networks deep learning",
        max_chunks=20,
        num_questions=8
    )
    
    if result1['success']:
        print(f"✅ Generated {len(result1['quiz'])} questions")
        print(f"📊 Metadata: {result1['metadata']}")
        
        # Show sample question
        first_q = list(result1['quiz'].values())[0]
        print(f"📝 Sample Q: {first_q['question'][:80]}...")
    else:
        print(f"❌ Failed: {result1['error']}")
    
    print("\n" + "="*50 + "\n")
    
    # Test 2: Specific technical topic
    print("🧪 Test 2: Specific Technical Quiz")
    result2 = await generate_knowledge_base_quiz(
        topic_description="Database Management Systems",
        search_query="SQL database normalization ACID properties",
        max_chunks=15,
        num_questions=6,
        difficulty_mix={"easy": 2, "medium": 3, "hard": 1}
    )
    
    if result2['success']:
        print(f"✅ Generated {len(result2['quiz'])} questions")
        print(f"📊 Metadata: {result2['metadata']}")
    else:
        print(f"❌ Failed: {result2['error']}")

if __name__ == "__main__":
    asyncio.run(test_kb_quiz_generation())
