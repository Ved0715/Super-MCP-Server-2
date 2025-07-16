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

@dataclass
class KBQuizRequest:
    """Knowledge Base quiz generation request parameters"""
    topic_description: str  # User's topic or description
    search_query: str  # Specific query to search knowledge base
    max_chunks: int = 50  # Maximum chunks to analyze from KB
    num_questions: int = 10  # Number of questions to generate
    difficulty_mix: Optional[Dict[str, int]] = None  # {easy: 4, medium: 4, hard: 2}
    namespace: Optional[str] = None  # Specific namespace to search (optional)

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
        self.index_name = "all-pdfs-index"
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
        """Retrieve relevant chunks from knowledge base using semantic search"""
        
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
            
            # Search knowledge base
            search_kwargs = {
                "vector": query_vector,
                "top_k": min(quiz_request.max_chunks, 100),  # Pinecone limit
                "include_metadata": True,
                "include_values": False
            }
            
            # Add namespace if specified
            if quiz_request.namespace:
                search_kwargs["namespace"] = quiz_request.namespace
                print(f"🎯 Searching in namespace: {quiz_request.namespace}")
            
            result = self.index.query(**search_kwargs)
            
            chunks = []
            if result and hasattr(result, 'matches'):
                matches = getattr(result, 'matches', [])
                if matches:
                    for match in matches:
                        # Extract text content from metadata
                        content = self._extract_text_content(match.metadata)
                        if content and len(content.strip()) > 50:  # Filter out very short content
                            chunks.append({
                                "id": match.id,
                                "content": content,
                                "score": match.score,
                                "metadata": match.metadata,
                                "source": match.metadata.get('source', 'Unknown'),
                                "page": match.metadata.get('page_number', 'Unknown')
                            })
            
            # Sort by relevance score (highest first)
            chunks = sorted(chunks, key=lambda x: x['score'], reverse=True)
            
            print(f"📊 Retrieved {len(chunks)} relevant chunks with scores: {[round(c['score'], 3) for c in chunks[:5]]}")
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
        
        # Comprehensive quiz generation prompt
        quiz_prompt = f"""
You are an expert knowledge base quiz generator. Generate {num_questions} multiple choice questions based on the knowledge base content below.

TOPIC CONTEXT:
- Main Topic: {topic_description}
- Search Query: {search_query}

KNOWLEDGE BASE CONTENT:
{full_content}

QUIZ REQUIREMENTS:
1. Total Questions: {num_questions}
2. Difficulty Distribution:
{difficulty_str}

3. Question Categories to Cover:
   - Factual Knowledge: Direct facts, definitions, specific information
   - Conceptual Understanding: Core concepts, principles, relationships
   - Application Knowledge: How concepts are applied, used, or implemented
   - Analytical Thinking: Analysis, comparison, evaluation of information
   - Synthesis: Combining multiple pieces of information

DIFFICULTY GUIDELINES:
- Easy: Basic facts, definitions, direct recall from content
- Medium: Understanding concepts, relationships, straightforward application
- Hard: Analysis, synthesis, complex reasoning, cross-content connections

QUESTION REQUIREMENTS:
- Each question must have exactly 4 options (a, b, c, d)
- Only ONE option should be correct
- Incorrect options should be plausible but clearly distinguishable
- Questions must be based on the PROVIDED content blocks
- Include variety in question types (what/who/when/where/why/how)
- Reference specific information from the content when possible
- Avoid questions that could be answered without the provided content

CONTENT UTILIZATION:
- Use information from multiple content blocks when possible
- Ensure questions test understanding of the topic area
- Create questions that demonstrate comprehension of the material
- Balance questions across different content blocks

OUTPUT FORMAT:
Return a valid JSON object with the exact structure below. Do not include any text outside the JSON.

{{
    "Q1": {{
        "question": "Question text here?",
        "options": {{
            "a": "Option A text",
            "b": "Option B text",
            "c": "Option C text", 
            "d": "Option D text"
        }},
        "answer": {{
            "a": "Option A text"
        }},
        "difficulty": "easy",
        "topic": "factual_knowledge",
        "source_info": "Based on content from [source reference]"
    }},
    "Q2": {{
        "question": "Question text here?",
        "options": {{
            "a": "Option A text",
            "b": "Option B text",
            "c": "Option C text",
            "d": "Option D text"
        }},
        "answer": {{
            "b": "Option B text"
        }},
        "difficulty": "medium",
        "topic": "conceptual_understanding",
        "source_info": "Based on content from [source reference]"
    }},
    ... continue for all {num_questions} questions
}}

CRITICAL REQUIREMENTS:
- Ensure the answer field contains the correct option letter as key and the full text as value
- Distribute questions across all topic categories
- Follow the exact difficulty distribution specified above
- Include source_info field indicating which content was used
- Return only valid JSON, no additional text
- Questions must be answerable from the provided content
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
                    
                    # Convert to KBQuizQuestion objects
                    questions = []
                    topics_cycle = ["factual_knowledge", "conceptual_understanding", "application_knowledge", "analytical_thinking", "synthesis"]
                    
                    for i, (q_key, q_data) in enumerate(quiz_json.items()):
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
                        
                        # Get topic from response or cycle through defaults
                        topic = q_data.get("topic", topics_cycle[i % len(topics_cycle)])
                        difficulty = q_data.get("difficulty", "medium")
                        source_info = q_data.get("source_info", "Knowledge base content")
                        
                        question = KBQuizQuestion(
                            question=q_data.get("question", f"Question {i+1}"),
                            options=q_data.get("options", {
                                "a": "Option A", "b": "Option B", 
                                "c": "Option C", "d": "Option D"
                            }),
                            answer=answer_key,
                            topic=topic,
                            difficulty=difficulty,
                            source_info=source_info
                        )
                        questions.append(question)
                    
                    print(f"✅ Successfully parsed {len(questions)} questions from single API response")
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
                }
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
