import os
import json
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from openai import OpenAI
from pinecone import Pinecone

# Add config import - adjust path as needed
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config import config

@dataclass
class QuizQuestion:
    """Single MCQ question with options and answer"""
    question: str
    options: Dict[str, str]  # {a: option1, b: option2, c: option3, d: option4}
    answer: str  # correct option key (a, b, c, or d)
    topic: str   # topic/section this question covers
    difficulty: str  # easy, medium, hard
    page: str = "unknown"  # page number
    reasoning: str = ""    # reasoning for the answer

@dataclass
class QueryBasedQuizRequest:
    """Quiz generation request parameters with user query"""
    user_id: str
    document_uuid: str
    user_query: str  # Topics or specific areas user wants to focus on
    num_questions: int
    difficulty_mix: Optional[Dict[str, int]] = None  # {easy: 2, medium: 2, hard: 1}

class QueryBasedPaperQuizGenerator:
    """
    Query-Based Paper Quiz Generation System
    
    Generates MCQ quizzes based on user-specified topics or queries,
    retrieving relevant content from the document and focusing
    questions on the areas of interest specified by the user.
    """
    
    def __init__(self):
        """Initialize the query-based quiz generator"""
        print("📝 Initializing Query-Based Paper Quiz Generator...")
        
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Configuration
        self.index_name = "test"
        self.embedding_model = config.embedding_model
        self.response_model = config.response_model
        
        # Quiz generation settings
        self.min_questions = 5
        self.max_questions = 20
        self.option_keys = ['a', 'b', 'c', 'd']
        
        # Connect to index
        self.setup_pinecone()
        
        print("✅ Query-Based Paper Quiz Generator initialized successfully")
    
    def setup_pinecone(self):
        """Setup Pinecone index connection"""
        try:
            self.index = self.pc.Index(self.index_name)
            print(f"✅ Connected to quiz generation index: {self.index_name}")
        except Exception as e:
            print(f"❌ Failed to connect to Pinecone index: {e}")
            self.index = None
    
    def _build_namespace(self, user_id: str, document_uuid: str) -> str:
        """Build the namespace for user-specific document"""
        return f"user_{user_id}_doc_{document_uuid}"
    
    async def _check_namespace_exists(self, user_id: str, document_uuid: str) -> bool:
        """Check if the namespace exists and has content"""
        namespace = self._build_namespace(user_id, document_uuid)
        
        if not self.index:
            print("❌ No Pinecone index available")
            return False
        
        try:
            # Get index stats to check namespace
            stats = self.index.describe_index_stats()
            namespaces = getattr(stats, 'namespaces', {})
            
            if namespace not in namespaces:
                print(f"❌ Namespace '{namespace}' not found in index")
                return False
            
            namespace_stats = namespaces[namespace]
            vector_count = getattr(namespace_stats, 'vector_count', 0)
            
            if vector_count == 0:
                print(f"❌ Namespace '{namespace}' exists but has no vectors")
                return False
            
            print(f"✅ Namespace '{namespace}' found with {vector_count} vectors")
            return True
            
        except Exception as e:
            print(f"❌ Failed to check namespace existence: {e}")
            return False
    
    async def _generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for user query"""
        try:
            print(f"🔄 Generating embedding for query: '{query[:50]}...'")
            
            response = await asyncio.to_thread(
                self.openai_client.embeddings.create,
                model=self.embedding_model,
                input=query
            )
            
            embedding = response.data[0].embedding
            print(f"✅ Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            print(f"❌ Failed to generate query embedding: {e}")
            return None
    
    async def _retrieve_relevant_chunks(self, user_id: str, document_uuid: str, 
                                      query_embedding: List[float], user_query: str) -> List[Dict[str, Any]]:
        """Retrieve chunks relevant to user query using vector similarity"""
        namespace = self._build_namespace(user_id, document_uuid)
        
        if not self.index:
            print("❌ No Pinecone index available")
            return []
        
        try:
            print(f"🔍 Searching for content relevant to: '{user_query}'")
            
            result = self.index.query(
                vector=query_embedding,
                top_k=50,  # Get top 50 most relevant chunks
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            
            relevant_chunks = []
            if result and hasattr(result, 'matches'):
                matches = getattr(result, 'matches', [])
                if matches:
                    for match in matches:
                        # Only include chunks with reasonable similarity score
                        if match.score > 0.5:  # Adjust threshold as needed
                            content = self._extract_text_content(match.metadata)
                            if content and len(content.strip()) > 20:  # Filter out very short content
                                relevant_chunks.append({
                                    "id": match.id,
                                    "content": content,
                                    "metadata": match.metadata,
                                    "page": match.metadata.get('page_number', 'unknown'),
                                    "similarity_score": match.score
                                })
            
            # Sort by similarity score (highest first)
            relevant_chunks.sort(key=lambda x: x.get("similarity_score", 0), reverse=True)
            
            print(f"📊 Retrieved {len(relevant_chunks)} relevant chunks from namespace: {namespace}")
            return relevant_chunks[:30]  # Limit to top 30 for processing efficiency
            
        except Exception as e:
            print(f"❌ Failed to retrieve relevant chunks: {e}")
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
                content = str(metadata.get('_node_content', ''))[:500]
        
        return content.strip() if content else ""
    
    async def generate_query_based_quiz(self, quiz_request: QueryBasedQuizRequest) -> Dict[str, Any]:
        """
        Generate a quiz based on user query/topics
        
        Args:
            quiz_request: Quiz generation parameters including user query
            
        Returns:
            Formatted quiz dictionary
        """
        try:
            # Validate parameters
            if not self._validate_request(quiz_request):
                return self._error_response("Invalid quiz request parameters")
            
            print(f"📚 Generating query-based quiz for document: {quiz_request.document_uuid}")
            print(f"👤 User: {quiz_request.user_id}")
            print(f"🔍 User Query: {quiz_request.user_query}")
            print(f"❓ Questions requested: {quiz_request.num_questions}")
            
            # Step 1: Check if namespace exists and has content
            namespace_exists = await self._check_namespace_exists(quiz_request.user_id, quiz_request.document_uuid)
            if not namespace_exists:
                return self._error_response(f"Document not found or has no content for user {quiz_request.user_id}")
            
            # Step 2: Generate embeddings for user query
            query_embedding = await self._generate_query_embedding(quiz_request.user_query)
            if not query_embedding:
                return self._error_response("Failed to process user query")
            
            # Step 3: Retrieve relevant chunks based on query
            relevant_chunks = await self._retrieve_relevant_chunks(
                quiz_request.user_id, 
                quiz_request.document_uuid, 
                query_embedding,
                quiz_request.user_query
            )
            
            if not relevant_chunks:
                return self._error_response("No relevant content found for the specified query")
            
            print(f"📄 Found {len(relevant_chunks)} relevant chunks for query: '{quiz_request.user_query}'")
            
            # Step 4: Generate quiz questions focused on query topics
            questions = await self._generate_query_focused_questions(
                relevant_chunks, 
                quiz_request.user_query,
                quiz_request.num_questions,
                quiz_request.difficulty_mix or self._default_difficulty_mix(quiz_request.num_questions)
            )
            
            # Step 5: Format output
            formatted_quiz = self._format_quiz_output(questions)
            
            print(f"✅ Successfully generated {len(questions)} query-focused quiz questions")
            
            return {
                "success": True,
                "quiz": formatted_quiz,
                "metadata": {
                    "user_id": quiz_request.user_id,
                    "document_uuid": quiz_request.document_uuid,
                    "user_query": quiz_request.user_query,
                    "total_questions": len(questions),
                    "chunks_analyzed": len(relevant_chunks),
                    "topics_covered": list(set(q.topic for q in questions))
                }
            }
            
        except Exception as e:
            print(f"❌ Query-based quiz generation failed: {e}")
            return self._error_response(f"Quiz generation error: {str(e)}")
    
    def _validate_request(self, request: QueryBasedQuizRequest) -> bool:
        """Validate quiz request parameters"""
        if not (self.min_questions <= request.num_questions <= self.max_questions):
            print(f"❌ Invalid number of questions: {request.num_questions} (must be {self.min_questions}-{self.max_questions})")
            return False
        
        if not request.user_id or not request.document_uuid:
            print("❌ Missing user_id or document_uuid")
            return False
        
        if not request.user_query.strip():
            print("❌ Missing or empty user query")
            return False
            
        return True
    
    def _default_difficulty_mix(self, num_questions: int) -> Dict[str, int]:
        """Generate default difficulty distribution"""
        if num_questions <= 5:
            return {"easy": 2, "medium": 2, "hard": 1}
        elif num_questions <= 10:
            return {"easy": 4, "medium": 4, "hard": 2}
        elif num_questions <= 15:
            return {"easy": 5, "medium": 6, "hard": 4}
        else:  # 16-20
            return {"easy": 6, "medium": 8, "hard": 6}
    
    async def _generate_query_focused_questions(self, chunks: List[Dict[str, Any]], 
                                              user_query: str, num_questions: int, 
                                              difficulty_mix: Dict[str, int]) -> List[QuizQuestion]:
        """Generate quiz questions focused on user query using relevant content"""
        print(f"🔄 Generating {num_questions} questions focused on: '{user_query}'")
        
        # Combine relevant content, prioritizing higher-scoring chunks
        focused_content = "\n\n".join([
            f"Page {chunk['page']} (Score: {chunk.get('similarity_score', 0):.2f}): {chunk['content']}" 
            for chunk in chunks[:15]  # Use top 15 most relevant chunks
        ])
        
        # Limit content size for API call
        max_chars = 25000
        if len(focused_content) > max_chars:
            focused_content = focused_content[:max_chars] + "..."
        
        # Create quiz generation prompt
        quiz_prompt = f"""
You are an expert quiz generator. Generate {num_questions} multiple choice questions based on the paper content below, focused specifically on the user's query: "{user_query}".

RELEVANT PAPER CONTENT:
{focused_content}

REQUIREMENTS:
- All questions must relate directly to "{user_query}"
- Each question has exactly 4 options (a, b, c, d)
- Only ONE correct answer per question
- Include page references and reasoning

OUTPUT FORMAT (JSON only):
{{
    "Q1": {{
        "question": "Question about {user_query}?",
        "options": {{
            "a": "Option A",
            "b": "Option B",
            "c": "Option C",
            "d": "Option D"
        }},
        "answer": {{
            "a": "Option A"
        }},
        "difficulty": "easy",
        "topic": "{user_query.lower().replace(' ', '_')}",
        "page": "4",
        "reasoning": "Explanation why this is correct..."
    }}
}}
"""

        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.response_model,
                messages=[
                    {"role": "system", "content": "Generate focused quiz questions as JSON only."},
                    {"role": "user", "content": quiz_prompt}
                ],
                temperature=0.6,
                max_tokens=8000
            )
            
            response_text = response.choices[0].message.content or ""
            response_text = response_text.strip()
            
            # Clean and parse JSON
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()
            
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                quiz_json = json.loads(response_text[json_start:json_end])
                return self._parse_questions_from_json(quiz_json, user_query, chunks, num_questions)
            else:
                return self._generate_fallback_questions(chunks, user_query, num_questions)
                
        except Exception as e:
            print(f"❌ Question generation failed: {e}")
            return self._generate_fallback_questions(chunks, user_query, num_questions)
    
    def _parse_questions_from_json(self, quiz_json: Dict[str, Any], user_query: str, 
                                  chunks: List[Dict[str, Any]], num_questions: int) -> List[QuizQuestion]:
        """Parse questions from JSON response"""
        questions = []
        query_topic = user_query.lower().replace(' ', '_')
        
        for i, (q_key, q_data) in enumerate(quiz_json.items()):
            if len(questions) >= num_questions:
                break
                
            # Extract answer key
            answer_dict = q_data.get("answer", {})
            answer_key = list(answer_dict.keys())[0] if answer_dict else "a"
            if answer_key not in ['a', 'b', 'c', 'd']:
                answer_key = "a"
            
            question = QuizQuestion(
                question=q_data.get("question", f"Question about {user_query}"),
                options=q_data.get("options", {
                    "a": f"Option A about {user_query}", 
                    "b": f"Option B about {user_query}", 
                    "c": f"Option C about {user_query}", 
                    "d": f"Option D about {user_query}"
                }),
                answer=answer_key,
                topic=q_data.get("topic", query_topic),
                difficulty=q_data.get("difficulty", "medium"),
                page=q_data.get("page", str(chunks[0]["page"]) if chunks else "unknown"),
                reasoning=q_data.get("reasoning", f"Answer relates to {user_query}")
            )
            questions.append(question)
        
        return questions
    
    def _generate_fallback_questions(self, chunks: List[Dict[str, Any]], user_query: str, 
                                   num_questions: int) -> List[QuizQuestion]:
        """Generate fallback questions when API fails"""
        print(f"🔄 Generating fallback questions about '{user_query}'...")
        
        questions = []
        query_topic = user_query.lower().replace(' ', '_')
        
        for i in range(min(num_questions, 5)):
            question = QuizQuestion(
                question=f"Based on the paper content about {user_query}, which statement is most accurate?",
                options={
                    "a": f"Aspect A related to {user_query}",
                    "b": f"Aspect B related to {user_query}", 
                    "c": f"Aspect C related to {user_query}",
                    "d": f"Aspect D related to {user_query}"
                },
                answer="a",
                topic=query_topic,
                difficulty="easy" if i < 2 else "medium",
                page=str(chunks[0]["page"]) if chunks else "unknown",
                reasoning=f"This answer relates to {user_query} based on the paper content."
            )
            questions.append(question)
        
        return questions
    
    def _format_quiz_output(self, questions: List[QuizQuestion]) -> Dict[str, Any]:
        """Format quiz questions for output"""
        formatted_quiz = {}
        for i, question in enumerate(questions, 1):
            question_key = f"Q{i}"
            formatted_quiz[question_key] = {
                "question": question.question,
                "options": question.options,
                "answer": {
                    question.answer: question.options[question.answer]
                },
                "page": question.page,
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

# Convenience function
async def generate_query_based_quiz(user_id: str, document_uuid: str, user_query: str, 
                                  num_questions: int = 10) -> Dict[str, Any]:
    """
    Convenience function to generate a query-based quiz from a paper
    
    Args:
        user_id: User identifier
        document_uuid: Document UUID
        user_query: Topics or areas user wants to focus on
        num_questions: Number of questions to generate (5-20)
    
    Returns:
        Formatted quiz dictionary
    """
    generator = QueryBasedPaperQuizGenerator()
    
    request = QueryBasedQuizRequest(
        user_id=user_id,
        document_uuid=document_uuid,
        user_query=user_query,
        num_questions=num_questions
    )
    
    return await generator.generate_query_based_quiz(request)

# Test function for development
async def test_query_based_quiz_generation():
    """Test function for query-based quiz generation"""
    result = await generate_query_based_quiz(
        user_id="5",
        document_uuid="9e51cc77-49ed-403e-906e-9882ef8a236d",
        user_query="Lists in Python",
        num_questions=5
    )
    
    print("Query-Based Quiz Generation Test Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_query_based_quiz_generation())