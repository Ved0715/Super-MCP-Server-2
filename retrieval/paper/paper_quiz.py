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
class QuizRequest:
    """Quiz generation request parameters"""
    user_id: str
    document_uuid: str
    num_questions: int
    difficulty_mix: Optional[Dict[str, int]] = None  # {easy: 2, medium: 2, hard: 1}

class PaperQuizGenerator:
    """
    Advanced Research Paper Quiz Generation System
    
    Analyzes all chunks in a user's document namespace and generates
    comprehensive MCQ quizzes covering key concepts, methodologies,
    and findings from the research paper.
    """
    
    def __init__(self):
        """Initialize the quiz generator"""
        print("📝 Initializing Research Paper Quiz Generator...")
        
        self.openai_client = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        
        # Configuration
        self.index_name = "all-pdfs-index"
        self.embedding_model = config.embedding_model
        self.response_model = config.response_model
        
        # Quiz generation settings
        self.min_questions = 5
        self.max_questions = 20
        self.option_keys = ['a', 'b', 'c', 'd']
        
        # Connect to index
        self.setup_pinecone()
        
        print("✅ Research Paper Quiz Generator initialized successfully")
    
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
    
    async def generate_quiz(self, quiz_request: QuizRequest) -> Dict[str, Any]:
        """
        Generate a comprehensive quiz from research paper
        
        Args:
            quiz_request: Quiz generation parameters
            
        Returns:
            Formatted quiz dictionary as specified
        """
        try:
            # Validate parameters
            if not self._validate_request(quiz_request):
                return self._error_response("Invalid quiz request parameters")
            
            print(f"📚 Generating quiz for document: {quiz_request.document_uuid}")
            print(f"👤 User: {quiz_request.user_id}")
            print(f"❓ Questions requested: {quiz_request.num_questions}")
            
            # Step 1: Retrieve all document chunks
            chunks = await self._retrieve_all_chunks(quiz_request.user_id, quiz_request.document_uuid)
            if not chunks:
                return self._error_response("No content found in the research paper")
            
            print(f"📄 Analyzed {len(chunks)} content chunks from research paper")
            
            # Step 2: Generate ALL quiz questions in one API call
            questions = await self._generate_all_questions_single_call(
                chunks, 
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
                    "user_id": quiz_request.user_id,
                    "document_uuid": quiz_request.document_uuid,
                    "total_questions": len(questions),
                    "chunks_analyzed": len(chunks),
                    "topics_covered": list(set(q.topic for q in questions))
                }
            }
            
        except Exception as e:
            print(f"❌ Quiz generation failed: {e}")
            return self._error_response(f"Quiz generation error: {str(e)}")
    
    def _validate_request(self, request: QuizRequest) -> bool:
        """Validate quiz request parameters"""
        if not (self.min_questions <= request.num_questions <= self.max_questions):
            print(f"❌ Invalid number of questions: {request.num_questions} (must be {self.min_questions}-{self.max_questions})")
            return False
        
        if not request.user_id or not request.document_uuid:
            print("❌ Missing user_id or document_uuid")
            return False
            
        return True
    
    async def _retrieve_all_chunks(self, user_id: str, document_uuid: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks from the user's document namespace"""
        namespace = self._build_namespace(user_id, document_uuid)
        
        if not self.index:
            print("❌ No Pinecone index available")
            return []
        
        try:
            # Use a dummy vector to get all chunks in namespace
            dummy_vector = [0.0] * 3072  # Assuming 3072 dimensions
            
            # Get maximum possible chunks (Pinecone limit is usually 10k)
            result = self.index.query(
                vector=dummy_vector,
                top_k=10000,  # Get all chunks
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            
            chunks = []
            if result and hasattr(result, 'matches'):
                matches = getattr(result, 'matches', [])
                if matches:
                    for match in matches:
                        # Extract text content from _node_content JSON
                        content = self._extract_text_content(match.metadata)
                        if content:
                            chunks.append({
                                "id": match.id,
                                "content": content,
                                "metadata": match.metadata,
                                "page": match.metadata.get('page_number', 'unknown')
                            })
            
            print(f"📊 Retrieved {len(chunks)} chunks from namespace: {namespace}")
            return chunks
            
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
                content = str(metadata.get('_node_content', ''))[:500]
        
        return content.strip() if content else ""
    
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

    async def _generate_all_questions_single_call(self, chunks: List[Dict[str, Any]], num_questions: int, difficulty_mix: Dict[str, int]) -> List[QuizQuestion]:
        """
        Generate ALL quiz questions in a single API call using a comprehensive prompt
        """
        print(f"🔄 Generating {num_questions} questions in single API call...")
        
        # Combine all content for analysis
        full_content = "\n\n".join([f"Page {chunk['page']}: {chunk['content']}" for chunk in chunks])
        
        # Limit content size for API call (OpenAI has token limits)
        max_chars = 20000  # Reasonable limit for comprehensive analysis
        if len(full_content) > max_chars:
            full_content = full_content[:max_chars] + "..."
        
        # Create detailed difficulty breakdown string
        difficulty_breakdown = []
        for difficulty, count in difficulty_mix.items():
            difficulty_breakdown.append(f"- {count} {difficulty} questions")
        difficulty_str = "\n".join(difficulty_breakdown)
        
        # Comprehensive quiz generation prompt
        quiz_prompt = f"""
You are an expert research paper quiz generator. Generate {num_questions} multiple choice questions based on the research paper content below.

RESEARCH PAPER CONTENT:
{full_content}

QUIZ REQUIREMENTS:
1. Total Questions: {num_questions}
2. Difficulty Distribution:
{difficulty_str}

3. Question Categories to Cover:
   - Main Concepts: Core ideas, definitions, theories
   - Methodologies: Research methods, algorithms, approaches  
   - Findings: Key results, conclusions, discoveries
   - Technical Details: Specific facts, numbers, procedures
   - Applications: Use cases, implementations, examples

DIFFICULTY GUIDELINES:
- Easy: Basic facts, definitions, what/who/when questions
- Medium: Understanding concepts, how/why questions, application
- Hard: Analysis, synthesis, comparison, evaluation

QUESTION REQUIREMENTS:
- Each question must have exactly 4 options (a, b, c, d)
- Only ONE option should be correct
- Incorrect options should be plausible but clearly wrong
- Questions must be specific to THIS paper's content
- Avoid generic questions that could apply to any paper
- Use varied question stems and formats
- For each question, provide a brief reasoning for the correct answer and  why the answer is correc and referance page number for the answer where the answer can be found.

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
        "topic": "main_concepts",
        "page": "4",
        "reasoning": "This is why the answer is correct, you can find the context of the answer in th page number "4" ..."
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
        "topic": "methodologies",
        "page": "5",
        "reasoning": "Reason why the answer is correct out of all and you can refer from the page number "5" ..."
    }},
    ... continue for all {num_questions} questions
}}

CRITICAL: Ensure the answer field contains the correct option letter as key and the full text as value.
CRITICAL: Distribute questions across all topic categories.
CRITICAL: Follow the exact difficulty distribution specified above.
CRITICAL: Return only valid JSON, no additional text.
"""

        try:
            print("🤖 Making single API call to generate all questions...")
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=self.response_model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert research paper quiz generator. Generate comprehensive MCQ quizzes that test deep understanding of academic papers. Always respond with valid JSON only."
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
                    
                    # Convert to QuizQuestion objects
                    questions = []
                    topics_cycle = ["main_concepts", "methodologies", "findings", "technical_details", "applications"]
                    
                    for i, (q_key, q_data) in enumerate(quiz_json.items()):
                        # Extract the correct answer key and ensure it's valid
                        answer_dict = q_data.get("answer", {})
                        if answer_dict:
                            answer_key = list(answer_dict.keys())[0]
                        else:
                            answer_key = "a"  # Default fallback
                        if answer_key not in ['a', 'b', 'c', 'd']:
                            answer_key = "a"
                        topic = q_data.get("topic", topics_cycle[i % len(topics_cycle)])
                        difficulty = q_data.get("difficulty", "medium")
                        page = q_data.get("page", "unknown")
                        reasoning = q_data.get("reasoning", "")
                        question = QuizQuestion(
                            question=q_data.get("question", f"Question {i+1}"),
                            options=q_data.get("options", {
                                "a": "Option A", "b": "Option B", 
                                "c": "Option C", "d": "Option D"
                            }),
                            answer=answer_key,
                            topic=topic,
                            difficulty=difficulty,
                            page=page,
                            reasoning=reasoning
                        )
                        # If LLM did not provide page/reasoning, assign from the most relevant chunk
                        if question.page == "unknown" or not question.reasoning:
                            # Fallback: assign from the first chunk (or you can improve this with keyword matching)
                            if chunks:
                                question.page = str(chunks[0]["page"])
                                question.reasoning = f"This answer is based on content from Page {question.page}."
                        questions.append(question)
                    print(f"✅ Successfully parsed {len(questions)} questions from single API response")
                    return questions[:num_questions]  # Ensure we don't exceed requested count
                else:
                    raise json.JSONDecodeError("No valid JSON found", response_text, 0)
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse quiz JSON: {e}")
                print(f"Response preview: {response_text[:500]}...")
                return self._generate_fallback_questions(chunks, num_questions, difficulty_mix)
        except Exception as e:
            print(f"❌ Quiz generation API call failed: {e}")
            return self._generate_fallback_questions(chunks, num_questions, difficulty_mix)

    def _generate_fallback_questions(self, chunks: List[Dict[str, Any]], num_questions: int, difficulty_mix: Dict[str, int]) -> List[QuizQuestion]:
        """Generate basic fallback questions when API fails"""
        print("🔄 Generating fallback questions...")
        
        questions = []
        content_sample = " ".join([chunk["content"][:100] for chunk in chunks[:3]])
        
        for i in range(min(num_questions, 5)):  # Generate up to 5 fallback questions
            difficulty = "easy" if i < 2 else "medium"
            topic = ["main_concepts", "methodologies", "findings", "technical_details", "applications"][i % 5]
            
            question = QuizQuestion(
                question=f"Based on the research paper content, which of the following is most relevant to {topic.replace('_', ' ')}?",
                options={
                    "a": f"Content related to {topic.replace('_', ' ')} aspect A",
                    "b": f"Content related to {topic.replace('_', ' ')} aspect B", 
                    "c": f"Content related to {topic.replace('_', ' ')} aspect C",
                    "d": f"Content related to {topic.replace('_', ' ')} aspect D"
                },
                answer="a",
                topic=topic,
                difficulty=difficulty
            )
            questions.append(question)
        
        return questions
    
    def _format_quiz_output(self, questions: List[QuizQuestion]) -> Dict[str, Any]:
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

# Example usage function
async def generate_paper_quiz(user_id: str, document_uuid: str, num_questions: int = 10) -> Dict[str, Any]:
    """
    Convenience function to generate a quiz from a research paper
    
    Args:
        user_id: User identifier
        document_uuid: Document UUID
        num_questions: Number of questions to generate (5-20)
    
    Returns:
        Formatted quiz dictionary
    """
    generator = PaperQuizGenerator()
    
    request = QuizRequest(
        user_id=user_id,
        document_uuid=document_uuid,
        num_questions=num_questions
    )
    
    return await generator.generate_quiz(request)

# Test function for development
async def test_quiz_generation():
    """Test function for quiz generation"""
    result = await generate_paper_quiz(
        user_id="5",
        document_uuid="7346b737-9b41-4d9a-a652-4c7b2757bb06", 
        num_questions=5
    )
    
    print("Quiz Generation Test Result:")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(test_quiz_generation())

