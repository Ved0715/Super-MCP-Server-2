import logging
import json
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from pinecone import Pinecone
from openai import OpenAI


@dataclass
class DocumentChunk:
    """Document chunk with metadata"""
    id: str
    content: str
    metadata: Dict = None


@dataclass
class TopicQuestion:
    """Generated question based on topic"""
    question: str
    answer: str
    topic: str
    difficulty: str
    keywords: List[str]


class PaperTopicQuestionGenerator:
    """
    Simple and Optimal Topic-Based Question Generator
    
    Generates relevant questions from document content based on a query
    with minimal API calls and maximum efficiency.
    """
    
    def __init__(self, config):
        """Initialize with config"""
        self.config = config
        self.openai = OpenAI(api_key=self.config.openai_api_key)
        self.pinecone = Pinecone(api_key=self.config.pinecone_api_key)
        self.index = self.pinecone.Index("test")
        self.model = self.config.response_model
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _build_namespace(self, user_id: str, document_uuid: str) -> str:
        """Build namespace for user document"""
        return f"user_{user_id}_doc_{document_uuid}"
    
    def _namespace_exists(self, namespace: str) -> bool:
        """Check if namespace exists"""
        try:
            stats = self.index.describe_index_stats()
            return namespace in stats.get("namespaces", {})
        except Exception as e:
            self.logger.error(f"Error checking namespace: {e}")
            return False
    
    def _search_relevant_content(self, query: str, namespace: str, top_k: int = 15) -> List[DocumentChunk]:
        """Search for relevant content using vector similarity"""
        try:
            # Get query embedding
            query_embedding = self.openai.embeddings.create(
                model="text-embedding-3-large",
                input=query
            )
            query_vector = query_embedding.data[0].embedding
            
            # Search Pinecone
            response = self.index.query(
                vector=query_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            
            return [
                DocumentChunk(
                    id=match["id"],
                    content=match["metadata"].get("text", ""),
                    metadata=match["metadata"]
                )
                for match in response.get("matches", [])
                if match["metadata"].get("text", "").strip()
            ]
            
        except Exception as e:
            self.logger.error(f"Error searching content: {e}")
            return []
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text using simple regex patterns"""
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        # Extract words that are likely important (capitalized, technical terms, etc.)
        words = re.findall(r'\b[A-Za-z]{3,}\b', text.lower())
        keywords = [word for word in words if word not in stop_words]
        
        # Get unique keywords, prioritize longer words and those that appear multiple times
        keyword_counts = {}
        for word in keywords:
            keyword_counts[word] = keyword_counts.get(word, 0) + 1
        
        # Sort by frequency and length, return top keywords
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: (x[1], len(x[0])), reverse=True)
        return [keyword for keyword, _ in sorted_keywords[:10]]
    
    def _generate_questions_efficiently(self, 
                                      query: str, 
                                      chunks: List[DocumentChunk], 
                                      num_questions: int = 5) -> List[TopicQuestion]:
        """Generate questions with a single API call"""
        try:
            # Combine all relevant content
            combined_content = "\n\n".join([chunk.content for chunk in chunks])
            
            # No need to extract page numbers anymore
            
            # Extract keywords locally (no API call needed)
            keywords = self._extract_keywords(combined_content)
            
            # Single API call to generate all questions with answers
            prompt = f"""
You are an expert educator. Based on the query "{query}", generate {num_questions} educational questions with detailed answers from the following document content.

Document Content:
{combined_content[:4000]}  

Key Terms Found: {', '.join(keywords[:15])}

Generate questions that:
1. Are directly answerable from the content
2. Test understanding of "{query}" topic
3. Vary in difficulty (easy, medium, hard)
4. Include comprehensive answers based on the document content

Return ONLY a JSON array with this format:
[
  {{
    "question": "Question text ending with ?",
    "answer": "Write a detailed, comprehensive answer (minimum 4-5 sentences) using only information from the document. Include: definitions of key terms, step-by-step explanations of processes, specific examples mentioned in the text, and the significance of the concept. Quote or reference specific details from the document content provided.",
    "topic": "Specific subtopic",
    "difficulty": "easy|medium|hard",
    "keywords": ["key1", "key2", "key3"]
  }}
]

CRITICAL REQUIREMENTS FOR ANSWERS:
- Minimum 4-5 sentences per answer (detailed explanations)
- Use ONLY information found in the provided document content
- Include specific definitions, processes, and examples from the text
- Explain step-by-step where applicable
- Reference specific details and facts from the document
- Make answers educational and comprehensive for learning
- Connect related concepts mentioned in the document
- Focus on generating diverse, high-quality questions about "{query}"
"""
            
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            questions_data = json.loads(content)
            
            # Convert to TopicQuestion objects
            questions = []
            for q_data in questions_data:
                question = TopicQuestion(
                    question=q_data["question"],
                    answer=q_data.get("answer", "Answer not available"),
                    topic=q_data.get("topic", query),
                    difficulty=q_data.get("difficulty", "medium"),
                    keywords=q_data.get("keywords", keywords[:3])
                )
                questions.append(question)
            
            return questions
            
        except Exception as e:
            self.logger.error(f"Error generating questions: {e}")
            return []
    
    def generate_questions_from_query(self, 
                                    query: str, 
                                    user_id: str, 
                                    document_uuid: str,
                                    num_questions: int = 5) -> Dict[str, Any]:
        """
        Main method to generate questions based on query
        
        Args:
            query: Topic or query to generate questions about
            user_id: User identifier  
            document_uuid: Document UUID
            num_questions: Number of questions to generate (max 5)
            
        Returns:
            Dictionary with generated questions and answers
        """
        try:
            # Limit to max 5 questions
            num_questions = min(num_questions, 5)
            self.logger.info(f"Generating {num_questions} questions for query: '{query}'")
            
            # Check namespace exists
            namespace = self._build_namespace(user_id, document_uuid)
            if not self._namespace_exists(namespace):
                return {
                    "success": False,
                    "error": f"Document not found for user {user_id}",
                    "query": query,
                    "questions": []
                }
            
            # Search for relevant content (1 API call)
            relevant_chunks = self._search_relevant_content(query, namespace, top_k=20)
            
            if not relevant_chunks:
                return {
                    "success": False,
                    "error": "No relevant content found for the query",
                    "query": query, 
                    "questions": []
                }
            
            self.logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            
            # Generate questions efficiently (1 API call)
            questions = self._generate_questions_efficiently(query, relevant_chunks, num_questions)
            
            if not questions:
                return {
                    "success": False,
                    "error": "Failed to generate questions",
                    "query": query,
                    "questions": []
                }
            
            # Prepare response
            result = {
                "success": True,
                "query": query,
                "total_questions": len(questions),
                "questions": [
                    {
                        "question": q.question,
                        "answer": q.answer,
                        "topic": q.topic,
                        "difficulty": q.difficulty,
                        "keywords": q.keywords
                    }
                    for q in questions
                ],
                "metadata": {
                    "chunks_analyzed": len(relevant_chunks),
                    "namespace": namespace,
                    "api_calls_used": 2  # 1 embedding + 1 chat completion
                }
            }
            
            self.logger.info(f"Successfully generated {len(questions)} questions with 2 API calls")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in question generation: {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "questions": []
            }
    
    def get_quick_topic_preview(self, query: str, user_id: str, document_uuid: str) -> Dict[str, Any]:
        """
        Quick preview of topic coverage without API calls to language model
        Uses only vector search to assess content availability
        """
        try:
            namespace = self._build_namespace(user_id, document_uuid)
            
            if not self._namespace_exists(namespace):
                return {"error": "Document not found"}
            
            # Only use vector search - no LLM API call
            relevant_chunks = self._search_relevant_content(query, namespace, top_k=10)
            
            if not relevant_chunks:
                return {
                    "query": query,
                    "coverage": "none",
                    "message": "No content found for this topic"
                }
            
            # Simple analysis without API calls
            total_content_length = sum(len(chunk.content) for chunk in relevant_chunks)
            page_numbers = set()
            
            for chunk in relevant_chunks:
                if chunk.metadata and 'page_number' in chunk.metadata:
                    page_num = chunk.metadata['page_number']
                    if page_num:
                        page_numbers.add(page_num)
            
            # Simple coverage assessment
            if len(relevant_chunks) >= 8:
                coverage = "excellent"
            elif len(relevant_chunks) >= 5:
                coverage = "good"  
            elif len(relevant_chunks) >= 3:
                coverage = "moderate"
            else:
                coverage = "limited"
            
            return {
                "query": query,
                "coverage": coverage,
                "chunks_found": len(relevant_chunks),
                "content_length": total_content_length,
                "pages_covered": len(page_numbers),
                "recommended_questions": min(max(len(relevant_chunks), 3), 15),
                "api_calls_used": 1  # Only embedding for search
            }
            
        except Exception as e:
            self.logger.error(f"Error in topic preview: {e}")
            return {"error": str(e)}


# Example usage
def main():
    """Example usage"""
    print("Hi")
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import AdvancedConfig
    
    config = AdvancedConfig()
    generator = PaperTopicQuestionGenerator(config)
    
    # Generate questions
    result = generator.generate_questions_from_query(
        query="photosynthesis process",
        user_id="61",
        document_uuid="0443d74c-d23b-4360-858a-d9dcd7b60fb8",
        num_questions=5
    )
    
    print("\n🎯 Generated Questions:\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()