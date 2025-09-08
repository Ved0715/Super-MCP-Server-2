import logging
import json
import re
from typing import List, Dict
from dataclasses import dataclass
from pinecone import Pinecone
from openai import OpenAI


@dataclass
class DocumentChunk:
    id: str
    content: str
    metadata: Dict = None


class QuestionPaperAnalysis:
    """
    Analyzes question papers by parsing numbered questions from text and providing detailed answers with key topics and references.
    """
    def __init__(self, config):
        self.config = config
        self.openai = OpenAI(api_key=self.config.openai_api_key)
        self.pinecone = Pinecone(api_key=self.config.pinecone_api_key)
        self.index = self.pinecone.Index("test")
        self.model = self.config.response_model

    def _build_namespace(self, user_id: str, document_uuid: str) -> str:
        return f"user_{user_id}_doc_{document_uuid}"

    def _list_namespaces(self, user_id: str) -> List[str]:
        return [ns.name for ns in self.index.list_namespaces() if ns.name.startswith(f"user_{user_id}_doc_")]

    def _namespace_exists(self, namespace: str) -> bool:
        try:
            stats = self.index.describe_index_stats()
            return namespace in stats.get("namespaces", {})
        except Exception as e:
            logging.error(f"Error checking namespace existence: {e}")
            return False

    def _parse_questions(self, questions_text: str) -> List[str]:
        """Parse numbered questions from a single text string."""
        questions = []
        
        # Split by patterns like "Q1.", "Q2.", "1.", "2.", etc.
        question_pattern = r'(?:Q)?(\d+)\.\s*(.+?)(?=(?:Q)?\d+\.|$)'
        matches = re.findall(question_pattern, questions_text, re.DOTALL)
        
        for _, question_content in matches:
            # Clean up the question content - remove extra newlines and trailing Q
            question = re.sub(r'\s*\n\s*Q\s*$', '', question_content.strip())
            question = re.sub(r'\s+', ' ', question)  # Normalize whitespace
            if question:
                questions.append(question)
        
        return questions

    def _load_chunks(self, namespace: str) -> List[DocumentChunk]:
        """Fetch all text chunks from the given Pinecone namespace."""
        try:
            dummy = [0.0] * 3072
            resp = self.index.query(
                vector=dummy,
                top_k=10000,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            return [
                DocumentChunk(id=m["id"], content=m["metadata"].get("text", ""), metadata=m["metadata"])
                for m in resp.get("matches", [])
                if m["metadata"].get("text", "").strip()
            ]
        except Exception as e:
            logging.error(f"Error loading chunks: {e}")
            return []

    def _search_relevant_content(self, question: str, namespace: str, top_k: int = 15) -> List[DocumentChunk]:
        """Search for content relevant to the question using vector similarity."""
        try:
            question_embedding_resp = self.openai.embeddings.create(
                model="text-embedding-3-large",
                input=question
            )
            question_vector = question_embedding_resp.data[0].embedding
            
            resp = self.index.query(
                vector=question_vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                include_values=False
            )
            
            return [
                DocumentChunk(id=m["id"], content=m["metadata"].get("text", ""), metadata=m["metadata"])
                for m in resp.get("matches", [])
                if m["metadata"].get("text", "").strip()
            ]
        except Exception as e:
            logging.error(f"Error searching relevant content: {e}")
            return []

    def _analyze_question_with_topics(self, question: str, relevant_chunks: List[DocumentChunk]) -> Dict:
        """Analyze a single question and extract key topics with references."""
        try:
            combined_text = "\n\n".join([chunk.content for chunk in relevant_chunks])
            
            # Extract actual page numbers from chunk metadata
            page_numbers = []
            for chunk in relevant_chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata and 'page_number' in chunk.metadata:
                    page_num = chunk.metadata['page_number']
                    if page_num and page_num not in page_numbers:
                        page_numbers.append(page_num)
            
            page_references = [f"Page {num}" for num in sorted(page_numbers) if num]
            page_refs_str = ", ".join(page_references) if page_references else "document"
            
            prompt = f"""
            You are an expert academic analyst. Analyze the following question using the provided document content.

            Question: {question}

            Document Content:
            {combined_text}

            Available Page References: {page_refs_str}

            Provide a comprehensive analysis including:
            1. A detailed, well-structured answer to the question
            2. A key concept statement that includes evidence and page references from the document
            3. Page references where relevant information is found

            Return your response in this exact JSON format:
            {{
              "question": "{question}",
              "answer": "Provide a comprehensive, detailed answer to the question using information from the document. Include definitions, explanations, types, significance, and any relevant details.",
              "key_concept": "The main concept with evidence from the document. For example: 'Chemical reactions follow the law of conservation of mass, as discussed in the document on {page_refs_str}. The content demonstrates how iron reacts with copper sulfate to produce iron sulfate and copper, showing the transformation of substances while maintaining total mass.'",
              "page_references": {json.dumps(page_references)}
            }}

            Important formatting requirements for key_concept:
            - Start with the core concept/definition
            - Include "as discussed in the document" or similar phrasing
            - Reference specific evidence from the provided page references
            - Include page references within the concept statement
            - End with significance or what the concept demonstrates/illustrates

            Make sure to:
            - Use information directly from the document content provided
            - Use the actual page references provided: {page_refs_str}
            - Include specific page references in both the key_concept and page_references fields
            - Provide evidence-based explanations
            """
            
            resp = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
            )
            
            content = resp.choices[0].message.content.strip()
            
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
            
        except Exception as e:
            logging.error(f"Error analyzing question: {e}")
            return {
                "question": question,
                "answer": "Error analyzing question",
                "key_concept": "",
                "page_references": []
            }

    def analyze_question_paper(self, questions_text: str, user_id: str, document_uuid: str) -> Dict:
        """
        Main method to analyze a question paper from text format.
        
        Args:
            questions_text: String containing numbered questions like "1. What is...? 2. Explain..."
            user_id: User ID for namespace
            document_uuid: Document UUID for namespace
            
        Returns:
            JSON with analysis for each question including answer, key topics, and references
        """
        namespace = self._build_namespace(user_id, document_uuid)
        
        if not self._namespace_exists(namespace):
            logging.warning(f"Namespace {namespace} does not exist")
            return {"error": f"Document not found for user {user_id} and document {document_uuid}"}
        
        # Parse individual questions
        questions = self._parse_questions(questions_text)
        if not questions:
            return {"error": "No valid questions found in the input text"}
        
        logging.info(f"Found {len(questions)} questions to analyze")
        
        results = []
        for i, question in enumerate(questions, 1):
            logging.info(f"Analyzing question {i}: {question[:50]}...")
            
            relevant_chunks = self._search_relevant_content(question, namespace)
            if not relevant_chunks:
                logging.warning(f"No relevant content found for question: {question}")
                results.append({
                    "question_number": i,
                    "question": question,
                    "answer": "No relevant content found in the document",
                    "key_concept": "",
                    "page_references": []
                })
                continue
            
            analysis = self._analyze_question_with_topics(question, relevant_chunks)
            analysis["question_number"] = i
            results.append(analysis)
        
        return {"question_paper_analysis": results}


def main():
    """Example usage of QuestionPaperAnalysis"""
    from config import AdvancedConfig
    
    config = AdvancedConfig()
    analyzer = QuestionPaperAnalysis(config)
    
    # Example questions text
    questions_text = """
    Q1. How a Thunderstorm become a cyclone ?
    Q2. Name of all the acids and where they are found in.
    Q3. Explain the process of photosynthesis on the plant.
    Q4. What are all the indian breeds of sheep ?
    """
    
    user_id = "61"
    document_uuid = "0443d74c-d23b-4360-858a-d9dcd7b60fb8"
    
    result = analyzer.analyze_question_paper(questions_text, user_id, document_uuid)
    
    print("\n📝 Question Paper Analysis Results:\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

    