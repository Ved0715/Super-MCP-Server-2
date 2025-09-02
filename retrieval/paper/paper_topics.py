import os
import logging
import json
from typing import List, Dict
from dataclasses import dataclass
from pinecone import Pinecone
from openai import OpenAI
from config import AdvancedConfig
 

@dataclass
class DocumentChunk:
    id: str
    content: str





class DocumentTopics:
    """
    Retrieves all text chunks of a single document from Pinecone
    and find all the Topics ad there summary in from the document.   
    """ 
    def __init__(self, config):
        """
        Args:
            config.openai_api_key: OpenAI API key
            config.pinecone_api_key: Pinecone API key
            config.response_model: Chat model name (e.g. "gpt-o4-mini")
            config.index_name: Pinecone index name (default "test")
        """
        logging.basicConfig(level=logging.INFO)
        self.config = AdvancedConfig()
        self.openai = OpenAI(api_key=self.config.openai_api_key)
        self.pc = Pinecone(api_key=self.config.pinecone_api_key)
        self.index = self.pc.Index("test")
        self.model = self.config.response_model
        self.embedding_dimension = self.config.embedding_dimension

    def _build_namespace(self, user_id: str, document_uuid: str) -> str:
        """Build the namespace for user-specific document"""
        return f"user_{user_id}_doc_{document_uuid}"
    
    def _list_namespaces(self, user_id: str) -> List[str]:
        """List all namespaces for a user."""
        try:
            return [
                ns.name for ns in self.index.list_namespaces()
                if ns.name.startswith(f"user_{user_id}_doc_")
            ]
        except Exception as e:
            logging.error(f"Error listing namespaces: {e}")
            return []
    
    def _namespace_exists(self, namespace: str) -> bool:
        """Check if namespace exists in the index"""
        try:
            stats = self.index.describe_index_stats()
            return namespace in stats.get("namespaces", {})
        except Exception as e:
            logging.error(f"Error checking namespace existence: {e}")
            return False
    
    def _load_chunks(self, namespace:str) -> List[DocumentChunk]:
        """ Fectch all the text chunks from the given Pinecone namespace."""
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
                DocumentChunk(id=m["id"], content=m["metadata"].get("text", ""))
                for m in resp.get("matches", [])
                if m["metadata"].get("text", "").strip()
            ]
        except Exception as e:
            logging.error(f"Error loading chunks: {e}")
            return []

    def extract_topics(self, user_id: str, document_uuid: str) -> Dict:
        """Send all chunks to OpenAI in one call to extract topics + summaries."""
        namespace = self._build_namespace(user_id, document_uuid)
        
        # Check if namespace exists
        if not self._namespace_exists(namespace):
            logging.warning(f"Namespace {namespace} does not exist")
            return {"topics": [], "error": f"Document not found for user {user_id} and document {document_uuid}"}
        
        chunks = self._load_chunks(namespace)
        if not chunks:
            return {"topics": [], "error": "No content found in document"}

        combined_text = "\n".join([c.content for c in chunks])

        prompt = f"""
        You are an expert in document analysis.
        Analyze the following full document text and extract the **main topics** in the exact order they appear.
        
        For each topic, write a natural summary like:
        "This topic consists of ...", "This section explains ...", or "Here the document covers ...".
        
        Guidelines:
        - Maintain document flow (topics in sequence).
        - Merge repeated ideas automatically.
        - Keep summaries short and natural.
        - Do not invent new topics.

        Return strictly valid JSON in this format:
        {{
          "topics": [
             {{"topic": "Topic Title", "summary": "This topic consists of ..."}},
             ...
          ]
        }}

        Document Text:
        {combined_text}
        """

        try:
            resp = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            content = resp.choices[0].message.content.strip()
            logging.info(f"OpenAI response content: {content[:500]}...")
            
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            # Parse JSON response
            result = json.loads(content)
            
            # Validate response structure
            if not isinstance(result, dict) or "topics" not in result:
                logging.error(f"Invalid response structure: {result}")
                return {"topics": [], "error": "Invalid response structure from OpenAI"}
            
            if not isinstance(result["topics"], list):
                logging.error(f"Topics is not a list: {type(result['topics'])}")
                return {"topics": [], "error": "Topics field is not a list"}
            
            logging.info(f"Successfully extracted {len(result['topics'])} topics")
            return result

        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            logging.error(f"Raw content: {content}")
            return {"topics": [], "error": f"JSON decode error: {str(e)}"}
        except Exception as e:
            logging.error(f"Error extracting topics: {e}")
            return {"topics": [], "error": str(e)}


def main():
    from config import AdvancedConfig
    
    config = AdvancedConfig()
    user_id = "44"
    document_uuid = "4109a094-a6a0-4ec1-bba7-9ce5eb8dcace"     
    
    doc_topics = DocumentTopics(config)
    result = doc_topics.extract_topics(user_id=user_id, document_uuid=document_uuid)

    print("\n📑 Extracted Topics & Summaries:\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()