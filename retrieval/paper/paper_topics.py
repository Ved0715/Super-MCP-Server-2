import logging
import json
from typing import List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
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

    def _calculate_optimal_batch_size(self, chunks: List[DocumentChunk]) -> int:
        """Calculate optimal batch size based on content and token limits."""
        total_chars = sum(len(chunk.content) for chunk in chunks)
        avg_chunk_size = total_chars / len(chunks) if chunks else 0
        
        # Aim for ~100K characters per batch (roughly 25K tokens)
        # Leave room for prompt and response
        target_chars_per_batch = 100000
        optimal_batch_size = max(10, int(target_chars_per_batch / avg_chunk_size)) if avg_chunk_size > 0 else 50
        
        # Cap at reasonable limits
        return min(optimal_batch_size, 100)

    def _process_batch(self, batch: List[DocumentChunk]) -> Dict:
        """Send a batch of chunks to OpenAI to extract broader, main topics."""
        try:
            combined_text = "\n".join([c.content for c in batch])
            
            prompt = f"""
            You are an expert in document analysis.
            Analyze the following text and extract only the **topics/sections** in the exact order they appear.
            
            For each major topic, write a VERY COMPREHENSIVE and DETAILED summary that captures EVERYTHING covered in that topic section. The summary should be extensive and include:
            - All key concepts, definitions, and explanations
            - All examples, code snippets, and practical demonstrations
            - All subtopics and related concepts discussed
            - All important details, procedures, and methodologies
            - All syntax, rules, and guidelines mentioned
            - All use cases and applications described

            Use natural language patterns like:
            "This topic consists of comprehensive coverage of...", "This section explains in detail...", "this topic covers extensively..."

            Guidelines:
            - Extract MAJOR and Important topics maximum.
            - Focus on broad themes that span multiple paragraphs.
            - Merge related concepts into single comprehensive topics.
            - Maintain document flow (topics in sequence).
            - Make summaries VERY DETAILED and COMPREHENSIVE - include everything discussed.
            - Summaries should be 3-5 sentences minimum, capturing all content.
            - Do not extract trivial or overly specific topics.
            - Each summary should be a complete representation of everything in that topic area.

            Return strictly valid JSON:
            {{
              "topics": [
                 {{"topic": "Broad Topic Title", "summary": "This topic consists of comprehensive and detailed coverage of [include everything - concepts, examples, procedures, syntax, rules, use cases, etc.]"}},
                 ...
              ]
            }}

            Text:
            {combined_text}
            """
            
            resp = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
            )
            
            content = resp.choices[0].message.content.strip()
            logging.info(f"Batch response content: {content[:200]}...")
            
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
                logging.error(f"Invalid batch response structure: {result}")
                return {"topics": []}
            
            return result

        except Exception as e:
            logging.error(f"Error processing batch: {e}")
            return {"topics": []}

    def _merge_and_deduplicate(self, topics: List[Dict]) -> List[Dict]:
        """Use OpenAI to merge duplicate/overlapping topics and preserve flow order."""
        if not topics:
            return []
        
        try:
            topics_text = json.dumps({"topics": topics}, indent=2)

            prompt = f"""
            You are an expert document analyst tasked with merging and consolidating topics from different document sections.
            
            CRITICAL REQUIREMENTS:
            1. **CAPTURE ALL TOPICS**: Ensure every unique topic from the input is represented in the output
            2. **MERGE STRATEGICALLY**: Only merge topics that are truly duplicates or very closely related
            3. **COMPREHENSIVE SUMMARIES**: Create detailed, extensive summaries that include EVERYTHING from merged topics
            
            Your task is to:
            - Identify genuinely duplicate or overlapping topics and merge them intelligently
            - For merged topics, combine ALL content from related summaries into one comprehensive summary
            - Keep unique topics separate - do NOT over-merge distinct concepts
            - Create VERY DETAILED and COMPREHENSIVE summaries that capture:
              • All key concepts, definitions, and explanations from merged topics
              • All examples, code snippets, and practical demonstrations
              • All subtopics, procedures, methodologies, and syntax rules
              • All use cases, applications, and important details
            - Preserve the original document flow/order
            - Use natural language patterns like "This topic consists of comprehensive coverage of...", "Here the document explains extensively..."
            
            MERGING GUIDELINES:
            - Only merge if topics are truly about the same concept (e.g., "Functions" with "Function Definition")
            - Keep distinct concepts separate (e.g., "Functions" vs "Loops" vs "Data Types")
            - When merging, create summaries that are 4-8 sentences long and capture ALL details
            - If uncertain whether to merge, keep topics separate to avoid information loss
            
            Input Topics:
            {topics_text}

            Return strictly valid JSON with ALL topics represented (merged where appropriate):
            {{
              "topics": [
                 {{"topic": "Comprehensive Topic Title", "summary": "This topic consists of extensive and detailed coverage of [include EVERYTHING from all merged related topics - all concepts, examples, procedures, syntax, rules, use cases, applications, etc. Make this very comprehensive]"}},
                 ...
              ]
            }}
            """

            resp = self.openai.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
            
            content = resp.choices[0].message.content.strip()
            logging.info(f"Merge response content: {content[:200]}...")
            
            # Handle markdown code blocks
            if content.startswith("```json"):
                content = content[7:]  # Remove ```json
            if content.startswith("```"):
                content = content[3:]   # Remove ```
            if content.endswith("```"):
                content = content[:-3]  # Remove trailing ```
            content = content.strip()
            
            result = json.loads(content)
            return result.get("topics", topics)

        except Exception as e:
            logging.error(f"Error merging topics: {e}")
            return topics

    def extract_topics(self, user_id: str, document_uuid: str, batch_size: int = None, max_workers: int = 3) -> Dict:
        """Extract important topics + summaries with optimized batching."""
        namespace = self._build_namespace(user_id, document_uuid)
        
        # Check if namespace exists
        if not self._namespace_exists(namespace):
            logging.warning(f"Namespace {namespace} does not exist")
            return {"topics": [], "error": f"Document not found for user {user_id} and document {document_uuid}"}
        
        chunks = self._load_chunks(namespace)
        if not chunks:
            return {"topics": [], "error": "No content found in document"}

        # Calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = self._calculate_optimal_batch_size(chunks)
        
        logging.info(f"Processing {len(chunks)} chunks with optimal batch size of {batch_size}")
        
        # Create batches
        batches = [chunks[i:i+batch_size] for i in range(0, len(chunks), batch_size)]
        logging.info(f"Created {len(batches)} batches for processing")

        # If small document (1-2 batches), process without parallel processing
        if len(batches) <= 2:
            logging.info("Small document detected, processing sequentially")
            results = []
            for i, batch in enumerate(batches):
                batch_result = self._process_batch(batch)
                if batch_result and "topics" in batch_result:
                    logging.info(f"Batch {i} completed with {len(batch_result['topics'])} topics")
                    results.extend(batch_result["topics"])
            
            # For small documents, minimal deduplication needed
            if len(results) <= 15:
                return {"topics": results}
            else:
                merged_topics = self._merge_and_deduplicate(results)
                return {"topics": merged_topics}

        # Process larger documents in parallel
        logging.info("Large document detected, processing in parallel")
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {executor.submit(self._process_batch, b): i for i, b in enumerate(batches)}
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                batch_result = future.result()
                if batch_result and "topics" in batch_result:
                    # logging.info(f"Batch {batch_idx} completed with {len(batch_result['topics'])} topics")
                    results.extend(batch_result["topics"])

        if not results:
            return {"topics": [], "error": "No topics extracted from any batch"}

        logging.info(f"Merging and deduplicating {len(results)} topics from all batches")
        # Final merge + deduplication
        merged_topics = self._merge_and_deduplicate(results)
        
        logging.info(f"Successfully extracted {len(merged_topics)} final topics")
        return {"topics": merged_topics}


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