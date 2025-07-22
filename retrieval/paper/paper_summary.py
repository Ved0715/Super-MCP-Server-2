import os
import logging
from typing import List
from dataclasses import dataclass
from pinecone import Pinecone
from openai import OpenAI

@dataclass
class DocumentChunk:
    id: str
    content: str

class DocumentSummarizer:
    """
    Retrieves all text chunks of a single document from Pinecone
    and generates a comprehensive summary, analysis, and conclusion
    in one well-defined LLM prompt.
    """

    def __init__(self, config):
        """
        Args:
            config.openai_api_key: OpenAI API key
            config.pinecone_api_key: Pinecone API key
            config.response_model: Chat model name (e.g. "gpt-4")
            config.index_name: Pinecone index name (default "all-pdfs-index")
        """
        logging.basicConfig(level=logging.INFO)
        self.openai = OpenAI(api_key=config.openai_api_key)
        self.pc     = Pinecone(api_key=config.pinecone_api_key)
        self.index  = self.pc.Index(config.index_name or "all-pdfs-index")
        self.model  = config.response_model

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

    def _load_chunks(self, namespace: str) -> List[DocumentChunk]:
        """Fetch all text chunks from the given Pinecone namespace."""
        stats = self.index.describe_index_stats()  # no filter on Starter plan
        dim   = stats["namespaces"].get(namespace, {}).get("dimension")
        dummy = [0.0] * dim
        resp  = self.index.query(
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

    def summarize(self, user_id: str, doc_uuid: str) -> str:
        """
        Public method: retrieves all chunks for one document
        and returns a detailed summary, analysis, and conclusion.
        """
        # 1. Resolve namespace
        ns = f"user_{user_id}_doc_{doc_uuid}"
        available = set(self._list_namespaces(user_id))
        if ns not in available:
            raise ValueError(f"Document namespace not found: {ns}")

        # 2. Load chunks
        chunks = self._load_chunks(ns)
        if not chunks:
            raise ValueError("No text chunks found in document.")

        # 3. Concatenate text with clear separators
        full_text = "\n\n---\n\n".join(c.content for c in chunks)
        # Truncate to fit model context if needed
        snippet = full_text[:6000] + ("..." if len(full_text) > 6000 else "")

        # 4. Build comprehensive prompt
        prompt = f"""
You are a senior document analyst. Below is the full content of a single document (chunks separated by '---'):

{snippet}

Please provide:
1. A concise summary of what this document is about.
2. A detailed analysis highlighting key points, structure, and themes.
3. A clear conclusion on the document's overall message or purpose.

Write in clear, human-friendly language and organize your response under headings: "Summary", "Analysis", and "Conclusion".
"""

        # 5. Single API call
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert document summarizer."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=800,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()

# Usage Example
if __name__ == "__main__":
    from config import config

    summarizer = DocumentSummarizer(config)
    report = summarizer.summarize(
        user_id="123",
        doc_uuid="abcd-ef12-3456"
    )
    print(report)
