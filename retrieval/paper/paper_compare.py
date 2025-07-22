import os
import logging
from typing import List, Dict
from dataclasses import dataclass
from pinecone import Pinecone
from openai import OpenAI
from config import config

@dataclass
class DocumentChunk:
    id: str
    content: str

class PDFComparator:
    """
    Loads all chunks for two PDFs from Pinecone and sends them directly
    to a single OpenAI prompt which performs a comprehensive comparison,
    describing similarities and differences in human‐friendly language.
    """

    def __init__(self):
        self.embedding_dimension = config.embedding_dimension
        self.openai = OpenAI(api_key=config.openai_api_key)
        self.pc = Pinecone(api_key=config.pinecone_api_key)
        self.index = self.pc.Index("all-pdfs-index")
        self.model = config.response_model
        logging.basicConfig(level=logging.INFO)

    def _list_namespaces(self, user_id: str) -> List[str]:
        return [
            ns.name for ns in self.index.list_namespaces()
            if ns.name.startswith(f"user_{user_id}_doc_")
        ]

    def _load_chunks(self, namespace: str) -> List[DocumentChunk]:
        dummy = [0.0] * self.embedding_dimension
        resp = self.index.query(
            vector=dummy,
            top_k=10000,
            namespace=namespace,
            include_metadata=True,
            include_values=False
        )
        return [
            DocumentChunk(id=m["id"], content=m["metadata"].get("text",""))
            for m in resp["matches"]
        ]

    def compare_with_llm(
        self,
        user_id: str,
        doc1_uuid: str,
        doc2_uuid: str
    ) -> str:
        # 1. Build namespaces
        ns1 = f"user_{user_id}_doc_{doc1_uuid}"
        ns2 = f"user_{user_id}_doc_{doc2_uuid}"
        available = set(self._list_namespaces(user_id))
        if ns1 not in available or ns2 not in available:
            raise ValueError("One or both document namespaces not found.")

        # 2. Load all chunks
        chunks1 = self._load_chunks(ns1)
        chunks2 = self._load_chunks(ns2)

        # 3. Aggregate text
        text1 = "\n\n---\n\n".join(c.content for c in chunks1)
        text2 = "\n\n---\n\n".join(c.content for c in chunks2)

        # 4. Compose comprehensive prompt
        prompt = f"""
You are an expert document analyst. Compare the two documents below and provide:

1. A clear, human‐friendly overview of how the documents relate.
2. Key points where they share similar themes or information.
3. Detailed differences in content, structure, or phrasing.
4. Any notable observations or recommendations.

Document 1 (all chunks concatenated):
{text1[:3000]}...

Document 2 (all chunks concatenated):
{text2[:3000]}...

Do not use labels like “Text A” or “Text B.” Instead refer to “the first document” and “the second document.” Provide a cohesive narrative that integrates both similarity and distinction.
"""

        # 5. Single API call
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a senior document comparison expert."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()


    def compare(self,
        user_id: str,
        doc1_uuid: str,
        doc2_uuid: str
        ) -> str:

        if not user_id or not doc1_uuid or not doc2_uuid:
            raise ValueError("Please provide user_id, doc1_uuid and doc2_uuid.")

        # Delegate to the LLM-based comparator
        report = self.compare_with_llm(
            user_id=user_id,
            doc1_uuid=doc1_uuid,
            doc2_uuid=doc2_uuid
        )
        return report

    
# Usage example
if __name__ == "__main__":
    from config import config
    comparator = PDFComparator(config)
    report = comparator.compare_with_llm(
        user_id="123",
        doc1_uuid="doc-uuid-1",
        doc2_uuid="doc-uuid-2"
    )
    print(report)
