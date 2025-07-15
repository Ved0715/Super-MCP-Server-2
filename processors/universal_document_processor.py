import os
import re
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
from tqdm import tqdm

# LlamaParse imports
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.schema import Document

# Local imports
from enhanced_pdf_processor import EnhancedPDFProcessor
from vector_storage import AdvancedVectorStorage, DocumentChunk
from config import Config

@dataclass
class ProcessingConfig:
    """Configuration for different document processing types"""
    document_type: str
    index_name: str
    chunk_size: int
    chunk_overlap: int
    enable_analysis: bool
    extract_citations: bool
    processing_mode: str
    namespace: str
    enable_llamaparse: bool = False
    semantic_density_threshold: float = 0.1

@dataclass
class ProcessedChunk:
    """Enhanced chunk with mathematical and semantic context for knowledge base"""
    id: str
    content: str
    metadata: Dict
    chunk_type: str  # 'text', 'formula', 'table', 'mixed'
    mathematical_entities: List[str]
    semantic_density: float
    page_references: List[int]
    section_hierarchy: Dict

class UniversalDocumentProcessor:
    """Universal document processor that handles both research papers and knowledge base content"""
    
    def __init__(self):
        """Initialize the universal processor with both processing pipelines"""
        self.config = Config()
        
        # Initialize both processors
        self.research_processor = EnhancedPDFProcessor(self.config)
        self.vector_storage = AdvancedVectorStorage(self.config)
        
        # Initialize LlamaParse for knowledge base processing
        self.llama_parser = None
        self.processed_cache = {}
        self.setup_llamaparse()
        self.setup_directories()
    
    def setup_llamaparse(self):
        """Initialize LlamaParse for superior academic content extraction"""
        llama_parse_key = os.getenv("LLAMA_PARSE_API_KEY")
        if not llama_parse_key:
            print("⚠️ LlamaParse API key not found. Knowledge base processing will use fallback.")
            return
            
        try:
            self.llama_parser = LlamaParse(
                api_key=llama_parse_key,
                result_type="text",
                verbose=True,
                language="en",
                system_prompt="""
                Extract technical/academic content preserving:
                1. Mathematical expressions and formulas
                2. Table structure and relationships  
                3. Section hierarchies and headers
                4. Citations and references
                5. Figure and table captions
                """,
                max_timeout=60000,
            )
            print("✅ LlamaParse initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing LlamaParse: {e}")
            self.llama_parser = None
    
    def setup_directories(self):
        """Create necessary directories"""
        for directory in ["cache", "exports"]:
            Path(directory).mkdir(exist_ok=True)
    
    def get_processing_config(self, document_type: str, index_name: str) -> ProcessingConfig:
        """Get processing configuration based on document type"""
        if document_type == "research_paper":
            return ProcessingConfig(
                document_type="research_paper",
                index_name=index_name,
                chunk_size=1500,
                chunk_overlap=200,
                enable_analysis=True,
                extract_citations=True,
                processing_mode="academic",
                namespace="default",
                enable_llamaparse=False,
                semantic_density_threshold=0.0
            )
        elif document_type == "knowledge_base":
            return ProcessingConfig(
                document_type="knowledge_base",
                index_name=index_name,
                chunk_size=800,
                chunk_overlap=100,
                enable_analysis=False,
                extract_citations=False,
                processing_mode="knowledge_extraction",
                namespace="knowledge-base",
                enable_llamaparse=True,
                semantic_density_threshold=0.1
            )
        else:
            raise ValueError(f"Unknown document type: {document_type}")
    
    async def process_document(self, 
                              file_content: bytes,
                              document_type: str, 
                              index_name: str,
                              paper_id: Optional[str] = None,
                              book_name: Optional[str] = None) -> Dict[str, Any]:
        """Universal document processing entry point"""
        
        # Get appropriate processing configuration
        processing_config = self.get_processing_config(document_type, index_name)
        
        # Route to appropriate processor
        if document_type == "research_paper":
            return await self._process_research_paper(
                file_content, processing_config, paper_id
            )
        elif document_type == "knowledge_base":
            return await self._process_knowledge_content(
                file_content, processing_config, book_name
            )
        else:
            raise ValueError(f"Unsupported document type: {document_type}")
    
    async def _process_research_paper(self, 
                                    file_content: bytes, 
                                    config: ProcessingConfig,
                                    paper_id: Optional[str] = None) -> Dict[str, Any]:
        """Process research papers using existing enhanced PDF processor"""
        try:
            print(f"🔬 Processing research paper with enhanced PDF processor...")
            
            # Use existing research paper processing pipeline
            result = await self.research_processor.extract_content_from_bytes(
                file_content=file_content,
                file_name=f"{paper_id or 'research_paper'}.pdf"
            )
            
            # Create chunks from extracted content and store in vector database
            if result.get("success"):
                content = result.get("content", "")
                if content:
                    # Create intelligent chunks for research papers
                    chunks = self._create_research_paper_chunks(
                        content=content,
                        metadata=result.get("metadata", {}),
                        sections=result.get("sections", {}),
                        paper_id=paper_id,
                        config=config
                    )
                    
                    if chunks:
                        # Store in vector database  
                        await self.vector_storage.store_document_chunks(
                            chunks=chunks,
                            namespace=config.namespace,
                            index_name=config.index_name
                        )
                        
                        print(f"✅ Research paper stored in {config.index_name} with {len(chunks)} chunks")
                        
                        # Add chunks info to result
                        result["chunks_created"] = len(chunks)
                        result["chunks"] = [{"id": c.id, "content": c.content[:100] + "...", "metadata": c.metadata} for c in chunks[:3]]
            
            return result
            
        except Exception as e:
            print(f"❌ Research paper processing failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_research_paper_chunks(self, content: str, metadata: Dict, sections: Dict, 
                                    paper_id: Optional[str], config: ProcessingConfig) -> List[DocumentChunk]:
        """Create intelligent chunks for research papers"""
        chunks = []
        chunk_id_counter = 0
        
        # Process sections if available
        if sections:
            for section_name, section_content in sections.items():
                if not section_content or len(section_content.strip()) < 50:
                    continue
                
                # Split section into chunks
                section_chunks = self._split_text_into_chunks(
                    text=section_content,
                    chunk_size=config.chunk_size,
                    chunk_overlap=config.chunk_overlap
                )
                
                for chunk_text in section_chunks:
                    chunk = DocumentChunk(
                        id=f"{paper_id or 'paper'}_{chunk_id_counter:03d}",
                        content=chunk_text,
                        embedding=None,
                        metadata={
                            **metadata,
                            "paper_id": paper_id,
                            "section": section_name,
                            "document_type": "research_paper",
                            "processing_method": "enhanced_pdf_processor",
                            "chunk_type": "text"
                        },
                        section=section_name,
                        chunk_index=chunk_id_counter
                    )
                    chunks.append(chunk)
                    chunk_id_counter += 1
        else:
            # Fallback: split full content
            content_chunks = self._split_text_into_chunks(
                text=content,
                chunk_size=config.chunk_size,
                chunk_overlap=config.chunk_overlap
            )
            
            for chunk_text in content_chunks:
                chunk = DocumentChunk(
                    id=f"{paper_id or 'paper'}_{chunk_id_counter:03d}",
                    content=chunk_text,
                    embedding=None,
                    metadata={
                        **metadata,
                        "paper_id": paper_id,
                        "document_type": "research_paper",
                        "processing_method": "enhanced_pdf_processor",
                        "chunk_type": "text"
                    },
                    chunk_index=chunk_id_counter
                )
                chunks.append(chunk)
                chunk_id_counter += 1
        
        return chunks

    def _split_text_into_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings near the end
                for i in range(end, max(start + chunk_size - 200, start), -1):
                    if text[i:i+2] in ['. ', '.\n', '!\n', '?\n']:
                        end = i + 1
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def _process_knowledge_content(self, 
                                       file_content: bytes, 
                                       config: ProcessingConfig,
                                       book_name: Optional[str] = None) -> Dict[str, Any]:
        """Process knowledge base content using optimized pipeline"""
        try:
            print(f"📚 Processing knowledge base content with optimized pipeline...")
            
            # Save file temporarily for LlamaParse processing
            temp_file = f"temp/temp_book_{int(asyncio.get_event_loop().time())}.pdf"
            Path("temp").mkdir(exist_ok=True)
            
            with open(temp_file, "wb") as f:
                f.write(file_content)
            
            try:
                # Parse document with LlamaParse or fallback
                documents = await self._parse_document_with_llamaparse(temp_file)
                
                if not documents:
                    return {"success": False, "error": "Failed to extract content from document"}
                
                # Extract book name if not provided
                if not book_name:
                    book_name = self._extract_book_name(temp_file)
                
                # Create enhanced chunks
                processed_chunks = self._create_enhanced_chunks(documents, book_name)
                
                if not processed_chunks:
                    return {"success": False, "error": "No content chunks created"}
                
                # Convert to DocumentChunk format and store in vector database
                doc_chunks = []
                for chunk in processed_chunks:
                    doc_chunk = DocumentChunk(
                        id=chunk.id,
                        content=chunk.content,
                        embedding=None,  # Will be generated during storage
                        metadata={
                            **chunk.metadata,
                            "book_name": book_name,
                            "chunk_type": chunk.chunk_type,
                            "semantic_density": chunk.semantic_density,
                            "mathematical_entities": chunk.mathematical_entities,
                            "has_formulas": len(chunk.mathematical_entities) > 0,
                            "math_entities_count": len(chunk.mathematical_entities),
                            "chapters_found": chunk.section_hierarchy.get("chapters", []),
                            "sections_found": chunk.section_hierarchy.get("sections", [])
                        },
                        chunk_index=0  # Will be set properly during storage
                    )
                    doc_chunks.append(doc_chunk)
                
                # Store in optimized knowledge base index
                await self.vector_storage.store_document_chunks(
                    chunks=doc_chunks,
                    namespace=config.namespace,
                    index_name=config.index_name
                )
                
                print(f"✅ Knowledge base content stored in {config.index_name} with {len(doc_chunks)} chunks")
                
                return {
                    "success": True,
                    "book_name": book_name,
                    "chunks_created": len(doc_chunks),
                    "processing_stats": {
                        "total_chunks": len(processed_chunks),
                        "mathematical_chunks": sum(1 for c in processed_chunks if c.mathematical_entities),
                        "average_semantic_density": sum(c.semantic_density for c in processed_chunks) / len(processed_chunks),
                        "chunk_types": {chunk_type: sum(1 for c in processed_chunks if c.chunk_type == chunk_type) 
                                       for chunk_type in set(c.chunk_type for c in processed_chunks)}
                    }
                }
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    
        except Exception as e:
            print(f"❌ Knowledge base processing failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _parse_document_with_llamaparse(self, file_path: str) -> List[Document]:
        """Parse document using LlamaParse for superior extraction"""
        if not self.llama_parser:
            return self._fallback_parsing(file_path)
        
        try:
            print(f"🦙 Processing with LlamaParse: {Path(file_path).name}")
            
            reader = SimpleDirectoryReader(
                input_files=[file_path],
                file_extractor={".pdf": self.llama_parser}
            )
            
            documents = await asyncio.to_thread(reader.load_data)
            print(f"✅ LlamaParse extracted {len(documents)} document sections")
            return documents
            
        except Exception as e:
            print(f"❌ LlamaParse failed: {e}. Using fallback processing.")
            return self._fallback_parsing(file_path)
    
    def _fallback_parsing(self, file_path: str) -> List[Document]:
        """Fallback parsing using basic PDF processing"""
        try:
            import pdfplumber
            
            content = []
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text()
                        if text and len(text.strip()) > 50:
                            content.append(f"Page {page_num + 1}:\n{text}")
                    except Exception as e:
                        print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                        continue
            
            if content:
                full_text = "\n\n".join(content)
                doc = Document(
                    text=full_text,
                    metadata={
                        "file_path": file_path,
                        "extraction_method": "fallback",
                        "total_pages": len(content)
                    }
                )
                return [doc]
            
        except Exception as e:
            print(f"❌ Fallback parsing also failed: {e}")
        
        return []
    
    def _create_enhanced_chunks(self, documents: List[Document], book_name: str) -> List[ProcessedChunk]:
        """Create enhanced chunks optimized for knowledge base content"""
        all_chunks = []
        
        for doc_idx, document in enumerate(documents):
            sections = self._split_into_semantic_sections(document.text)
            
            for section_idx, section_text in enumerate(sections):
                chunks = self._create_chunks_from_section(
                    section_text, doc_idx, section_idx, book_name, document.metadata or {}
                )
                all_chunks.extend(chunks)
        
        print(f"📊 Created {len(all_chunks)} enhanced chunks for {book_name}")
        return all_chunks
    
    def _split_into_semantic_sections(self, text: str) -> List[str]:
        """Split text into semantic sections based on headers and natural breaks"""
        # Split on chapter/section headers
        section_pattern = r'\n(?=(?:Chapter|Section|\d+\.|[A-Z][^a-z]*\n))'
        sections = re.split(section_pattern, text)
        
        # Filter out very short sections
        sections = [s.strip() for s in sections if len(s.strip()) > 100]
        
        return sections if sections else [text]
    
    def _create_chunks_from_section(self, section_text: str, doc_idx: int, section_idx: int, 
                                  book_name: str, doc_metadata: Dict) -> List[ProcessedChunk]:
        """Create optimized chunks from a section"""
        chunks = []
        chunk_size = 800  # Optimized for knowledge base
        chunk_overlap = 100
        
        # Clean text
        clean_text = self._clean_text_artifacts(section_text)
        
        # Split into chunks with overlap
        words = clean_text.split()
        if len(words) <= chunk_size:
            # Single chunk
            chunk = self._create_processed_chunk(
                clean_text, doc_idx, section_idx, 0, book_name, doc_metadata
            )
            chunks.append(chunk)
        else:
            # Multiple chunks with overlap
            chunk_idx = 0
            start = 0
            
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_words = words[start:end]
                chunk_text = " ".join(chunk_words)
                
                chunk = self._create_processed_chunk(
                    chunk_text, doc_idx, section_idx, chunk_idx, book_name, doc_metadata
                )
                chunks.append(chunk)
                
                # Move to next chunk with overlap
                start = end - chunk_overlap
                chunk_idx += 1
                
                if end >= len(words):
                    break
        
        return chunks
    
    def _create_processed_chunk(self, text: str, doc_idx: int, section_idx: int, 
                              chunk_idx: int, book_name: str, doc_metadata: Dict) -> ProcessedChunk:
        """Create a processed chunk with enhanced metadata"""
        
        # Extract mathematical entities
        math_entities = self._extract_mathematical_entities(text)
        
        # Calculate semantic density
        semantic_density = self._calculate_semantic_density(text)
        
        # Detect section hierarchy
        section_hierarchy = self._detect_section_hierarchy(text)
        
        # Extract page references
        page_refs = self._extract_page_references(text)
        
        # Determine chunk type
        chunk_type = self._determine_chunk_type(text, math_entities)
        
        chunk_id = f"{book_name.replace(' ', '_')}_{doc_idx}_{section_idx}_{chunk_idx}"
        
        return ProcessedChunk(
            id=chunk_id,
            content=text,
            metadata={
                "word_count": len(text.split()),
                "book_name": book_name,
                "doc_index": doc_idx,
                "section_index": section_idx,
                "chunk_index": chunk_idx,
                **doc_metadata
            },
            chunk_type=chunk_type,
            mathematical_entities=math_entities,
            semantic_density=semantic_density,
            page_references=page_refs,
            section_hierarchy=section_hierarchy
        )
    
    def _extract_mathematical_entities(self, text: str) -> List[str]:
        """Extract mathematical formulas, equations, and technical terms"""
        math_entities = []
        
        # LaTeX math expressions
        latex_patterns = [
            r'\$[^$]+\$',  # Inline math
            r'\$\$[^$]+\$\$',  # Display math
            r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}',  # LaTeX environments
            r'\\[a-zA-Z]+(?:\[[^\]]*\])?(?:\{[^}]*\})*',  # LaTeX commands
        ]
        
        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            math_entities.extend(matches)
        
        # Mathematical symbols and expressions
        math_symbols = [
            r'[α-ωΑ-Ω]',  # Greek letters
            r'[∀∃∈∉⊂⊃∪∩∅∞≤≥≠≈≡∝∼∂∇∫∮∏∑]',  # Mathematical symbols
            r'\b(?:theorem|lemma|proof|corollary|proposition)\b',
            r'\b(?:matrix|vector|tensor|eigenvalue|eigenvector)\b',
            r'\b(?:probability|distribution|variance|correlation)\b',
            r'\b(?:algorithm|complexity|optimization|gradient)\b',
        ]
        
        for pattern in math_symbols:
            matches = re.findall(pattern, text, re.IGNORECASE)
            math_entities.extend(matches)
        
        return list(set(math_entities))
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density based on technical content"""
        technical_terms = [
            'algorithm', 'function', 'variable', 'parameter', 'hypothesis',
            'theorem', 'proof', 'lemma', 'corollary', 'definition',
            'analysis', 'optimization', 'regression', 'classification',
            'probability', 'distribution', 'matrix', 'vector', 'tensor'
        ]
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        technical_count = sum(1 for word in words if any(term in word for term in technical_terms))
        math_entities_count = len(self._extract_mathematical_entities(text))
        
        semantic_score = (technical_count + math_entities_count * 2) / len(words)
        return min(semantic_score, 1.0)
    
    def _detect_section_hierarchy(self, text: str) -> Dict:
        """Detect section hierarchy and structure"""
        hierarchy = {
            "chapters": [],
            "sections": [],
            "level": 0
        }
        
        # Chapter patterns
        chapter_patterns = [
            r'^Chapter\s+(\d+)[:\.]?\s*([A-Z][A-Za-z\s\-\(\):]{10,100})',
            r'^Part\s+([IVX]+)[:\.]?\s*([A-Z][A-Za-z\s\-\(\):]{10,100})',
            r'^(\d{1,2})\.\s+([A-Z][A-Za-z\s\-\(\):]{20,100})',
        ]
        
        # Section patterns  
        section_patterns = [
            r'^(\d+\.\d+)\s*(.+)$',
            r'^Section\s+(\d+\.\d+)[:\.]?\s*(.+)$',
        ]
        
        lines = text.split('\n')[:10]
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Check for chapters
            for pattern in chapter_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    chapter_title = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    hierarchy["chapters"].append(chapter_title.strip())
                    hierarchy["level"] = 1
                    break
            
            # Check for sections
            for pattern in section_patterns:
                match = re.match(pattern, line, re.MULTILINE)
                if match:
                    section_title = match.group(2) if len(match.groups()) > 1 else match.group(1)
                    hierarchy["sections"].append(section_title.strip())
                    hierarchy["level"] = max(hierarchy["level"], 2)
                    break
        
        return hierarchy
    
    def _determine_chunk_type(self, text: str, math_entities: List[str]) -> str:
        """Determine the type of chunk based on content"""
        if len(math_entities) > 5:
            return "mathematical"
        elif "table" in text.lower() or "|" in text:
            return "table"
        elif any(keyword in text.lower() for keyword in ["figure", "diagram", "chart"]):
            return "figure"
        else:
            return "text"
    
    def _extract_page_references(self, text: str) -> List[int]:
        """Extract page references from text"""
        page_refs = []
        page_patterns = [
            r'page\s+(\d+)',
            r'p\.\s*(\d+)',
            r'\[(\d+)\]'
        ]
        
        for pattern in page_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            page_refs.extend([int(match) for match in matches])
        
        return sorted(list(set(page_refs)))
    
    def _clean_text_artifacts(self, text: str) -> str:
        """Clean text artifacts from PDF extraction"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page headers/footers patterns
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Clean up line breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text.strip()
    
    def _extract_book_name(self, file_path: str) -> str:
        """Extract book name from file path"""
        name = Path(file_path).stem
        
        # Clean common suffixes
        name = re.sub(r'_(?:book|pdf|final|v\d+)$', '', name, flags=re.IGNORECASE)
        
        # Replace underscores with spaces and title case
        name = name.replace('_', ' ').replace('-', ' ')
        name = ' '.join(word.capitalize() for word in name.split())
        
        return name 