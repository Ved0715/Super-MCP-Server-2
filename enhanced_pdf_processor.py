"""
📄 Enhanced PDF Processor
Advanced PDF processing with LlamaParse and multi-modal extraction
"""

import os
import logging
import tempfile
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import io

from llama_parse import LlamaParse
from pypdf import PdfReader
import pandas as pd
import re

logger = logging.getLogger(__name__)

class EnhancedPDFProcessor:
    """Enhanced PDF processor with academic research focus"""
    
    def __init__(self, config):
        self.config = config
        
        # Initialize LlamaParse if API key available
        if config.LLAMA_PARSE_API_KEY:
            self.llama_parser = LlamaParse(
                api_key=config.LLAMA_PARSE_API_KEY,
                result_type="markdown",
                instructions=self._get_parsing_instructions(),
                max_timeout=60,
                split_by_page=True,
                use_vendor_multimodal_model=True
            )
        else:
            self.llama_parser = None
            logger.warning("LlamaParse not available - using fallback PDF processing")

    def _get_parsing_instructions(self) -> str:
        """Get specialized parsing instructions for academic papers"""
        return """
        This is an academic research paper. Please extract and preserve:
        
        1. STRUCTURE: Maintain clear section headers (Abstract, Introduction, Methods, Results, Discussion, Conclusion)
        2. TABLES: Extract all tables with proper formatting and preserve data relationships
        3. FIGURES: Describe figure content and captions in detail
        4. EQUATIONS: Preserve mathematical equations and formulas
        5. CITATIONS: Maintain all in-text citations and reference formatting
        6. METADATA: Extract author information, title, abstract, keywords
        7. STATISTICS: Preserve all statistical results, p-values, confidence intervals
        8. METHODOLOGY: Extract detailed methodological procedures
        
        Format the output in clean markdown with proper headers and preserve the academic structure.
        """

    async def extract_content_from_file(self, file_path: str) -> Dict[str, Any]:
        """Extract content from PDF file with advanced processing"""
        try:
            if not os.path.exists(file_path):
                return {"success": False, "error": f"File not found: {file_path}"}
            
            with open(file_path, 'rb') as file:
                file_content = file.read()
            
            return await self.extract_content_from_bytes(file_content, os.path.basename(file_path))
            
        except Exception as e:
            logger.error(f"Error reading PDF file {file_path}: {e}")
            return {"success": False, "error": f"Failed to read file: {str(e)}"}

    async def extract_content_from_bytes(self, file_content: bytes, file_name: str = "research_paper.pdf") -> Dict[str, Any]:
        """Extract content from PDF bytes with comprehensive processing"""
        try:
            # Primary extraction with LlamaParse
            if self.llama_parser:
                primary_result = await self._extract_with_llamaparse(file_content, file_name)
                if primary_result["success"]:
                    # Enhance with additional processing
                    enhanced_result = await self._enhance_extraction(primary_result, file_content, file_name)
                    return enhanced_result
                else:
                    logger.warning("LlamaParse failed, falling back to pypdf")
            
            # Fallback extraction with pypdf
            fallback_result = await self._extract_with_pypdf(file_content, file_name)
            return fallback_result
            
        except Exception as e:
            logger.error(f"Error in PDF extraction: {e}")
            return {"success": False, "error": f"PDF extraction failed: {str(e)}"}

    async def _extract_with_llamaparse(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Extract content using LlamaParse"""
        try:
            # Create temporary file for LlamaParse
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_content)
                tmp_path = tmp_file.name
            
            try:
                # Parse with LlamaParse
                documents = self.llama_parser.load_data(tmp_path)
                
                # Process LlamaParse results
                result = await self._process_llamaparse_output(documents, file_name)
                
                return result
                
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                    
        except Exception as e:
            logger.error(f"LlamaParse extraction failed: {e}")
            return {"success": False, "error": str(e)}

    async def _process_llamaparse_output(self, documents: List, file_name: str) -> Dict[str, Any]:
        """Process LlamaParse output into structured format"""
        try:
            # Combine all document text
            full_text = ""
            pages = []
            
            for i, doc in enumerate(documents):
                page_text = doc.text if hasattr(doc, 'text') else str(doc)
                full_text += page_text + "\n\n"
                
                pages.append({
                    "page_number": i + 1,
                    "text": page_text,
                    "word_count": len(page_text.split()),
                    "extracted_with": "llamaparse"
                })
            
            # Extract metadata
            metadata = self._extract_enhanced_metadata(full_text, file_name, len(pages))
            
            # Extract sections with advanced parsing
            sections = await self._extract_advanced_sections(full_text)
            
            # Extract tables and figures
            tables = self._extract_tables_from_markdown(full_text)
            figures = self._extract_figures_from_markdown(full_text)
            
            # Extract research elements
            research_elements = await self._extract_research_elements(full_text, sections)
            
            return {
                "success": True,
                "content": full_text,
                "pages": pages,
                "sections": sections,
                "metadata": metadata,
                "tables": tables,
                "figures": figures,
                "research_elements": research_elements,
                "summary_stats": {
                    "total_pages": len(pages),
                    "total_words": len(full_text.split()),
                    "total_characters": len(full_text),
                    "extraction_method": "llamaparse_enhanced"
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing LlamaParse output: {e}")
            return {"success": False, "error": str(e)}

    async def _extract_with_pypdf(self, file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Fallback extraction using pypdf"""
        try:
            reader = PdfReader(io.BytesIO(file_content))
            
            # Extract basic metadata
            metadata = self._extract_basic_metadata(reader, file_name)
            
            # Extract text from all pages
            pages = []
            full_text = []
            
            for page_num, page in enumerate(reader.pages, 1):
                try:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        pages.append({
                            "page_number": page_num,
                            "text": page_text,
                            "word_count": len(page_text.split()),
                            "extracted_with": "pypdf"
                        })
                        full_text.append(page_text)
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue
            
            combined_text = "\n\n".join(full_text)
            
            # Extract sections with basic parsing
            sections = await self._extract_basic_sections(combined_text)
            
            # Extract research elements
            research_elements = await self._extract_research_elements(combined_text, sections)
            
            return {
                "success": True,
                "content": combined_text,
                "pages": pages,
                "sections": sections,
                "metadata": metadata,
                "tables": [],
                "figures": [],
                "research_elements": research_elements,
                "summary_stats": {
                    "total_pages": len(pages),
                    "total_words": len(combined_text.split()),
                    "total_characters": len(combined_text),
                    "extraction_method": "pypdf_basic"
                }
            }
            
        except Exception as e:
            logger.error(f"Error in pypdf extraction: {e}")
            return {"success": False, "error": str(e)}

    async def _enhance_extraction(self, primary_result: Dict[str, Any], file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Enhance extraction with additional processing"""
        try:
            # Add citation analysis
            citations = self._analyze_citations(primary_result["content"])
            primary_result["citations"] = citations
            
            # Add statistical analysis
            statistics = self._extract_statistical_content(primary_result["content"])
            primary_result["statistics"] = statistics
            
            # Add academic structure validation
            structure_quality = self._assess_academic_structure(primary_result["sections"])
            primary_result["structure_quality"] = structure_quality
            
            # Add content quality metrics
            quality_metrics = self._calculate_quality_metrics(primary_result["content"])
            primary_result["quality_metrics"] = quality_metrics
            
            return primary_result
            
        except Exception as e:
            logger.error(f"Error enhancing extraction: {e}")
            return primary_result

    async def _extract_advanced_sections(self, full_text: str) -> Dict[str, str]:
        """Extract sections with advanced academic structure awareness"""
        sections = {}
        
        # Enhanced section patterns for academic papers
        section_patterns = {
            "abstract": [
                r"(?i)^#*\s*abstract\s*#*\s*\n(.*?)(?=\n#|\Z)",
                r"(?i)abstract[:.\s]*\n(.*?)(?=\n\n[A-Z]|\Z)"
            ],
            "introduction": [
                r"(?i)^#*\s*(?:1\.?\s*)?introduction\s*#*\s*\n(.*?)(?=\n#|\Z)",
                r"(?i)^#*\s*background\s*#*\s*\n(.*?)(?=\n#|\Z)"
            ],
            "literature_review": [
                r"(?i)^#*\s*(?:2\.?\s*)?(?:literature review|related work|previous work)\s*#*\s*\n(.*?)(?=\n#|\Z)"
            ],
            "methodology": [
                r"(?i)^#*\s*(?:3\.?\s*)?(?:methodology|methods|materials and methods|approach)\s*#*\s*\n(.*?)(?=\n#|\Z)"
            ],
            "results": [
                r"(?i)^#*\s*(?:4\.?\s*)?(?:results|findings|experiments|evaluation)\s*#*\s*\n(.*?)(?=\n#|\Z)"
            ],
            "discussion": [
                r"(?i)^#*\s*(?:5\.?\s*)?(?:discussion|analysis|interpretation)\s*#*\s*\n(.*?)(?=\n#|\Z)"
            ],
            "conclusion": [
                r"(?i)^#*\s*(?:6\.?\s*)?(?:conclusion|conclusions|summary|future work)\s*#*\s*\n(.*?)(?=\n#|\Z)"
            ],
            "references": [
                r"(?i)^#*\s*(?:references|bibliography|works cited)\s*#*\s*\n(.*?)(?=\Z)",
            ]
        }
        
        for section_name, patterns in section_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, full_text, re.MULTILINE | re.DOTALL)
                if match:
                    content = match.group(1).strip()
                    if len(content) > 100:  # Only substantial sections
                        sections[section_name] = content
                        break
        
        return sections

    async def _extract_basic_sections(self, full_text: str) -> Dict[str, str]:
        """Basic section extraction for pypdf fallback"""
        sections = {}
        lines = full_text.split('\n')
        
        current_section = None
        current_content = []
        
        section_keywords = {
            "abstract": ["abstract", "summary"],
            "introduction": ["introduction", "1. introduction", "1 introduction"],
            "methodology": ["methodology", "methods", "materials and methods"],
            "results": ["results", "findings", "experiments"],
            "discussion": ["discussion", "analysis"],
            "conclusion": ["conclusion", "conclusions", "summary"],
            "references": ["references", "bibliography", "works cited"]
        }
        
        for line in lines:
            line_lower = line.lower().strip()
            
            # Check if line is a section header
            detected_section = None
            for section_name, keywords in section_keywords.items():
                if any(keyword in line_lower for keyword in keywords):
                    detected_section = section_name
                    break
            
            if detected_section:
                # Save previous section
                if current_section and current_content:
                    sections[current_section] = "\n".join(current_content).strip()
                
                # Start new section
                current_section = detected_section
                current_content = []
            elif current_section and line.strip():
                current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            sections[current_section] = "\n".join(current_content).strip()
        
        return sections

    async def _extract_research_elements(self, full_text: str, sections: Dict[str, str]) -> Dict[str, Any]:
        """Extract specific research elements"""
        research_elements = {}
        
        # Extract research questions
        research_questions = self._extract_research_questions(full_text, sections)
        research_elements["research_questions"] = research_questions
        
        # Extract hypotheses
        hypotheses = self._extract_hypotheses(full_text)
        research_elements["hypotheses"] = hypotheses
        
        # Extract key findings
        key_findings = self._extract_key_findings(sections.get("results", ""))
        research_elements["key_findings"] = key_findings
        
        # Extract methodology type
        methodology_type = self._classify_methodology(sections.get("methodology", ""))
        research_elements["methodology_type"] = methodology_type
        
        # Extract limitations
        limitations = self._extract_limitations(full_text, sections)
        research_elements["limitations"] = limitations
        
        return research_elements

    def _extract_enhanced_metadata(self, full_text: str, file_name: str, num_pages: int) -> Dict[str, Any]:
        """Extract enhanced metadata from PDF content"""
        metadata = {
            "file_name": file_name,
            "num_pages": num_pages,
            "extraction_timestamp": pd.Timestamp.now().isoformat()
        }
        
        # Extract title (usually first substantial line)
        lines = [line.strip() for line in full_text.split('\n') if line.strip()]
        if lines:
            potential_title = lines[0]
            if 10 < len(potential_title) < 200:
                metadata["title"] = potential_title
        
        # Extract authors (look for common patterns)
        author_patterns = [
            r"(?i)authors?\s*[:]\s*([^\n]+)",
            r"(?i)by\s+([^\n]+)",
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, full_text[:2000])  # Look in first 2000 chars
            if match:
                metadata["authors"] = match.group(1).strip()
                break
        
        # Extract publication info
        year_match = re.search(r"\b(20[0-2]\d)\b", full_text[:1000])
        if year_match:
            metadata["year"] = year_match.group(1)
        
        # Extract keywords
        keywords_match = re.search(r"(?i)keywords?\s*[:]\s*([^\n]+)", full_text)
        if keywords_match:
            metadata["keywords"] = keywords_match.group(1).strip()
        
        return metadata

    def _extract_basic_metadata(self, reader: PdfReader, file_name: str) -> Dict[str, Any]:
        """Extract basic metadata from PDF reader"""
        metadata = {"file_name": file_name, "num_pages": len(reader.pages)}
        
        if reader.metadata:
            pdf_meta = reader.metadata
            metadata.update({
                "title": pdf_meta.get("/Title", ""),
                "author": pdf_meta.get("/Author", ""),
                "subject": pdf_meta.get("/Subject", ""),
                "creator": pdf_meta.get("/Creator", ""),
                "creation_date": str(pdf_meta.get("/CreationDate", ""))
            })
        
        return metadata

    # Additional helper methods for comprehensive extraction
    def _extract_tables_from_markdown(self, text: str) -> List[Dict[str, Any]]:
        """Extract tables from markdown text"""
        tables = []
        table_pattern = r'\|.*\|.*\n\|[-\s\|:]*\|.*\n(\|.*\|.*\n)+'
        
        table_matches = re.finditer(table_pattern, text, re.MULTILINE)
        for i, match in enumerate(table_matches):
            table_text = match.group()
            tables.append({
                "table_id": i + 1,
                "raw_text": table_text,
                "location": "extracted_from_markdown"
            })
        
        return tables

    def _extract_figures_from_markdown(self, text: str) -> List[Dict[str, Any]]:
        """Extract figure descriptions from markdown"""
        figures = []
        figure_pattern = r'(?i)figure\s+\d+[:\.]?\s*([^\n]+)'
        
        figure_matches = re.finditer(figure_pattern, text)
        for i, match in enumerate(figure_matches):
            figures.append({
                "figure_id": i + 1,
                "description": match.group(1).strip(),
                "location": "extracted_from_text"
            })
        
        return figures

    def _analyze_citations(self, text: str) -> Dict[str, Any]:
        """Analyze citation patterns"""
        citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[\d+[,\s\-\d]*\]',    # [1, 2, 3]
        ]
        
        citations = []
        for pattern in citation_patterns:
            matches = re.findall(pattern, text)
            citations.extend(matches)
        
        return {
            "total_citations": len(citations),
            "citation_density": len(citations) / len(text.split()) if text else 0,
            "sample_citations": citations[:10]
        }

    def _extract_statistical_content(self, text: str) -> Dict[str, Any]:
        """Extract statistical content"""
        stat_patterns = {
            "p_values": r"p\s*[<>=]\s*(0\.\d+)",
            "correlations": r"r\s*=\s*([-]?0\.\d+)",
            "means": r"(?:mean|M)\s*=\s*([\d.]+)"
        }
        
        statistics = {}
        for stat_type, pattern in stat_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            statistics[stat_type] = matches
        
        return statistics

    def _assess_academic_structure(self, sections: Dict[str, str]) -> Dict[str, Any]:
        """Assess the quality of academic structure"""
        expected_sections = ["abstract", "introduction", "methodology", "results", "discussion", "conclusion"]
        present_sections = [s for s in expected_sections if s in sections]
        
        return {
            "completeness_score": len(present_sections) / len(expected_sections),
            "present_sections": present_sections,
            "missing_sections": [s for s in expected_sections if s not in sections],
            "has_references": "references" in sections
        }

    def _calculate_quality_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate text quality metrics"""
        import textstat
        
        return {
            "word_count": len(text.split()),
            "readability_score": textstat.flesch_reading_ease(text),
            "grade_level": textstat.flesch_kincaid_grade(text),
            "sentence_count": len(re.split(r'[.!?]+', text))
        }

    # Research element extraction helpers
    def _extract_research_questions(self, full_text: str, sections: Dict[str, str]) -> List[str]:
        """Extract research questions"""
        patterns = [
            r'research question[s]?[:\-\s]+([^.]*\?)',
            r'we ask[:\-\s]+([^.]*\?)',
            r'investigate[s]?\s+whether[:\-\s]+([^.]*)'
        ]
        
        questions = []
        for pattern in patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            questions.extend(matches)
        
        return questions

    def _extract_hypotheses(self, text: str) -> List[str]:
        """Extract hypotheses"""
        patterns = [
            r'hypothes[ie]s?\s*[:]\s*([^.]*)',
            r'we hypothesize\s+that\s+([^.]*)',
            r'predict\s+that\s+([^.]*)'
        ]
        
        hypotheses = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            hypotheses.extend(matches)
        
        return hypotheses

    def _extract_key_findings(self, results_text: str) -> List[str]:
        """Extract key findings from results"""
        if not results_text:
            return []
        
        sentences = re.split(r'[.!?]+', results_text)
        significance_indicators = ['significant', 'p <', 'correlation', 'effect']
        
        findings = []
        for sentence in sentences:
            if any(indicator in sentence.lower() for indicator in significance_indicators):
                findings.append(sentence.strip())
        
        return findings

    def _classify_methodology(self, methodology_text: str) -> str:
        """Classify methodology type"""
        text_lower = methodology_text.lower()
        
        if any(word in text_lower for word in ["experiment", "controlled", "randomized"]):
            return "experimental"
        elif any(word in text_lower for word in ["survey", "observational", "cross-sectional"]):
            return "observational"
        elif any(word in text_lower for word in ["meta-analysis", "systematic review"]):
            return "meta_analysis"
        elif any(word in text_lower for word in ["qualitative", "interview", "focus group"]):
            return "qualitative"
        elif any(word in text_lower for word in ["simulation", "computational", "modeling"]):
            return "computational"
        else:
            return "unclassified"

    def _extract_limitations(self, full_text: str, sections: Dict[str, str]) -> List[str]:
        """Extract research limitations"""
        limitation_patterns = [r'limitation[s]?', r'constraint[s]?', r'shortcoming[s]?']
        
        limitations = []
        for pattern in limitation_patterns:
            matches = re.finditer(pattern, full_text, re.IGNORECASE)
            for match in matches:
                start = max(0, match.start() - 100)
                end = min(len(full_text), match.end() + 200)
                context = full_text[start:end].strip()
                limitations.append(context)
        
        return limitations

    def get_paper_summary(self, extracted_content: Dict[str, Any]) -> str:
        """Create comprehensive paper summary"""
        if not extracted_content.get("success"):
            return f"Error: {extracted_content.get('error', 'Unknown error')}"
        
        metadata = extracted_content.get("metadata", {})
        sections = extracted_content.get("sections", {})
        stats = extracted_content.get("summary_stats", {})
        research_elements = extracted_content.get("research_elements", {})
        
        summary_parts = []
        
        # Title and basic info
        if metadata.get("title"):
            summary_parts.append(f"**Title:** {metadata['title']}")
        
        if metadata.get("authors"):
            summary_parts.append(f"**Authors:** {metadata['authors']}")
        
        # Abstract
        if sections.get("abstract"):
            abstract = sections["abstract"][:400] + "..." if len(sections["abstract"]) > 400 else sections["abstract"]
            summary_parts.append(f"**Abstract:** {abstract}")
        
        # Research elements
        if research_elements.get("research_questions"):
            questions = research_elements["research_questions"][:3]  # First 3 questions
            summary_parts.append(f"**Research Questions:** {'; '.join(questions)}")
        
        if research_elements.get("methodology_type"):
            summary_parts.append(f"**Methodology:** {research_elements['methodology_type']}")
        
        # Statistics
        summary_parts.append(f"**Pages:** {stats.get('total_pages', 'Unknown')}")
        summary_parts.append(f"**Words:** {stats.get('total_words', 'Unknown')}")
        summary_parts.append(f"**Extraction Method:** {stats.get('extraction_method', 'Unknown')}")
        
        # Available sections
        if sections:
            section_list = list(sections.keys())
            summary_parts.append(f"**Available Sections:** {', '.join(section_list)}")
        
        return "\n\n".join(summary_parts) 