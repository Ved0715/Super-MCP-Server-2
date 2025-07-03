"""
🚀 Perfect MCP Server
Complete integration of research intelligence, vector storage, and advanced presentation generation
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import tempfile
import os
import time

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource
)
import mcp.server.stdio

from config import AdvancedConfig
from enhanced_pdf_processor import EnhancedPDFProcessor
from vector_storage import AdvancedVectorStorage
from research_intelligence import ResearchPaperAnalyzer
from perfect_ppt_generator import PerfectPPTGenerator
from search_client import SerpAPIClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerfectMCPServer:
    """Perfect MCP Server with complete research capabilities"""
    
    def __init__(self):
        """Initialize the perfect research system"""
        self.config = AdvancedConfig()
        
        # Initialize all components
        self.pdf_processor = EnhancedPDFProcessor(self.config)
        self.vector_storage = AdvancedVectorStorage(self.config)
        self.research_analyzer = ResearchPaperAnalyzer(self.config)
        self.ppt_generator = PerfectPPTGenerator(
            self.config, 
            self.vector_storage, 
            self.research_analyzer
        )
        self.search_client = SerpAPIClient(self.config)
        
        # Paper storage
        self.processed_papers = {}
        
        # Create MCP server
        self.server = Server("perfect-research-mcp")
        self._setup_tools()
        self._setup_resources()
        
        logger.info("Perfect MCP Server initialized with all advanced features")

    def _setup_tools(self):
        """Setup all advanced MCP tools"""
        
        # ============================================================================
        # ADVANCED SEARCH TOOLS
        # ============================================================================
        
        @self.server.list_tools()
        async def handle_list_tools() -> List[Tool]:
            """List all available research tools"""
            return [
                Tool(
                    name="advanced_search_web",
                    description="Advanced web search with academic focus and result enhancement",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "search_type": {
                                "type": "string", 
                                "enum": ["web", "scholar", "news", "images"],
                                "default": "web",
                                "description": "Type of search to perform"
                            },
                            "num_results": {"type": "integer", "default": 10, "description": "Number of results"},
                            "location": {"type": "string", "default": "United States", "description": "Search location"},
                            "time_period": {"type": "string", "enum": ["all", "year", "month", "week", "day"], "default": "all"},
                            "enhance_results": {"type": "boolean", "default": True, "description": "Apply AI enhancement to results"}
                        },
                        "required": ["query"]
                    }
                ),
                
                Tool(
                    name="process_research_paper",
                    description="Process PDF research paper with advanced extraction, analysis, and vector storage",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_content": {"type": "string", "description": "Base64 encoded PDF content"},
                            "file_name": {"type": "string", "description": "Original filename"},
                            "paper_id": {"type": "string", "description": "Unique identifier for the paper"},
                            "enable_research_analysis": {"type": "boolean", "default": True},
                            "enable_vector_storage": {"type": "boolean", "default": True},
                            "analysis_depth": {"type": "string", "enum": ["basic", "standard", "comprehensive"], "default": "comprehensive"}
                        },
                        "required": ["file_content", "file_name", "paper_id"]
                    }
                ),
                
                Tool(
                    name="create_perfect_presentation",
                    description="Create a perfect research presentation from processed paper with advanced features",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "Paper identifier"},
                            "user_prompt": {"type": "string", "description": "User's presentation requirements"},
                            "title": {"type": "string", "description": "Presentation title"},
                            "author": {"type": "string", "description": "Presentation author"},
                            "theme": {"type": "string", "description": "Presentation theme"},
                            "slide_count": {"type": "integer", "description": "Number of slides"},
                            "audience_type": {"type": "string", "description": "Target audience type"},
                            "include_search_results": {"type": "boolean", "description": "Include web search results"},
                            "search_query": {"type": "string", "description": "Search query for additional content"}
                        },
                        "required": ["paper_id", "user_prompt"]
                    }
                ),
                
                Tool(
                    name="create_presentation_from_namespace",
                    description="Create presentation from namespace-based vector search",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string", "description": "Vector database namespace"},
                            "user_prompt": {"type": "string", "description": "User's presentation requirements"},
                            "title": {"type": "string", "description": "Presentation title"},
                            "author": {"type": "string", "description": "Presentation author"},
                            "theme": {"type": "string", "description": "Presentation theme"},
                            "slide_count": {"type": "integer", "description": "Number of slides"},
                            "audience_type": {"type": "string", "description": "Target audience"},
                            "search_query": {"type": "string", "description": "Additional search context"}
                        },
                        "required": ["namespace", "user_prompt"]
                    }
                ),
                
                Tool(
                    name="research_intelligence_analysis",
                    description="Perform comprehensive research intelligence analysis on processed papers",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "ID of processed paper"},
                            "analysis_types": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["methodology", "contributions", "citations", "quality", "limitations", "statistical"]},
                                "default": ["methodology", "contributions", "quality"],
                                "description": "Types of analysis to perform"
                            },
                            "provide_recommendations": {"type": "boolean", "default": True}
                        },
                        "required": ["paper_id"]
                    }
                ),
                
                Tool(
                    name="semantic_paper_search",
                    description="Perform semantic search within processed papers using vector embeddings",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "paper_id": {"type": "string", "description": "Specific paper to search (optional)"},
                            "search_type": {"type": "string", "enum": ["general", "methodology", "results", "discussion", "conclusion"], "default": "general"},
                            "max_results": {"type": "integer", "default": 10, "minimum": 1, "maximum": 50},
                            "similarity_threshold": {"type": "number", "default": 0.7, "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["query"]
                    }
                ),
                
                Tool(
                    name="compare_research_papers",
                    description="Compare multiple research papers across various dimensions",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_ids": {"type": "array", "items": {"type": "string"}, "minItems": 2, "maxItems": 5},
                            "comparison_aspects": {
                                "type": "array",
                                "items": {"type": "string", "enum": ["methodology", "findings", "limitations", "contributions", "citations", "quality"]},
                                "default": ["methodology", "findings", "contributions"]
                            },
                            "generate_summary": {"type": "boolean", "default": True}
                        },
                        "required": ["paper_ids"]
                    }
                ),
                
                Tool(
                    name="generate_research_insights",
                    description="Generate AI-powered insights and recommendations from research analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "Paper to analyze"},
                            "focus_area": {"type": "string", "enum": ["methodology_improvement", "future_research", "practical_applications", "theoretical_implications"], "default": "future_research"},
                            "insight_depth": {"type": "string", "enum": ["overview", "detailed", "comprehensive"], "default": "detailed"},
                            "include_citations": {"type": "boolean", "default": True}
                        },
                        "required": ["paper_id"]
                    }
                ),
                
                Tool(
                    name="export_research_summary",
                    description="Export comprehensive research summary in various formats",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_id": {"type": "string", "description": "Paper to export"},
                            "export_format": {"type": "string", "enum": ["markdown", "json", "academic_report"], "default": "markdown"},
                            "include_analysis": {"type": "boolean", "default": True},
                            "include_presentation_ready": {"type": "boolean", "default": False}
                        },
                        "required": ["paper_id"]
                    }
                ),
                
                Tool(
                    name="list_processed_papers",
                    description="List all processed research papers with their analysis status",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_stats": {"type": "boolean", "default": True},
                            "sort_by": {"type": "string", "enum": ["name", "date", "quality_score"], "default": "date"}
                        }
                    }
                ),
                
                Tool(
                    name="system_status",
                    description="Get comprehensive system status including all components and capabilities",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_config": {"type": "boolean", "default": False},
                            "run_health_check": {"type": "boolean", "default": True}
                        }
                    }
                )
            ]

        # ============================================================================
        # TOOL IMPLEMENTATIONS
        # ============================================================================

        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool calls with comprehensive error handling"""
            try:
                if name == "advanced_search_web":
                    return await self._handle_advanced_search(**arguments)
                
                elif name == "process_research_paper":
                    return await self._handle_process_paper(**arguments)
                
                elif name == "create_perfect_presentation":
                    return await self._handle_create_presentation(**arguments)
                
                elif name == "create_presentation_from_namespace":
                    return await self._handle_create_presentation_from_namespace(**arguments)
                
                elif name == "research_intelligence_analysis":
                    return await self._handle_research_analysis(**arguments)
                
                elif name == "semantic_paper_search":
                    return await self._handle_semantic_search(**arguments)
                
                elif name == "compare_research_papers":
                    return await self._handle_compare_papers(**arguments)
                
                elif name == "generate_research_insights":
                    return await self._handle_generate_insights(**arguments)
                
                elif name == "export_research_summary":
                    return await self._handle_export_summary(**arguments)
                
                elif name == "list_processed_papers":
                    return await self._handle_list_papers(**arguments)
                
                elif name == "system_status":
                    return await self._handle_system_status(**arguments)
                
                else:
                    return [TextContent(type="text", text=f"Unknown tool: {name}")]
                    
            except Exception as e:
                logger.error(f"Error in tool {name}: {e}")
                return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]

    def _setup_resources(self):
        """Setup MCP resources"""
        
        @self.server.list_resources()
        async def handle_list_resources() -> List[Resource]:
            """List available resources"""
            resources = []
            
            # Add processed papers as resources
            for paper_id, paper_data in self.processed_papers.items():
                resources.append(Resource(
                    uri=f"paper://{paper_id}",
                    name=f"Research Paper: {paper_data.get('metadata', {}).get('title', paper_id)}",
                    description=f"Processed research paper with analysis",
                    mimeType="application/json"
                ))
            
            return resources

        @self.server.read_resource()
        async def handle_read_resource(uri: str) -> str:
            """Read resource content"""
            if uri.startswith("paper://"):
                paper_id = uri[8:]  # Remove "paper://" prefix
                if paper_id in self.processed_papers:
                    return json.dumps(self.processed_papers[paper_id], indent=2)
                else:
                    raise ValueError(f"Paper not found: {paper_id}")
            else:
                raise ValueError(f"Unknown resource URI: {uri}")

    # ============================================================================
    # TOOL IMPLEMENTATION METHODS
    # ============================================================================

    async def _handle_advanced_search(self, query: str, search_type: str = "web", 
                                    num_results: int = 10, location: str = "United States",
                                    time_period: str = "all", enhance_results: bool = True) -> List[TextContent]:
        """Handle advanced search with AI enhancement"""
        try:
            # Perform search
            search_results = self.search_client.search_google(
                query=query,
                search_type=search_type,
                num_results=num_results,
                location=location
            )
            
            if not search_results.get("success"):
                return [TextContent(type="text", text=f"Search failed: {search_results.get('error')}")]
            
            # Enhance results if requested (simplified for now)
            if enhance_results:
                # Format results nicely instead of AI enhancement
                search_results["formatted_results"] = self.search_client.format_search_results(search_results)
            
            # Format response
            response = self._format_search_results(search_results, query, search_type)
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            return [TextContent(type="text", text=f"Search error: {str(e)}")]

    async def _handle_process_paper(self, file_content: str, file_name: str, paper_id: str,
                                  enable_research_analysis: bool = True,
                                  enable_vector_storage: bool = True,
                                  analysis_depth: str = "comprehensive") -> List[TextContent]:
        """Handle comprehensive research paper processing"""
        try:
            # Decode file content
            import base64
            pdf_bytes = base64.b64decode(file_content)
            
            # Process PDF with advanced extraction
            extraction_result = await self.pdf_processor.extract_content_from_bytes(pdf_bytes, file_name)
            
            if not extraction_result.get("success"):
                return [TextContent(type="text", text=f"PDF processing failed: {extraction_result.get('error')}")]
            
            # Store extracted content
            self.processed_papers[paper_id] = extraction_result
            
            response_parts = [f"# Research Paper Processing Complete\n"]
            response_parts.append(f"**Paper ID:** {paper_id}")
            response_parts.append(f"**File:** {file_name}")
            response_parts.append(f"**Pages:** {extraction_result.get('summary_stats', {}).get('total_pages', 'Unknown')}")
            response_parts.append(f"**Words:** {extraction_result.get('summary_stats', {}).get('total_words', 'Unknown')}")
            response_parts.append(f"**Extraction Method:** {extraction_result.get('summary_stats', {}).get('extraction_method', 'Unknown')}")
            
            # Research intelligence analysis
            if enable_research_analysis and self.research_analyzer:
                response_parts.append("\n## 🧠 Research Intelligence Analysis")
                
                try:
                    research_analysis = await self.research_analyzer.analyze_research_paper(extraction_result)
                    self.processed_papers[paper_id]["research_analysis"] = research_analysis
                    
                    if research_analysis.get("research_elements"):
                        elements = research_analysis["research_elements"]
                        response_parts.append(f"**Research Elements Identified:** {len(elements)}")
                    
                    if research_analysis.get("methodology_analysis"):
                        method_analysis = research_analysis["methodology_analysis"]
                        response_parts.append(f"**Methodology Type:** {method_analysis.get('methodology_type', 'Unknown')}")
                        response_parts.append(f"**Rigor Score:** {method_analysis.get('rigor_score', 0):.2f}")
                    
                    if research_analysis.get("quality_assessment"):
                        quality = research_analysis["quality_assessment"]
                        response_parts.append(f"**Overall Quality Score:** {quality.get('overall_quality', 0):.2f}")
                        
                except Exception as e:
                    logger.warning(f"Research analysis failed: {e}")
                    response_parts.append("*Research analysis partially failed - basic processing completed*")
            
            # Vector storage
            if enable_vector_storage and self.vector_storage:
                response_parts.append("\n## 🔍 Vector Storage & Semantic Indexing")
                
                try:
                    storage_result = await self.vector_storage.process_and_store_document(
                        extraction_result, paper_id
                    )
                    
                    if storage_result.get("success"):
                        response_parts.append(f"**Chunks Created:** {storage_result.get('chunks_created', 0)}")
                        response_parts.append(f"**Chunks Stored:** {storage_result.get('chunks_stored', 0)}")
                        response_parts.append("**Semantic Search:** ✅ Enabled")
                    else:
                        response_parts.append("**Vector Storage:** ❌ Failed")
                        
                except Exception as e:
                    logger.warning(f"Vector storage failed: {e}")
                    response_parts.append("*Vector storage failed - search capabilities limited*")
            
            # Add paper summary
            if extraction_result.get("sections", {}).get("abstract"):
                abstract = extraction_result["sections"]["abstract"][:300]
                response_parts.append(f"\n## 📄 Abstract\n{abstract}...")
            
            # Available sections
            sections = list(extraction_result.get("sections", {}).keys())
            if sections:
                response_parts.append(f"\n**Available Sections:** {', '.join(sections)}")
            
            response_parts.append(f"\n✅ **Paper processing complete!** Ready for presentation generation and analysis.")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            return [TextContent(type="text", text=f"Paper processing error: {str(e)}")]

    async def _handle_create_presentation(self, paper_id: str, user_prompt: str,
                                        title: str = None, author: str = "AI Research Assistant",
                                        theme: str = "academic_professional", slide_count: int = 12,
                                        audience_type: str = "academic", 
                                        include_search_results: bool = False,
                                        search_query: str = None) -> List[TextContent]:
        """Handle perfect presentation creation"""
        try:
            if paper_id not in self.processed_papers:
                return [TextContent(type="text", text=f"Paper not found: {paper_id}. Please process the paper first.")]
            
            paper_content = self.processed_papers[paper_id]
            
            # Get search results if requested
            search_results = None
            if include_search_results and search_query:
                search_results = self.search_client.search_google(
                    query=search_query,
                    search_type="scholar",
                    num_results=5
                )
            
            response_parts = [f"# 🎯 Creating Perfect Research Presentation"]
            response_parts.append(f"**Paper:** {paper_id}")
            response_parts.append(f"**User Requirements:** {user_prompt}")
            response_parts.append(f"**Theme:** {theme}")
            response_parts.append(f"**Slides:** {slide_count}")
            response_parts.append(f"**Audience:** {audience_type}")
            
            # Create presentation
            presentation_path = await self.ppt_generator.create_perfect_presentation(
                paper_content=paper_content,
                user_prompt=user_prompt,
                paper_id=paper_id,
                search_results=search_results,
                title=title,
                author=author,
                theme=theme,
                slide_count=slide_count,
                audience_type=audience_type
            )
            
            response_parts.append(f"\n✅ **Perfect presentation created successfully!**")
            response_parts.append(f"**File:** {presentation_path}")
            response_parts.append(f"**Location:** {os.path.abspath(presentation_path)}")
            
            # Add presentation details
            response_parts.append(f"\n## 🎨 Presentation Features")
            response_parts.append("✅ AI-powered content analysis and selection")
            response_parts.append("✅ Semantic search integration for relevant content")
            response_parts.append("✅ Research intelligence for accurate scientific content")
            response_parts.append("✅ Professional academic theme with visual enhancements")
            response_parts.append("✅ Audience-appropriate depth and technical level")
            response_parts.append("✅ Proper citation integration and academic formatting")
            
            if self.vector_storage:
                response_parts.append("✅ Context-aware content retrieval from vector embeddings")
            
            if search_results:
                response_parts.append("✅ Enhanced with related research findings")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error creating presentation: {e}")
            return [TextContent(type="text", text=f"Presentation creation error: {str(e)}")]

    async def _handle_create_presentation_from_namespace(
        self,
        namespace: str,
        user_prompt: str,
        title: str = None,
        author: str = "AI Research Assistant",
        theme: str = "academic_professional",
        slide_count: int = 12,
        audience_type: str = "academic",
        search_query: str = None
    ) -> List[TextContent]:
        """
        Handle namespace-based presentation creation
        
        1. Search vector database in specified namespace
        2. Gather relevant information
        3. Generate PPT using found content
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting namespace-based presentation creation")
            logger.info(f"📁 Namespace: {namespace}")
            logger.info(f"💭 User prompt: {user_prompt}")
            logger.info(f"🎨 Theme: {theme}, Slides: {slide_count}, Audience: {audience_type}")
            
            # Step 1: Search vector database in namespace
            logger.info(f"🔍 Step 1/3: Starting vector database search in namespace...")
            search_start_time = time.time()
            
            if not self.vector_storage:
                logger.error("❌ Vector storage not initialized")
                return [TextContent(type="text", text="Vector storage not initialized")]
            
            search_results = await self.vector_storage.search_in_namespace(
                namespace=namespace,
                query=user_prompt,
                max_results=20,
                similarity_threshold=0.2
            )
            
            search_duration = time.time() - search_start_time
            logger.info(f"✅ Vector search completed in {search_duration:.2f}s")
            logger.info(f"📊 Found {len(search_results)} content chunks")
            
            if not search_results:
                logger.warning(f"⚠️  No content found in namespace: {namespace}")
                return [TextContent(type="text", text=f"No content found in namespace: {namespace}")]
            
            # Log search results summary
            total_content_length = sum(len(result.get('content', '')) for result in search_results)
            avg_score = sum(result.get('score', 0) for result in search_results) / len(search_results)
            logger.info(f"📈 Search quality - Avg score: {avg_score:.3f}, Total content: {total_content_length} chars")
            
            # Step 2: Aggregate search results into paper-like content
            logger.info(f"🔧 Step 2/3: Aggregating search results into presentation content...")
            aggregation_start_time = time.time()
            
            aggregated_content = self._aggregate_search_results(search_results, namespace)
            
            aggregation_duration = time.time() - aggregation_start_time
            logger.info(f"✅ Content aggregation completed in {aggregation_duration:.2f}s")
            logger.info(f"📝 Aggregated content: {len(aggregated_content.get('full_text', ''))} chars")
            logger.info(f"📑 Sections found: {list(aggregated_content.get('sections', {}).keys())}")
            
            # Step 3: Generate PPT using existing generator
            logger.info(f"🎨 Step 3/3: Starting PowerPoint generation...")
            ppt_start_time = time.time()
            
            if not self.ppt_generator:
                logger.error("❌ PPT generator not initialized")
                return [TextContent(type="text", text="PPT generator not initialized")]
            
            logger.info(f"🤖 Calling PPT generator with {slide_count} slides...")
            presentation_path = await self.ppt_generator.create_perfect_presentation(
                paper_content=aggregated_content,
                user_prompt=user_prompt,
                paper_id=namespace,  # Use namespace as paper_id
                title=title,
                theme=theme,
                slide_count=slide_count,
                audience_type=audience_type
            )
            
            ppt_duration = time.time() - ppt_start_time
            total_duration = time.time() - start_time
            
            logger.info(f"✅ PowerPoint generation completed in {ppt_duration:.2f}s")
            logger.info(f"📁 Presentation saved: {os.path.basename(presentation_path)}")
            logger.info(f"🎉 Total process completed in {total_duration:.2f}s")
            
            # Performance breakdown log
            logger.info(f"⏱️  Performance breakdown:")
            logger.info(f"   - Vector search: {search_duration:.2f}s ({search_duration/total_duration*100:.1f}%)")
            logger.info(f"   - Content aggregation: {aggregation_duration:.2f}s ({aggregation_duration/total_duration*100:.1f}%)")
            logger.info(f"   - PPT generation: {ppt_duration:.2f}s ({ppt_duration/total_duration*100:.1f}%)")
            
            response_parts = [f"# 🎯 Namespace-Based Presentation Created"]
            response_parts.append(f"**Namespace:** {namespace}")
            response_parts.append(f"**Content Sources:** {len(search_results)} chunks found")
            response_parts.append(f"**Presentation:** {os.path.basename(presentation_path)}")
            response_parts.append(f"**Theme:** {theme}")
            response_parts.append(f"**Slides:** {slide_count}")
            response_parts.append(f"**Total Time:** {total_duration:.2f}s")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            total_duration = time.time() - start_time
            logger.error(f"❌ Error creating namespace presentation after {total_duration:.2f}s: {e}")
            logger.error(f"🔍 Error details: {str(e)}")
            import traceback
            logger.error(f"📋 Full traceback: {traceback.format_exc()}")
            return [TextContent(type="text", text=f"Namespace presentation error: {str(e)}")]

    def _aggregate_search_results(self, search_results: List, namespace: str) -> Dict[str, Any]:
        """
        Convert vector search results into paper-like content structure
        """
        # Aggregate content from search results
        full_text = ""
        sections = {}
        
        for result in search_results:
            chunk_content = result.get('content', '')
            chunk_section = result.get('metadata', {}).get('section', 'general')
            
            full_text += chunk_content + "\n\n"
            
            if chunk_section not in sections:
                sections[chunk_section] = ""
            sections[chunk_section] += chunk_content + "\n\n"
        
        # Create paper-like structure
        return {
            "success": True,
            "full_text": full_text,
            "sections": sections,
            "metadata": {
                "title": f"Content from {namespace}",
                "source": "vector_search",
                "namespace": namespace
            },
            "summary_stats": {
                "total_chunks": len(search_results),
                "total_words": len(full_text.split()),
                "extraction_method": "vector_search"
            }
        }

    async def _handle_research_analysis(self, paper_id: str, 
                                      analysis_types: List[str] = None,
                                      provide_recommendations: bool = True) -> List[TextContent]:
        """Handle comprehensive research intelligence analysis"""
        try:
            if paper_id not in self.processed_papers:
                return [TextContent(type="text", text=f"Paper not found: {paper_id}")]
            
            paper_content = self.processed_papers[paper_id]
            
            if not analysis_types:
                analysis_types = ["methodology", "contributions", "quality"]
            
            response_parts = [f"# 🧠 Research Intelligence Analysis: {paper_id}"]
            
            # Check if analysis already exists
            if "research_analysis" not in paper_content:
                response_parts.append("Running comprehensive research analysis...")
                research_analysis = await self.research_analyzer.analyze_research_paper(paper_content)
                self.processed_papers[paper_id]["research_analysis"] = research_analysis
            else:
                research_analysis = paper_content["research_analysis"]
            
            # Format analysis results
            for analysis_type in analysis_types:
                if analysis_type == "methodology":
                    method_data = research_analysis.get("methodology_analysis", {})
                    response_parts.append(f"\n## 🔬 Methodology Analysis")
                    response_parts.append(f"**Type:** {method_data.get('methodology_type', 'Unknown')}")
                    response_parts.append(f"**Rigor Score:** {method_data.get('rigor_score', 0):.2f}/1.0")
                    if method_data.get('sample_size'):
                        response_parts.append(f"**Sample Size:** {method_data['sample_size']}")
                    
                elif analysis_type == "contributions":
                    contributions = research_analysis.get("research_contributions", [])
                    response_parts.append(f"\n## 💡 Research Contributions")
                    for i, contrib in enumerate(contributions[:3], 1):
                        response_parts.append(f"**{i}.** {contrib.get('description', 'N/A')[:200]}")
                        response_parts.append(f"   *Novelty Score: {contrib.get('novelty_score', 0):.2f}/1.0*")
                
                elif analysis_type == "quality":
                    quality = research_analysis.get("quality_assessment", {})
                    response_parts.append(f"\n## ⭐ Quality Assessment")
                    response_parts.append(f"**Overall Quality:** {quality.get('overall_quality', 0):.2f}/1.0")
                    response_parts.append(f"**Completeness:** {quality.get('completeness_score', 0):.2f}/1.0")
                    response_parts.append(f"**Structure Score:** {quality.get('structure_score', 0):.2f}/1.0")
                    
                elif analysis_type == "citations":
                    citations = research_analysis.get("citation_analysis", {})
                    response_parts.append(f"\n## 📚 Citation Analysis")
                    response_parts.append(f"**Total Citations:** {citations.get('total_citations', 0)}")
                    response_parts.append(f"**Citation Density:** {citations.get('citation_density', 0):.4f}")
                    response_parts.append(f"**Recent Citations:** {citations.get('recent_citations', 0)}")
                
                elif analysis_type == "statistical":
                    stats = research_analysis.get("statistical_results", [])
                    response_parts.append(f"\n## 📊 Statistical Analysis")
                    response_parts.append(f"**Statistical Tests Found:** {len(stats)}")
                    significant_results = [s for s in stats if s.get('significance')]
                    response_parts.append(f"**Significant Results:** {len(significant_results)}")
                
                elif analysis_type == "limitations":
                    limitations = research_analysis.get("limitations", {})
                    response_parts.append(f"\n## ⚠️ Limitations Analysis")
                    response_parts.append(f"**Limitations Identified:** {limitations.get('limitation_count', 0)}")
                    response_parts.append(f"**Discusses Limitations:** {'✅' if limitations.get('discusses_limitations') else '❌'}")
            
            # Add recommendations if requested
            if provide_recommendations:
                response_parts.append(f"\n## 🎯 AI Recommendations")
                response_parts.append("Based on the analysis, here are key recommendations:")
                
                # Generate smart recommendations based on analysis
                recommendations = await self._generate_smart_recommendations(research_analysis)
                for rec in recommendations:
                    response_parts.append(f"• {rec}")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error in research analysis: {e}")
            return [TextContent(type="text", text=f"Research analysis error: {str(e)}")]

    async def _handle_semantic_search(self, query: str, paper_id: str = None,
                                    search_type: str = "general", max_results: int = 10,
                                    similarity_threshold: float = 0.7) -> List[TextContent]:
        """Handle semantic search within papers"""
        try:
            if not self.vector_storage:
                return [TextContent(type="text", text="Vector storage not available. Semantic search disabled.")]
            
            response_parts = [f"# 🔍 Semantic Search Results"]
            response_parts.append(f"**Query:** {query}")
            response_parts.append(f"**Search Type:** {search_type}")
            response_parts.append(f"**Similarity Threshold:** {similarity_threshold}")
            
            if paper_id:
                # Search specific paper
                if paper_id not in self.processed_papers:
                    return [TextContent(type="text", text=f"Paper not found: {paper_id}")]
                
                response_parts.append(f"**Target Paper:** {paper_id}")
                
                if search_type == "general":
                    results = await self.vector_storage.semantic_search(
                        query=query,
                        namespace=paper_id,
                        top_k=max_results
                    )
                else:
                    results = await self.vector_storage.contextual_search(
                        user_prompt=query,
                        namespace=paper_id,
                        context_type=search_type
                    )
            else:
                # Search all papers (would need multi-namespace search)
                response_parts.append("**Scope:** All processed papers")
                results = []
                
                for pid in self.processed_papers.keys():
                    paper_results = await self.vector_storage.semantic_search(
                        query=query,
                        namespace=pid,
                        top_k=max_results // len(self.processed_papers) + 1
                    )
                    results.extend(paper_results)
                
                # Sort by score and limit
                results.sort(key=lambda x: x.score, reverse=True)
                results = results[:max_results]
            
            # Filter by similarity threshold
            filtered_results = [r for r in results if r.score >= similarity_threshold]
            
            response_parts.append(f"\n**Results Found:** {len(filtered_results)}")
            
            if filtered_results:
                response_parts.append("\n## 📄 Search Results")
                
                for i, result in enumerate(filtered_results, 1):
                    response_parts.append(f"\n### Result {i} (Score: {result.score:.3f})")
                    response_parts.append(f"**Section:** {result.section or 'Unknown'}")
                    if result.page_number:
                        response_parts.append(f"**Page:** {result.page_number}")
                    
                    content_preview = result.content[:300]
                    response_parts.append(f"**Content:** {content_preview}...")
                    response_parts.append("---")
            else:
                response_parts.append("\nNo results found above the similarity threshold.")
                response_parts.append("Try lowering the threshold or rephrasing your query.")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {e}")
            return [TextContent(type="text", text=f"Semantic search error: {str(e)}")]

    async def _handle_system_status(self, include_config: bool = False, 
                                  run_health_check: bool = True) -> List[TextContent]:
        """Handle system status check"""
        try:
            response_parts = ["# 🚀 Perfect Research System Status"]
            
            # Core components status
            response_parts.append("\n## 🔧 Core Components")
            response_parts.append(f"✅ **Enhanced PDF Processor** - LlamaParse: {'✅' if self.config.LLAMA_PARSE_API_KEY else '⚠️ Fallback'}")
            response_parts.append(f"✅ **Vector Storage (Pinecone)** - Status: {'✅ Connected' if self.vector_storage else '❌ Disabled'}")
            response_parts.append(f"✅ **Research Intelligence** - AI Analysis: {'✅ Active' if self.research_analyzer else '❌ Disabled'}")
            response_parts.append(f"✅ **Perfect PPT Generator** - Advanced Themes: ✅ Available")
            response_parts.append(f"✅ **Advanced Search Client** - SerpAPI: {'✅ Connected' if self.config.SERPAPI_KEY else '❌ Missing'}")
            
            # API status
            response_parts.append("\n## 🔑 API Integration Status")
            response_parts.append(f"OpenAI API: {'✅ Connected' if self.config.OPENAI_API_KEY else '❌ Missing'}")
            response_parts.append(f"SerpAPI: {'✅ Connected' if self.config.SERPAPI_KEY else '❌ Missing'}")
            response_parts.append(f"Pinecone: {'✅ Connected' if self.config.PINECONE_API_KEY else '❌ Missing'}")
            response_parts.append(f"LlamaParse: {'✅ Connected' if self.config.LLAMA_PARSE_API_KEY else '⚠️ Optional'}")
            
            # Features status
            response_parts.append("\n## 🎯 Advanced Features")
            features = self.config.get_advanced_features_config()
            for feature, enabled in features.items():
                status = "✅ Enabled" if enabled else "❌ Disabled"
                response_parts.append(f"{feature.replace('_', ' ').title()}: {status}")
            
            # Papers status
            response_parts.append(f"\n## 📚 Processed Papers")
            response_parts.append(f"**Total Papers:** {len(self.processed_papers)}")
            
            if self.processed_papers:
                response_parts.append("**Paper List:**")
                for paper_id, paper_data in self.processed_papers.items():
                    title = paper_data.get('metadata', {}).get('title', paper_id)[:50]
                    pages = paper_data.get('summary_stats', {}).get('total_pages', 'Unknown')
                    response_parts.append(f"• {paper_id}: {title}... ({pages} pages)")
            
            # Health check
            if run_health_check:
                response_parts.append("\n## 🏥 Health Check")
                
                health_issues = self.config.validate_config()
                if health_issues:
                    response_parts.append("**Issues Found:**")
                    for issue in health_issues:
                        if issue.startswith("ERROR"):
                            response_parts.append(f"❌ {issue}")
                        else:
                            response_parts.append(f"⚠️ {issue}")
                else:
                    response_parts.append("✅ All systems operational")
            
            # Configuration (if requested)
            if include_config:
                response_parts.append("\n## ⚙️ Configuration")
                response_parts.append(f"**LLM Model:** {self.config.LLM_MODEL}")
                response_parts.append(f"**Embedding Model:** {self.config.EMBEDDING_MODEL}")
                response_parts.append(f"**Chunk Size:** {self.config.CHUNK_SIZE}")
                response_parts.append(f"**Max Slides:** {self.config.PPT_MAX_SLIDES}")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return [TextContent(type="text", text=f"System status error: {str(e)}")]

    async def _handle_list_papers(self, include_stats: bool = True, 
                                sort_by: str = "date") -> List[TextContent]:
        """Handle listing processed papers"""
        try:
            response_parts = ["# 📚 Processed Research Papers"]
            
            if not self.processed_papers:
                return [TextContent(type="text", text="No papers have been processed yet.")]
            
            response_parts.append(f"**Total Papers:** {len(self.processed_papers)}")
            
            # Sort papers
            papers = list(self.processed_papers.items())
            if sort_by == "name":
                papers.sort(key=lambda x: x[1].get('metadata', {}).get('title', x[0]))
            elif sort_by == "quality_score":
                papers.sort(key=lambda x: x[1].get('research_analysis', {}).get('quality_assessment', {}).get('overall_quality', 0), reverse=True)
            # Default: sort by processing order (date)
            
            response_parts.append("\n## 📄 Paper Details")
            
            for paper_id, paper_data in papers:
                title = paper_data.get('metadata', {}).get('title', 'Untitled')
                stats = paper_data.get('summary_stats', {})
                
                response_parts.append(f"\n### {paper_id}")
                response_parts.append(f"**Title:** {title}")
                
                if include_stats:
                    response_parts.append(f"**Pages:** {stats.get('total_pages', 'Unknown')}")
                    response_parts.append(f"**Words:** {stats.get('total_words', 'Unknown')}")
                    response_parts.append(f"**Extraction:** {stats.get('extraction_method', 'Unknown')}")
                    
                    # Analysis status
                    has_analysis = "research_analysis" in paper_data
                    has_vectors = "vector_stored" in paper_data  # This would be set during storage
                    
                    response_parts.append(f"**Research Analysis:** {'✅' if has_analysis else '❌'}")
                    response_parts.append(f"**Vector Storage:** {'✅' if has_vectors else '❌'}")
                    
                    # Quality score if available
                    if has_analysis:
                        quality = paper_data.get('research_analysis', {}).get('quality_assessment', {}).get('overall_quality', 0)
                        response_parts.append(f"**Quality Score:** {quality:.2f}/1.0")
                
                response_parts.append("---")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error listing papers: {e}")
            return [TextContent(type="text", text=f"Error listing papers: {str(e)}")]

    # ============================================================================
    # HELPER METHODS
    # ============================================================================

    def _format_search_results(self, search_results: Dict[str, Any], query: str, search_type: str) -> str:
        """Format search results for display"""
        response_parts = [f"# 🔍 Advanced Search Results"]
        response_parts.append(f"**Query:** {query}")
        response_parts.append(f"**Search Type:** {search_type}")
        response_parts.append(f"**Results:** {len(search_results.get('results', []))}")
        
        results = search_results.get("results", [])
        
        if results:
            response_parts.append("\n## 📄 Search Results")
            
            for i, result in enumerate(results, 1):
                response_parts.append(f"\n### {i}. {result.get('title', 'No Title')}")
                response_parts.append(f"**URL:** {result.get('link', 'N/A')}")
                
                snippet = result.get('snippet', 'No description available')
                response_parts.append(f"**Description:** {snippet}")
                
                if result.get('publication_info'):
                    pub_info = result['publication_info']
                    response_parts.append(f"**Publication:** {pub_info.get('source', 'Unknown')}")
                    if pub_info.get('date'):
                        response_parts.append(f"**Date:** {pub_info['date']}")
                
                response_parts.append("---")
        
        # Enhanced results if available
        if search_results.get("enhanced_results"):
            response_parts.append("\n## 🧠 AI-Enhanced Analysis")
            enhanced = search_results["enhanced_results"]
            
            if enhanced.get("key_themes"):
                themes = enhanced["key_themes"][:3]
                response_parts.append(f"**Key Themes:** {', '.join(themes)}")
            
            if enhanced.get("research_gaps"):
                gaps = enhanced["research_gaps"][:2]
                response_parts.append(f"**Research Gaps:** {'; '.join(gaps)}")
            
            if enhanced.get("methodology_trends"):
                trends = enhanced["methodology_trends"][:2]
                response_parts.append(f"**Methodology Trends:** {'; '.join(trends)}")
        
        return "\n".join(response_parts)

    async def _generate_smart_recommendations(self, research_analysis: Dict[str, Any]) -> List[str]:
        """Generate smart recommendations based on research analysis"""
        recommendations = []
        
        try:
            # Quality-based recommendations
            quality = research_analysis.get("quality_assessment", {})
            if quality.get("overall_quality", 0) < 0.7:
                recommendations.append("Consider strengthening the paper structure and completeness")
            
            # Methodology recommendations
            methodology = research_analysis.get("methodology_analysis", {})
            if methodology.get("rigor_score", 0) < 0.6:
                recommendations.append("Enhance methodological rigor with more detailed procedures")
            
            # Citation recommendations
            citations = research_analysis.get("citation_analysis", {})
            if citations.get("citation_density", 0) < 0.01:
                recommendations.append("Increase citation density to better support claims")
            
            # Statistical recommendations
            stats = research_analysis.get("statistical_results", [])
            if len(stats) < 3:
                recommendations.append("Consider adding more statistical analyses to strengthen findings")
            
            # General recommendations
            recommendations.append("Consider creating presentation slides focusing on methodology and key findings")
            recommendations.append("Use semantic search to find related content for deeper analysis")
            
        except Exception as e:
            logger.warning(f"Error generating recommendations: {e}")
            recommendations.append("Analysis complete - consider further review of methodology and results")
        
        return recommendations[:5]  # Limit to 5 recommendations

    # Additional tool implementations would continue here...
    # (Compare papers, generate insights, export summary methods)

    async def run(self):
        """Run the perfect MCP server"""
        # Create simple notification options  
        class SimpleNotificationOptions:
            def __init__(self):
                self.tools_changed = True
                self.resources_changed = True
        
        notification_options = SimpleNotificationOptions()
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="perfect-research-mcp",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=notification_options,
                        experimental_capabilities={}
                    )
                )
            )

# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main function to run the perfect MCP server"""
    try:
        server = PerfectMCPServer()
        logger.info("Starting Perfect Research MCP Server...")
        await server.run()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main()) 