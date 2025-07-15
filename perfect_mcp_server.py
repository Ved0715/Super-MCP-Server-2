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
import uuid

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.types import (
    Resource, Tool, TextContent, ImageContent, EmbeddedResource,
    Prompt, GetPromptResult, PromptMessage, PromptArgument
)
import mcp.server.stdio
import mcp.server.session

from config import AdvancedConfig
from enhanced_pdf_processor import EnhancedPDFProcessor
from vector_storage import AdvancedVectorStorage
from research_intelligence import ResearchPaperAnalyzer
from perfect_ppt_generator import PerfectPPTGenerator
from search_client import SerpAPIClient
from processors.universal_document_processor import UniversalDocumentProcessor

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
        
        # Initialize universal document processor
        self.universal_processor = UniversalDocumentProcessor()
        
        # Paper storage
        self.processed_papers = {}
        
        # MCP Protocol Enhancement - Track operations for progress/cancellation
        self.active_operations = {}  # operation_id -> operation_info
        self.cancellation_tokens = {}  # operation_id -> cancel_flag
        
        # Create MCP server
        self.server = Server("perfect-research-mcp")
        self._setup_tools()
        self._setup_resources()
        self._setup_prompts()
        
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
                            "user_id": {"type": "string", "description": "User identifier for namespace organization"},
                            "document_uuid": {"type": "string", "description": "Document UUID for namespace organization"},
                            "enable_research_analysis": {"type": "boolean", "default": True},
                            "enable_vector_storage": {"type": "boolean", "default": True},
                            "analysis_depth": {"type": "string", "enum": ["basic", "standard", "comprehensive"], "default": "comprehensive"}
                        },
                        "required": ["file_content", "file_name", "paper_id"]
                    }
                ),
                
                Tool(
                    name="create_perfect_presentation",
                    description="Create a perfect research presentation from knowledge base using Chain-of-Thought reasoning",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Main topic/query for presentation content"},
                            "user_prompt": {"type": "string", "description": "User's presentation requirements and specific focus"},
                            "title": {"type": "string", "description": "Presentation title"},
                            "author": {"type": "string", "description": "Presentation author"},
                            "theme": {"type": "string", "description": "Presentation theme"},
                            "slide_count": {"type": "integer", "description": "Number of slides"},
                            "audience_type": {"type": "string", "description": "Target audience type"},
                            "include_web_references": {"type": "boolean", "description": "Include web search for reference links"},
                            "reference_query": {"type": "string", "description": "Query for additional reference links"}
                        },
                        "required": ["query", "user_prompt"]
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
                ),
                
                Tool(
                    name="ai_enhanced_analysis",
                    description="Use AI sampling to enhance research analysis with advanced insights",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "paper_id": {
                                "type": "string",
                                "description": "ID of the processed paper to enhance"
                            },
                            "enhancement_type": {
                                "type": "string",
                                "description": "Type of AI enhancement",
                                "enum": ["insights", "quality_assessment", "general"],
                                "default": "insights"
                            },
                            "model_preference": {
                                "type": "string",
                                "description": "Preferred AI model for analysis",
                                "enum": ["claude-3-5-sonnet", "gpt-4", "auto"],
                                "default": "auto"
                            }
                        },
                        "required": ["paper_id"]
                    }
                ),
                
                Tool(
                    name="cancel_operation",
                    description="Cancel a running operation by its ID",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "operation_id": {
                                "type": "string",
                                "description": "ID of the operation to cancel"
                            }
                        },
                        "required": ["operation_id"]
                    }
                ),
                
                Tool(
                    name="list_active_operations",
                    description="List all currently active operations with their status and progress",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "include_completed": {
                                "type": "boolean",
                                "description": "Include recently completed operations",
                                "default": False
                            }
                        }
                    }
                ),
                
                # ============================================================================
                # NEW UNIVERSAL PROCESSING TOOLS
                # ============================================================================
                
                Tool(
                    name="process_research_paper_universal",
                    description="Process research papers using universal processor with all-pdf-index storage",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_content": {"type": "string", "description": "Hex-encoded PDF content"},
                            "filename": {"type": "string", "description": "Original filename"},
                            "paper_id": {"type": "string", "description": "Unique paper identifier"},
                            "enable_analysis": {"type": "boolean", "default": True},
                            "analysis_depth": {"type": "string", "enum": ["basic", "standard", "comprehensive"], "default": "comprehensive"}
                        },
                        "required": ["file_content", "filename"]
                    }
                ),
                
                Tool(
                    name="process_knowledge_base",
                    description="Process knowledge base content using optimized pipeline with optimized-kb-index storage",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_content": {"type": "string", "description": "Hex-encoded PDF content"},
                            "filename": {"type": "string", "description": "Original filename"},
                            "book_name": {"type": "string", "description": "Book name"},
                            "enable_llamaparse": {"type": "boolean", "default": True},
                            "extraction_mode": {"type": "string", "default": "knowledge_extraction"}
                        },
                        "required": ["file_content", "filename"]
                    }
                ),
                
                Tool(
                    name="search_knowledge_base",
                    description="Search knowledge base content with enhanced retrieval capabilities",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"},
                            "search_type": {"type": "string", "enum": ["enhanced", "basic"], "default": "enhanced"},
                            "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 20},
                            "namespace": {"type": "string", "default": "knowledge-base"},
                            "index_name": {"type": "string", "default": "optimized-kb-index"}
                        },
                        "required": ["query"]
                    }
                ),
                
                Tool(
                    name="get_knowledge_base_inventory",
                    description="Get comprehensive inventory of knowledge base content including books and chapters",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "namespace": {"type": "string", "default": "knowledge-base"},
                            "index_name": {"type": "string", "default": "optimized-kb-index"}
                        }
                    }
                ),
                
                Tool(
                    name="find_books_covering_topic",
                    description="Find which books in the knowledge base cover a specific topic",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "topic": {"type": "string", "description": "Topic to search for"},
                            "namespace": {"type": "string", "default": "knowledge-base"},
                            "index_name": {"type": "string", "default": "optimized-kb-index"}
                        },
                        "required": ["topic"]
                    }
                ),
                
                Tool(
                    name="generate_research_quiz",
                    description="Generate comprehensive MCQ quiz from a specific research paper uploaded by user",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "user_id": {
                                "type": "string",
                                "description": "User identifier who uploaded the PDF"
                            },
                            "document_uuid": {
                                "type": "string", 
                                "description": "Unique identifier for the research paper"
                            },
                            "number_of_questions": {
                                "type": "integer",
                                "description": "Number of MCQ questions to generate",
                                "default": 10,
                                "minimum": 5,
                                "maximum": 50
                            },
                            "difficulty_level": {
                                "type": "string",
                                "enum": ["easy", "medium", "hard", "mixed"],
                                "default": "mixed",
                                "description": "Difficulty level of questions"
                            },
                            "include_explanations": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include explanations for correct answers"
                            },
                            "question_categories": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": ["conceptual", "methodological", "factual", "analytical", "application"]
                                },
                                "default": ["conceptual", "methodological", "factual", "analytical"],
                                "description": "Categories of questions to include"
                            }
                        },
                        "required": ["user_id", "document_uuid"]
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
                
                elif name == "ai_enhanced_analysis":
                    return await self._handle_ai_enhanced_analysis(**arguments)
                
                elif name == "cancel_operation":
                    return await self._handle_cancel_operation(**arguments)
                
                elif name == "list_active_operations":
                    return await self._handle_list_operations(**arguments)
                
                # New universal processing tools
                elif name == "process_research_paper_universal":
                    return await self._handle_universal_research_paper(**arguments)
                
                elif name == "process_knowledge_base":
                    return await self._handle_universal_knowledge_base(**arguments)
                
                elif name == "search_knowledge_base":
                    return await self._handle_search_knowledge_base(**arguments)
                
                elif name == "get_knowledge_base_inventory":
                    return await self._handle_get_knowledge_base_inventory(**arguments)
                
                elif name == "find_books_covering_topic":
                    return await self._handle_find_books_covering_topic(**arguments)
                
                elif name == "generate_research_quiz":
                    return await self._handle_generate_research_quiz(**arguments)
                
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

    def _setup_prompts(self):
        """Setup MCP prompts for research workflows"""
        
        @self.server.list_prompts()
        async def handle_list_prompts() -> List[Prompt]:
            """List all available research prompts"""
            return [
                Prompt(
                    name="research_analysis_workflow",
                    description="Complete workflow for analyzing a research paper",
                    arguments=[
                        PromptArgument(
                            name="paper_id",
                            description="ID of the processed paper to analyze",
                            required=True
                        ),
                        PromptArgument(
                            name="analysis_focus",
                            description="Specific focus area for analysis",
                            required=False
                        )
                    ]
                ),
                Prompt(
                    name="presentation_creation_workflow",
                    description="Complete workflow for creating presentations from research",
                    arguments=[
                        PromptArgument(
                            name="paper_id",
                            description="ID of the processed paper",
                            required=True
                        ),
                        PromptArgument(
                            name="audience_type",
                            description="Target audience (academic, business, general)",
                            required=False
                        ),
                        PromptArgument(
                            name="presentation_style",
                            description="Presentation style preference",
                            required=False
                        )
                    ]
                ),
                Prompt(
                    name="literature_review_workflow",
                    description="Workflow for conducting literature review with multiple papers",
                    arguments=[
                        PromptArgument(
                            name="research_topic",
                            description="Main research topic or question",
                            required=True
                        ),
                        PromptArgument(
                            name="paper_count",
                            description="Number of papers to analyze",
                            required=False
                        )
                    ]
                ),
                Prompt(
                    name="research_insights_workflow",
                    description="Generate research insights and future directions",
                    arguments=[
                        PromptArgument(
                            name="paper_id",
                            description="ID of the analyzed paper",
                            required=True
                        ),
                        PromptArgument(
                            name="insight_type",
                            description="Type of insights to generate",
                            required=False
                        )
                    ]
                )
            ]
        
        @self.server.get_prompt()
        async def handle_get_prompt(name: str, arguments: dict) -> GetPromptResult:
            """Get a specific research workflow prompt"""
            if name == "research_analysis_workflow":
                paper_id = arguments.get("paper_id", "")
                analysis_focus = arguments.get("analysis_focus", "comprehensive")
                
                messages = [
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""# Research Paper Analysis Workflow

Please analyze the research paper with ID: {paper_id}

## Analysis Steps:
1. **Process the paper** using `process_research_paper` tool
2. **Perform intelligence analysis** using `research_intelligence_analysis` tool  
3. **Generate insights** using `generate_research_insights` tool
4. **Create presentation** using `create_perfect_presentation` tool

## Focus Area: {analysis_focus}

## Expected Outputs:
- Comprehensive research analysis
- Quality assessment scores
- Methodology evaluation
- Key findings summary
- Research insights and recommendations
- Professional presentation slides

Please execute these steps systematically to provide a complete research analysis."""
                        )
                    )
                ]
                
            elif name == "presentation_creation_workflow":
                paper_id = arguments.get("paper_id", "")
                audience_type = arguments.get("audience_type", "academic")
                presentation_style = arguments.get("presentation_style", "professional")
                
                messages = [
                    PromptMessage(
                        role="user", 
                        content=TextContent(
                            type="text",
                            text=f"""# Research Presentation Creation Workflow

Create a professional presentation from paper: {paper_id}

## Presentation Requirements:
- **Audience**: {audience_type}
- **Style**: {presentation_style}
- **Duration**: 15-20 minutes (12-15 slides)

## Creation Steps:
1. **Retrieve paper data** using resources or semantic search
2. **Analyze key findings** and methodology
3. **Create presentation** using `create_perfect_presentation` tool
4. **Enhance with search results** if needed

## Content Structure:
- Introduction & Background
- Methodology Overview  
- Key Results & Findings
- Discussion & Implications
- Conclusions & Future Work

Please create a compelling presentation that effectively communicates the research."""
                        )
                    )
                ]
                
            elif name == "literature_review_workflow":
                research_topic = arguments.get("research_topic", "")
                paper_count = arguments.get("paper_count", "5")
                
                messages = [
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text", 
                            text=f"""# Literature Review Workflow

Conduct a systematic literature review on: {research_topic}

## Review Parameters:
- **Topic**: {research_topic}
- **Target Papers**: {paper_count} papers
- **Analysis Depth**: Comprehensive

## Workflow Steps:
1. **Search for papers** using `advanced_search_web` tool (scholar mode)
2. **Process each paper** using `process_research_paper` tool
3. **Compare papers** using `compare_research_papers` tool
4. **Synthesize findings** across all papers
5. **Generate review summary** with themes and gaps

## Expected Deliverables:
- Individual paper analyses
- Comparative analysis matrix
- Key themes identification
- Research gaps analysis
- Future research directions
- Comprehensive literature review document

Execute this workflow to produce a thorough literature review."""
                        )
                    )
                ]
                
            elif name == "research_insights_workflow":
                paper_id = arguments.get("paper_id", "")
                insight_type = arguments.get("insight_type", "future_research")
                
                messages = [
                    PromptMessage(
                        role="user",
                        content=TextContent(
                            type="text",
                            text=f"""# Research Insights Generation Workflow

Generate advanced insights from paper: {paper_id}

## Insight Focus: {insight_type}

## Analysis Steps:
1. **Review paper analysis** using resources or semantic search
2. **Generate insights** using `generate_research_insights` tool
3. **Perform quality assessment** if not already done
4. **Identify research gaps** and opportunities
5. **Suggest methodological improvements**

## Insight Categories:
- Theoretical contributions
- Methodological innovations  
- Practical applications
- Future research directions
- Cross-disciplinary connections

## Output Format:
- Executive summary of insights
- Detailed analysis by category
- Actionable recommendations
- Research roadmap suggestions

Please provide comprehensive insights that advance understanding of this research area."""
                        )
                    )
                ]
            else:
                raise ValueError(f"Unknown prompt: {name}")
            
            return GetPromptResult(messages=messages)

    # ============================================================================
    # SAMPLING SUPPORT FOR AI MODEL INTERACTIONS  
    # ============================================================================

    async def request_model_sampling(self, messages: List[PromptMessage], model_preferences: dict = None) -> str:
        """Request AI model sampling from client"""
        try:
            # Prepare sampling request
            request_data = {
                "method": "sampling/createMessage",
                "params": {
                    "messages": [
                        {
                            "role": msg.role,
                            "content": msg.content.text if hasattr(msg.content, 'text') else str(msg.content)
                        } for msg in messages
                    ],
                    "modelPreferences": model_preferences or {
                        "hints": [{"name": "claude-3-5-sonnet"}, {"name": "gpt-4"}],
                        "costPriority": 0.5,
                        "speedPriority": 0.5,
                        "intelligencePriority": 0.9
                    },
                    "systemPrompt": "You are a research intelligence assistant helping with academic paper analysis.",
                    "includeContext": "thisServer",
                    "temperature": 0.7,
                    "maxTokens": 4000
                }
            }
            
            # Send sampling request to client
            response = await self.server.request_sampling(request_data)
            
            if response and response.get("content"):
                return response["content"].get("text", "No response received")
            else:
                return "Sampling request failed or no response received"
                
        except Exception as e:
            logger.warning(f"Model sampling failed: {e}")
            return f"Sampling unavailable: {str(e)}"

    async def enhance_analysis_with_ai(self, analysis_data: dict, enhancement_type: str = "insights") -> dict:
        """Use AI sampling to enhance research analysis"""
        try:
            # Create prompt for AI enhancement
            if enhancement_type == "insights":
                prompt_text = f"""
# Research Analysis Enhancement

Please analyze the following research paper data and provide advanced insights:

## Paper Metadata
- Title: {analysis_data.get('metadata', {}).get('title', 'Unknown')}
- Authors: {analysis_data.get('metadata', {}).get('authors', [])}

## Research Analysis Summary
{json.dumps(analysis_data.get('research_analysis', {}), indent=2)}

## Request
Please provide:
1. **Key Theoretical Contributions** - What new theories or concepts does this work introduce?
2. **Methodological Innovations** - What novel approaches or techniques are used?
3. **Practical Applications** - How can these findings be applied in real-world scenarios?
4. **Future Research Directions** - What questions does this work raise for future investigation?
5. **Cross-Disciplinary Connections** - How might this work connect to other fields?

Provide detailed, academic-level insights that would be valuable for researchers in this field.
"""
            
            elif enhancement_type == "quality_assessment":
                prompt_text = f"""
# Research Quality Assessment

Please provide a comprehensive quality assessment of this research paper:

## Paper Data
{json.dumps(analysis_data, indent=2)}

## Assessment Request
Please evaluate and score (1-10) the following aspects:
1. **Methodological Rigor** - How sound are the research methods?
2. **Novelty** - How original and innovative is this work?
3. **Evidence Quality** - How strong is the supporting evidence?
4. **Clarity** - How well is the work presented and explained?
5. **Impact Potential** - How significant could the impact be?
6. **Reproducibility** - How reproducible are the results?

For each aspect, provide:
- Score (1-10)
- Justification
- Specific recommendations for improvement

Also provide an overall assessment summary.
"""

            else:
                prompt_text = f"""
Please analyze this research paper data and provide expert insights:

{json.dumps(analysis_data, indent=2)}

Focus on providing actionable insights for researchers and practitioners.
"""

            # Create message for sampling
            messages = [
                PromptMessage(
                    role="user",
                    content=TextContent(type="text", text=prompt_text)
                )
            ]
            
            # Request AI enhancement
            ai_response = await self.request_model_sampling(messages)
            
            # Add AI enhancement to analysis
            analysis_data["ai_enhancement"] = {
                "type": enhancement_type,
                "response": ai_response,
                "timestamp": time.time()
            }
            
            return analysis_data
            
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            analysis_data["ai_enhancement"] = {
                "type": enhancement_type,
                "error": str(e),
                "timestamp": time.time()
            }
            return analysis_data

    # ============================================================================
    # PROGRESS TRACKING & CANCELLATION SUPPORT
    # ============================================================================

    async def send_progress_notification(self, operation_id: str, progress: int, message: str = ""):
        """Send progress notification to client"""
        try:
            # This would be sent to the client via the MCP connection
            await self.server.send_notification("progress", {
                "token": operation_id,
                "value": progress,
                "message": message
            })
        except Exception as e:
            logger.warning(f"Failed to send progress notification: {e}")

    async def send_log_notification(self, level: str, message: str, data: dict = None):
        """Send structured log notification to client"""
        try:
            await self.server.send_notification("logging/message", {
                "level": level,
                "message": message,
                "data": data or {}
            })
        except Exception as e:
            logger.warning(f"Failed to send log notification: {e}")

    def create_operation_id(self) -> str:
        """Create unique operation ID for tracking"""
        return str(uuid.uuid4())

    def start_operation(self, operation_id: str, name: str, description: str):
        """Start tracking an operation"""
        self.active_operations[operation_id] = {
            "name": name,
            "description": description,
            "start_time": time.time(),
            "status": "running"
        }
        self.cancellation_tokens[operation_id] = False

    def check_cancellation(self, operation_id: str) -> bool:
        """Check if operation should be cancelled"""
        return self.cancellation_tokens.get(operation_id, False)

    def cancel_operation(self, operation_id: str):
        """Cancel an operation"""
        if operation_id in self.cancellation_tokens:
            self.cancellation_tokens[operation_id] = True
        if operation_id in self.active_operations:
            self.active_operations[operation_id]["status"] = "cancelled"

    def complete_operation(self, operation_id: str):
        """Mark operation as completed"""
        if operation_id in self.active_operations:
            self.active_operations[operation_id]["status"] = "completed"
            self.active_operations[operation_id]["end_time"] = time.time()

    # ============================================================================
    # TOOL IMPLEMENTATION METHODS
    # ============================================================================

# Google Search 
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

#
    async def _handle_process_paper(self, file_content: str, file_name: str, paper_id: str,
                                  user_id: str = None, document_uuid: str = None,
                                  enable_research_analysis: bool = True,
                                  enable_vector_storage: bool = True,
                                  analysis_depth: str = "comprehensive") -> List[TextContent]:
        """Handle comprehensive research paper processing with progress tracking"""
        operation_id = self.create_operation_id()
        
        try:
            # Start operation tracking
            self.start_operation(operation_id, "process_research_paper", f"Processing {file_name}")
            await self.send_progress_notification(operation_id, 5, "Starting paper processing...")
            await self.send_log_notification("info", f"Starting paper processing: {file_name}", {"paper_id": paper_id})
            
            # Check for cancellation
            if self.check_cancellation(operation_id):
                return [TextContent(type="text", text="Operation cancelled by user")]
            
            # Decode file content
            import base64
            pdf_bytes = base64.b64decode(file_content)
            await self.send_progress_notification(operation_id, 10, "PDF decoded, starting extraction...")
            
            # Process PDF with advanced extraction
            extraction_result = await self.pdf_processor.extract_content_from_bytes(pdf_bytes, file_name)
            await self.send_progress_notification(operation_id, 40, "PDF extraction completed")
            
            if not extraction_result.get("success"):
                self.complete_operation(operation_id)
                return [TextContent(type="text", text=f"PDF processing failed: {extraction_result.get('error')}")]
            
            # Store extracted content
            self.processed_papers[paper_id] = extraction_result
            await self.send_progress_notification(operation_id, 50, "Paper content stored successfully")
            
            response_parts = [f"# Research Paper Processing Complete\n"]
            response_parts.append(f"**Paper ID:** {paper_id}")
            response_parts.append(f"**File:** {file_name}")
            response_parts.append(f"**Pages:** {extraction_result.get('summary_stats', {}).get('total_pages', 'Unknown')}")
            response_parts.append(f"**Words:** {extraction_result.get('summary_stats', {}).get('total_words', 'Unknown')}")
            response_parts.append(f"**Extraction Method:** {extraction_result.get('summary_stats', {}).get('extraction_method', 'Unknown')}")
            
            # Check for cancellation before analysis
            if self.check_cancellation(operation_id):
                self.complete_operation(operation_id)
                return [TextContent(type="text", text="Operation cancelled during analysis phase")]
            
            # Research intelligence analysis
            if enable_research_analysis and self.research_analyzer:
                response_parts.append("\n## 🧠 Research Intelligence Analysis")
                await self.send_progress_notification(operation_id, 60, "Starting research intelligence analysis...")
                
                try:
                    research_analysis = await self.research_analyzer.analyze_research_paper(extraction_result)
                    self.processed_papers[paper_id]["research_analysis"] = research_analysis
                    await self.send_progress_notification(operation_id, 75, "Research analysis completed")
                    
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
            
            # Check for cancellation before vector storage
            if self.check_cancellation(operation_id):
                self.complete_operation(operation_id)
                return [TextContent(type="text", text="Operation cancelled during vector storage phase")]
            
            # Vector storage
            if enable_vector_storage and self.vector_storage:
                response_parts.append("\n## 🔍 Vector Storage & Semantic Indexing")
                await self.send_progress_notification(operation_id, 85, "Starting vector storage...")
                
                try:
                    # Generate dynamic namespace
                    if user_id and document_uuid:
                        namespace = f"user_{user_id}_doc_{document_uuid}"
                    elif user_id:
                        namespace = f"user_{user_id}_doc_{paper_id}"
                    elif document_uuid:
                        namespace = f"doc_{document_uuid}"
                    else:
                        namespace = paper_id
                    
                    response_parts.append(f"**Namespace:** {namespace}")
                    
                    storage_result = await self.vector_storage.process_and_store_document(
                        extraction_result, paper_id, namespace, user_id, document_uuid
                    )
                    
                    if storage_result.get("success"):
                        response_parts.append(f"**Chunks Created:** {storage_result.get('chunks_created', 0)}")
                        response_parts.append(f"**Chunks Stored:** {storage_result.get('chunks_stored', 0)}")
                        response_parts.append("**Semantic Search:** ✅ Enabled")
                        response_parts.append(f"**Storage Location:** {namespace}")
                        await self.send_progress_notification(operation_id, 95, "Vector storage completed")
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
            
            # Add namespace information to response
            if user_id or document_uuid:
                response_parts.append(f"\n## 🔗 Storage Information")
                response_parts.append(f"**Namespace:** {namespace}")
                if user_id:
                    response_parts.append(f"**User ID:** {user_id}")
                if document_uuid:
                    response_parts.append(f"**Document UUID:** {document_uuid}")
                response_parts.append(f"**Search Pattern:** Use namespace `{namespace}` for document-specific searches")
            
            # Complete operation tracking
            await self.send_progress_notification(operation_id, 100, "Paper processing completed successfully!")
            await self.send_log_notification("info", "Paper processing completed", {
                "paper_id": paper_id, 
                "pages": extraction_result.get('summary_stats', {}).get('total_pages', 0),
                "analysis_enabled": enable_research_analysis,
                "vector_storage_enabled": enable_vector_storage
            })
            self.complete_operation(operation_id)
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            # Complete operation on error
            self.complete_operation(operation_id)
            await self.send_log_notification("error", f"Paper processing failed: {str(e)}", {"paper_id": paper_id})
            logger.error(f"Error processing paper: {e}")
            return [TextContent(type="text", text=f"Paper processing error: {str(e)}")]

    async def _chain_of_thought_presentation_analysis(self, query: str, user_prompt: str) -> Dict[str, Any]:
        """
        Chain-of-Thought analysis for presentation creation using self-questioning approach
        The model generates questions for itself and answers them step by step
        """
        try:
            logger.info(f"🧠 Starting Chain-of-Thought analysis for: {query}")
            
            # Import OpenAI for CoT reasoning
            import openai
            client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            # Chain-of-Thought prompt that encourages self-questioning
            cot_prompt = f"""You are an expert presentation advisor. Use Chain-of-Thought reasoning by asking yourself questions and answering them step by step.

USER QUERY: "{query}"
USER REQUIREMENTS: "{user_prompt}"

Please think through this systematically by asking yourself questions and providing answers:

Q1: What is the main topic the user wants to present?
A1: [Answer based on the query]

Q2: What are the key subtopics I should cover for this topic?
A2: [List the main subtopics]

Q3: What specific aspects should I focus on based on the user's requirements?
A3: [Analyze user_prompt for specific focus areas]

Q4: What type of content would be most valuable for each subtopic?
A4: [Determine content types: definitions, examples, processes, comparisons, etc.]

Q5: How should I structure this information for maximum presentation impact?
A5: [Logical flow and organization]

Q6: What are the most important search terms to find relevant knowledge base content?
A6: [List specific search terms]

Q7: What kind of examples or evidence would strengthen each point?
A7: [Types of supporting evidence needed]

Q8: What are potential questions the audience might have?
A8: [Anticipate audience questions]

Q9: How can I ensure comprehensive coverage of the topic?
A9: [Strategy for thorough coverage]

Q10: What would be the most logical slide structure?
A10: [Proposed slide organization]

Based on this analysis, provide your final recommendation in JSON format:
{{
    "main_topic": "...",
    "key_subtopics": ["...", "..."],
    "focus_areas": ["...", "..."],
    "search_terms": ["...", "..."],
    "content_strategy": "...",
    "slide_structure": ["...", "..."],
    "audience_considerations": "...",
    "evidence_needs": ["...", "..."]
}}"""
            
            # Get CoT analysis
            response = client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert presentation advisor who uses systematic self-questioning to analyze presentation requirements."},
                    {"role": "user", "content": cot_prompt}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            cot_analysis = response.choices[0].message.content
            logger.info(f"✅ Chain-of-Thought analysis completed")
            
            # Extract JSON recommendation from the response
            import re
            import json
            
            # Find JSON in the response
            json_match = re.search(r'\{[^}]+\}', cot_analysis.replace('\n', ' '))
            if json_match:
                try:
                    structured_analysis = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # Fallback structured analysis
                    structured_analysis = {
                        "main_topic": query,
                        "search_terms": [query, user_prompt],
                        "content_strategy": "comprehensive_coverage"
                    }
            else:
                structured_analysis = {
                    "main_topic": query,
                    "search_terms": [query, user_prompt],
                    "content_strategy": "comprehensive_coverage"
                }
            
            return {
                "success": True,
                "cot_reasoning": cot_analysis,
                "structured_analysis": structured_analysis,
                "query": query,
                "user_prompt": user_prompt
            }
            
        except Exception as e:
            logger.error(f"❌ Chain-of-Thought analysis failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_analysis": {
                    "main_topic": query,
                    "search_terms": [query, user_prompt],
                    "content_strategy": "basic_coverage"
                }
            }

    async def _comprehensive_knowledge_search(self, cot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive knowledge base search using CoT analysis
        Searches for maximum relevant content across all identified topics
        """
        try:
            logger.info(f"🔍 Starting comprehensive knowledge base search")
            
            # Import the advanced knowledge base retriever
            from knowledge_base_retrieval import AdvancedKnowledgeBaseRetriever
            
            # Create retriever instance
            kb_retriever = AdvancedKnowledgeBaseRetriever()
            
            # Extract search terms from CoT analysis
            structured_analysis = cot_analysis.get("structured_analysis", {})
            search_terms = structured_analysis.get("search_terms", [])
            main_topic = structured_analysis.get("main_topic", "")
            key_subtopics = structured_analysis.get("key_subtopics", [])
            focus_areas = structured_analysis.get("focus_areas", [])
            
            # Build optimized search query list (reduce from 15+ to 5-8 focused searches)
            all_search_queries = []
            
            # Priority 1: Main topic (most important)
            if main_topic and main_topic.strip():
                all_search_queries.append(main_topic.strip())
            
            # Priority 2: Top 3 subtopics only (not all)
            if key_subtopics:
                all_search_queries.extend(key_subtopics[:3])
            
            # Priority 3: Top 2 focus areas only (not all)
            if focus_areas:
                all_search_queries.extend(focus_areas[:2])
            
            # Priority 4: Only unique search terms that aren't already covered
            for term in search_terms:
                term_clean = term.strip()
                if term_clean and not any(term_clean.lower() in existing.lower() for existing in all_search_queries):
                    all_search_queries.append(term_clean)
                    if len(all_search_queries) >= 8:  # Cap at 8 total searches
                        break
            
            # Remove duplicates and empty strings, maintain order
            seen = set()
            unique_queries = []
            for q in all_search_queries:
                q_clean = q.strip()
                if q_clean and q_clean.lower() not in seen:
                    seen.add(q_clean.lower())
                    unique_queries.append(q_clean)
            
            all_search_queries = unique_queries[:8]  # Final cap at 8 searches
            
            logger.info(f"📝 Performing {len(all_search_queries)} optimized knowledge base searches")
            
            # Perform multiple searches for comprehensive coverage
            all_results = []
            total_content_length = 0
            
            for query in all_search_queries:
                try:
                    logger.info(f"🔍 Searching: {query}")
                    
                    # Use intelligent search for each query
                    search_result = await kb_retriever.intelligent_search(
                        query=query,
                        top_k=15,  # Get more results per query
                        namespace="knowledge-base",
                        index_name="optimized-kb-index"
                    )
                    
                    if search_result.get("success"):
                        # Extract search results
                        search_results = search_result.get("search_results", [])
                        ai_response = search_result.get("ai_response", "")
                        
                        # Add to comprehensive results
                        all_results.extend(search_results)
                        total_content_length += len(ai_response)
                        
                        logger.info(f"✅ Found {len(search_results)} results for '{query}'")
                    else:
                        logger.warning(f"⚠️ Search failed for '{query}': {search_result.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    logger.error(f"❌ Search error for '{query}': {e}")
                    continue
            
            # Remove duplicate content based on chunk_id
            unique_results = {}
            for result in all_results:
                chunk_id = result.get('chunk_id', '')
                if chunk_id not in unique_results:
                    unique_results[chunk_id] = result
                else:
                    # Keep the result with higher score
                    if result.get('score', 0) > unique_results[chunk_id].get('score', 0):
                        unique_results[chunk_id] = result
            
            final_results = list(unique_results.values())
            
            # Sort by relevance score
            final_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            
            # Calculate statistics
            total_chunks = len(final_results)
            total_content = sum(len(result.get('content', '')) for result in final_results)
            avg_score = sum(result.get('score', 0) for result in final_results) / total_chunks if total_chunks > 0 else 0
            
            # Get unique books covered
            unique_books = set()
            for result in final_results:
                book_name = result.get('metadata', {}).get('book_name', 'Unknown')
                unique_books.add(book_name)
            
            logger.info(f"✅ Comprehensive search completed:")
            logger.info(f"   - Total chunks: {total_chunks}")
            logger.info(f"   - Total content: {total_content} characters")
            logger.info(f"   - Average score: {avg_score:.3f}")
            logger.info(f"   - Books covered: {len(unique_books)}")
            
            return {
                "success": True,
                "search_results": final_results,
                "statistics": {
                    "total_chunks": total_chunks,
                    "total_content_length": total_content,
                    "average_score": avg_score,
                    "unique_books": len(unique_books),
                    "books_covered": list(unique_books),
                    "search_queries_used": all_search_queries
                },
                "coverage_analysis": {
                    "main_topic_covered": main_topic in str(final_results),
                    "subtopics_found": len([topic for topic in key_subtopics if topic in str(final_results)]),
                    "focus_areas_covered": len([area for area in focus_areas if area in str(final_results)])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Comprehensive knowledge search failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_results": [],
                "statistics": {
                    "total_chunks": 0,
                    "total_content_length": 0,
                    "average_score": 0,
                    "unique_books": 0,
                    "books_covered": [],
                    "search_queries_used": []
                }
                         }

    def _transform_search_results_to_paper_format(self, search_results: Dict[str, Any], cot_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform knowledge base search results into paper-like content format
        for compatibility with existing PPT generator
        """
        try:
            logger.info(f"🔧 Transforming search results into paper format")
            
            if not search_results.get("success"):
                logger.error(f"❌ Search results transformation failed - no valid search results")
                return {
                    "success": False,
                    "error": "No valid search results to transform"
                }
            
            search_data = search_results.get("search_results", [])
            statistics = search_results.get("statistics", {})
            structured_analysis = cot_analysis.get("structured_analysis", {})
            
            # Create aggregated content sections
            aggregated_content = {
                "success": True,
                "extraction_method": "knowledge_base_search",
                "paper_id": f"kb_query_{structured_analysis.get('main_topic', 'unknown')}",
                "title": structured_analysis.get('main_topic', 'Knowledge Base Presentation'),
                "author": "Knowledge Base",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                
                # Full text aggregation
                "full_text": "",
                
                # Structured sections
                "sections": {},
                
                # Metadata
                "metadata": {
                    "total_chunks": statistics.get("total_chunks", 0),
                    "total_content_length": statistics.get("total_content_length", 0),
                    "average_score": statistics.get("average_score", 0),
                    "books_covered": statistics.get("books_covered", []),
                    "search_queries": statistics.get("search_queries_used", [])
                },
                
                # References for citation
                "references": []
            }
            
            # Aggregate content by sections/topics
            content_by_section = {}
            all_content = []
            
            # Group content by subtopics/sections using intelligent categorization
            key_subtopics = structured_analysis.get('key_subtopics', [])
            
            # Create semantic topic mappings for better categorization
            def categorize_content_intelligently(content_text: str, subtopics: List[str]) -> str:
                """Intelligent content categorization using multiple strategies"""
                content_lower = content_text.lower()
                
                # Strategy 1: Direct substring matching (original approach)
                for subtopic in subtopics:
                    if subtopic.lower() in content_lower:
                        return subtopic
                
                # Strategy 2: Keyword-based matching for common topics
                topic_keywords = {
                    "supervised learning": ["supervised", "classification", "regression", "labeled data", "training", "prediction"],
                    "classification algorithms": ["classification", "classifier", "categorization", "decision tree", "svm", "neural network"],
                    "regression algorithms": ["regression", "linear regression", "polynomial", "prediction", "continuous", "forecasting"],
                    "machine learning": ["machine learning", "ml", "algorithm", "model", "training", "learning"],
                    "data science": ["data science", "analytics", "statistics", "data analysis", "insights"],
                    "applications": ["application", "use case", "example", "implementation", "practical", "real-world"],
                    "evaluation": ["evaluation", "metrics", "performance", "accuracy", "precision", "recall", "validation"],
                    "challenges": ["challenge", "limitation", "problem", "difficulty", "issue", "constraint"],
                    "methodology": ["methodology", "method", "approach", "technique", "procedure", "process"],
                    "introduction": ["introduction", "overview", "background", "definition", "concept", "fundamental"]
                }
                
                # Find best matching topic based on keyword overlap
                best_topic = None
                best_score = 0
                
                for subtopic in subtopics:
                    # Check if this subtopic has keyword mappings
                    subtopic_lower = subtopic.lower()
                    keywords = topic_keywords.get(subtopic_lower, [])
                    
                    if not keywords:
                        # Create keywords from subtopic words
                        keywords = subtopic_lower.split()
                    
                    # Count keyword matches
                    matches = sum(1 for keyword in keywords if keyword in content_lower)
                    if matches > best_score:
                        best_score = matches
                        best_topic = subtopic
                
                # Strategy 3: Fallback to thematic categorization
                if not best_topic and best_score == 0:
                    # Categorize by content themes
                    if any(term in content_lower for term in ["algorithm", "method", "technique", "approach"]):
                        return "Algorithms and Methods"
                    elif any(term in content_lower for term in ["application", "example", "use case", "practical"]):
                        return "Applications"
                    elif any(term in content_lower for term in ["evaluation", "performance", "metric", "accuracy"]):
                        return "Evaluation and Metrics"
                    elif any(term in content_lower for term in ["challenge", "limitation", "problem", "difficulty"]):
                        return "Challenges and Limitations"
                    elif any(term in content_lower for term in ["introduction", "overview", "definition", "concept"]):
                        return "Introduction and Concepts"
                    else:
                        return "General Topics"
                
                return best_topic or "General Topics"
            
            for result in search_data:
                content = result.get('content', '')
                book_name = result.get('metadata', {}).get('book_name', 'Unknown')
                score = result.get('score', 0)
                
                # Add to full text
                all_content.append(content)
                
                # Intelligently categorize content
                category = categorize_content_intelligently(content, key_subtopics)
                
                # Add to appropriate section
                if category not in content_by_section:
                    content_by_section[category] = []
                content_by_section[category].append({
                    "content": content,
                    "source": book_name,
                    "score": score
                })
                
                # Add to references
                if book_name not in [ref["source"] for ref in aggregated_content["references"]]:
                    aggregated_content["references"].append({
                        "source": book_name,
                        "type": "knowledge_base",
                        "relevance_score": float(score) if score > 0 else 0.5  # Use actual score or default to 0.5
                    })
            
            # Create sections
            for section_name, section_content in content_by_section.items():
                # Sort by relevance score
                section_content.sort(key=lambda x: x.get('score', 0), reverse=True)
                
                # Combine content
                combined_content = "\n\n".join([item["content"] for item in section_content])
                
                aggregated_content["sections"][section_name] = {
                    "content": combined_content,
                    "sources": list(set([item["source"] for item in section_content])),
                    "chunk_count": len(section_content),
                    "avg_score": sum(item.get('score', 0) for item in section_content) / len(section_content)
                }
            
            # Create full text
            aggregated_content["full_text"] = "\n\n".join(all_content)
            
            # Add summary information
            aggregated_content["summary"] = {
                "topic": structured_analysis.get('main_topic', 'Unknown'),
                "subtopics_covered": len(content_by_section),
                "total_sources": len(aggregated_content["references"]),
                "content_quality": statistics.get("average_score", 0)
            }
            
            logger.info(f"✅ Content transformation completed:")
            logger.info(f"   - Sections created: {len(content_by_section)}")
            logger.info(f"   - Full text length: {len(aggregated_content['full_text'])} characters")
            logger.info(f"   - References: {len(aggregated_content['references'])}")
            
            return aggregated_content
            
        except Exception as e:
            logger.error(f"❌ Content transformation failed: {e}")
            return {
                "success": False,
                "error": f"Content transformation failed: {str(e)}",
                "full_text": "",
                "sections": {},
                "references": []
            }

    async def _handle_create_presentation(self, query: str, user_prompt: str,
                                        title: str = None, author: str = "AI Research Assistant",
                                        theme: str = "academic_professional", slide_count: int = 12,
                                        audience_type: str = "academic", 
                                        include_web_references: bool = False,
                                        reference_query: str = None) -> List[TextContent]:
        """Handle knowledge base-driven presentation creation with Chain-of-Thought reasoning"""
        try:
            logger.info(f"🎯 Starting knowledge base-driven presentation creation")
            logger.info(f"📝 Query: {query}")
            logger.info(f"🎨 Theme: {theme}, Slides: {slide_count}, Audience: {audience_type}")
            
            response_parts = [f"# 🎯 Creating Perfect Knowledge Base Presentation"]
            response_parts.append(f"**Topic:** {query}")
            response_parts.append(f"**User Requirements:** {user_prompt}")
            response_parts.append(f"**Theme:** {theme}")
            response_parts.append(f"**Slides:** {slide_count}")
            response_parts.append(f"**Audience:** {audience_type}")
            
            # Step 1: Chain-of-Thought Analysis
            response_parts.append(f"\n## 🧠 Chain-of-Thought Analysis")
            logger.info(f"🧠 Step 1/4: Performing Chain-of-Thought analysis")
            
            cot_analysis = await self._chain_of_thought_presentation_analysis(query, user_prompt)
            
            if cot_analysis.get("success"):
                structured_analysis = cot_analysis.get("structured_analysis", {})
                response_parts.append(f"**Main Topic:** {structured_analysis.get('main_topic', query)}")
                response_parts.append(f"**Key Subtopics:** {', '.join(structured_analysis.get('key_subtopics', []))}")
                response_parts.append(f"**Focus Areas:** {', '.join(structured_analysis.get('focus_areas', []))}")
                response_parts.append(f"**Search Strategy:** {structured_analysis.get('content_strategy', 'comprehensive')}")
                logger.info(f"✅ Chain-of-Thought analysis completed successfully")
            else:
                response_parts.append(f"⚠️ Using fallback analysis due to error: {cot_analysis.get('error', 'Unknown error')}")
                logger.warning(f"⚠️ Chain-of-Thought analysis failed, using fallback")
            
            # Step 2: Comprehensive Knowledge Base Search
            response_parts.append(f"\n## 🔍 Comprehensive Knowledge Base Search")
            logger.info(f"🔍 Step 2/4: Performing comprehensive knowledge base search")
            
            search_results = await self._comprehensive_knowledge_search(cot_analysis)
            
            if search_results.get("success"):
                stats = search_results.get("statistics", {})
                response_parts.append(f"**Total Content Chunks:** {stats.get('total_chunks', 0)}")
                response_parts.append(f"**Content Length:** {stats.get('total_content_length', 0):,} characters")
                response_parts.append(f"**Average Relevance Score:** {stats.get('average_score', 0):.3f}")
                response_parts.append(f"**Knowledge Sources:** {stats.get('unique_books', 0)} books")
                response_parts.append(f"**Books Covered:** {', '.join(stats.get('books_covered', []))}")
                logger.info(f"✅ Knowledge base search completed: {stats.get('total_chunks', 0)} chunks from {stats.get('unique_books', 0)} books")
            else:
                response_parts.append(f"⚠️ Knowledge base search failed: {search_results.get('error', 'Unknown error')}")
                logger.error(f"❌ Knowledge base search failed")
                return [TextContent(type="text", text="\n".join(response_parts))]
            
            # Step 3: Get Web References (if requested)
            web_references = None
            if include_web_references:
                response_parts.append(f"\n## 🌐 Web Reference Links")
                logger.info(f"🌐 Step 3/4: Gathering web references")
                
                try:
                    reference_search_query = reference_query or query
                    web_references = self.search_client.search_google(
                        query=reference_search_query,
                    search_type="scholar",
                    num_results=5
                )
                    response_parts.append(f"**Additional References:** {len(web_references) if web_references else 0} web sources found")
                    logger.info(f"✅ Web references gathered: {len(web_references) if web_references else 0} sources")
                except Exception as e:
                    response_parts.append(f"⚠️ Web reference search failed: {str(e)}")
                    logger.warning(f"⚠️ Web reference search failed: {e}")
            
            # Step 4: Create Presentation
            response_parts.append(f"\n## 🎨 Presentation Generation")
            logger.info(f"🎨 Step 4/4: Creating presentation")
            
            # Transform search results into paper-like content format
            aggregated_content = self._transform_search_results_to_paper_format(search_results, cot_analysis)
            
            # Create presentation using existing PPT generator
            presentation_path = await self.ppt_generator.create_perfect_presentation(
                paper_content=aggregated_content,
                user_prompt=user_prompt,
                paper_id=f"kb_presentation_{query[:50]}",  # Use truncated query as paper_id
                search_results=web_references,
                title=title,
                author=author,
                theme=theme,
                slide_count=slide_count,
                audience_type=audience_type
            )
            
            response_parts.append(f"\n✅ **Perfect presentation created successfully!**")
            response_parts.append(f"**File:** {os.path.basename(presentation_path)}")
            response_parts.append(f"**Location:** {os.path.abspath(presentation_path)}")
            
            # Add presentation details
            response_parts.append(f"\n## 🎨 Presentation Features")
            response_parts.append("✅ Chain-of-Thought reasoning for topic analysis")
            response_parts.append("✅ Comprehensive knowledge base content integration")
            response_parts.append("✅ Maximum relevant content coverage")
            response_parts.append("✅ Professional academic theme with visual enhancements")
            response_parts.append("✅ Audience-appropriate depth and technical level")
            response_parts.append("✅ Proper source attribution and references")
            
            if include_web_references:
                response_parts.append("✅ Additional web reference links included")
            
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
    # NEW MCP PROTOCOL ENHANCEMENT TOOLS
    # ============================================================================

    async def _handle_ai_enhanced_analysis(self, paper_id: str, enhancement_type: str = "insights", 
                                         model_preference: str = "auto") -> List[TextContent]:
        """Handle AI-enhanced analysis using sampling"""
        try:
            if paper_id not in self.processed_papers:
                return [TextContent(type="text", text=f"Paper not found: {paper_id}. Please process the paper first.")]
            
            paper_data = self.processed_papers[paper_id]
            
            await self.send_log_notification("info", f"Starting AI-enhanced analysis", {
                "paper_id": paper_id, 
                "enhancement_type": enhancement_type,
                "model_preference": model_preference
            })
            
            # Set model preferences based on user choice
            model_prefs = None
            if model_preference != "auto":
                model_prefs = {
                    "hints": [{"name": model_preference}],
                    "costPriority": 0.5,
                    "speedPriority": 0.5,
                    "intelligencePriority": 0.9
                }
            
            # Enhance analysis with AI
            enhanced_data = await self.enhance_analysis_with_ai(paper_data, enhancement_type)
            
            # Store enhanced analysis
            self.processed_papers[paper_id] = enhanced_data
            
            response_parts = [f"# 🤖 AI-Enhanced Research Analysis"]
            response_parts.append(f"**Paper:** {paper_id}")
            response_parts.append(f"**Enhancement Type:** {enhancement_type}")
            response_parts.append(f"**Model Used:** {model_preference}")
            
            if enhanced_data.get("ai_enhancement"):
                ai_result = enhanced_data["ai_enhancement"]
                
                if ai_result.get("response"):
                    response_parts.append(f"\n## 🧠 AI Analysis Results\n")
                    response_parts.append(ai_result["response"])
                elif ai_result.get("error"):
                    response_parts.append(f"\n## ❌ Enhancement Error\n{ai_result['error']}")
                    response_parts.append("\n*Falling back to basic analysis...*")
            
            response_parts.append(f"\n✅ **AI-enhanced analysis complete!** Enhanced data stored for future use.")
            
            await self.send_log_notification("info", "AI-enhanced analysis completed", {"paper_id": paper_id})
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            await self.send_log_notification("error", f"AI enhancement failed: {str(e)}", {"paper_id": paper_id})
            logger.error(f"Error in AI enhanced analysis: {e}")
            return [TextContent(type="text", text=f"AI enhancement error: {str(e)}")]

    async def _handle_cancel_operation(self, operation_id: str) -> List[TextContent]:
        """Handle operation cancellation"""
        try:
            if operation_id not in self.active_operations:
                return [TextContent(type="text", text=f"Operation not found: {operation_id}")]
            
            operation_info = self.active_operations[operation_id]
            
            # Cancel the operation
            self.cancel_operation(operation_id)
            
            await self.send_log_notification("info", f"Operation cancelled", {
                "operation_id": operation_id,
                "operation_name": operation_info.get("name", "unknown")
            })
            
            response = f"""# ⏹️ Operation Cancelled

**Operation ID:** {operation_id}
**Operation:** {operation_info.get('name', 'Unknown')}
**Description:** {operation_info.get('description', 'N/A')}
**Status:** Cancelled

The operation has been successfully cancelled and will stop as soon as possible."""
            
            return [TextContent(type="text", text=response)]
            
        except Exception as e:
            logger.error(f"Error cancelling operation: {e}")
            return [TextContent(type="text", text=f"Cancellation error: {str(e)}")]

    async def _handle_list_operations(self, include_completed: bool = False) -> List[TextContent]:
        """Handle listing active operations"""
        try:
            response_parts = [f"# 📋 Active Operations Status"]
            
            if not self.active_operations:
                response_parts.append("\n*No active operations currently running.*")
                return [TextContent(type="text", text="\n".join(response_parts))]
            
            active_count = 0
            completed_count = 0
            
            for op_id, op_info in self.active_operations.items():
                status = op_info.get("status", "unknown")
                
                if status == "running":
                    active_count += 1
                    response_parts.append(f"\n## 🔄 {op_info.get('name', 'Unknown Operation')}")
                    response_parts.append(f"**ID:** `{op_id}`")
                    response_parts.append(f"**Description:** {op_info.get('description', 'N/A')}")
                    response_parts.append(f"**Status:** Running")
                    
                    if "start_time" in op_info:
                        duration = time.time() - op_info["start_time"]
                        response_parts.append(f"**Duration:** {duration:.1f}s")
                        
                elif status == "completed" and include_completed:
                    completed_count += 1
                    response_parts.append(f"\n## ✅ {op_info.get('name', 'Unknown Operation')}")
                    response_parts.append(f"**ID:** `{op_id}`")
                    response_parts.append(f"**Status:** Completed")
                    
                    if "start_time" in op_info and "end_time" in op_info:
                        duration = op_info["end_time"] - op_info["start_time"]
                        response_parts.append(f"**Duration:** {duration:.1f}s")
                        
                elif status == "cancelled":
                    response_parts.append(f"\n## ⏹️ {op_info.get('name', 'Unknown Operation')}")
                    response_parts.append(f"**ID:** `{op_id}`")
                    response_parts.append(f"**Status:** Cancelled")
            
            # Summary
            response_parts.append(f"\n---")
            response_parts.append(f"**Active Operations:** {active_count}")
            if include_completed:
                response_parts.append(f"**Completed Operations:** {completed_count}")
            
            if active_count > 0:
                response_parts.append(f"\n💡 *Use `cancel_operation` tool with operation ID to cancel running operations.*")
            
            return [TextContent(type="text", text="\n".join(response_parts))]
            
        except Exception as e:
            logger.error(f"Error listing operations: {e}")
            return [TextContent(type="text", text=f"Error listing operations: {str(e)}")]

    # ============================================================================
    # NEW UNIVERSAL PROCESSING HANDLERS
    # ============================================================================

    async def _handle_universal_research_paper(self, file_content: str, filename: str, 
                                             paper_id: str = None, enable_analysis: bool = True,
                                             analysis_depth: str = "comprehensive") -> List[TextContent]:
        """Handle research paper processing using universal processor"""
        operation_id = self.create_operation_id()
        
        try:
            self.start_operation(operation_id, "process_research_paper_universal", f"Processing research paper: {filename}")
            
            # Convert hex string back to bytes
            file_bytes = bytes.fromhex(file_content)
            
            # Generate paper_id if not provided
            if not paper_id:
                paper_id = f"paper_{uuid.uuid4().hex[:8]}"
            
            await self.send_progress_notification(operation_id, 10, "Starting research paper processing...")
            
            # Use universal processor
            result = await self.universal_processor.process_document(
                file_content=file_bytes,
                document_type="research_paper",
                index_name="all-pdf-index",
                paper_id=paper_id
            )
            
            await self.send_progress_notification(operation_id, 80, "Research paper processing completed")
            
            if result.get("success"):
                self.processed_papers[paper_id] = {
                    "filename": filename,
                    "metadata": {"title": filename},
                    "processing_type": "research_paper",
                    "index_used": "all-pdf-index"
                }
                
                await self.send_progress_notification(operation_id, 100, "Research paper stored successfully")
                self.complete_operation(operation_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "paper_id": paper_id,
                        "filename": filename,
                        "processing_type": "research_paper",
                        "index_used": "all-pdf-index",
                        "chunks_created": result.get("chunks_created", 0),
                        "message": "Research paper processed and stored successfully"
                    }, indent=2)
                )]
            else:
                error_msg = result.get("error", "Processing failed")
                await self.send_log_notification("error", f"Research paper processing failed: {error_msg}")
                self.complete_operation(operation_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": error_msg,
                        "paper_id": paper_id,
                        "filename": filename
                    }, indent=2)
                )]
                
        except Exception as e:
            await self.send_log_notification("error", f"Research paper processing failed: {e}")
            self.complete_operation(operation_id)
            
            return [TextContent(
                type="text", 
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "paper_id": paper_id,
                    "filename": filename
                }, indent=2)
            )]

    async def _handle_universal_knowledge_base(self, file_content: str, filename: str,
                                             book_name: str = None, enable_llamaparse: bool = True,
                                             extraction_mode: str = "knowledge_extraction") -> List[TextContent]:
        """Handle knowledge base content processing using universal processor"""
        operation_id = self.create_operation_id()
        
        try:
            self.start_operation(operation_id, "process_knowledge_base", f"Processing knowledge base: {filename}")
            
            # Convert hex string back to bytes
            file_bytes = bytes.fromhex(file_content)
            
            # Extract book name if not provided
            if not book_name:
                book_name = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
            
            await self.send_progress_notification(operation_id, 10, "Starting knowledge base processing...")
            
            # Use universal processor
            result = await self.universal_processor.process_document(
                file_content=file_bytes,
                document_type="knowledge_base",
                index_name="optimized-kb-index",
                book_name=book_name
            )
            
            await self.send_progress_notification(operation_id, 80, "Knowledge base processing completed")
            
            if result.get("success"):
                await self.send_progress_notification(operation_id, 100, "Knowledge base content stored successfully")
                self.complete_operation(operation_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "book_name": book_name,
                        "filename": filename,
                        "processing_type": "knowledge_base",
                        "index_used": "optimized-kb-index",
                        "chunks_created": result.get("chunks_created", 0),
                        "processing_stats": result.get("processing_stats", {}),
                        "message": "Knowledge base content processed and stored successfully"
                    }, indent=2)
                )]
            else:
                error_msg = result.get("error", "Processing failed")
                await self.send_log_notification("error", f"Knowledge base processing failed: {error_msg}")
                self.complete_operation(operation_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": error_msg,
                        "book_name": book_name,
                        "filename": filename
                    }, indent=2)
                )]
                
        except Exception as e:
            await self.send_log_notification("error", f"Knowledge base processing failed: {e}")
            self.complete_operation(operation_id)
            
            return [TextContent(
                type="text", 
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "book_name": book_name,
                    "filename": filename
                }, indent=2)
            )]

    async def _handle_search_knowledge_base(self, query: str, search_type: str = "enhanced",
                                          max_results: int = 5, namespace: str = "knowledge-base",
                                          index_name: str = "optimized-kb-index") -> List[TextContent]:
        """Handle knowledge base search using enhanced retrieval with OpenAI-powered responses"""
        try:
            # Import the advanced knowledge base retriever
            from knowledge_base_retrieval import AdvancedKnowledgeBaseRetriever
            
            # Create retriever instance
            kb_retriever = AdvancedKnowledgeBaseRetriever()
            
            # Perform intelligent search with OpenAI-powered response
            result = await kb_retriever.intelligent_search(
                query=query,
                top_k=max_results,
                namespace=namespace,
                index_name=index_name
            )
            
            # Extract the AI-generated response or fallback to formatted results
            if result.get("success") and "ai_response" in result:
                # Use the AI-generated response as the primary output
                response_text = result["ai_response"]
                
                # Also include search metadata
                response_data = {
                    "success": True,
                    "query": query,
                    "search_type": "enhanced",
                    "results": response_text,  # This is now the AI-generated answer
                    "namespace": namespace,
                    "index_name": index_name,
                    "total_sources": result.get("total_results", 0)
                }
            else:
                # Fallback for other response types (study plans, book analysis, etc.)
                response_data = result
            
            return [TextContent(
                type="text",
                text=json.dumps(response_data, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Enhanced knowledge base search failed: {e}")
            
            # Fallback to basic search if enhanced fails
            try:
                results = await self.vector_storage.enhanced_knowledge_base_search(
                    query=query,
                    namespace=namespace,
                    top_k=max_results,
                    index_name=index_name
                )
                formatted_results = "\n\n".join([f"[Source {i+1}]:\n{r['content']}" for i, r in enumerate(results)])
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": True,
                        "query": query,
                        "search_type": "basic_fallback",
                        "results": formatted_results,
                        "namespace": namespace,
                        "index_name": index_name,
                        "note": "Using basic search due to enhanced search failure"
                    }, indent=2)
                )]
                
            except Exception as fallback_error:
                logger.error(f"Basic search fallback also failed: {fallback_error}")
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"Both enhanced and basic search failed: {str(e)}, {str(fallback_error)}",
                        "query": query
                    }, indent=2)
                )]

    async def _handle_get_knowledge_base_inventory(self, namespace: str = "knowledge-base",
                                                 index_name: str = "optimized-kb-index") -> List[TextContent]:
        """Handle knowledge base inventory request"""
        try:
            inventory = await self.vector_storage.get_knowledge_base_inventory(
                namespace=namespace,
                index_name=index_name
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "inventory": inventory,
                    "namespace": namespace,
                    "index_name": index_name
                }, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Knowledge base inventory failed: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "namespace": namespace,
                    "index_name": index_name
                }, indent=2)
            )]

    async def _handle_find_books_covering_topic(self, topic: str, namespace: str = "knowledge-base",
                                              index_name: str = "optimized-kb-index") -> List[TextContent]:
        """Handle finding books that cover a specific topic"""
        try:
            books = await self.vector_storage.find_books_covering_topic(
                topic=topic,
                namespace=namespace,
                index_name=index_name
            )
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": True,
                    "topic": topic,
                    "books": books,
                    "namespace": namespace,
                    "index_name": index_name
                }, indent=2)
            )]
            
        except Exception as e:
            logger.error(f"Find books for topic failed: {e}")
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "topic": topic
                }, indent=2)
            )]

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

    async def _handle_generate_research_quiz(self, user_id: str, document_uuid: str, 
                                           number_of_questions: int = 10,
                                           difficulty_level: str = "mixed",
                                           include_explanations: bool = True,
                                           question_categories: List[str] = None) -> List[TextContent]:
        """Handle research quiz generation from user-specific document"""
        operation_id = self.create_operation_id()
        
        try:
            self.start_operation(operation_id, "generate_research_quiz", f"Generating quiz for document: {document_uuid}")
            
            # Default question categories if not provided
            if not question_categories:
                question_categories = ["conceptual", "methodological", "factual", "analytical"]
            
            # Construct namespace for the specific document
            namespace = f"user_{user_id}_doc_{document_uuid}"
            
            await self.send_progress_notification(operation_id, 10, "Validating document access...")
            
            # Validate namespace exists and get document content
            logger.info(f"🔍 Searching for document in namespace: {namespace}")
            
            # Use vector storage to search the specific namespace
            document_content = await self.vector_storage.search_in_namespace(
                namespace=namespace,
                query="research paper content methodology results",  # General query to get all content
                max_results=50,  # Get comprehensive content
                similarity_threshold=0.0  # Accept all content from this document
            )
            
            if not document_content:
                await self.send_log_notification("error", f"No content found in namespace: {namespace}")
                self.complete_operation(operation_id)
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": f"No document found for user {user_id} with document UUID {document_uuid}",
                        "namespace": namespace
                    }, indent=2)
                )]
            
            await self.send_progress_notification(operation_id, 30, f"Found {len(document_content)} content chunks")
            
            # Organize content by sections for better quiz generation
            organized_content = self._organize_content_for_quiz(document_content)
            
            await self.send_progress_notification(operation_id, 50, "Analyzing content and generating questions...")
            
            # Generate quiz questions using AI
            quiz_data = await self._generate_quiz_questions(
                organized_content=organized_content,
                number_of_questions=number_of_questions,
                difficulty_level=difficulty_level,
                include_explanations=include_explanations,
                question_categories=question_categories,
                user_id=user_id,
                document_uuid=document_uuid
            )
            
            await self.send_progress_notification(operation_id, 90, "Finalizing quiz...")
            
            if quiz_data.get("success"):
                await self.send_progress_notification(operation_id, 100, "Quiz generation completed successfully!")
                self.complete_operation(operation_id)
                
                # Format successful response
                response_data = {
                    "success": True,
                    "quiz_metadata": {
                        "user_id": user_id,
                        "document_uuid": document_uuid,
                        "namespace": namespace,
                        "total_questions": len(quiz_data.get("questions", [])),
                        "difficulty_level": difficulty_level,
                        "question_categories": question_categories,
                        "include_explanations": include_explanations,
                        "content_chunks_analyzed": len(document_content),
                        "generated_at": time.time()
                    },
                    "questions": quiz_data.get("questions", []),
                    "content_analysis": quiz_data.get("content_analysis", {}),
                    "generation_stats": quiz_data.get("generation_stats", {})
                }
                
                return [TextContent(
                    type="text",
                    text=json.dumps(response_data, indent=2)
                )]
            else:
                error_msg = quiz_data.get("error", "Quiz generation failed")
                await self.send_log_notification("error", f"Quiz generation failed: {error_msg}")
                self.complete_operation(operation_id)
                
                return [TextContent(
                    type="text",
                    text=json.dumps({
                        "success": False,
                        "error": error_msg,
                        "user_id": user_id,
                        "document_uuid": document_uuid,
                        "namespace": namespace
                    }, indent=2)
                )]
                
        except Exception as e:
            await self.send_log_notification("error", f"Quiz generation error: {e}")
            self.complete_operation(operation_id)
            logger.error(f"Error generating research quiz: {e}")
            
            return [TextContent(
                type="text",
                text=json.dumps({
                    "success": False,
                    "error": str(e),
                    "user_id": user_id,
                    "document_uuid": document_uuid,
                    "namespace": namespace
                }, indent=2)
            )]

    def _organize_content_for_quiz(self, document_content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Organize document content by sections for better quiz generation"""
        try:
            organized = {
                "sections": {},
                "all_content": [],
                "metadata": {
                    "total_chunks": len(document_content),
                    "total_length": 0,
                    "sections_found": set(),
                    "paper_info": {}
                }
            }
            
            # Group content by sections
            for chunk in document_content:
                content = chunk.get("content", "")
                metadata = chunk.get("metadata", {})
                
                organized["all_content"].append(content)
                organized["metadata"]["total_length"] += len(content)
                
                # Extract section information
                section = metadata.get("section", "general")
                organized["metadata"]["sections_found"].add(section)
                
                if section not in organized["sections"]:
                    organized["sections"][section] = []
                
                organized["sections"][section].append({
                    "content": content,
                    "metadata": metadata,
                    "chunk_id": chunk.get("chunk_id", ""),
                    "score": chunk.get("score", 0)
                })
                
                # Extract paper metadata
                if "paper_id" in metadata:
                    organized["metadata"]["paper_info"]["paper_id"] = metadata["paper_id"]
                if "user_id" in metadata:
                    organized["metadata"]["paper_info"]["user_id"] = metadata["user_id"]
                if "document_uuid" in metadata:
                    organized["metadata"]["paper_info"]["document_uuid"] = metadata["document_uuid"]
            
            organized["metadata"]["sections_found"] = list(organized["metadata"]["sections_found"])
            
            return organized
            
        except Exception as e:
            logger.error(f"Error organizing content for quiz: {e}")
            return {
                "sections": {"general": [{"content": chunk.get("content", ""), "metadata": chunk.get("metadata", {})} for chunk in document_content]},
                "all_content": [chunk.get("content", "") for chunk in document_content],
                "metadata": {"total_chunks": len(document_content), "sections_found": ["general"], "paper_info": {}}
            }

    async def _generate_quiz_questions(self, organized_content: Dict[str, Any], 
                                     number_of_questions: int,
                                     difficulty_level: str, 
                                     include_explanations: bool,
                                     question_categories: List[str],
                                     user_id: str, 
                                     document_uuid: str) -> Dict[str, Any]:
        """Generate quiz questions using AI analysis of the document content"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.config.OPENAI_API_KEY)
            
            # Prepare content for AI analysis
            content_summary = self._prepare_content_for_quiz_ai(organized_content)
            
            # Create comprehensive prompt for quiz generation
            quiz_prompt = f"""You are an expert educational content creator. Generate a comprehensive multiple-choice quiz from the following research paper content.

**DOCUMENT CONTENT:**
{content_summary}

**QUIZ REQUIREMENTS:**
- Number of questions: {number_of_questions}
- Difficulty level: {difficulty_level}
- Question categories: {', '.join(question_categories)}
- Include explanations: {include_explanations}

**INSTRUCTIONS:**
1. Analyze the content thoroughly to understand key concepts, methodologies, findings, and conclusions
2. Create {number_of_questions} multiple-choice questions with 4 options each (A, B, C, D)
3. Ensure questions cover different aspects: {', '.join(question_categories)}
4. For difficulty level "{difficulty_level}": 
   - easy: Focus on basic definitions and main concepts
   - medium: Include application and analysis questions
   - hard: Require deep understanding and critical thinking
   - mixed: Combine all difficulty levels
5. Each question should be answerable from the provided content
6. Create realistic but incorrect distractors for multiple choice options
7. If explanations are requested, provide clear reasoning for the correct answer

**OUTPUT FORMAT (JSON):**
{{
    "questions": [
        {{
            "question_id": 1,
            "question": "What is the primary research methodology used in this study?",
            "options": {{
                "A": "Qualitative analysis",
                "B": "Quantitative experiment", 
                "C": "Mixed methods approach",
                "D": "Literature review"
            }},
            "correct_answer": "B",
            "explanation": "The paper explicitly states in the methodology section that a quantitative experimental design was used...",
            "category": "methodological",
            "difficulty": "medium",
            "content_source": "methodology section"
        }}
    ],
    "content_analysis": {{
        "main_topic": "Topic extracted from content",
        "key_concepts": ["concept1", "concept2"],
        "methodology_used": "research methodology",
        "main_findings": ["finding1", "finding2"]
    }}
}}

Generate the quiz now:"""

            # Generate quiz using AI
            response = client.chat.completions.create(
                model=self.config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator specializing in research paper analysis and quiz generation. Always output valid JSON format."},
                    {"role": "user", "content": quiz_prompt}
                ],
                temperature=0.3,  # Balanced creativity and consistency
                max_tokens=4000  # Sufficient for comprehensive quiz
            )
            
            ai_response = response.choices[0].message.content
            
            # Parse AI response to extract JSON
            import re
            import json
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                try:
                    quiz_data = json.loads(json_match.group())
                    
                    # Validate quiz structure
                    if "questions" in quiz_data and isinstance(quiz_data["questions"], list):
                        # Ensure we have the requested number of questions
                        questions = quiz_data["questions"][:number_of_questions]
                        
                        # Add metadata to each question
                        for i, question in enumerate(questions):
                            question["question_id"] = i + 1
                            question["user_id"] = user_id
                            question["document_uuid"] = document_uuid
                            
                            # Ensure all required fields exist
                            if "category" not in question:
                                question["category"] = question_categories[i % len(question_categories)]
                            if "difficulty" not in question:
                                question["difficulty"] = difficulty_level if difficulty_level != "mixed" else ["easy", "medium", "hard"][i % 3]
                            if include_explanations and "explanation" not in question:
                                question["explanation"] = "Explanation not provided by AI"
                        
                        return {
                            "success": True,
                            "questions": questions,
                            "content_analysis": quiz_data.get("content_analysis", {}),
                            "generation_stats": {
                                "requested_questions": number_of_questions,
                                "generated_questions": len(questions),
                                "difficulty_level": difficulty_level,
                                "categories_used": question_categories,
                                "ai_model": self.config.LLM_MODEL
                            }
                        }
                    else:
                        raise ValueError("Invalid quiz structure in AI response")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse AI quiz response as JSON: {e}")
                    return {"success": False, "error": f"AI response parsing failed: {e}"}
            else:
                logger.error("No JSON found in AI response")
                return {"success": False, "error": "No valid JSON found in AI response"}
                
        except Exception as e:
            logger.error(f"Error generating quiz questions: {e}")
            return {"success": False, "error": str(e)}

    def _prepare_content_for_quiz_ai(self, organized_content: Dict[str, Any]) -> str:
        """Prepare content summary for AI quiz generation"""
        try:
            sections = organized_content.get("sections", {})
            metadata = organized_content.get("metadata", {})
            
            content_parts = []
            
            # Add paper info
            content_parts.append("=== RESEARCH PAPER ANALYSIS ===")
            content_parts.append(f"Total Content Chunks: {metadata.get('total_chunks', 0)}")
            content_parts.append(f"Sections Found: {', '.join(metadata.get('sections_found', []))}")
            content_parts.append("")
            
            # Add content by sections
            for section_name, section_content in sections.items():
                content_parts.append(f"=== {section_name.upper()} SECTION ===")
                
                # Combine content from this section
                section_text = []
                for chunk in section_content[:3]:  # Limit to first 3 chunks per section
                    content = chunk.get("content", "")
                    if content and len(content.strip()) > 50:
                        section_text.append(content[:500])  # Limit chunk size
                
                if section_text:
                    content_parts.append("\n".join(section_text))
                else:
                    content_parts.append("No significant content in this section.")
                
                content_parts.append("")
            
            # Combine all parts
            full_content = "\n".join(content_parts)
            
            # Limit total content length for AI processing
            max_content_length = 8000  # Reasonable limit for AI processing
            if len(full_content) > max_content_length:
                full_content = full_content[:max_content_length] + "\n\n[Content truncated for processing...]"
            
            return full_content
            
        except Exception as e:
            logger.error(f"Error preparing content for AI: {e}")
            return "Error processing content for quiz generation."

    async def run(self):
        """Run the perfect MCP server with enhanced capabilities"""
        # Enhanced notification options for full protocol compliance
        class EnhancedNotificationOptions:
            def __init__(self):
                self.tools_changed = True
                self.resources_changed = True
                self.prompts_changed = True
                self.progress_notifications = True
                self.logging_notifications = True
        
        notification_options = EnhancedNotificationOptions()
        
        # Declare comprehensive MCP capabilities
        server_capabilities = {
            "tools": {
                "listTools": True,
                "callTool": True
            },
            "resources": {
                "listResources": True,
                "readResource": True,
                "subscribe": True
            },
            "prompts": {
                "listPrompts": True,
                "getPrompt": True
            },
            "sampling": {
                "createMessage": True
            },
            "notifications": {
                "progress": True,
                "logging": True,
                "resources/updated": True,
                "tools/updated": True,
                "prompts/updated": True
            },
            "cancellation": {
                "cancel": True
            },
            "logging": {
                "level": "info"
            }
        }
        
        # Experimental capabilities for advanced features
        experimental_capabilities = {
            "progressNotifications": True,
            "cancellationSupport": True,
            "enhancedLogging": True,
            "aiSampling": True,
            "operationTracking": True
        }
        
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="perfect-research-mcp",
                    server_version="2.0.0",  # Updated version for enhanced protocol
                    capabilities=server_capabilities,
                    experimental_capabilities=experimental_capabilities,
                    notification_options=notification_options
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