import os
from dotenv import load_dotenv
from typing import Dict, List, Any

load_dotenv()

class AdvancedConfig:
    """Advanced configuration for perfect research PPT system"""
    
    def __init__(self):
        # ============================================================================
        # REQUIRED API KEYS
        # ============================================================================
        self.OPENAI_API_KEY = self._get_env_var("OPENAI_API_KEY")
        self.SERPAPI_KEY = self._get_env_var("SERPAPI_KEY")
        self.PINECONE_API_KEY = self._get_env_var("PINECONE_API_KEY")
        
        # ============================================================================
        # OPTIONAL API KEYS
        # ============================================================================
        self.LLAMA_PARSE_API_KEY = os.getenv("LLAMA_PARSE_API_KEY")
        self.UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY")
        self.PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
        
        # ============================================================================
        # AI MODEL SETTINGS
        # ============================================================================
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.0))
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.EMBEDDING_DIMENSIONS = 3072
        
        # ============================================================================
        # VECTOR STORAGE SETTINGS
        # ============================================================================
        self.PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")  # For research papers
        self.PINECONE_KB_INDEX_NAME = os.getenv("PINECONE_KB_INDEX_NAME", "optimized-kb-index")  # For knowledge base
        self.PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
        self.VECTOR_SIMILARITY_THRESHOLD = float(os.getenv("VECTOR_SIMILARITY_THRESHOLD", 0.7))
        self.MAX_RETRIEVAL_RESULTS = int(os.getenv("MAX_RETRIEVAL_RESULTS", 20))
        
        # ============================================================================
        # DOCUMENT PROCESSING SETTINGS
        # ============================================================================
        self.CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 300))
        self.MIN_CHUNK_SIZE = int(os.getenv("MIN_CHUNK_SIZE", 100))
        self.MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", 3000))
        
        # ============================================================================
        # RESEARCH ANALYSIS SETTINGS
        # ============================================================================
        self.ENABLE_RESEARCH_INTELLIGENCE = os.getenv("ENABLE_RESEARCH_INTELLIGENCE", "true").lower() == "true"
        self.ENABLE_CITATION_ANALYSIS = os.getenv("ENABLE_CITATION_ANALYSIS", "true").lower() == "true"
        self.ENABLE_STATISTICAL_EXTRACTION = os.getenv("ENABLE_STATISTICAL_EXTRACTION", "true").lower() == "true"
        self.ENABLE_METHODOLOGY_ANALYSIS = os.getenv("ENABLE_METHODOLOGY_ANALYSIS", "true").lower() == "true"
        
        # ============================================================================
        # PRESENTATION GENERATION SETTINGS
        # ============================================================================
        self.PPT_OUTPUT_DIR = os.getenv("PPT_OUTPUT_DIR", "presentations")
        self.PPT_MAX_SLIDES = int(os.getenv("PPT_MAX_SLIDES", 20))
        self.PPT_MIN_SLIDES = int(os.getenv("PPT_MIN_SLIDES", 5))
        self.ENABLE_ACADEMIC_FORMATTING = os.getenv("ENABLE_ACADEMIC_FORMATTING", "true").lower() == "true"
        self.ENABLE_AUTO_CITATIONS = os.getenv("ENABLE_AUTO_CITATIONS", "true").lower() == "true"
        
        # ============================================================================
        # SEARCH SETTINGS
        # ============================================================================
        self.MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 15))
        self.SEARCH_LOCATION = os.getenv("SEARCH_LOCATION", "United States")
        self.ENABLE_ACADEMIC_SEARCH = os.getenv("ENABLE_ACADEMIC_SEARCH", "true").lower() == "true"
        
        # ============================================================================
        # ADVANCED FEATURES
        # ============================================================================
        self.ENABLE_MULTI_MODAL_EXTRACTION = os.getenv("ENABLE_MULTI_MODAL_EXTRACTION", "true").lower() == "true"
        self.ENABLE_FIGURE_ANALYSIS = os.getenv("ENABLE_FIGURE_ANALYSIS", "false").lower() == "true"
        self.ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"
        self.ENABLE_EQUATION_PROCESSING = os.getenv("ENABLE_EQUATION_PROCESSING", "false").lower() == "true"
        
        # ============================================================================
        # QUALITY ASSURANCE SETTINGS
        # ============================================================================
        self.ENABLE_FACT_CHECKING = os.getenv("ENABLE_FACT_CHECKING", "true").lower() == "true"
        self.ENABLE_CITATION_VALIDATION = os.getenv("ENABLE_CITATION_VALIDATION", "true").lower() == "true"
        self.ENABLE_CONTENT_COHERENCE_CHECK = os.getenv("ENABLE_CONTENT_COHERENCE_CHECK", "true").lower() == "true"
        
        # ============================================================================
        # PERFORMANCE SETTINGS
        # ============================================================================
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 5))
        self.REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))
        self.ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"
        self.CACHE_EXPIRY_HOURS = int(os.getenv("CACHE_EXPIRY_HOURS", 24))
        
        # ============================================================================
        # LOGGING SETTINGS
        # ============================================================================
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_DIR = os.getenv("LOG_DIR", "logs")
        self.ENABLE_DETAILED_LOGGING = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"
        
        # Create necessary directories
        self._create_directories()

    def _get_env_var(self, var_name: str) -> str:
        """Get required environment variable"""
        value = os.getenv(var_name)
        if not value:
            raise ValueError(f"Missing required environment variable: {var_name}")
        return value

    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.PPT_OUTPUT_DIR,
            self.LOG_DIR,
            "cache",
            "temp",
            "exports"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def get_research_analysis_config(self) -> Dict[str, Any]:
        """Get research analysis configuration"""
        return {
            "enable_research_intelligence": self.ENABLE_RESEARCH_INTELLIGENCE,
            "enable_citation_analysis": self.ENABLE_CITATION_ANALYSIS,
            "enable_statistical_extraction": self.ENABLE_STATISTICAL_EXTRACTION,
            "enable_methodology_analysis": self.ENABLE_METHODOLOGY_ANALYSIS,
            "similarity_threshold": self.VECTOR_SIMILARITY_THRESHOLD,
            "max_retrieval_results": self.MAX_RETRIEVAL_RESULTS
        }

    def get_presentation_config(self) -> Dict[str, Any]:
        """Get presentation generation configuration"""
        return {
            "output_dir": self.PPT_OUTPUT_DIR,
            "max_slides": self.PPT_MAX_SLIDES,
            "min_slides": self.PPT_MIN_SLIDES,
            "enable_academic_formatting": self.ENABLE_ACADEMIC_FORMATTING,
            "enable_auto_citations": self.ENABLE_AUTO_CITATIONS,
            "enable_multi_modal": self.ENABLE_MULTI_MODAL_EXTRACTION
        }

    def get_vector_config(self) -> Dict[str, Any]:
        """Get vector storage configuration"""
        return {
            "index_name": self.PINECONE_INDEX_NAME,
            "environment": self.PINECONE_ENVIRONMENT,
            "dimensions": self.EMBEDDING_DIMENSIONS,
            "similarity_threshold": self.VECTOR_SIMILARITY_THRESHOLD,
            "chunk_size": self.CHUNK_SIZE,
            "chunk_overlap": self.CHUNK_OVERLAP
        }

    def get_advanced_features_config(self) -> Dict[str, Any]:
        """Get advanced features configuration"""
        return {
            "multi_modal_extraction": self.ENABLE_MULTI_MODAL_EXTRACTION,
            "figure_analysis": self.ENABLE_FIGURE_ANALYSIS,
            "table_extraction": self.ENABLE_TABLE_EXTRACTION,
            "equation_processing": self.ENABLE_EQUATION_PROCESSING,
            "fact_checking": self.ENABLE_FACT_CHECKING,
            "citation_validation": self.ENABLE_CITATION_VALIDATION,
            "content_coherence": self.ENABLE_CONTENT_COHERENCE_CHECK
        }

    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check required API keys
        required_keys = ["OPENAI_API_KEY", "SERPAPI_KEY", "PINECONE_API_KEY"]
        for key in required_keys:
            try:
                getattr(self, key)
            except ValueError:
                issues.append(f"ERROR: {key} is required")
        
        # Check optional API keys
        if not self.LLAMA_PARSE_API_KEY:
            issues.append("WARNING: LLAMA_PARSE_API_KEY not set - using fallback PDF parsing")
        
        if not self.UNSPLASH_ACCESS_KEY:
            issues.append("WARNING: UNSPLASH_ACCESS_KEY not set - presentations will not include images")
        
        # Validate numeric ranges
        if not 0 <= self.LLM_TEMPERATURE <= 2:
            issues.append("WARNING: LLM_TEMPERATURE should be between 0 and 2")
        
        if not 100 <= self.CHUNK_SIZE <= 5000:
            issues.append("WARNING: CHUNK_SIZE should be between 100 and 5000")
        
        if not 0 <= self.VECTOR_SIMILARITY_THRESHOLD <= 1:
            issues.append("WARNING: VECTOR_SIMILARITY_THRESHOLD should be between 0 and 1")
        
        # Check directory permissions
        for directory in [self.PPT_OUTPUT_DIR, self.LOG_DIR]:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                except PermissionError:
                    issues.append(f"ERROR: Cannot create directory {directory} - permission denied")
        
        return issues

    def get_model_config(self) -> Dict[str, Any]:
        """Get AI model configuration"""
        return {
            "llm_model": self.LLM_MODEL,
            "llm_temperature": self.LLM_TEMPERATURE,
            "embedding_model": self.EMBEDDING_MODEL,
            "embedding_dimensions": self.EMBEDDING_DIMENSIONS,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "request_timeout": self.REQUEST_TIMEOUT
        }

# Backward compatibility
Config = AdvancedConfig 