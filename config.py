import os
from dotenv import load_dotenv
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

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
        # AWS S3 CONFIGURATION
        # ============================================================================
        self.AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
        self.S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "knoiprocessedimagesbucket")
        self.S3_BUCKET_REGION = os.getenv("S3_BUCKET_REGION", "ap-south-1")
        
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
        self.DEFAULT_TIMEOUT = int(os.getenv('DEFAULT_TIMEOUT', '30'))
        
        # Create necessary directories
        self._create_directories()

        # Add compatibility attributes for HybridRetriever
        self.openai_api_key = self.OPENAI_API_KEY
        self.pinecone_api_key = self.PINECONE_API_KEY
        self.index_name = self.PINECONE_KB_INDEX_NAME
        self.embedding_model = self.EMBEDDING_MODEL
        self.embedding_dimension = self.EMBEDDING_DIMENSIONS
        self.namespace = "knowledge-base"  # Default namespace for knowledge base
        self.cross_encoder_reranking = True  # Enable cross-encoder reranking
        self.hybrid_search = True  # Enable hybrid search
        self.response_model = self.LLM_MODEL
        self.max_response_tokens = 1500  # Maximum tokens for response generation
        self.temperature = self.LLM_TEMPERATURE  # Temperature for response generation


    
    
    # Local storage functions deprecated - keeping for compatibility
    def get_pdf_image_dir(self, pdf_name: str) -> Path:
        """Deprecated - kept for backward compatibility. Use S3 storage instead."""
        import tempfile
        return Path(tempfile.mkdtemp())  # Return temp directory for legacy code
    
    def get_pdf_s3_prefix(self, pdf_name: str) -> str:
        """Get S3 prefix for PDF images (replaces local directory for S3 storage)."""
        # Clean PDF name for S3 prefix (remove .pdf extension and clean special chars)
        clean_name = Path(pdf_name).stem  # Remove extension
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_').replace('-', '_')
        
        return f"images/{clean_name}/"
    
    # Table directories removed - tables stored only in Pinecone
    
    def generate_pdf_id_from_name(pdf_name: str) -> str:
        """Generate PDF ID based on filename for reuse detection."""
        clean_name = Path(pdf_name).stem  # Remove extension
        clean_name = "".join(c for c in clean_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
        clean_name = clean_name.replace(' ', '_').replace('-', '_').lower()
        return clean_name  # Just return the clean name, no "pdf_" prefix
    
    def get_s3_url(self, s3_key: str) -> str:
        """Generate full public S3 URL from S3 key."""
        if not self.S3_BUCKET_NAME or not self.AWS_REGION:
            raise ValueError("S3_BUCKET_NAME and AWS_REGION must be configured")
        return f"https://{self.S3_BUCKET_NAME}.s3.{self.AWS_REGION}.amazonaws.com/{s3_key}"
    
    def get_s3_key_from_url(self, s3_url: str) -> str:
        """Extract S3 key from full S3 URL."""
        # Extract key from URL like: https://bucket.s3.region.amazonaws.com/key
        parts = s3_url.split(f"{self.S3_BUCKET_NAME}.s3.{self.AWS_REGION}.amazonaws.com/")
        return parts[1] if len(parts) > 1 else ""
    
    def check_pdf_already_processed(self, pdf_name: str) -> bool:
        """Check if PDF has been processed (DEPRECATED - use check_pdf_already_processed_s3)."""
        # Deprecated function - always return False to force S3 check
        return False
    
    def check_pdf_already_processed_s3(self, pdf_name: str) -> bool:
        """Check if PDF with this name has already been processed in S3."""
        try:
            from s3_handler import S3Handler
            import logging
            
            logger = logging.getLogger(__name__)
            s3_handler = S3Handler()
            
            # Check if any images exist for this PDF in S3
            s3_prefix = self.get_pdf_s3_prefix(pdf_name)
            images = s3_handler.list_objects(s3_prefix)
            
            return len(images) > 0
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Error checking S3 for processed PDF: {e}")
            return False   

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
        
        # Check S3 configuration
        if not self.AWS_ACCESS_KEY_ID or not self.AWS_SECRET_ACCESS_KEY:
            issues.append("WARNING: AWS credentials not set - S3 storage will not be available")
        else:
            if not self.S3_BUCKET_NAME:
                issues.append("WARNING: S3_BUCKET_NAME not set - using default bucket")
        
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
    
    def get_s3_config(self) -> Dict[str, Any]:
        """Get S3 configuration"""
        return {
            "aws_access_key_id": self.AWS_ACCESS_KEY_ID,
            "aws_secret_access_key": self.AWS_SECRET_ACCESS_KEY,
            "aws_region": self.AWS_REGION,
            "s3_bucket_name": self.S3_BUCKET_NAME,
            "s3_bucket_region": self.S3_BUCKET_REGION
        }

# Backward compatibility
Config = AdvancedConfig 

# Create a global config instance
config = AdvancedConfig()

# Export AWS credentials for backward compatibility
AWS_ACCESS_KEY_ID = config.AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY = config.AWS_SECRET_ACCESS_KEY
AWS_REGION = config.AWS_REGION
S3_BUCKET_NAME = config.S3_BUCKET_NAME 