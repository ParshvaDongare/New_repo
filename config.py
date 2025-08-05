"""
Configuration module for HackRX Query System.
Handles environment variables and application settings for high-performance RAG.
"""

import os
from typing import Optional, List, Dict, Any
from functools import lru_cache
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # API Keys - Multiple keys for parallel processing
    # Groq API Keys (5 keys for LLM parallelism)
    groq_api_key_1: Optional[str] = Field(None, env='GROQ_API_KEYS_1')
    groq_api_key_2: Optional[str] = Field(None, env='GROQ_API_KEYS_2')
    groq_api_key_3: Optional[str] = Field(None, env='GROQ_API_KEYS_3')
    groq_api_key_4: Optional[str] = Field(None, env='GROQ_API_KEYS_4')
    groq_api_key_5: Optional[str] = Field(None, env='GROQ_API_KEYS_5')
    
    # Jina API Keys (6 keys for embedding parallelism)
    jina_api_key_1: Optional[str] = Field(None, env='JINA_API_KEY_1')
    jina_api_key_2: Optional[str] = Field(None, env='JINA_API_KEY_2')
    jina_api_key_3: Optional[str] = Field(None, env='JINA_API_KEY_3')
    jina_api_key_4: Optional[str] = Field(None, env='JINA_API_KEY_4')
    jina_api_key_5: Optional[str] = Field(None, env='JINA_API_KEY_5')
    jina_api_key_6: Optional[str] = Field(None, env='JINA_API_KEY_6')
    
    # Legacy single Jina key support (fallback)
    jina_api_key: Optional[str] = Field(None, env='JINA_API_KEY')
    
    # Pinecone
    pinecone_api_key: str = Field(..., env='PINECONE_API_KEY')
    pinecone_environment: str = Field(default='gcp-starter', env='PINECONE_ENVIRONMENT')
    
    # Database (optional for local development)
    database_url: Optional[str] = Field(None, env='DATABASE_URL')
    database_pool_size: int = Field(default=5, env='DATABASE_POOL_SIZE')
    database_max_overflow: int = Field(default=10, env='DATABASE_MAX_OVERFLOW')
    
    # Authentication
    bearer_token: Optional[str] = Field(None, env='BEARER_TOKEN')
    
    # Application Settings
    app_name: str = Field(default='HackRX Query System')
    api_version: str = Field(default='v1')
    debug: bool = Field(default=False, env='DEBUG')
    clean_logs: bool = Field(default=True, env='CLEAN_LOGS')
    log_level: str = Field(default='INFO', env='LOG_LEVEL')
    
    # Model Configuration
    llm_model: str = Field(default='llama3-8b-8192', env='LLM_MODEL')
    embedding_model: str = Field(default='jina-embeddings-v2-base-en', env='EMBEDDING_MODEL')
    
    # On-Demand Processing Settings
    enable_on_demand: bool = Field(default=True, env='ENABLE_ON_DEMAND')
    min_pages_per_batch: int = Field(default=5, env='MIN_PAGES_PER_BATCH')
    max_pages_per_batch: int = Field(default=50, env='MAX_PAGES_PER_BATCH')
    page_processing_timeout: int = Field(default=30, env='PAGE_PROCESSING_TIMEOUT')
    
    # Performance Settings
    max_chunk_size: int = Field(default=1500, env='MAX_CHUNK_SIZE')  # Optimized for accuracy
    chunk_overlap: int = Field(default=200, env='CHUNK_OVERLAP')  # Increased for better context
    min_chunk_size: int = Field(default=300, env='MIN_CHUNK_SIZE')
    semantic_chunking_threshold: float = Field(default=0.7, env='SEMANTIC_CHUNKING_THRESHOLD')
    
    # Retrieval Settings
    top_k_results: int = Field(default=10, env='TOP_K_RESULTS')  # More candidates for reranking
    rerank_top_k: int = Field(default=5, env='RERANK_TOP_K')  # Final results after reranking
    similarity_threshold: float = Field(default=0.6, env='SIMILARITY_THRESHOLD')
    enable_hybrid_search: bool = Field(default=True, env='ENABLE_HYBRID_SEARCH')
    
    # LLM Settings
    max_tokens_per_response: int = Field(default=500, env='MAX_TOKENS_PER_RESPONSE')
    llm_temperature: float = Field(default=0.1, env='LLM_TEMPERATURE')  # Low for accuracy
    llm_timeout: int = Field(default=30, env='LLM_TIMEOUT')
    enable_cot_prompting: bool = Field(default=True, env='ENABLE_COT_PROMPTING')
    
    # Rate Limiting (per key)
    groq_rpm_limit: int = Field(default=30, env='GROQ_RPM_LIMIT')  # Per key
    groq_concurrent_requests: int = Field(default=3, env='GROQ_CONCURRENT_REQUESTS')
    jina_rpm_limit: int = Field(default=100, env='JINA_RPM_LIMIT')  # Per key
    jina_concurrent_requests: int = Field(default=5, env='JINA_CONCURRENT_REQUESTS')
    pinecone_rpm_limit: int = Field(default=100, env='PINECONE_RPM_LIMIT')
    
    # Pinecone Settings
    pinecone_index_name: str = Field(default='hackrx-docs', env='PINECONE_INDEX_NAME')
    pinecone_dimension: int = Field(default=768, env='PINECONE_DIMENSION')
    pinecone_metric: str = Field(default='cosine', env='PINECONE_METRIC')
    pinecone_batch_size: int = Field(default=100, env='PINECONE_BATCH_SIZE')
    
    # Document Processing
    max_document_size_mb: int = Field(default=100, env='MAX_DOCUMENT_SIZE_MB')
    pdf_extraction_timeout: int = Field(default=60, env='PDF_EXTRACTION_TIMEOUT')
    enable_ocr: bool = Field(default=False, env='ENABLE_OCR')  # Disabled for speed
    # New ingestion toggles
    use_llm_planner: bool = Field(default=False, env='USE_LLM_PLANNER')  # Disable page planner by default
    extraction_workers: int = Field(default_factory=lambda: os.cpu_count() or 4, env='EXTRACTION_WORKERS')
    
    # Caching Configuration
    enable_query_cache: bool = Field(default=True, env='ENABLE_QUERY_CACHE')
    enable_embedding_cache: bool = Field(default=True, env='ENABLE_EMBEDDING_CACHE')
    enable_document_cache: bool = Field(default=True, env='ENABLE_DOCUMENT_CACHE')
    enable_llm_cache: bool = Field(default=True, env='ENABLE_LLM_CACHE')
    
    # Cache TTLs (in seconds)
    query_cache_ttl: int = Field(default=7200, env='QUERY_CACHE_TTL')  # 2 hours
    embedding_cache_ttl: int = Field(default=86400, env='EMBEDDING_CACHE_TTL')  # 24 hours
    document_cache_ttl: int = Field(default=3600, env='DOCUMENT_CACHE_TTL')  # 1 hour
    llm_cache_ttl: int = Field(default=3600, env='LLM_CACHE_TTL')  # 1 hour
    
    # Memory Management
    max_memory_mb: int = Field(default=400, env='MAX_MEMORY_MB')  # Leave headroom from 512MB
    embedding_cache_max_size: int = Field(default=1000, env='EMBEDDING_CACHE_MAX_SIZE')
    query_cache_max_size: int = Field(default=500, env='QUERY_CACHE_MAX_SIZE')
    enable_memory_monitoring: bool = Field(default=True, env='ENABLE_MEMORY_MONITORING')
    
    # Background Task Settings
    enable_background_tasks: bool = Field(default=True, env='ENABLE_BACKGROUND_TASKS')
    task_queue_max_size: int = Field(default=100, env='TASK_QUEUE_MAX_SIZE')
    task_worker_threads: int = Field(default=2, env='TASK_WORKER_THREADS')
    
    # Domain-Specific Settings
    enable_domain_adaptation: bool = Field(default=True, env='ENABLE_DOMAIN_ADAPTATION')
    detect_document_domain: bool = Field(default=True, env='DETECT_DOCUMENT_DOMAIN')
    
    @property
    def groq_api_keys(self) -> List[str]:
        """Get list of Groq API keys for parallel processing."""
        keys = []
        
        # Check numbered keys
        for i in range(1, 6):
            key = getattr(self, f'groq_api_key_{i}', None) or os.getenv(f'GROQ_API_KEYS_{i}')
            if key and key.strip():
                keys.append(key.strip())
        
        if not keys:
            raise ValueError("No Groq API keys found. Please set GROQ_API_KEYS_1 through GROQ_API_KEYS_5")
        
        return keys
    
    @property
    def jina_api_keys(self) -> List[str]:
        """Get list of Jina API keys for parallel embedding generation."""
        keys = []
        
        # Check numbered keys first
        for i in range(1, 7):
            key = getattr(self, f'jina_api_key_{i}', None) or os.getenv(f'JINA_API_KEY_{i}')
            if key and key.strip():
                keys.append(key.strip())
        
        # Fallback to single key if no numbered keys found
        if not keys and self.jina_api_key:
            logger.warning("Using single Jina API key. For optimal performance, configure 6 keys.")
            keys = [self.jina_api_key]
        
        if not keys:
            raise ValueError("No Jina API keys found. Please set JINA_API_KEY_1 through JINA_API_KEY_6")
        
        return keys
    
    @property
    def domain_prompts(self) -> Dict[str, Any]:
        """Domain-specific prompt configurations."""
        return {
            "insurance": {
                "chunk_size": 1200,
                "focus_keywords": ["policy", "coverage", "claim", "premium", "deductible"],
                "prompt_style": "precise_technical"
            },
            "legal": {
                "chunk_size": 1500,
                "focus_keywords": ["article", "section", "clause", "provision", "statute"],
                "prompt_style": "formal_structured"
            },
            "scientific": {
                "chunk_size": 1800,
                "focus_keywords": ["hypothesis", "methodology", "results", "conclusion", "abstract"],
                "prompt_style": "analytical_detailed"
            },
            "general": {
                "chunk_size": self.max_chunk_size,
                "focus_keywords": [],
                "prompt_style": "conversational"
            }
        }
    
    @validator('max_memory_mb')
    def validate_memory_limit(cls, v):
        """Ensure memory limit is reasonable for Railway free tier."""
        if v > 450:
            logger.warning(f"Memory limit {v}MB is close to Railway's 512MB limit. Setting to 400MB.")
            return 400
        return v
    
    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'extra': 'allow',
        'env_file_encoding': 'utf-8'
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export commonly used settings
settings = get_settings()

# Validate critical settings on module load
try:
    _ = settings.groq_api_keys
    _ = settings.jina_api_keys
    logger.info(f"Loaded {len(settings.groq_api_keys)} Groq keys and {len(settings.jina_api_keys)} Jina keys")
except Exception as e:
    logger.error(f"Critical configuration error: {e}")
    raise