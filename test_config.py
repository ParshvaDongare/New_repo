"""
Mock configuration for testing purposes.
This allows tests to run without requiring actual API keys.
"""

import os
from typing import List, Dict, Any, Optional
from unittest.mock import Mock


class MockSettings:
    """Mock settings class that mimics the real Settings class for testing."""
    
    def __init__(self):
        # Basic app settings
        self.app_name = "HackRX Query System"
        self.api_version = "v1"
        self.debug = True
        self.log_level = "INFO"
        
        # Mock API keys
        self._mock_groq_keys = ["mock_groq_key_1", "mock_groq_key_2", "mock_groq_key_3"]
        self._mock_jina_keys = ["mock_jina_key_1", "mock_jina_key_2", "mock_jina_key_3"]
        
        # Database settings
        self.database_url = "sqlite:///test.db"
        self.database_pool_size = 1
        self.database_max_overflow = 1
        
        # Authentication
        self.bearer_token = "mock_bearer_token"
        
        # Model configuration
        self.llm_model = "mock_llm_model"
        self.embedding_model = "mock_embedding_model"
        
        # Performance settings
        self.max_chunk_size = 1500
        self.chunk_overlap = 200
        self.min_chunk_size = 300
        
        # Retrieval settings
        self.top_k_results = 10
        self.similarity_threshold = 0.6
        self.enable_hybrid_search = True
        
        # LLM settings
        self.max_tokens_per_response = 500
        self.llm_temperature = 0.1
        self.llm_timeout = 30
        
        # Rate limiting
        self.groq_rpm_limit = 30
        self.groq_concurrent_requests = 3
        self.jina_rpm_limit = 100
        self.jina_concurrent_requests = 5
        self.pinecone_rpm_limit = 100
        
        # Pinecone settings
        self.pinecone_api_key = "mock_pinecone_key"
        self.pinecone_environment = "mock-env"
        self.pinecone_index_name = "mock-index"
        self.pinecone_dimension = 768
        self.pinecone_metric = "cosine"
        self.pinecone_batch_size = 100
        
        # Processing settings
        self.enable_on_demand = True
        self.max_pages_per_batch = 50
        self.use_llm_planner = False
        
        # Cache settings
        self.enable_query_cache = True
        self.enable_embedding_cache = True
        self.query_cache_ttl = 3600
        self.embedding_cache_ttl = 86400
        
        # Memory settings
        self.max_memory_mb = 400
        self.embedding_cache_max_size = 100
        self.query_cache_max_size = 50
    
    @property
    def groq_api_keys(self) -> List[str]:
        """Get list of mock Groq API keys."""
        return self._mock_groq_keys
    
    @property
    def jina_api_keys(self) -> List[str]:
        """Get list of mock Jina API keys."""
        return self._mock_jina_keys
    
    @property
    def domain_prompts(self) -> Dict[str, Any]:
        """Domain-specific prompt configurations."""
        return {
            "insurance": {
                "chunk_size": 1200,
                "focus_keywords": ["policy", "coverage", "claim"],
                "prompt_style": "precise_technical"
            },
            "legal": {
                "chunk_size": 1500,
                "focus_keywords": ["article", "section", "clause"],
                "prompt_style": "formal_structured"
            },
            "general": {
                "chunk_size": self.max_chunk_size,
                "focus_keywords": [],
                "prompt_style": "conversational"
            }
        }


# Mock services for testing
class MockGroqClient:
    """Mock Groq client for testing."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.chat = Mock()
        self.chat.completions = Mock()
        self.chat.completions.create = Mock()
    
    def create_completion(self, **kwargs):
        """Mock completion creation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Mock LLM response"
        mock_response.usage = Mock()
        mock_response.usage.total_tokens = 10
        return mock_response


class MockPineconeClient:
    """Mock Pinecone client for testing."""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self._indexes = {}
    
    def create_index(self, name: str, **kwargs):
        """Mock index creation."""
        self._indexes[name] = MockPineconeIndex(name)
        return self._indexes[name]
    
    def list_indexes(self):
        """Mock index listing."""
        return list(self._indexes.keys())
    
    def Index(self, name: str):
        """Mock index retrieval."""
        if name not in self._indexes:
            self._indexes[name] = MockPineconeIndex(name)
        return self._indexes[name]


class MockPineconeIndex:
    """Mock Pinecone index for testing."""
    
    def __init__(self, name: str):
        self.name = name
        self._vectors = {}
    
    def upsert(self, vectors, **kwargs):
        """Mock vector upsert."""
        for vector in vectors:
            self._vectors[vector['id']] = vector
        return {"upserted_count": len(vectors)}
    
    def query(self, vector=None, id=None, top_k=10, **kwargs):
        """Mock vector query."""
        mock_matches = []
        for i in range(min(top_k, 3)):  # Return up to 3 mock matches
            mock_matches.append({
                'id': f'mock_id_{i}',
                'score': 0.9 - (i * 0.1),
                'metadata': {'text': f'Mock text {i}', 'page': i + 1}
            })
        
        return {
            'matches': mock_matches,
            'namespace': kwargs.get('namespace', '')
        }
    
    def describe_index_stats(self):
        """Mock index stats."""
        return {
            'dimension': 768,
            'index_fullness': 0.1,
            'namespaces': {'': {'vector_count': len(self._vectors)}},
            'total_vector_count': len(self._vectors)
        }


# Mock embedding service
class MockEmbeddingService:
    """Mock embedding service for testing."""
    
    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Mock text embedding."""
        # Return mock embeddings (random-ish values)
        import random
        embeddings = []
        for text in texts:
            # Create deterministic "embedding" based on text hash
            seed = hash(text) % 1000
            random.seed(seed)
            embedding = [random.random() for _ in range(768)]
            embeddings.append(embedding)
        return embeddings
    
    async def init(self):
        """Mock initialization."""
        pass


# Function to create mock settings
def get_mock_settings() -> MockSettings:
    """Get mock settings for testing."""
    return MockSettings()


# Function to patch settings in tests
def patch_settings_for_testing():
    """
    Patch the settings module to use mock settings for testing.
    This should be called at the beginning of test files that need to import
    modules that depend on the settings.
    """
    import sys
    
    # Create a mock config module
    mock_config = type(sys)('mock_config')  # Create a module-like object
    mock_config.settings = get_mock_settings()
    
    # Replace the config module in sys.modules
    sys.modules['config'] = mock_config
    
    return mock_config.settings
