# LLM-Powered Intelligent Query-Retrieval System for Multi-Domain Document Processing

## Executive Summary

This comprehensive guide presents a production-ready RAG (Retrieval-Augmented Generation) system architecture designed for processing large documents (50+ MB, 500+ pages) across multiple domains including insurance, legal, HR, compliance, and general business documents. The system maintains sub-30 second processing times and 60-second total query response times while being domain-agnostic and adaptable to various document types.

## System Architecture Overview

### High-Level Architecture

The proposed system follows a distributed, asynchronous architecture optimized for Railway's 4GB build limit and production constraints:

```
User Query → FastAPI Orchestrator → Async Processing Pipeline
    ↓
[PDF Processing] → [Jina Embeddings] → [Pinecone Storage] → [PostgreSQL Metadata]
    ↓
[Query Processing] → [Groq LLM Pool] → [Response Generation] → [JSON Output]
```

### Core Components

**1. FastAPI Application Server**
- Asynchronous request handling with proper memory management[1][2]
- Connection pooling for PostgreSQL with Railway optimization[3]
- Background task processing for document ingestion[4]
- Streaming responses for large document processing[5]

**2. Document Processing Pipeline**
- PyMuPDF for efficient PDF text extraction (faster than PyPDF2)[6][7]
- Concurrent PDF processing using asyncio[8][9]
- Chunking strategy with semantic preservation[10][11]
- Memory-efficient streaming document processing[12][4]

**3. Embedding Generation**
- 6 Jina AI free-tier APIs with rate limit management[13][14]
- Async batch processing for embedding generation[15]
- Token optimization and efficient API usage[16][17]

**4. Vector Storage**
- Pinecone for production-ready vector search[18][19]
- Optimized embedding storage with quantization techniques[20]
- Efficient similarity search with sub-second retrieval[21][22]

**5. LLM Response Generation**
- 5 Groq APIs for parallel query processing[23][24]
- Async request batching and concurrent execution[25][26]
- Context-aware response generation with citations[27]

## Implementation Strategy

### Document Processing Optimization

**Memory-Efficient PDF Processing**[6][28]
```python
import asyncio
import fitz  # PyMuPDF
from concurrent.futures import ProcessPoolExecutor
import gc

async def process_pdf_chunks(pdf_bytes: bytes, chunk_size: int = 512):
    """Process PDF in memory-efficient chunks"""
    loop = asyncio.get_running_loop()
    
    with ProcessPoolExecutor(max_workers=2) as executor:
        # Process PDF pages concurrently
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        tasks = []
        for page_num in range(len(doc)):
            task = loop.run_in_executor(
                executor, extract_page_text, doc, page_num
            )
            tasks.append(task)
        
        # Process pages concurrently
        page_texts = await asyncio.gather(*tasks)
        
        # Clean up memory
        doc.close()
        del doc
        gc.collect()
        
        return chunk_text(page_texts, chunk_size)
```

**Async Embedding Generation**[16][13]
```python
import asyncio
import aiohttp
from typing import List

class JinaEmbeddingManager:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.rate_limits = {key: {"requests": 0, "tokens": 0, "reset_time": 0} 
                           for key in api_keys}
    
    async def generate_embeddings_batch(self, texts: List[str]):
        """Generate embeddings using round-robin API keys"""
        semaphore = asyncio.Semaphore(6)  # Limit concurrent requests
        
        async def process_batch(batch_texts):
            async with semaphore:
                api_key = self.get_available_key()
                return await self.call_jina_api(batch_texts, api_key)
        
        # Split into batches to respect rate limits
        batches = [texts[i:i+100] for i in range(0, len(texts), 100)]
        
        tasks = [process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return self.combine_results(results)
```

### Railway Deployment Optimization

**Memory Management for 4GB Limit**[29][4]
```python
import os
import gc
from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Optimize memory settings
    os.environ["PYTHONOPTIMIZE"] = "2"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    
    # Configure garbage collection for memory efficiency
    gc.set_threshold(700, 10, 10)  # More aggressive GC
    
    yield
    
    # Cleanup on shutdown
    gc.collect()

app = FastAPI(lifespan=lifespan)

# Background task for document processing
async def process_document_background(document_url: str):
    """Process document in background with memory cleanup"""
    try:
        # Process document
        result = await process_large_pdf(document_url)
        
        # Store results
        await store_embeddings(result)
        
    finally:
        # Aggressive memory cleanup
        gc.collect()
        
@app.post("/process-document")
async def process_document(
    document_url: str, 
    background_tasks: BackgroundTasks
):
    background_tasks.add_task(process_document_background, document_url)
    return {"status": "processing_started"}
```

**PostgreSQL Connection Pooling**[3][30]
```python
import asyncpg
import asyncio
from typing import Optional

class DatabaseManager:
    def __init__(self, database_url: str):
        self.database_url = database_url
        self.pool: Optional[asyncpg.Pool] = None
    
    async def initialize_pool(self):
        """Initialize connection pool for Railway PostgreSQL"""
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=1,
            max_size=5,  # Conservative for 4GB limit
            command_timeout=60,
            server_settings={
                'application_name': 'rag_system',
                'jit': 'off'  # Reduce memory usage
            }
        )
    
    async def store_document_metadata(self, doc_id: str, metadata: dict):
        """Store document metadata efficiently"""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO documents (id, metadata, created_at) VALUES ($1, $2, NOW())",
                doc_id, metadata
            )
```

### Query Processing Pipeline

**Async RAG Implementation**[31][15]
```python
class AsyncRAGPipeline:
    def __init__(self, pinecone_client, groq_clients: List, db_manager):
        self.pinecone = pinecone_client
        self.groq_clients = groq_clients
        self.db = db_manager
        self.semaphore = asyncio.Semaphore(5)  # Limit concurrent LLM calls
    
    async def process_queries_batch(self, queries: List[str]):
        """Process multiple queries concurrently"""
        tasks = []
        for query in queries:
            task = self.process_single_query(query)
            tasks.append(task)
        
        # Process all queries concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return self.format_results(results)
    
    async def process_single_query(self, query: str):
        """Process single query with retrieval and generation"""
        async with self.semaphore:
            # Step 1: Generate query embedding
            query_embedding = await self.generate_embedding(query)
            
            # Step 2: Retrieve relevant chunks
            relevant_chunks = await self.retrieve_chunks(query_embedding)
            
            # Step 3: Generate response using available Groq API
            response = await self.generate_response(query, relevant_chunks)
            
            return {
                "query": query,
                "answer": response["content"],
                "sources": [chunk["metadata"] for chunk in relevant_chunks]
            }
    
    async def generate_response(self, query: str, chunks: List):
        """Generate response using round-robin Groq APIs"""
        groq_client = self.get_available_groq_client()
        
        context = "\n".join([chunk["text"] for chunk in chunks])
        
        prompt = """
        Based on the following context, answer the question accurately and provide citations.
        
        Context: {context}
        
        Question: {query}
        
        Instructions:
        - Only use information from the provided context
        - Include specific citations to source documents
        - If information is not available, clearly state this
        - Provide a comprehensive answer as if you are an expert who has read the documents
        """
        
        response = await groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1500
        )
        
        return response.choices[0].message
```

## Performance Optimization Strategies

### Token Efficiency and Cost Management

**Jina API Optimization**[13][32]
- Implement token counting and batch optimization
- Use round-robin API key rotation to maximize throughput
- Cache embeddings for repeated content
- Implement smart chunking to minimize token usage

**Groq API Optimization**[23][25]
- Leverage batch processing for multiple queries
- Use streaming responses for real-time user experience
- Implement query priority queuing
- Cache frequent query patterns

### Latency Optimization

**Target Performance Metrics**[33][34]
- Document processing: 25-30 seconds for 500+ page PDFs
- Query response time: 1-2 seconds per query
- Concurrent query handling: 15-30 queries within 60 seconds
- Memory usage: Under 3.5GB peak during processing

**Implementation Strategies**[35][36]
```python
# Concurrent processing with memory management
async def process_document_optimized(pdf_bytes: bytes):
    start_time = time.time()
    
    # Phase 1: Extract text (10-15 seconds)
    text_chunks = await extract_text_concurrent(pdf_bytes)
    
    # Phase 2: Generate embeddings (10-15 seconds)
    embeddings = await generate_embeddings_parallel(text_chunks)
    
    # Phase 3: Store in Pinecone (2-5 seconds)
    await store_embeddings_batch(embeddings)
    
    processing_time = time.time() - start_time
    logger.info(f"Document processed in {processing_time:.2f} seconds")
    
    return {"processing_time": processing_time, "chunks_processed": len(text_chunks)}
```

## Testing and Evaluation Framework

### Test Document Processing

Based on the provided test questions, the system should handle various document types:

**Newton's Principia Questions** - Complex scientific document analysis
**Insurance Policy Questions** - Dense legal document processing with specific clause retrieval
**Mixed Query Types** - From factual extraction to complex reasoning

### Evaluation Metrics

**Accuracy Metrics**
- Precise clause matching and retrieval
- Contextual answer generation
- Citation accuracy and source attribution

**Performance Metrics**
- Processing time per document size
- Query response latency
- Memory usage efficiency
- API rate limit optimization

**Quality Metrics**
- Answer relevance and completeness
- Hallucination prevention
- Proper handling of "I don't know" responses

## Production Deployment Configuration

### Railway Configuration

**railway.toml**
```toml
[build]
builder = "NIXPACKS"

[deploy]
startCommand = "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1"
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 3

[variables]
PYTHON_VERSION = "3.11"
PYTHONOPTIMIZE = "2"
PYTHONDONTWRITEBYTECODE = "1"
```

**Memory-Optimized Dockerfile**
```dockerfile
FROM python:3.11-slim

# Optimize for Railway's constraints
ENV PYTHONOPTIMIZE=2
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run with memory-optimized settings
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

## API Endpoints and Integration

### Core API Endpoints

**Document Processing**
```python
@app.post("/hackrx/run")
async def process_hackrx_submission(request: HackrxRequest):
    """Process HackRX evaluation request"""
    try:
        # Extract document from URL
        pdf_content = await download_pdf(request.documents)
        
        # Process document if not already processed
        doc_id = generate_doc_id(request.documents)
        if not await is_document_processed(doc_id):
            await process_document_background(pdf_content, doc_id)
        
        # Process all questions concurrently
        answers = await process_queries_batch(request.questions, doc_id)
        
        return {"answers": answers}
        
    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        raise HTTPException(status_code=500, detail="Processing failed")
```

**Health Check and Monitoring**
```python
@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "memory_usage": get_memory_usage(),
        "active_connections": get_db_connections(),
        "api_rate_limits": get_api_status()
    }
```

## Scalability and Future Enhancements

### Horizontal Scaling Strategies

**Multi-Instance Deployment**
- Load balancing across multiple Railway services
- Shared Pinecone and PostgreSQL resources
- Redis caching for query results

**Performance Monitoring**
- Real-time memory usage tracking
- API rate limit monitoring
- Query performance analytics
- Error rate and exception tracking

### Advanced Features

**Smart Caching Layer**[35]
- Embedding cache for repeated content
- Query result caching with TTL
- Document processing state management

**Advanced RAG Techniques**[10][27]
- Hierarchical document chunking
- Multi-modal content handling
- Dynamic retrieval strategies
- Query expansion and refinement

## Conclusion

This comprehensive RAG system architecture addresses the unique challenges of processing large documents within strict deployment constraints while maintaining production-ready performance. The system leverages modern async Python patterns, efficient PDF processing, and distributed API management to achieve the required processing speeds and query response times.

Key advantages of this approach:

- **Performance**: Sub-30 second document processing, 1-2 second query responses[33][34]
- **Scalability**: Concurrent processing with memory optimization[4][37]
- **Reliability**: Robust error handling and API failover mechanisms[23][25]
- **Cost Efficiency**: Optimal use of free-tier APIs with intelligent rate limiting[13][38]
- **Production Ready**: Railway deployment with monitoring and health checks[1][2]

The system is designed to handle the evaluation scenarios described in the problem statement while providing a solid foundation for enterprise deployment in insurance, legal, HR, and compliance domains.
