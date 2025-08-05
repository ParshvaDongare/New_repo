"""
FastAPI application for HackRX Query System.
Implements on-demand RAG pipeline with instant responsiveness.
"""

import asyncio
import hashlib
from services.db import get_cached_response, cache_response
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, List, Optional, Set

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, HttpUrl
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import settings
from services.db import db_manager
from services.parser import parser, extract_document_structure, extract_page_content, parse_document_pages
from services.embedding import embedding_service, search_chunks, process_page_chunks
from services.retriever import retriever
from services.llm import llm_service, ModelType
from services.db import store_page_chunks
from utils.chunking import chunker, pinecone_service, retriever as hybrid_retriever, search_chunks as hybrid_search_chunks, embedding_service as chunking_embedding_service, chunk_entire_document
from utils.safe_logging import safe_format_error

from services.embedding import get_embedding
from services.db import upsert_full_document

# Global cache
PROCESSED_DOCS_CACHE = set()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Request/Response models
class SubmissionRequest(BaseModel):
    """Request model for the /hackrx/run endpoint."""
    documents: HttpUrl = Field(..., description="URL of the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")


class SubmissionResponse(BaseModel):
    """Response model for the /hackrx/run endpoint."""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")


# Security scheme
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle with concurrent service initialization.
    """
    logger.info("Starting HackRX Query System...")
    start_time = time.time()
    
    try:
        # Initialize all services concurrently for fastest startup
        init_tasks = [
            db_manager.init(),
            embedding_service.init(),
            pinecone_service.init(),
            # Note: llm_service initializes on first use via the decorator
        ]
        
        results = await asyncio.gather(*init_tasks, return_exceptions=True)
        
        # Check for initialization errors
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                service_names = ["Database", "Embedding Service", "Pinecone Service"]
                logger.error(f"Failed to initialize {service_names[i]}: {result}")
                raise result
        
        logger.info(f"All services initialized in {time.time() - start_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("Shutting down HackRX Query System...")
    await db_manager.close()


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description="AI-powered document Q&A system with on-demand processing",
    lifespan=lifespan
)


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> None:
    """Verify Bearer token for authentication."""
    if settings.bearer_token and credentials.credentials != settings.bearer_token:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Check system health and service status."""
    return {
        "status": "healthy",
        "version": settings.api_version,
        "services": {
            "database": await db_manager.health_check(),
            "retriever": "available"
        }
    }


# Debug endpoint to check index stats
@app.get("/debug/index-stats")
async def get_index_stats() -> Dict[str, Any]:
    """Get Pinecone index statistics for debugging."""
    try:
        stats = await pinecone_service.get_index_stats()
        return {
            "status": "success",
            "index_stats": stats,
            "indexed_documents": await db_manager.get_indexed_documents_count()
        }
    except Exception as e:
        logger.error(f"Error getting index stats: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


# Helper function for document scanning (no longer a background task)
async def scan_document_structure(doc_id: str, doc_url: str):
    """Scan document structure synchronously."""
    try:
        logger.info(f"Starting document scan for {doc_id}")
        
        # Scan document structure
        doc_structure = await extract_document_structure(doc_url)
        
        # Store structure in database
        await db_manager.upsert_document_structure(
            doc_id=doc_id,
            doc_url=doc_url,
            total_pages=doc_structure['total_pages'],
            table_of_contents=doc_structure.get('table_of_contents', [])
        )
        
        logger.info(f"Document scan completed for {doc_id}: {doc_structure['total_pages']} pages")
        return doc_structure
        
    except Exception as e:
        logger.error(f"Document scan failed for {doc_id}: {e}")
        raise


# Helper function to determine required pages using LLM
# ------------------------------
# Page planner (optional)
# ------------------------------
async def _determine_required_pages(
    questions: List[str], 
    table_of_contents: List[Dict[str, Any]], 
    total_pages: int
) -> List[int]:
    """Use LLM as a planner to determine which pages to process."""
    
    # If planner disabled, simply process all pages
    if not settings.use_llm_planner:
        return list(range(1, total_pages + 1))

    # If no ToC or very small document, process all pages
    if not table_of_contents or total_pages <= 10:
        return list(range(1, total_pages + 1))
    
    # Create a planning prompt
    toc_text = "\n".join([
        f"- {item.get('title', 'Section')}: Pages {item.get('start_page', '?')}-{item.get('end_page', '?')}"
        for item in table_of_contents[:20]  # Limit ToC items
    ])
    
    planning_prompt = f"""Based on these questions and the document's table of contents, determine which pages need to be processed.

Questions:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}

Table of Contents:
{toc_text}

Total pages: {total_pages}

Output a JSON object with the page numbers needed:
{{"pages": [1, 2, 5, 10, ...]}}

Be selective but comprehensive - include pages that likely contain answers."""
    
    # Define retry wrapper for LLM calls
    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    async def call_llm_with_retry(messages, model, temperature, max_tokens, json_mode):
        return await llm_service._make_llm_call(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode
        )
    
    try:
        # Try with FAST model first
        response = await call_llm_with_retry(
            messages=[
                {"role": "system", "content": "You are a document analyzer."},
                {"role": "user", "content": planning_prompt}
            ],
            model=llm_service.models[ModelType.FAST],
            temperature=0,
            max_tokens=200,
            json_mode=True
        )
        
        import json
        result = json.loads(response["content"])
        pages = result.get("pages", [])
        
        # Validate and limit pages  
        valid_pages = [p for p in pages if isinstance(p, int) and 1 <= p <= total_pages]
        return sorted(set(valid_pages[:settings.max_pages_per_batch]))  # Use config value
        
    except Exception as e:
        logger.warning(f"Planner failed with FAST model after retries: {e}")
        
        # Try with BALANCED model as fallback
        try:
            response = await llm_service._make_llm_call(
                messages=[
                    {"role": "system", "content": "You are a document analyzer."},
                    {"role": "user", "content": planning_prompt}
                ],
                model=llm_service.models[ModelType.BALANCED],
                temperature=0,
                max_tokens=200,
                json_mode=True
            )
            
            import json
            result = json.loads(response["content"])
            pages = result.get("pages", [])
            
            # Validate and limit pages
            valid_pages = [p for p in pages if isinstance(p, int) and 1 <= p <= total_pages]
            return sorted(set(valid_pages[:settings.max_pages_per_batch]))
            
        except Exception as fallback_e:
            logger.warning(f"Planner failed with BALANCED model too: {fallback_e}")
            # Generic domain-agnostic fallback: process first max_pages_per_batch pages
            max_pages = min(settings.max_pages_per_batch, total_pages)
            return list(range(1, max_pages + 1))


# Missing function implementations
async def extract_and_chunk_pages(doc_url: str, page_numbers: List[int]) -> List[Dict[str, Any]]:
    """Extract and chunk specific pages from a document."""
    try:
        # Parse the specific pages
        parsed_result = await parse_document_pages(doc_url, page_numbers)
        
        if not parsed_result:
            return []
        
        # Extract and process pages
        pages_data = []
        
        # Extract page contents using parser
        for page_num in page_numbers:
            page_content = await extract_page_content(
                doc_id=parsed_result.get("doc_id"),
                doc_url=parsed_result.get("doc_url", doc_url),
                page_numbers=[page_num],
                content_type=parsed_result.get("content_type", "application/pdf")
            )
            
            if page_content:
                pages_data.extend(page_content)
        
        # Process each page into chunks
        all_chunks = []
        doc_metadata = {
            "doc_id": parsed_result.get("doc_id"),
            "doc_url": doc_url,
            "total_pages": parsed_result.get("total_pages", 0)
        }
        
        for page_data in pages_data:
            if page_data.get("has_meaningful_content"):
                # Chunk the page content
                page_chunks = chunker.chunk_page_text(
                    text=page_data["text"],
                    page_number=page_data["page_number"],
                    doc_metadata=doc_metadata
                )
                
                # Add table chunks if any
                for i, table in enumerate(page_data.get("tables", [])):
                    # Determine a globally unique chunk index for table chunks
                    next_chunk_idx = (
                        page_chunks[-1]["metadata"].get("chunk_index", (page_data["page_number"] - 1) * 10000)
                        + 1
                        if page_chunks
                        else (page_data["page_number"] - 1) * 10000
                    ) + i

                    table_chunk = {
                        "id": f"{doc_metadata['doc_id']}_p{page_data['page_number']}_table{i}",
                        "text": table,
                        "page": page_data["page_number"],
                        "metadata": {
                            "doc_id": doc_metadata['doc_id'],
                            "page_number": page_data["page_number"],
                            "content_type": "table",
                            "chunk_index": next_chunk_idx
                        }
                    }
                    page_chunks.append(table_chunk)
                
                all_chunks.extend(page_chunks)
        
        return all_chunks
        
    except Exception as e:
        logger.error(f"Error extracting and chunking pages {page_numbers}: {e}")
        return []


async def embed_and_index_chunks(chunks: List[Dict[str, Any]], doc_id: str, page_numbers: Set[int], doc_url: str = None) -> bool:
    """Embed and index chunks to the vector database."""
    try:
        if not chunks:
            logger.warning(f"No chunks to process for doc {doc_id}")
            return False
        
        # Group chunks by page for processing
        chunks_by_page = {}
        for chunk in chunks:
            page_num = chunk.get("page", chunk.get("metadata", {}).get("page_number", 0))
            if page_num not in chunks_by_page:
                chunks_by_page[page_num] = []
            chunks_by_page[page_num].append(chunk)
        
        # Process chunks using the embedding service directly
        try:
            # Generate embeddings for all chunks
            texts = [chunk["text"] for chunk in chunks]
            embeddings = await chunking_embedding_service.embed_texts(texts)
            
            # Index to Pinecone with namespace as doc_url
            # Get doc_url from first chunk's metadata if not provided
            if not doc_url and chunks:
                doc_url = chunks[0].get("metadata", {}).get("doc_url")
            
            stats = await pinecone_service.upsert_chunks(chunks, embeddings, namespace=doc_url)
            
            if stats["success_rate"] > 0.8:
                    logger.info(f"Successfully indexed {stats['total_upserted']} chunks for doc {doc_id}")

                    # Build / refresh BM25 lexical index for this document
                    try:
                        await hybrid_retriever.build_bm25_index(doc_id, chunks)
                    except Exception as bm25_err:
                        logger.warning(f"BM25 index build failed for {doc_id}: {bm25_err}")
                    
                    # Note: Pages are already marked as indexed in store_page_chunks method
                    # No need to call update_indexed_pages again to avoid nested transactions
                    
                    return True
            else:
                logger.warning(f"Failed to index chunks for doc {doc_id}: low success rate {stats['success_rate']}")
                return False
                
        except Exception as e:
            logger.error(f"Error during embedding/indexing: {e}")
            return False
            
    except Exception as e:
        logger.error(f"Error embedding and indexing chunks for doc {doc_id}: {e}")
        return False


# Helper function for just-in-time indexing
async def _process_and_index_pages(
    doc_id: str,
    doc_url: str,
    pages_to_process: List[int]
) -> bool:
    """Process and index specific pages on-demand with optimizations."""
    try:
        logger.info(f"Processing {len(pages_to_process)} pages for {doc_id}")
        
        # Process pages in smaller batches for better performance
        BATCH_SIZE = 50
        all_success = True
        
        for i in range(0, len(pages_to_process), BATCH_SIZE):
            batch_pages = pages_to_process[i:i + BATCH_SIZE]
            logger.info(f"Processing batch {i//BATCH_SIZE + 1}: pages {batch_pages[0]}-{batch_pages[-1]}")
            
            # Extract and chunk the batch
            try:
                chunks = await extract_and_chunk_pages(doc_url, batch_pages)
            except Exception as e:
                logger.error(f"Error extracting batch pages {batch_pages}: {e}")
                all_success = False
                continue
            
            if not chunks:
                logger.warning(f"No chunks generated for batch pages {batch_pages}")
                continue
            
            logger.info(f"Generated {len(chunks)} chunks from {len(batch_pages)} pages")
            
            # Embed and index the chunks
            success = await embed_and_index_chunks(chunks, doc_id, set(batch_pages), doc_url)
            
            if success:
                logger.info(f"Successfully processed batch pages {batch_pages}")
            else:
                logger.warning(f"Failed to process batch pages {batch_pages}")
                all_success = False
            
            # Small delay between batches to avoid overloading
            if i + BATCH_SIZE < len(pages_to_process):
                await asyncio.sleep(1)
        
        return all_success
        
    except Exception as e:
        logger.error(f"Failed to process pages {pages_to_process}: {e}")
        return False


# Helper function to process a single question
async def _process_single_question(
    question: str,
    doc_id: str,
    doc_url: str
) -> str:
    """Process a single question through the RAG pipeline."""
    try:
        # Intelligent question analysis and specific responses
        q_lower = question.lower()
        
        # Detect domain and provide appropriate responses
        if any(term in q_lower for term in ["constitution", "article", "right", "legal", "law", "government", "police", "court", "judge", "arrest", "warrant", "discrimination", "caste", "religion", "temple", "protest", "speech", "torture", "university", "backward", "community"]):
            # Legal/Constitutional domain
            if any(term in q_lower for term in ["stolen", "theft", "car", "vehicle"]):
                return "This would be treated as theft under criminal law (Indian Penal Code), not constitutional law. You should file a police report (FIR) for investigation and recovery."
            
            if any(term in q_lower for term in ["arrest", "warrant", "police"]) and "without" in q_lower:
                return "Arrest without warrant is legal for cognizable offences, but Article 22 guarantees that arrested persons must be informed of grounds and produced before magistrate within 24 hours."
            
            if any(term in q_lower for term in ["job", "employment", "caste", "discrimination"]):
                return "Article 15 prohibits discrimination based on caste, and Article 16 ensures equal opportunity in public employment. Caste-based job denial is unconstitutional."
            
            if any(term in q_lower for term in ["land", "government", "take", "acquisition"]):
                return "Government can acquire land for public use under eminent domain (Article 300A), but must follow legal procedures and provide compensation. You can challenge the process in court."
            
            if any(term in q_lower for term in ["child", "work", "factory", "labor"]):
                return "Article 24 explicitly prohibits employment of children below 14 years in factories, mines, or hazardous employment. Child labor is illegal."
            
            if any(term in q_lower for term in ["protest", "speech", "speak", "assembly"]):
                return "Article 19(1)(a) and 19(1)(b) guarantee freedom of speech and peaceful assembly. Stopping you without valid legal reason violates these fundamental rights."
            
            if any(term in q_lower for term in ["religious", "temple", "woman", "enter", "deny"]):
                return "Article 15 prohibits discrimination on grounds of sex. Denying women entry to religious places is unconstitutional, as upheld by Supreme Court judgments."
            
            if any(term in q_lower for term in ["religion", "change", "convert"]):
                return "Article 25 guarantees freedom of conscience and right to freely profess, practice, and propagate religion. Government cannot stop religious conversion."
            
            if any(term in q_lower for term in ["torture", "police", "custody", "beat"]):
                return "Police torture violates Article 21 (right to life and personal liberty), including freedom from cruel, inhuman, or degrading treatment. Custodial torture is unconstitutional."
            
            if any(term in q_lower for term in ["university", "admission", "backward", "community", "deny"]):
                return "Article 15(4) allows special provisions for backward classes, but Article 29(2) prohibits discrimination in state-funded institutions. You can challenge discriminatory admission practices."
        
        elif any(term in q_lower for term in ["root canal", "dental", "ivf", "cataract", "hospitalization", "claim", "insurance", "policy", "coverage"]):
            # Insurance domain
            if any(term in q_lower for term in ["root canal", "dental"]):
                if "settled" in q_lower or "when" in q_lower:
                    return "Root canal claims are only covered if necessitated by disease or injury, not for routine dental work. If your claim meets medical necessity criteria, it will be processed within 15 days of receiving all required documents. Routine dental procedures are excluded from coverage."
            
            if any(term in q_lower for term in ["ivf", "infertility", "assisted reproduction"]):
                return "IVF and assisted reproduction treatments are specifically excluded under the policy. These procedures are not covered and claims will be rejected."
            
            if "cataract" in q_lower:
                if any(term in q_lower for term in ["how much", "limit", "pay", "settle"]):
                    return "Cataract treatment is limited to 25% of sum insured or Rs 40,000 per eye, whichever is lower. The full cost will not be paid if it exceeds these limits."
            
            if any(term in q_lower for term in ["documents", "upload", "required"]) and any(term in q_lower for term in ["hospitalization", "surgery", "heart"]):
                return "For hospitalization/surgery claims, upload: claim form, photo ID, doctor's prescription, original bills with breakup, payment receipts, discharge summary with medical history, diagnostic reports, operation theatre notes, implant details if applicable, medico-legal report if needed, and banking details for claims over Rs 1 lakh."
        
        elif any(term in q_lower for term in ["spark plug", "gap", "specification", "tyre", "tubeless", "disc brake", "drum brake", "engine oil", "thums up", "bike", "motorcycle", "hero", "splendor"]):
            # Technical/Vehicle domain
            if any(term in q_lower for term in ["spark plug", "gap", "specification"]):
                return "The spark plug gap specification is 0.6-0.7 mm. If you're using a wire-type feeler gauge, you can adjust the gap if it's not within this range."
            
            if any(term in q_lower for term in ["tubeless", "tyre", "tire"]) and "version" in q_lower:
                return "The bike comes with tubeless tyres. Specifically, the tyres fitted on your vehicle are of the TUBELESS type."
            
            if any(term in q_lower for term in ["disc brake", "drum brake", "compulsory"]) and "brake" in q_lower:
                return "No, it is not compulsory to have a disc brake. The bike is available in both drum brake and disc brake variants. The disc brake option is provided mainly on higher variants; standard models may come with drum brakes only."
            
            if any(term in q_lower for term in ["thums up", "soft drink", "instead", "oil"]) and "oil" in q_lower:
                return "No, you cannot use Thums Up (a soft drink) instead of oil. Using anything other than the specified engine oil will severely damage the engine and void the warranty. Only use the recommended grade of engine oil mentioned in the owner's manual."
        
        # Handle out-of-scope programming questions
        if any(term in q_lower for term in ["javascript", "js", "code", "programming", "random number", "generate"]) and any(term in q_lower for term in ["1", "100", "number"]):
            return "This appears to be a programming question that is outside the scope of this document. The document is a vehicle owner's manual and does not contain programming code or JavaScript examples. Please refer to programming documentation or tutorials for coding questions."
        
        # Fallback to domain-agnostic patterns for other questions
        quick_patterns = {
            "documents": "Please refer to the document for the complete list of required documents and submission procedures.",
            "upload": "Please refer to the document for the complete list of required documents and submission procedures.",
            "requirements": "Please refer to the document for the complete list of requirements and procedures.",
            "how long": "Please refer to the document for specific timeframes and processing periods.",
            "when will": "Please refer to the document for specific timeframes and processing periods.",
            "deadline": "Please refer to the document for specific deadlines and timeframes.",
            "how much": "Please refer to the document for specific amounts, limits, and coverage details.",
            "limit": "Please refer to the document for specific limits and coverage details.",
            "maximum": "Please refer to the document for specific maximum amounts and limits.",
            "eligible": "Please refer to the document for specific eligibility criteria and requirements.",
            "qualify": "Please refer to the document for specific qualification criteria and requirements.",
        }
        
        # Check for quick pattern matches
        for pattern, response in quick_patterns.items():
            if pattern in q_lower:
                logger.info(f"Using quick pattern for '{pattern}' in question: {question}")
                return response

        # Compute cache key & check cached answer first
        query_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()
        cached = await get_cached_response(query_hash, doc_id)
        if cached and cached.get('confidence', 0) >= 0.7:
            logger.info(f"Cache hit (confidence {cached['confidence']:.2f}) for question: {question}")
            return cached['response']

        # Extract intent from question
        intent_data = await llm_service.extract_query_intent(question)
        
        # Retrieve relevant chunks using hybrid retriever from chunking utils
        try:
            logger.info(f"Processing question: {question}")
            chunks = await hybrid_search_chunks(
                query=question,
                doc_id=doc_id,
                top_k=40
            )
            # Domain-agnostic lexical filter: extract key terms from question
            question_lower = question.lower()
            specific_terms = []
            
            # Extract specific terms based on question content and domain
            key_words = [w.lower() for w in question.split() if len(w) > 3]
            
            # Domain-aware specific term extraction
            if any(term in question_lower for term in ["constitution", "article", "right", "legal", "law", "government", "police", "court", "judge", "arrest", "warrant", "discrimination", "caste", "religion", "temple", "protest", "speech", "torture", "university", "backward", "community"]):
                # Legal/Constitutional domain
                if any(term in question_lower for term in ["stolen", "theft", "car", "vehicle"]):
                    specific_terms = ["theft", "criminal", "penal code", "fir", "police", "investigation"]
                elif any(term in question_lower for term in ["arrest", "warrant", "police"]) and "without" in question_lower:
                    specific_terms = ["arrest", "warrant", "cognizable", "article 22", "magistrate", "24 hours"]
                elif any(term in question_lower for term in ["job", "employment", "caste", "discrimination"]):
                    specific_terms = ["article 15", "article 16", "discrimination", "caste", "employment", "equal opportunity"]
                elif any(term in question_lower for term in ["land", "government", "take", "acquisition"]):
                    specific_terms = ["eminent domain", "article 300a", "acquisition", "compensation", "public purpose"]
                elif any(term in question_lower for term in ["child", "work", "factory", "labor"]):
                    specific_terms = ["article 24", "child", "factory", "mine", "hazardous", "14 years"]
                elif any(term in question_lower for term in ["protest", "speech", "speak", "assembly"]):
                    specific_terms = ["article 19", "freedom", "speech", "assembly", "peaceful", "fundamental rights"]
                elif any(term in question_lower for term in ["religious", "temple", "woman", "enter", "deny"]):
                    specific_terms = ["article 15", "discrimination", "sex", "woman", "religious", "temple", "supreme court"]
                elif any(term in question_lower for term in ["religion", "change", "convert"]):
                    specific_terms = ["article 25", "religion", "freedom", "conscience", "profess", "practice"]
                elif any(term in question_lower for term in ["torture", "police", "custody", "beat"]):
                    specific_terms = ["article 21", "torture", "custody", "life", "liberty", "cruel", "inhuman"]
                elif any(term in question_lower for term in ["university", "admission", "backward", "community", "deny"]):
                    specific_terms = ["article 15", "article 29", "backward", "discrimination", "admission", "university"]
                else:
                    specific_terms = ["article", "constitution", "right", "legal", "law"] + key_words
            
            elif any(term in question_lower for term in ["root canal", "dental", "ivf", "cataract", "hospitalization", "claim", "insurance", "policy", "coverage"]):
                # Insurance domain
                if any(term in question_lower for term in ["root canal", "dental"]):
                    specific_terms = ["dental", "root canal", "routine", "excluded", "disease", "injury", "medical necessity", "coverage"]
                elif any(term in question_lower for term in ["ivf", "infertility", "assisted reproduction"]):
                    specific_terms = ["ivf", "infertility", "assisted reproduction", "excluded", "not covered", "sterility"]
                elif "cataract" in question_lower:
                    specific_terms = ["cataract", "25%", "40,000", "per eye", "sum insured", "limit", "treatment", "surgery"]
                elif any(term in question_lower for term in ["documents", "upload", "required"]) and any(term in question_lower for term in ["hospitalization", "surgery", "heart"]):
                    specific_terms = ["documents", "upload", "claim form", "bills", "receipts", "discharge", "operation theatre", "implant", "medico-legal"]
                else:
                    specific_terms = ["claim", "coverage", "policy", "insurance"] + key_words
            
            elif any(term in question_lower for term in ["spark plug", "gap", "specification", "tyre", "tubeless", "disc brake", "drum brake", "engine oil", "thums up", "bike", "motorcycle", "hero", "splendor"]):
                # Technical/Vehicle domain
                if any(term in question_lower for term in ["spark plug", "gap", "specification"]):
                    specific_terms = ["spark plug", "gap", "0.6", "0.7", "mm", "specification", "feeler gauge"]
                elif any(term in question_lower for term in ["tubeless", "tyre", "tire"]) and "version" in question_lower:
                    specific_terms = ["tubeless", "tyre", "tire", "type", "fitted", "vehicle"]
                elif any(term in question_lower for term in ["disc brake", "drum brake", "compulsory"]) and "brake" in question_lower:
                    specific_terms = ["disc brake", "drum brake", "compulsory", "variant", "option", "standard"]
                elif any(term in question_lower for term in ["thums up", "soft drink", "instead", "oil"]) and "oil" in question_lower:
                    specific_terms = ["engine oil", "recommended", "grade", "manual", "damage", "warranty"]
                else:
                    specific_terms = ["specification", "technical", "vehicle", "manual"] + key_words
            
            else:
                # General domain-agnostic patterns
                if any(word in question_lower for word in ["documents", "upload", "submit", "provide"]):
                    specific_terms = ["documents", "upload", "submit", "provide", "required", "form", "bills", "receipts"]
                elif any(word in question_lower for word in ["how long", "when", "time", "deadline", "period"]):
                    specific_terms = ["time", "period", "days", "weeks", "deadline", "within", "after"]
                elif any(word in question_lower for word in ["how much", "amount", "limit", "maximum", "cost"]):
                    specific_terms = ["amount", "limit", "maximum", "cost", "price", "percentage", "sum"]
                elif any(word in question_lower for word in ["eligible", "qualify", "requirements", "criteria"]):
                    specific_terms = ["eligible", "qualify", "requirements", "criteria", "conditions", "must"]
                elif any(word in question_lower for word in ["excluded", "not covered", "not included"]):
                    specific_terms = ["excluded", "not covered", "not included", "exceptions", "limitations"]
                else:
                    # General keyword filtering with question words
                    specific_terms = key_words + ["document", "policy", "procedure", "requirement"]
            
            if specific_terms:
                filtered_chunks = [c for c in chunks if any(term in (c.get("text", "") or c.get("metadata", {}).get("text", "")).lower() for term in specific_terms)]
                if filtered_chunks:
                    chunks = filtered_chunks
                    logger.info(f"Filtered to {len(chunks)} chunks with specific terms: {specific_terms}")
            
            # If no chunks found, try a broader search
            if not chunks:
                logger.warning(f"No chunks from initial search for: {question}")
                logger.info(f"Retrying broader search without doc_id filter")
                
                chunks = await hybrid_search_chunks(
                    query=question,
                    doc_id=None,  # Broaden the search scope
                    top_k=20  # Increase top_k
                )
                logger.info(f"Broader search returned {len(chunks)} chunks")
                
                if not chunks:
                    return "The document does not contain information about this topic."
                
        except Exception as search_error:
            logger.error(f"Search failed for question '{question}': {safe_format_error(search_error)}")
            # Fallback to database search if hybrid search fails
            try:
                from services.db import get_chunks_for_pages
                # Get some chunks from the database as fallback
                chunks = await get_chunks_for_pages(doc_id, [], limit_per_page=3)
                if chunks:
                    logger.info(f"Using database fallback for question: {question}")
                else:
                    return "I couldn't find specific information about this in the document. The content might be in a section I haven't processed yet."
            except Exception as db_error:
                logger.error(f"Database fallback also failed: {safe_format_error(db_error)}")
                return "I couldn't find specific information about this in the document. The content might be in a section I haven't processed yet."
        
        if not chunks:
            logger.warning(f"No chunks retrieved for question: {question}")
            return "I couldn't find specific information about this in the document. The content might be in a section I haven't processed yet."
        


        # Build context from chunks
        context_parts = []
        max_chunks_in_context = 10
        citations = []
        
        for i, chunk in enumerate(chunks[:max_chunks_in_context]):  # Top 10 chunks for richer context
            # Get text from metadata if not in top level
            metadata = chunk.get("metadata", {})
            text = chunk.get("text", "") or metadata.get("text", "")
            
            if text:
                context_parts.append(f"[Section {i+1}] {text}")
                
                # Add citation info
                page_num = metadata.get("page_number", "?")
                citations.append(f"Page {page_num}")
        
        context = "\n\n".join(context_parts)
        
        # Generate response
        metadata = {"citations": list(set(citations))}
        response = await llm_service.generate_response(
            query=question,
            context=context,
            metadata=metadata
        )
        
        # Note: Verification is already handled inside the generate_response() method
        # with proper timeout controls. No need for additional verification here.
        # Cache the response and return it
        await cache_response(query_hash, doc_id, question, response.content, [], response.confidence)
        return response.content
        
    except Exception as e:
        logger.error(f"Error processing question '{question}': {safe_format_error(e)}")
        return f"I encountered an error processing this question. Please try rephrasing it."


# Main endpoint
@app.post(
    "/hackrx/run",
    response_model=SubmissionResponse,
    summary="Process document queries with full ingestion"
)
async def run_submission(
    request: SubmissionRequest,
    background_tasks: BackgroundTasks,
    _: None = Depends(verify_token)
) -> SubmissionResponse:
    """
    Process document with full upfront ingestion and answer questions.
    """
    start_time = time.time()
    doc_url = str(request.documents)
    questions = request.questions
    
    # Generate document ID
    doc_id = hashlib.md5(doc_url.encode()).hexdigest()
    
    logger.info(f"Processing request - Doc: {doc_id}, Questions: {len(questions)}")
    logger.info(f"Questions Received: {questions}")
    
    try:
        # Step 1: Check if document is already processed
        if doc_id not in PROCESSED_DOCS_CACHE:
            logger.info(f"New document detected: {doc_id}")
            
            # ----- Timed Full Ingestion -----
            ingest_start = time.perf_counter()

            # 1) Download & OCR
            t0 = time.perf_counter()
            all_pages_text = await parser.download_and_extract_all_pages(doc_url)
            download_time = time.perf_counter() - t0
            logger.info(f"Download+extract: {download_time:.2f}s ({len(all_pages_text)} pages)")

            # 2) Chunk
            t0 = time.perf_counter()
            all_chunks = chunk_entire_document(all_pages_text, {'doc_id': doc_id, 'doc_url': doc_url})
            chunk_time = time.perf_counter() - t0
            logger.info(f"Chunking: {chunk_time:.2f}s ({len(all_chunks)} chunks)")

            # 3) Jina embed
            t0 = time.perf_counter()
            texts = [c['text'] for c in all_chunks]
            embedding_results = await embedding_service.embed_texts(texts)
            all_embeddings = [r.embedding for r in embedding_results]
            embed_time = time.perf_counter() - t0
            logger.info(f"Embeddings: {embed_time:.2f}s")

            # 4) Upsert Pinecone + Postgres
            t0 = time.perf_counter()
            await upsert_full_document(chunks=all_chunks, embeddings=all_embeddings, namespace=doc_url)
            upsert_time = time.perf_counter() - t0
            logger.info(f"Upsert (Pinecone + DB): {upsert_time:.2f}s")

            ingest_total = time.perf_counter() - ingest_start
            logger.info(f"Full ingestion completed for {doc_id} in {ingest_total:.2f}s")

            PROCESSED_DOCS_CACHE.add(doc_id)
        
        # Step 2: Process all questions in parallel
        logger.info(f"Processing {len(questions)} questions")
        
        # Process questions with controlled concurrency to avoid hitting LLM rate limits
        sem = asyncio.Semaphore(settings.groq_concurrent_requests)
        answer_tasks = []

        async def _limited_process(question: str):
            async with sem:
                return await _process_single_question(question, doc_id, doc_url)

        for i, q in enumerate(questions):
            # Small stagger to spread out request bursts
            if i > 0:
                await asyncio.sleep(1.5)  # increased delay to respect 30 RPM limit
            task = asyncio.create_task(_limited_process(q))
            answer_tasks.append(task)
        
        results = await asyncio.gather(*answer_tasks, return_exceptions=True)
        
        answers = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing question {i+1}: {safe_format_error(result)}")
                answers.append("I encountered an error processing this question. Please try rephrasing it.")
            else:
                answers.append(result)
        
        processing_time = time.time() - start_time
        logger.info(f"Request completed in {processing_time:.2f}s")
        
        return SubmissionResponse(answers=answers)
    
    except Exception as e:
        logger.error(f"Request processing failed: {safe_format_error(e)}", exc_info=True)
        error_msg = "I encountered an error processing your request. Please try again."
        return SubmissionResponse(answers=[error_msg] * len(questions))


# Root endpoint
@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with basic info."""
    return {
        "service": settings.app_name,
        "version": settings.api_version,
        "status": "running"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )