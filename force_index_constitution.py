"""
Force index specific pages of any document for accurate answers.
"""

import asyncio
import hashlib
import logging
from typing import List, Dict, Any

from app import (
    extract_document_structure, 
    _process_and_index_pages,
    db_manager
)
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def force_index_specific_pages(doc_url: str, pages_to_index: List[int]):
    """Force index specific pages of any document for accurate answers."""
    
    doc_id = hashlib.md5(doc_url.encode()).hexdigest()
    
    logger.info(f"Starting force index for document: {doc_id}")
    logger.info(f"Document URL: {doc_url}")
    
    # Get current status
    status = await db_manager.get_document_status(doc_id)
    logger.info(f"Current status: {status}")
    
    # Remove already indexed pages
    if status and status.get('indexed_pages'):
        already_indexed = set(status['indexed_pages'])
        pages_to_process = [p for p in pages_to_index if p not in already_indexed]
    else:
        pages_to_process = pages_to_index
    
    if not pages_to_process:
        logger.info("All specified pages already indexed!")
        return
    
    logger.info(f"Need to index {len(pages_to_process)} pages: {pages_to_process[:10]}...")
    
    # Process pages in batches
    batch_size = 10
    for i in range(0, len(pages_to_process), batch_size):
        batch = pages_to_process[i:i + batch_size]
        logger.info(f"Processing batch {i//batch_size + 1}: pages {batch}")
        
        try:
            await _process_and_index_pages(doc_url, batch)
            logger.info(f"Successfully indexed batch: {batch}")
        except Exception as e:
            logger.error(f"Failed to index batch {batch}: {e}")
    
    logger.info("Page indexing completed!")

async def index_document_sections(doc_url: str, section_pages: Dict[str, List[int]]):
    """Index specific sections of a document based on content type."""
    
    logger.info(f"Indexing document sections for: {doc_url}")
    
    # Collect all pages from all sections
    all_pages = set()
    for section_name, pages in section_pages.items():
        logger.info(f"Section '{section_name}': pages {pages}")
        all_pages.update(pages)
    
    await force_index_specific_pages(doc_url, list(all_pages))

if __name__ == "__main__":
    # Example usage - can be customized for any document
    doc_url = "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2024/07/20240716890312078.pdf"
    
    # Example: Index specific page ranges for any document
    pages_to_index = list(range(15, 60))  # Example: pages 15-59
    
    asyncio.run(force_index_specific_pages(doc_url, pages_to_index)) 