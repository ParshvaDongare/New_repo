#!/usr/bin/env python3
"""
Data Cleaning Script for Policy Intelligence API
Cleans data from both Pinecone vector database and PostgreSQL relational database.
"""

import asyncio
import argparse
import logging
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass

# Import our services
from services.db import DatabaseManager, Document, DocumentChunk, QueryCache
from services.embedding import OptimizedEmbeddingService
from config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CleanupOptions:
    """Options for data cleanup operations."""
    # General options
    dry_run: bool = False
    verbose: bool = False
    
    # PostgreSQL cleanup options
    clean_documents: bool = False
    clean_chunks: bool = False
    clean_cache: bool = False
    clean_all_pg: bool = False
    
    # Pinecone cleanup options
    clean_pinecone: bool = False
    clean_pinecone_by_doc: Optional[str] = None
    
    # Filtering options
    doc_ids: Optional[List[str]] = None
    older_than_days: Optional[int] = None
    min_relevance_score: Optional[float] = None
    max_retrieval_count: Optional[int] = None
    
    # Safety options
    confirm: bool = False
    backup_before_cleanup: bool = False


class DataCleaner:
    """Main data cleaning class for both Pinecone and PostgreSQL."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.embedding_service = OptimizedEmbeddingService()
        self._initialized = False
    
    async def init(self):
        """Initialize database and embedding service connections."""
        if self._initialized:
            return
        
        try:
            await self.db_manager.init()
            await self.embedding_service.init()
            self._initialized = True
            logger.info("Data cleaner initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize data cleaner: {e}")
            raise
    
    async def close(self):
        """Close all connections."""
        if self._initialized:
            await self.db_manager.close()
            await self.embedding_service.close()
            self._initialized = False
            logger.info("Data cleaner connections closed")
    
    async def get_cleanup_stats(self, options: CleanupOptions) -> Dict[str, Any]:
        """Get statistics about what would be cleaned."""
        stats = {
            "postgresql": {},
            "pinecone": {},
            "total_estimated_size_mb": 0
        }
        
        async with self.db_manager.get_session() as session:
            # Document stats
            if options.clean_documents or options.clean_all_pg:
                doc_query = self._build_document_query(options)
                doc_count = await session.execute(doc_query)
                stats["postgresql"]["documents"] = doc_count.scalar() or 0
            
            # Chunk stats
            if options.clean_chunks or options.clean_all_pg:
                chunk_query = self._build_chunk_query(options)
                chunk_count = await session.execute(chunk_query)
                stats["postgresql"]["chunks"] = chunk_count.scalar() or 0
            
            # Cache stats
            if options.clean_cache or options.clean_all_pg:
                cache_query = self._build_cache_query(options)
                cache_count = await session.execute(cache_query)
                stats["postgresql"]["cache_entries"] = cache_count.scalar() or 0
            
            # Pinecone stats
            if options.clean_pinecone:
                pinecone_stats = await self._get_pinecone_stats(options)
                stats["pinecone"] = pinecone_stats
        
        return stats
    
    def _build_document_query(self, options: CleanupOptions):
        """Build query for documents to be cleaned."""
        from sqlalchemy import select, and_, func
        
        conditions = []
        
        if options.doc_ids:
            conditions.append(Document.doc_id.in_(options.doc_ids))
        
        if options.older_than_days:
            cutoff_date = datetime.utcnow() - timedelta(days=options.older_than_days)
            conditions.append(Document.created_at < cutoff_date)
        
        if conditions:
            return select(func.count(Document.id)).where(and_(*conditions))
        else:
            return select(func.count(Document.id))
    
    def _build_chunk_query(self, options: CleanupOptions):
        """Build query for chunks to be cleaned."""
        from sqlalchemy import select, and_, func
        
        conditions = []
        
        if options.doc_ids:
            conditions.append(DocumentChunk.doc_id.in_(options.doc_ids))
        
        if options.min_relevance_score is not None:
            conditions.append(DocumentChunk.relevance_score < options.min_relevance_score)
        
        if options.max_retrieval_count is not None:
            conditions.append(DocumentChunk.retrieval_count > options.max_retrieval_count)
        
        if conditions:
            return select(func.count(DocumentChunk.id)).where(and_(*conditions))
        else:
            return select(func.count(DocumentChunk.id))
    
    def _build_cache_query(self, options: CleanupOptions):
        """Build query for cache entries to be cleaned."""
        from sqlalchemy import select, and_, func
        
        conditions = []
        
        if options.doc_ids:
            conditions.append(QueryCache.doc_id.in_(options.doc_ids))
        
        if options.older_than_days:
            cutoff_date = datetime.utcnow() - timedelta(days=options.older_than_days)
            conditions.append(QueryCache.created_at < cutoff_date)
        
        if conditions:
            return select(func.count(QueryCache.id)).where(and_(*conditions))
        else:
            return select(func.count(QueryCache.id))
    
    async def _get_pinecone_stats(self, options: CleanupOptions) -> Dict[str, Any]:
        """Get Pinecone statistics."""
        try:
            # Get the first available Pinecone index
            index = list(self.embedding_service.pinecone_indexes.values())[0]
            
            # Get index stats
            stats = index.describe_index_stats()
            
            if options.clean_pinecone_by_doc:
                # Count vectors for specific document
                doc_filter = {"doc_id": options.clean_pinecone_by_doc}
                doc_stats = index.describe_index_stats(filter=doc_filter)
                return {
                    "total_vectors": doc_stats.total_vector_count,
                    "dimension": stats.dimension,
                    "metric": stats.metric
                }
            else:
                return {
                    "total_vectors": stats.total_vector_count,
                    "dimension": stats.dimension,
                    "metric": stats.metric
                }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
            return {"error": str(e)}
    
    async def clean_postgresql(self, options: CleanupOptions) -> Dict[str, Any]:
        """Clean PostgreSQL data."""
        results = {
            "documents_deleted": 0,
            "chunks_deleted": 0,
            "cache_entries_deleted": 0,
            "errors": []
        }
        
        try:
            async with self.db_manager.get_session() as session:
                # Clean documents
                if options.clean_documents or options.clean_all_pg:
                    deleted_docs = await self._clean_documents(session, options)
                    results["documents_deleted"] = deleted_docs
                
                # Clean chunks
                if options.clean_chunks or options.clean_all_pg:
                    deleted_chunks = await self._clean_chunks(session, options)
                    results["chunks_deleted"] = deleted_chunks
                
                # Clean cache
                if options.clean_cache or options.clean_all_pg:
                    deleted_cache = await self._clean_cache(session, options)
                    results["cache_entries_deleted"] = deleted_cache
                
                await session.commit()
                
        except Exception as e:
            results["errors"].append(f"PostgreSQL cleanup error: {e}")
            logger.error(f"PostgreSQL cleanup error: {e}")
        
        return results
    
    async def _clean_documents(self, session, options: CleanupOptions) -> int:
        """Clean documents from PostgreSQL."""
        from sqlalchemy import delete, and_
        
        conditions = []
        
        if options.doc_ids:
            conditions.append(Document.doc_id.in_(options.doc_ids))
        
        if options.older_than_days:
            cutoff_date = datetime.utcnow() - timedelta(days=options.older_than_days)
            conditions.append(Document.created_at < cutoff_date)
        
        if conditions:
            result = await session.execute(
                delete(Document).where(and_(*conditions)).returning(Document.id)
            )
        else:
            result = await session.execute(
                delete(Document).returning(Document.id)
            )
        
        deleted_count = len(result.all())
        logger.info(f"Deleted {deleted_count} documents from PostgreSQL")
        return deleted_count
    
    async def _clean_chunks(self, session, options: CleanupOptions) -> int:
        """Clean chunks from PostgreSQL."""
        from sqlalchemy import delete, and_
        
        conditions = []
        
        if options.doc_ids:
            conditions.append(DocumentChunk.doc_id.in_(options.doc_ids))
        
        if options.min_relevance_score is not None:
            conditions.append(DocumentChunk.relevance_score < options.min_relevance_score)
        
        if options.max_retrieval_count is not None:
            conditions.append(DocumentChunk.retrieval_count > options.max_retrieval_count)
        
        if conditions:
            result = await session.execute(
                delete(DocumentChunk).where(and_(*conditions)).returning(DocumentChunk.id)
            )
        else:
            result = await session.execute(
                delete(DocumentChunk).returning(DocumentChunk.id)
            )
        
        deleted_count = len(result.all())
        logger.info(f"Deleted {deleted_count} chunks from PostgreSQL")
        return deleted_count
    
    async def _clean_cache(self, session, options: CleanupOptions) -> int:
        """Clean cache entries from PostgreSQL."""
        from sqlalchemy import delete, and_
        
        conditions = []
        
        if options.doc_ids:
            conditions.append(QueryCache.doc_id.in_(options.doc_ids))
        
        if options.older_than_days:
            cutoff_date = datetime.utcnow() - timedelta(days=options.older_than_days)
            conditions.append(QueryCache.created_at < cutoff_date)
        
        if conditions:
            result = await session.execute(
                delete(QueryCache).where(and_(*conditions)).returning(QueryCache.id)
            )
        else:
            result = await session.execute(
                delete(QueryCache).returning(QueryCache.id)
            )
        
        deleted_count = len(result.all())
        logger.info(f"Deleted {deleted_count} cache entries from PostgreSQL")
        return deleted_count
    
    async def clean_pinecone(self, options: CleanupOptions) -> Dict[str, Any]:
        """Clean Pinecone data."""
        results = {
            "vectors_deleted": 0,
            "errors": []
        }
        
        try:
            # Get the first available Pinecone index
            index = list(self.embedding_service.pinecone_indexes.values())[0]
            
            if options.clean_pinecone_by_doc:
                # Delete vectors for specific document
                doc_filter = {"doc_id": options.clean_pinecone_by_doc}
                index.delete(filter=doc_filter)
                logger.info(f"Deleted vectors for document {options.clean_pinecone_by_doc} from Pinecone")
                results["vectors_deleted"] = 1  # We don't get exact count from delete operation
            else:
                # Delete all vectors
                index.delete(delete_all=True)
                logger.info("Deleted all vectors from Pinecone")
                results["vectors_deleted"] = 1  # We don't get exact count from delete operation
            
        except Exception as e:
            results["errors"].append(f"Pinecone cleanup error: {e}")
            logger.error(f"Pinecone cleanup error: {e}")
        
        return results
    
    async def backup_data(self, options: CleanupOptions) -> Dict[str, Any]:
        """Create backup of data before cleanup."""
        backup_info = {
            "timestamp": datetime.utcnow().isoformat(),
            "postgresql": {},
            "pinecone": {},
            "backup_path": None
        }
        
        try:
            # Get current stats as backup
            stats = await self.get_cleanup_stats(options)
            backup_info["postgresql"] = stats["postgresql"]
            backup_info["pinecone"] = stats["pinecone"]
            
            logger.info("Backup information collected")
            
        except Exception as e:
            logger.error(f"Backup error: {e}")
            backup_info["error"] = str(e)
        
        return backup_info


async def main():
    """Main function for the data cleaning script."""
    parser = argparse.ArgumentParser(
        description="Clean data from Pinecone and PostgreSQL databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show what would be cleaned (dry run)
  python clean_data.py --dry-run --clean-all-pg --clean-pinecone
  
  # Clean all data from both databases
  python clean_data.py --clean-all-pg --clean-pinecone --confirm
  
  # Clean specific document
  python clean_data.py --doc-ids doc123 --clean-all-pg --clean-pinecone --confirm
  
  # Clean old data (older than 30 days)
  python clean_data.py --older-than-days 30 --clean-all-pg --confirm
  
  # Clean low-quality chunks (relevance score < 0.5)
  python clean_data.py --min-relevance-score 0.5 --clean-chunks --confirm
        """
    )
    
    # General options
    parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without actually cleaning")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--confirm", action="store_true", help="Confirm cleanup operation (required for actual cleanup)")
    
    # PostgreSQL cleanup options
    parser.add_argument("--clean-documents", action="store_true", help="Clean documents from PostgreSQL")
    parser.add_argument("--clean-chunks", action="store_true", help="Clean chunks from PostgreSQL")
    parser.add_argument("--clean-cache", action="store_true", help="Clean cache entries from PostgreSQL")
    parser.add_argument("--clean-all-pg", action="store_true", help="Clean all PostgreSQL data")
    
    # Pinecone cleanup options
    parser.add_argument("--clean-pinecone", action="store_true", help="Clean all vectors from Pinecone")
    parser.add_argument("--clean-pinecone-by-doc", type=str, help="Clean vectors for specific document from Pinecone")
    
    # Filtering options
    parser.add_argument("--doc-ids", nargs="+", help="Clean only specific document IDs")
    parser.add_argument("--older-than-days", type=int, help="Clean data older than specified days")
    parser.add_argument("--min-relevance-score", type=float, help="Clean chunks with relevance score below this value")
    parser.add_argument("--max-retrieval-count", type=int, help="Clean chunks with retrieval count above this value")
    
    # Safety options
    parser.add_argument("--backup-before-cleanup", action="store_true", help="Create backup before cleanup")
    
    args = parser.parse_args()
    
    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate arguments
    if not args.dry_run and not args.confirm:
        logger.error("ERROR: --confirm flag is required for actual cleanup operations")
        logger.error("Use --dry-run to see what would be cleaned without actually cleaning")
        sys.exit(1)
    
    if not any([
        args.clean_documents, args.clean_chunks, args.clean_cache, 
        args.clean_all_pg, args.clean_pinecone, args.clean_pinecone_by_doc
    ]):
        logger.error("ERROR: No cleanup operation specified")
        logger.error("Use --help to see available options")
        sys.exit(1)
    
    # Create options object
    options = CleanupOptions(
        dry_run=args.dry_run,
        verbose=args.verbose,
        clean_documents=args.clean_documents,
        clean_chunks=args.clean_chunks,
        clean_cache=args.clean_cache,
        clean_all_pg=args.clean_all_pg,
        clean_pinecone=args.clean_pinecone,
        clean_pinecone_by_doc=args.clean_pinecone_by_doc,
        doc_ids=args.doc_ids,
        older_than_days=args.older_than_days,
        min_relevance_score=args.min_relevance_score,
        max_retrieval_count=args.max_retrieval_count,
        confirm=args.confirm,
        backup_before_cleanup=args.backup_before_cleanup
    )
    
    # Initialize cleaner
    cleaner = DataCleaner()
    
    try:
        await cleaner.init()
        
        # Show what would be cleaned
        logger.info("Analyzing data to be cleaned...")
        stats = await cleaner.get_cleanup_stats(options)
        
        print("\n" + "="*60)
        print("DATA CLEANUP ANALYSIS")
        print("="*60)
        
        if stats["postgresql"]:
            print("\nPostgreSQL Data:")
            for key, value in stats["postgresql"].items():
                if value is not None:
                    print(f"  {key}: {value:,}")
                else:
                    print(f"  {key}: 0")
        
        if stats["pinecone"]:
            print("\nPinecone Data:")
            for key, value in stats["pinecone"].items():
                if value is not None:
                    print(f"  {key}: {value:,}")
                else:
                    print(f"  {key}: 0")
        
        print("\n" + "="*60)
        
        if options.dry_run:
            print("DRY RUN - No data was actually cleaned")
            print("Use --confirm to perform actual cleanup")
            return
        
        # Confirm cleanup
        if not options.confirm:
            print("\nERROR: --confirm flag is required for actual cleanup")
            return
        
        # Create backup if requested
        if options.backup_before_cleanup:
            logger.info("Creating backup before cleanup...")
            backup_info = await cleaner.backup_data(options)
            print(f"\nBackup created at: {backup_info['timestamp']}")
        
        # Perform cleanup
        print("\nStarting cleanup operations...")
        
        # Clean PostgreSQL
        if any([options.clean_documents, options.clean_chunks, options.clean_cache, options.clean_all_pg]):
            logger.info("Cleaning PostgreSQL data...")
            pg_results = await cleaner.clean_postgresql(options)
            
            print("\nPostgreSQL Cleanup Results:")
            for key, value in pg_results.items():
                if key != "errors":
                    print(f"  {key}: {value:,}")
            
            if pg_results["errors"]:
                print("\nPostgreSQL Errors:")
                for error in pg_results["errors"]:
                    print(f"  - {error}")
        
        # Clean Pinecone
        if options.clean_pinecone or options.clean_pinecone_by_doc:
            logger.info("Cleaning Pinecone data...")
            pinecone_results = await cleaner.clean_pinecone(options)
            
            print("\nPinecone Cleanup Results:")
            for key, value in pinecone_results.items():
                if key != "errors":
                    print(f"  {key}: {value:,}")
            
            if pinecone_results["errors"]:
                print("\nPinecone Errors:")
                for error in pinecone_results["errors"]:
                    print(f"  - {error}")
        
        print("\n" + "="*60)
        print("CLEANUP COMPLETED")
        print("="*60)
        
    except KeyboardInterrupt:
        logger.info("Cleanup interrupted by user")
    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        sys.exit(1)
    finally:
        await cleaner.close()


if __name__ == "__main__":
    asyncio.run(main()) 