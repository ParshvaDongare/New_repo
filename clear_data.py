#!/usr/bin/env python3
"""
Data clearing script for PostgreSQL and Pinecone.
Clears all data while preserving table structure.
"""

import asyncio
import logging
import sys
from typing import Dict, Any

from sqlalchemy import text
from pinecone import Pinecone

from config import settings
from services.db import DatabaseManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataClearer:
    """Handles data clearing while preserving schema."""
    
    def __init__(self):
        self.db_manager = DatabaseManager()
        self.pinecone_client = None
    
    async def init_pinecone(self) -> None:
        """Initialize Pinecone client."""
        try:
            self.pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
            logger.info("Pinecone client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def clear_postgresql_data(self) -> None:
        """Clear all data from PostgreSQL tables while preserving schema."""
        try:
            logger.info("Starting PostgreSQL data clearing...")
            
            # Initialize database manager
            await self.db_manager.init()
            
            async with self.db_manager.get_session() as session:
                # Get current counts
                result = await session.execute(text("SELECT COUNT(*) FROM document_chunks"))
                chunk_count = result.scalar()
                
                result = await session.execute(text("SELECT COUNT(*) FROM documents"))
                doc_count = result.scalar()
                
                result = await session.execute(text("SELECT COUNT(*) FROM query_cache"))
                cache_count = result.scalar()
                
                logger.info(f"Current data counts:")
                logger.info(f"  - Document chunks: {chunk_count}")
                logger.info(f"  - Documents: {doc_count}")
                logger.info(f"  - Query cache entries: {cache_count}")
                
                if chunk_count == 0 and doc_count == 0 and cache_count == 0:
                    logger.info("Database is already empty")
                    return
                
                # Clear all tables
                logger.info("Clearing all tables...")
                
                await session.execute(text("DELETE FROM document_chunks"))
                logger.info("  ✅ Cleared document_chunks")
                
                await session.execute(text("DELETE FROM documents"))
                logger.info("  ✅ Cleared documents")
                
                await session.execute(text("DELETE FROM query_cache"))
                logger.info("  ✅ Cleared query_cache")
                
                # Reset sequences
                logger.info("Resetting sequences...")
                try:
                    await session.execute(text("ALTER SEQUENCE IF EXISTS document_chunks_id_seq RESTART WITH 1"))
                    logger.info("  ✅ Reset document_chunks sequence")
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not reset document_chunks sequence: {e}")
                
                try:
                    await session.execute(text("ALTER SEQUENCE IF EXISTS query_cache_id_seq RESTART WITH 1"))
                    logger.info("  ✅ Reset query_cache sequence")
                except Exception as e:
                    logger.warning(f"  ⚠️ Could not reset query_cache sequence: {e}")
                
                await session.commit()
                logger.info("PostgreSQL data cleared successfully")
                
        except Exception as e:
            logger.error(f"Error clearing PostgreSQL data: {e}")
            raise
    
    async def clear_pinecone_data(self) -> None:
        """Clear all data from Pinecone index."""
        try:
            logger.info("Starting Pinecone data clearing...")
            
            # Initialize Pinecone client
            await self.init_pinecone()
            
            # Get the index
            index = self.pinecone_client.Index(settings.pinecone_index_name)
            
            # Get index stats
            try:
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                logger.info(f"Found {total_vectors} vectors in index")
                
                if total_vectors == 0:
                    logger.info("Pinecone index is already empty")
                    return
                
                # Delete all vectors
                logger.info("Deleting all vectors from Pinecone index...")
                index.delete(delete_all=True)
                logger.info("All vectors deleted from Pinecone index")
                    
            except Exception as stats_error:
                logger.warning(f"Could not get index stats: {stats_error}")
                # Try to delete all anyway
                try:
                    index.delete(delete_all=True)
                    logger.info("All vectors deleted from Pinecone index")
                except Exception as delete_error:
                    logger.error(f"Error deleting vectors: {delete_error}")
                    raise
            
            logger.info("Pinecone data cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing Pinecone data: {e}")
            raise
    
    async def verify_clearance(self) -> Dict[str, Any]:
        """Verify that all data has been cleared."""
        results = {
            "postgresql": {"cleared": False, "error": None},
            "pinecone": {"cleared": False, "error": None}
        }
        
        # Check PostgreSQL
        try:
            async with self.db_manager.get_session() as session:
                # Check documents table
                result = await session.execute(text("SELECT COUNT(*) FROM documents"))
                doc_count = result.scalar()
                
                # Check chunks table
                result = await session.execute(text("SELECT COUNT(*) FROM document_chunks"))
                chunk_count = result.scalar()
                
                # Check cache table
                result = await session.execute(text("SELECT COUNT(*) FROM query_cache"))
                cache_count = result.scalar()
                
                if doc_count == 0 and chunk_count == 0 and cache_count == 0:
                    results["postgresql"]["cleared"] = True
                    logger.info("PostgreSQL verification: All tables are empty")
                else:
                    results["postgresql"]["error"] = f"Tables not empty: docs={doc_count}, chunks={chunk_count}, cache={cache_count}"
                    logger.warning(f"PostgreSQL verification failed: {results['postgresql']['error']}")
                    
        except Exception as e:
            results["postgresql"]["error"] = str(e)
            logger.error(f"PostgreSQL verification error: {e}")
        
        # Check Pinecone
        try:
            index = self.pinecone_client.Index(settings.pinecone_index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            
            if total_vectors == 0:
                results["pinecone"]["cleared"] = True
                logger.info("Pinecone verification: Index is empty")
            else:
                results["pinecone"]["error"] = f"Index not empty: {total_vectors} vectors found"
                logger.warning(f"Pinecone verification failed: {results['pinecone']['error']}")
                
        except Exception as e:
            results["pinecone"]["error"] = str(e)
            logger.error(f"Pinecone verification error: {e}")
        
        return results
    
    async def clear_all_data(self) -> Dict[str, Any]:
        """Clear all data from both PostgreSQL and Pinecone."""
        logger.info("Starting complete data clearing process...")
        
        try:
            # Clear PostgreSQL data
            await self.clear_postgresql_data()
            
            # Clear Pinecone data
            await self.clear_pinecone_data()
            
            # Verify clearance
            verification = await self.verify_clearance()
            
            # Close connections
            await self.db_manager.close()
            
            logger.info("Data clearing process completed")
            return verification
            
        except Exception as e:
            logger.error(f"Error during data clearing: {e}")
            raise


async def main():
    """Main function to clear data."""
    try:
        clearer = DataClearer()
        results = await clearer.clear_all_data()
        
        # Print results
        print("\n" + "="*50)
        print("DATA CLEARING RESULTS")
        print("="*50)
        
        for db_name, result in results.items():
            status = "✅ CLEARED" if result["cleared"] else "❌ FAILED"
            print(f"{db_name.upper()}: {status}")
            if result["error"]:
                print(f"  Error: {result['error']}")
        
        print("="*50)
        
        # Check if both are cleared
        all_cleared = all(result["cleared"] for result in results.values())
        
        if all_cleared:
            print("🎉 All data cleared successfully!")
            print("Your database schema is preserved and ready for fresh data.")
        else:
            print("⚠️  Some data may not be fully cleared.")
            print("Please check the errors above and try again if needed.")
        
        return 0 if all_cleared else 1
        
    except Exception as e:
        logger.error(f"Fatal error during data clearing: {e}")
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 