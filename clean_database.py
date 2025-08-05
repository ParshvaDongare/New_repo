#!/usr/bin/env python3
"""
Comprehensive database cleanup script for PostgreSQL and Pinecone.
Deletes all data AND tables/schema completely.
"""

import asyncio
import logging
import sys
from typing import Dict, Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from pinecone import Pinecone

from config import settings
from services.db import DatabaseManager, Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabaseCleaner:
    """Handles complete database cleanup including table deletion."""
    
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
    
    async def get_postgresql_stats(self) -> Dict[str, int]:
        """Get current PostgreSQL table statistics."""
        stats = {}
        try:
            async with self.db_manager.get_session() as session:
                # Check if tables exist and get counts
                tables = ['documents', 'document_chunks', 'query_cache']
                
                for table in tables:
                    try:
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        stats[table] = count
                    except Exception as e:
                        logger.warning(f"Could not get count for {table}: {e}")
                        stats[table] = 0
                
                # Check if tables exist
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('documents', 'document_chunks', 'query_cache')
                """))
                existing_tables = [row[0] for row in result.fetchall()]
                stats['existing_tables'] = existing_tables
                
        except Exception as e:
            logger.error(f"Error getting PostgreSQL stats: {e}")
            stats = {}
        
        return stats
    
    async def drop_postgresql_tables(self) -> None:
        """Drop all PostgreSQL tables and recreate them."""
        try:
            logger.info("Starting PostgreSQL table deletion...")
            
            # Initialize database manager
            await self.db_manager.init()
            
            # Get current stats
            stats = await self.get_postgresql_stats()
            logger.info(f"Current PostgreSQL state:")
            for table, count in stats.items():
                if table != 'existing_tables':
                    logger.info(f"  - {table}: {count} records")
            
            existing_tables = stats.get('existing_tables', [])
            if not existing_tables:
                logger.info("No tables found to drop")
                return
            
            logger.info(f"Found {len(existing_tables)} tables to drop: {existing_tables}")
            
            # Drop all tables using SQLAlchemy
            async with self.db_manager._engine.begin() as conn:
                # Drop all tables defined in our models
                await conn.run_sync(Base.metadata.drop_all)
                logger.info("✅ Dropped all SQLAlchemy-defined tables")
                
                # Also drop any additional tables that might exist
                for table in existing_tables:
                    try:
                        await conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
                        logger.info(f"  ✅ Dropped table: {table}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Could not drop {table}: {e}")
                
                # Drop sequences
                sequences = ['document_chunks_id_seq', 'query_cache_id_seq']
                for seq in sequences:
                    try:
                        await conn.execute(text(f"DROP SEQUENCE IF EXISTS {seq} CASCADE"))
                        logger.info(f"  ✅ Dropped sequence: {seq}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Could not drop sequence {seq}: {e}")
            
            logger.info("PostgreSQL tables dropped successfully")
            
        except Exception as e:
            logger.error(f"Error dropping PostgreSQL tables: {e}")
            raise
    
    async def recreate_postgresql_tables(self) -> None:
        """Recreate PostgreSQL tables with fresh schema."""
        try:
            logger.info("Recreating PostgreSQL tables...")
            
            async with self.db_manager._engine.begin() as conn:
                # Create all tables from scratch
                await conn.run_sync(Base.metadata.create_all)
                logger.info("✅ Recreated all tables with fresh schema")
                
                # Create additional indexes
                try:
                    await conn.execute(text("""
                        CREATE INDEX IF NOT EXISTS idx_chunks_text_search 
                        ON document_chunks USING gin(to_tsvector('english', full_text))
                    """))
                    logger.info("✅ Created text search index")
                except Exception as e:
                    logger.warning(f"⚠️ Could not create text search index: {e}")
            
            logger.info("PostgreSQL tables recreated successfully")
            
        except Exception as e:
            logger.error(f"Error recreating PostgreSQL tables: {e}")
            raise
    
    async def delete_pinecone_index(self) -> None:
        """Delete the entire Pinecone index."""
        try:
            logger.info("Starting Pinecone index deletion...")
            
            # Initialize Pinecone client
            await self.init_pinecone()
            
            # Get index stats before deletion
            try:
                index = self.pinecone_client.Index(settings.pinecone_index_name)
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                logger.info(f"Found {total_vectors} vectors in index")
            except Exception as e:
                logger.warning(f"Could not get index stats: {e}")
                total_vectors = 0
            
            # Delete the entire index
            try:
                self.pinecone_client.delete_index(settings.pinecone_index_name)
                logger.info(f"✅ Deleted Pinecone index: {settings.pinecone_index_name}")
            except Exception as e:
                if "NOT_FOUND" in str(e) or "404" in str(e):
                    logger.info(f"✅ Pinecone index {settings.pinecone_index_name} was already deleted")
                else:
                    logger.warning(f"Could not delete index: {e}")
                    # Try to clear all vectors instead
                    try:
                        index = self.pinecone_client.Index(settings.pinecone_index_name)
                        index.delete(delete_all=True)
                        logger.info("✅ Cleared all vectors from Pinecone index")
                    except Exception as clear_error:
                        if "NOT_FOUND" in str(clear_error) or "404" in str(clear_error):
                            logger.info("✅ Pinecone index was already empty")
                        else:
                            logger.error(f"Error clearing vectors: {clear_error}")
                            raise
            
            logger.info("Pinecone index deletion completed")
            
        except Exception as e:
            logger.error(f"Error deleting Pinecone index: {e}")
            raise
    
    async def recreate_pinecone_index(self) -> None:
        """Recreate the Pinecone index with fresh configuration."""
        try:
            logger.info("Recreating Pinecone index...")
            
            # Check if index already exists
            existing_indexes = self.pinecone_client.list_indexes()
            index_name = settings.pinecone_index_name
            
            if index_name in existing_indexes:
                logger.info(f"Index {index_name} already exists, skipping recreation")
                return
            
            # Create new index
            self.pinecone_client.create_index(
                name=index_name,
                dimension=settings.pinecone_dimension,
                metric=settings.pinecone_metric,
                spec=dict(
                    serverless=dict(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
            )
            logger.info(f"✅ Created new Pinecone index: {index_name}")
            
            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            import time
            time.sleep(10)  # Give it more time to initialize
            
            logger.info("Pinecone index recreated successfully")
            
        except Exception as e:
            logger.error(f"Error recreating Pinecone index: {e}")
            raise
    
    async def verify_cleanup(self) -> Dict[str, Any]:
        """Verify that all data and tables have been cleaned."""
        results = {
            "postgresql": {"cleaned": False, "error": None},
            "pinecone": {"cleaned": False, "error": None}
        }
        
        # Check PostgreSQL
        try:
            async with self.db_manager.get_session() as session:
                # Check if tables exist
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name IN ('documents', 'document_chunks', 'query_cache')
                """))
                existing_tables = [row[0] for row in result.fetchall()]
                
                if len(existing_tables) == 3:  # All tables should exist after recreation
                    # Check if tables are empty
                    for table in existing_tables:
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        if count > 0:
                            results["postgresql"]["error"] = f"Table {table} has {count} records"
                            break
                    else:
                        results["postgresql"]["cleaned"] = True
                        logger.info("PostgreSQL verification: All tables exist and are empty")
                else:
                    results["postgresql"]["error"] = f"Expected 3 tables, found {len(existing_tables)}: {existing_tables}"
                    logger.warning(f"PostgreSQL verification failed: {results['postgresql']['error']}")
                    
        except Exception as e:
            results["postgresql"]["error"] = str(e)
            logger.error(f"PostgreSQL verification error: {e}")
        
        # Check Pinecone
        try:
            existing_indexes = self.pinecone_client.list_indexes()
            index_name = settings.pinecone_index_name
            
            if index_name in existing_indexes:
                # Check if index is empty
                index = self.pinecone_client.Index(index_name)
                stats = index.describe_index_stats()
                total_vectors = stats.get('total_vector_count', 0)
                
                if total_vectors == 0:
                    results["pinecone"]["cleaned"] = True
                    logger.info("Pinecone verification: Index exists and is empty")
                else:
                    results["pinecone"]["error"] = f"Index has {total_vectors} vectors"
                    logger.warning(f"Pinecone verification failed: {results['pinecone']['error']}")
            else:
                results["pinecone"]["error"] = f"Index {index_name} not found"
                logger.warning(f"Pinecone verification failed: {results['pinecone']['error']}")
                
        except Exception as e:
            results["pinecone"]["error"] = str(e)
            logger.error(f"Pinecone verification error: {e}")
        
        return results
    
    async def clean_all_databases(self) -> Dict[str, Any]:
        """Complete cleanup of both PostgreSQL and Pinecone databases."""
        logger.info("Starting complete database cleanup process...")
        
        try:
            # Step 1: Drop PostgreSQL tables
            await self.drop_postgresql_tables()
            
            # Step 2: Delete Pinecone index
            await self.delete_pinecone_index()
            
            # Step 3: Recreate PostgreSQL tables
            await self.recreate_postgresql_tables()
            
            # Step 4: Recreate Pinecone index
            await self.recreate_pinecone_index()
            
            # Step 5: Verify cleanup
            verification = await self.verify_cleanup()
            
            # Close connections
            await self.db_manager.close()
            
            logger.info("Database cleanup process completed")
            return verification
            
        except Exception as e:
            logger.error(f"Error during database cleanup: {e}")
            raise


async def main():
    """Main function to clean databases."""
    try:
        cleaner = DatabaseCleaner()
        results = await cleaner.clean_all_databases()
        
        # Print results
        print("\n" + "="*60)
        print("DATABASE CLEANUP RESULTS")
        print("="*60)
        
        for db_name, result in results.items():
            status = "✅ CLEANED" if result["cleaned"] else "❌ FAILED"
            print(f"{db_name.upper()}: {status}")
            if result["error"]:
                print(f"  Error: {result['error']}")
        
        print("="*60)
        
        # Check if both are cleaned
        all_cleaned = all(result["cleaned"] for result in results.values())
        
        if all_cleaned:
            print("🎉 All databases cleaned successfully!")
            print("✅ PostgreSQL: Tables dropped and recreated")
            print("✅ Pinecone: Index deleted and recreated")
            print("\nYour databases are now completely fresh and ready for new data.")
        else:
            print("⚠️  Some databases may not be fully cleaned.")
            print("Please check the errors above and try again if needed.")
        
        return 0 if all_cleaned else 1
        
    except Exception as e:
        logger.error(f"Fatal error during database cleanup: {e}")
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 