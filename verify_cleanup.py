#!/usr/bin/env python3
"""
Simple verification script to check database status after cleanup.
"""

import asyncio
import logging
from sqlalchemy import text
from pinecone import Pinecone

from config import settings
from services.db import DatabaseManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def verify_databases():
    """Verify the current state of both databases."""
    
    print("🔍 VERIFYING DATABASE STATUS")
    print("=" * 50)
    
    # Check PostgreSQL
    print("\n📊 POSTGRESQL STATUS:")
    try:
        db_manager = DatabaseManager()
        await db_manager.init()
        
        async with db_manager.get_session() as session:
            # Check table counts
            tables = ['documents', 'document_chunks', 'query_cache']
            for table in tables:
                try:
                    result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    count = result.scalar()
                    print(f"  ✅ {table}: {count} records")
                except Exception as e:
                    print(f"  ❌ {table}: Error - {e}")
        
        await db_manager.close()
        print("  ✅ PostgreSQL connection successful")
        
    except Exception as e:
        print(f"  ❌ PostgreSQL error: {e}")
    
    # Check Pinecone
    print("\n🌲 PINECONE STATUS:")
    try:
        pc = Pinecone(api_key=settings.pinecone_api_key)
        
        # List all indexes
        indexes = pc.list_indexes()
        print(f"  📋 Available indexes: {indexes}")
        
        if settings.pinecone_index_name in indexes:
            index = pc.Index(settings.pinecone_index_name)
            stats = index.describe_index_stats()
            total_vectors = stats.get('total_vector_count', 0)
            print(f"  ✅ Index '{settings.pinecone_index_name}': {total_vectors} vectors")
        else:
            print(f"  ⚠️  Index '{settings.pinecone_index_name}' not found")
            
    except Exception as e:
        print(f"  ❌ Pinecone error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ VERIFICATION COMPLETE")


if __name__ == "__main__":
    asyncio.run(verify_databases()) 