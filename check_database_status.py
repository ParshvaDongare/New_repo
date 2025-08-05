#!/usr/bin/env python3
"""
Database status checker for PostgreSQL and Pinecone.
Shows current data counts and system health.
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


class DatabaseStatusChecker:
    """Checks status of both PostgreSQL and Pinecone databases."""
    
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
    
    async def check_postgresql_status(self) -> Dict[str, Any]:
        """Check PostgreSQL database status."""
        try:
            logger.info("Checking PostgreSQL status...")
            
            # Initialize database manager
            await self.db_manager.init()
            
            async with self.db_manager.get_session() as session:
                # Check if tables exist
                result = await session.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                """))
                tables = [row[0] for row in result]
                
                status = {
                    "connected": True,
                    "tables": tables,
                    "data_counts": {},
                    "error": None
                }
                
                # Get data counts for each table
                for table in tables:
                    try:
                        result = await session.execute(text(f"SELECT COUNT(*) FROM {table}"))
                        count = result.scalar()
                        status["data_counts"][table] = count
                    except Exception as e:
                        status["data_counts"][table] = f"Error: {e}"
                
                # Get additional stats
                try:
                    # Check document stats
                    result = await session.execute(text("""
                        SELECT 
                            COUNT(*) as total_docs,
                            COUNT(CASE WHEN is_fully_indexed = true THEN 1 END) as indexed_docs,
                            AVG(total_pages) as avg_pages,
                            AVG(total_chunks) as avg_chunks
                        FROM documents
                    """))
                    doc_stats = result.one()
                    status["document_stats"] = {
                        "total_documents": doc_stats.total_docs or 0,
                        "fully_indexed": doc_stats.indexed_docs or 0,
                        "avg_pages_per_doc": float(doc_stats.avg_pages or 0),
                        "avg_chunks_per_doc": float(doc_stats.avg_chunks or 0)
                    }
                except Exception as e:
                    status["document_stats"] = {"error": str(e)}
                
                logger.info("PostgreSQL status check completed")
                return status
                
        except Exception as e:
            logger.error(f"Error checking PostgreSQL status: {e}")
            return {
                "connected": False,
                "tables": [],
                "data_counts": {},
                "document_stats": {},
                "error": str(e)
            }
    
    async def check_pinecone_status(self) -> Dict[str, Any]:
        """Check Pinecone database status."""
        try:
            logger.info("Checking Pinecone status...")
            
            # Initialize Pinecone client
            await self.init_pinecone()
            
            # Get the index
            index = self.pinecone_client.Index(settings.pinecone_index_name)
            
            # Get index stats
            try:
                stats = index.describe_index_stats()
                status = {
                    "connected": True,
                    "index_name": settings.pinecone_index_name,
                    "total_vectors": stats.get('total_vector_count', 0),
                    "index_dimension": stats.get('dimension', 0),
                    "index_metric": stats.get('metric', 'unknown'),
                    "namespaces": stats.get('namespaces', {}),
                    "error": None
                }
                
                # Get namespace details
                namespaces = stats.get('namespaces', {})
                if namespaces:
                    status["namespace_details"] = {}
                    for ns_name, ns_stats in namespaces.items():
                        status["namespace_details"][ns_name] = {
                            "vector_count": ns_stats.get('vector_count', 0)
                        }
                
                logger.info("Pinecone status check completed")
                return status
                
            except Exception as stats_error:
                logger.error(f"Error getting Pinecone stats: {stats_error}")
                return {
                    "connected": False,
                    "index_name": settings.pinecone_index_name,
                    "total_vectors": 0,
                    "index_dimension": 0,
                    "index_metric": "unknown",
                    "namespaces": {},
                    "error": str(stats_error)
                }
                
        except Exception as e:
            logger.error(f"Error checking Pinecone status: {e}")
            return {
                "connected": False,
                "index_name": settings.pinecone_index_name,
                "total_vectors": 0,
                "index_dimension": 0,
                "index_metric": "unknown",
                "namespaces": {},
                "error": str(e)
            }
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            # Check PostgreSQL
            postgresql_status = await self.check_postgresql_status()
            
            # Check Pinecone
            pinecone_status = await self.check_pinecone_status()
            
            # Determine overall health
            postgresql_healthy = postgresql_status["connected"] and not postgresql_status["error"]
            pinecone_healthy = pinecone_status["connected"] and not pinecone_status["error"]
            
            overall_health = "healthy" if postgresql_healthy and pinecone_healthy else "unhealthy"
            
            return {
                "overall_health": overall_health,
                "postgresql": postgresql_status,
                "pinecone": pinecone_status,
                "timestamp": asyncio.get_event_loop().time()
            }
            
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            return {
                "overall_health": "error",
                "postgresql": {"connected": False, "error": str(e)},
                "pinecone": {"connected": False, "error": str(e)},
                "timestamp": asyncio.get_event_loop().time()
            }
        finally:
            # Close connections
            await self.db_manager.close()


async def main():
    """Main function to check database status."""
    try:
        checker = DatabaseStatusChecker()
        health = await checker.get_system_health()
        
        # Print results
        print("\n" + "="*60)
        print("DATABASE STATUS REPORT")
        print("="*60)
        
        # Overall health
        health_icon = "🟢" if health["overall_health"] == "healthy" else "🔴"
        print(f"Overall System Health: {health_icon} {health['overall_health'].upper()}")
        print()
        
        # PostgreSQL Status
        pg = health["postgresql"]
        pg_icon = "🟢" if pg["connected"] else "🔴"
        print(f"PostgreSQL: {pg_icon} {'CONNECTED' if pg['connected'] else 'DISCONNECTED'}")
        
        if pg["connected"]:
            print(f"  Tables found: {len(pg['tables'])}")
            for table in pg["tables"]:
                count = pg["data_counts"].get(table, 0)
                print(f"    - {table}: {count} records")
            
            if "document_stats" in pg and "error" not in pg["document_stats"]:
                stats = pg["document_stats"]
                print(f"  Document Statistics:")
                print(f"    - Total documents: {stats['total_documents']}")
                print(f"    - Fully indexed: {stats['fully_indexed']}")
                print(f"    - Avg pages per doc: {stats['avg_pages_per_doc']:.1f}")
                print(f"    - Avg chunks per doc: {stats['avg_chunks_per_doc']:.1f}")
        
        if pg["error"]:
            print(f"  Error: {pg['error']}")
        
        print()
        
        # Pinecone Status
        pc = health["pinecone"]
        pc_icon = "🟢" if pc["connected"] else "🔴"
        print(f"Pinecone: {pc_icon} {'CONNECTED' if pc['connected'] else 'DISCONNECTED'}")
        
        if pc["connected"]:
            print(f"  Index: {pc['index_name']}")
            print(f"  Total vectors: {pc['total_vectors']}")
            print(f"  Dimension: {pc['index_dimension']}")
            print(f"  Metric: {pc['index_metric']}")
            
            if pc["namespaces"]:
                print(f"  Namespaces:")
                for ns_name, ns_stats in pc["namespaces"].items():
                    print(f"    - {ns_name}: {ns_stats.get('vector_count', 0)} vectors")
        
        if pc["error"]:
            print(f"  Error: {pc['error']}")
        
        print("="*60)
        
        # Summary
        total_data = sum(pg["data_counts"].values()) if pg["connected"] else 0
        total_vectors = pc["total_vectors"] if pc["connected"] else 0
        
        print(f"Summary:")
        print(f"  - PostgreSQL records: {total_data}")
        print(f"  - Pinecone vectors: {total_vectors}")
        
        if total_data == 0 and total_vectors == 0:
            print("  - Status: Empty databases ready for fresh data")
        elif health["overall_health"] == "healthy":
            print("  - Status: System is healthy and operational")
        else:
            print("  - Status: System has issues that need attention")
        
        return 0 if health["overall_health"] == "healthy" else 1
        
    except Exception as e:
        logger.error(f"Fatal error during status check: {e}")
        print(f"❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code) 