"""
Script to fix missing doc_id column in document_chunks table.
This adds the column if it doesn't exist and ensures proper constraints.
"""

import asyncio
import logging
from sqlalchemy import text
from services.db import db_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def fix_missing_doc_id_column():
    """Add missing doc_id column to document_chunks table."""
    try:
        # Initialize database connection
        await db_manager.init()
        
        async with db_manager.get_session() as session:
            # Check if column exists
            check_query = text("""
                SELECT column_name 
                FROM information_schema.columns 
                WHERE table_name = 'document_chunks' 
                AND column_name = 'doc_id'
            """)
            
            result = await session.execute(check_query)
            column_exists = result.rowcount > 0
            
            if not column_exists:
                logger.info("Column doc_id does not exist. Adding it now...")
                
                # Add the column
                add_column_query = text("""
                    ALTER TABLE document_chunks 
                    ADD COLUMN doc_id VARCHAR(255)
                """)
                await session.execute(add_column_query)
                logger.info("Added doc_id column")
                
                # Add foreign key constraint
                add_fk_query = text("""
                    ALTER TABLE document_chunks
                    ADD CONSTRAINT fk_document_chunks_doc_id
                    FOREIGN KEY (doc_id)
                    REFERENCES documents(doc_id)
                    ON DELETE CASCADE
                """)
                await session.execute(add_fk_query)
                logger.info("Added foreign key constraint")
                
                # Add index
                add_index_query = text("""
                    CREATE INDEX idx_document_chunks_doc_id 
                    ON document_chunks(doc_id)
                """)
                await session.execute(add_index_query)
                logger.info("Added index on doc_id")
                
                await session.commit()
                logger.info("Successfully fixed missing doc_id column!")
                
            else:
                logger.info("Column doc_id already exists. No action needed.")
            
            # Verify the fix
            verify_query = text("""
                SELECT 
                    column_name, 
                    data_type, 
                    is_nullable
                FROM information_schema.columns
                WHERE table_name = 'document_chunks'
                ORDER BY ordinal_position
            """)
            
            result = await session.execute(verify_query)
            columns = result.fetchall()
            
            logger.info("\nCurrent document_chunks table structure:")
            for col in columns:
                logger.info(f"  - {col.column_name}: {col.data_type} (nullable: {col.is_nullable})")
                
    except Exception as e:
        logger.error(f"Error fixing missing column: {e}")
        raise
    finally:
        await db_manager.close()


if __name__ == "__main__":
    asyncio.run(fix_missing_doc_id_column())
