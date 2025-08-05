-- Check if the column exists first
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'document_chunks'
        AND column_name = 'doc_id'
    ) THEN
        -- Add the doc_id column
        ALTER TABLE document_chunks
        ADD COLUMN doc_id VARCHAR(255);
        
        -- Add the foreign key constraint
        ALTER TABLE document_chunks
        ADD CONSTRAINT fk_document_chunks_doc_id
        FOREIGN KEY (doc_id)
        REFERENCES documents(doc_id)
        ON DELETE CASCADE;
        
        -- Add index for performance
        CREATE INDEX idx_document_chunks_doc_id ON document_chunks(doc_id);
        
        RAISE NOTICE 'Column doc_id added successfully to document_chunks table';
    ELSE
        RAISE NOTICE 'Column doc_id already exists in document_chunks table';
    END IF;
END $$;

-- Verify the column was added
SELECT 
    column_name, 
    data_type, 
    is_nullable, 
    column_default
FROM information_schema.columns
WHERE table_name = 'document_chunks'
AND column_name = 'doc_id';
