from app.services.vector_store import VectorStore
from app.models import DocumentChunk
from app import db
from app.services.ai_service import AIService
import logging

def rebuild_index_from_db():
    """Rebuild the vector index from database documents on app startup"""
    print("🔄 Rebuilding vector index from database...")
    logging.info("Starting vector index rebuild from database")

    try:
        # Query all document chunks from the database
        chunks = DocumentChunk.query.all()

        if not chunks:
            print("⚠️ No chunks found in DB")
            logging.info("No document chunks found in database")
            return

        print(f"Found {len(chunks)} chunks in database")
        logging.info(f"Found {len(chunks)} document chunks to index")

        # 2. Extract text content and metadata from chunks using a stream
        # This prevents OOM errors for large databases
        from app.models import Document
        doc_map = {d.id: d for d in Document.query.all()}
        
        # 3. Get the singleton vector store instance
        vector_store = VectorStore.get_instance()
        vector_store.clear()

        print(f"Adding chunks to vector store in batches...")
        logging.info(f"Adding chunks to vector store in batches...")
        
        BATCH_SIZE = 64
        current_batch_texts = []
        current_batch_metas = []
        total_processed = 0
        
        # Use yield_per for memory-efficient streaming from DB
        for c in DocumentChunk.query.yield_per(100):
            current_batch_texts.append(c.chunk_text)
            current_batch_metas.append({
                'text': c.chunk_text,
                'doc_id': c.document_id,
                'chunk_id': c.id,
                'doc_type': doc_map[c.document_id].doc_type if c.document_id in doc_map else 'syllabus',
                'filename': doc_map[c.document_id].filename if c.document_id in doc_map else None
            })
            
            if len(current_batch_texts) >= BATCH_SIZE:
                vector_store.add_texts(current_batch_texts, current_batch_metas)
                total_processed += len(current_batch_texts)
                print(f"Progress: {total_processed} chunks processed...")
                current_batch_texts = []
                current_batch_metas = []
        
        # Final batch
        if current_batch_texts:
            vector_store.add_texts(current_batch_texts, current_batch_metas)
            total_processed += len(current_batch_texts)
        
        print(f"✅ Rebuilt index. Processed {total_processed} chunks successfully.")
        logging.info(f"Successfully rebuilt vector index. {total_processed} chunks processed.")
        
        # Log final stats
        stats = vector_store.get_stats()
        print(f"📊 Final vector store stats: {stats}")
        logging.info(f"Final vector store stats: {stats}")

    except Exception as e:
        print(f"❌ Error rebuilding index from DB: {e}")
        logging.error(f"Error rebuilding vector index from database: {e}", exc_info=True)
        raise