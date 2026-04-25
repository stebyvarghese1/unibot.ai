from app.services.vector_store import VectorStore
from app.models import DocumentChunk
from app import db
from app.utils.background_tasks import TaskTracker
import logging

def rebuild_index_from_db():
    """Rebuild the vector index from database documents on app startup"""
    print("🔄 Rebuilding vector index from database...")
    logging.info("Starting vector index rebuild from database")
    
    task_name = "rebuild"
    TaskTracker.update_progress(task_name, 0, 100, "Initializing...")

    try:
        # Query all document chunks from the database
        total_chunks = DocumentChunk.query.count()
        chunks = DocumentChunk.query.all()

        if not chunks:
            print("⚠️ No chunks found in DB")
            logging.info("No document chunks found in database")
            TaskTracker.complete_task(task_name, "No data to rebuild")
            return

        TaskTracker.update_progress(task_name, 0, total_chunks, f"Found {total_chunks} chunks. Mapping metadata...")
        
        # 2. Extract text content and metadata from chunks
        from app.models import Document
        doc_map = {d.id: d for d in Document.query.all()}
        
        # 3. Get the singleton vector store instance
        vector_store = VectorStore.get_instance()
        vector_store.clear()

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
                TaskTracker.update_progress(task_name, total_processed, total_chunks, "Re-indexing vectors...")
                current_batch_texts = []
                current_batch_metas = []
        
        # Final batch
        if current_batch_texts:
            vector_store.add_texts(current_batch_texts, current_batch_metas)
            total_processed += len(current_batch_texts)
        
        TaskTracker.complete_task(task_name, f"Successfully rebuilt {total_processed} chunks")
        logging.info(f"Successfully rebuilt vector index. {total_processed} chunks processed.")
        
    except Exception as e:
        logging.error(f"Error rebuilding vector index from database: {e}", exc_info=True)
        TaskTracker.complete_task(task_name, f"Error: {str(e)}")
        raise