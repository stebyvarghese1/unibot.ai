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
        docs = Document.query.all()
        # Store primitive values in dict to prevent SQLAlchemy lazy-loading after commits expire objects
        doc_map = {d.id: {
            'doc_type': d.doc_type, 
            'filename': d.filename,
            'course': d.course,
            'semester': d.semester,
            'subject': d.subject
        } for d in docs}
        
        # 3. Get the singleton vector store instance
        vector_store = VectorStore.get_instance()
        vector_store.clear()

        BATCH_SIZE = 64
        current_batch_texts = []
        current_batch_metas = []
        total_processed = 0
        
        # Iterate over the already-loaded chunks list
        # We avoid yield_per because db.session.commit() inside add_texts will kill server-side cursors
        for c in chunks:
            current_batch_texts.append(c.chunk_text)
            
            doc_info = doc_map.get(c.document_id, {'doc_type': 'syllabus', 'filename': None, 'course': None, 'semester': None, 'subject': None})
            current_batch_metas.append({
                'text': c.chunk_text,
                'doc_id': c.document_id,
                'chunk_id': c.id,
                'doc_type': doc_info['doc_type'],
                'filename': doc_info['filename'],
                'course': doc_info['course'],
                'semester': doc_info['semester'],
                'subject': doc_info['subject']
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
        
        # 4. Re-generate Intelligence Grounding (Syllabus Unit Summaries)
        # These are virtual vectors extracted from structure_json
        import json
        logging.info("🧠 Restoring Intelligence Grounding from syllabus structures...")
        TaskTracker.update_progress(task_name, total_processed, total_chunks, "Grounding syllabus intelligence...")
        
        unit_texts = []
        unit_metas = []
        
        for d in docs:
            if d.structure_json:
                try:
                    structure_data = json.loads(d.structure_json)
                    if 'units' in structure_data:
                        for unit in structure_data['units']:
                            title = unit.get('title', 'Unknown Unit')
                            topics = ", ".join(unit.get('topics', []))
                            unit_summary = f"UNIT SYLLABUS: {title}\nTOPICS: {topics}"
                            
                            unit_texts.append(unit_summary)
                            unit_metas.append({
                                'text': unit_summary,
                                'doc_id': d.id,
                                'filename': d.filename,
                                'doc_type': 'unit_summary',
                                'course': d.course,
                                'semester': d.semester,
                                'subject': d.subject,
                                'unit_title': title
                            })
                            
                            if len(unit_texts) >= BATCH_SIZE:
                                vector_store.add_texts(unit_texts, unit_metas)
                                unit_texts = []
                                unit_metas = []
                except Exception as je:
                    logging.error(f"Failed to parse structure for doc {d.id}: {je}")
        
        if unit_texts:
            vector_store.add_texts(unit_texts, unit_metas)

        TaskTracker.complete_task(task_name, f"Successfully rebuilt {total_processed} chunks and grounded syllabus maps")
        logging.info(f"Successfully rebuilt vector index. {total_processed} chunks processed.")
        
    except Exception as e:
        logging.error(f"Error rebuilding vector index from database: {e}", exc_info=True)
        TaskTracker.complete_task(task_name, f"Error: {str(e)}")
        raise
