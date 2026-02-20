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

        # 1. Fetch documents for metadata mapping
        from app.models import Document
        doc_map = {d.id: d for d in Document.query.all()}

        # 2. Extract text content and metadata from chunks in a single pass
        texts = []
        metadatas = []
        for c in chunks:
            texts.append(c.chunk_text)
            metadatas.append({
                'text': c.chunk_text,
                'doc_id': c.document_id,
                'chunk_id': c.id,
                'doc_type': doc_map[c.document_id].doc_type if c.document_id in doc_map else 'syllabus',
                'filename': doc_map[c.document_id].filename if c.document_id in doc_map else None,
                'page_num': getattr(c, 'page_num', None)
            })

        # 3. Get the singleton vector store instance
        vector_store = VectorStore.get_instance()
        
        # Clear existing index and rebuild from database content
        vector_store.clear()
        
        if texts:
            total_chunks = len(texts)
            print(f"Adding {total_chunks} texts to vector store in batches...")
            logging.info(f"Adding {total_chunks} texts to vector store in batches...")
            
            # Optimized batch processing with larger batches and progress tracking
            BATCH_SIZE = 64  # Increased from 32 for better throughput
            successful_batches = 0
            failed_batches = 0
            
            # Process in larger batches for better performance
            for i in range(0, total_chunks, BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                batch_metas = metadatas[i:i + BATCH_SIZE]
                
                try:
                    # Use the new add_texts method which handles embedding internally
                    vector_store.add_texts(batch_texts, batch_metas)
                    successful_batches += 1
                    
                    # Progress reporting every 5 batches instead of every batch
                    if (i // BATCH_SIZE) % 5 == 0 or i + BATCH_SIZE >= total_chunks:
                        progress = min(i + BATCH_SIZE, total_chunks)
                        print(f"Progress: {progress}/{total_chunks} chunks processed ({successful_batches} batches successful)")
                        logging.info(f"Progress: {progress}/{total_chunks} chunks processed")
                        
                except Exception as e:
                    failed_batches += 1
                    logging.error(f"Failed to process batch ending at index {i + BATCH_SIZE}: {e}")
                    # Continue with remaining batches instead of stopping
                    continue
            
            print(f"✅ Rebuilt index. Processed {successful_batches} batches successfully, {failed_batches} failed.")
            logging.info(f"Successfully rebuilt vector index. {successful_batches} batches successful, {failed_batches} failed.")
            
            # Log final stats - THIS IS CRITICAL FOR DEBUGGING
            stats = vector_store.get_stats()
            print(f"📊 Final vector store stats: {stats}")
            logging.info(f"Final vector store stats: {stats}")
            
            # Double-check that the index is actually populated
            if stats['total_vectors'] > 0:
                print(f"✅ Vector store is ready with {stats['total_vectors']} vectors")
                logging.info(f"✅ Vector store is ready with {stats['total_vectors']} vectors")
            else:
                print("❌ WARNING: Vector store has 0 vectors after rebuild!")
                logging.warning("❌ WARNING: Vector store has 0 vectors after rebuild!")
        else:
            print("⚠️ No text content to index")
            logging.info("No text content found to index")

    except Exception as e:
        print(f"❌ Error rebuilding index from DB: {e}")
        logging.error(f"Error rebuilding vector index from database: {e}", exc_info=True)
        raise