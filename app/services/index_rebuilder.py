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

        # Extract text content and metadata from chunks
        texts = [c.chunk_text for c in chunks]
        metadatas = [
            {
                'text': c.chunk_text,
                'doc_id': c.document_id,
                'chunk_id': c.id,
                'page_num': getattr(c, 'page_num', None)  # if page info exists
            } 
            for c in chunks
        ]

        # Get the singleton vector store instance
        vector_store = VectorStore.get_instance()
        
        # Clear existing index and rebuild from database content
        vector_store.clear()
        
        if texts:
            # Generate embeddings for all texts at once
            print(f"Generating embeddings for {len(texts)} texts...")
            logging.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = AIService.get_embeddings(texts)
            
            if embeddings and len(embeddings) > 0:
                print(f"Received embeddings of length: {len(embeddings)}")
                logging.info(f"Received embeddings of length: {len(embeddings)}")
                
                # Add documents to vector store
                vector_store.add_documents(embeddings, metadatas)
                
                # Save the rebuilt index to Supabase storage
                vector_store.save_index('vector_index')
                
                print(f"✅ Rebuilt index with {len(texts)} vectors")
                logging.info(f"Successfully rebuilt vector index with {len(texts)} vectors")
                
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
                print("❌ Failed to generate embeddings for document chunks")
                logging.error("Failed to generate embeddings for document chunks")
        else:
            print("⚠️ No text content to index")
            logging.info("No text content found to index")

    except Exception as e:
        print(f"❌ Error rebuilding index from DB: {e}")
        logging.error(f"Error rebuilding vector index from database: {e}", exc_info=True)
        raise
