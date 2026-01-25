from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from config import Config
import threading
import time
import os
import logging

db = SQLAlchemy()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    CORS(app)
    db.init_app(app)
    
    with app.app_context():
        from app import routes, models
        db.create_all()
        from app.models import User
        from werkzeug.security import generate_password_hash
        from config import Config
        admin = User.query.filter_by(email=Config.ADMIN_EMAIL).first()
        if not admin:
            db.session.add(User(email=Config.ADMIN_EMAIL, password_hash=generate_password_hash(Config.ADMIN_PASSWORD), role='admin'))
            db.session.commit()
        
        # Register Blueprints
        app.register_blueprint(routes.bp)
        
        # Automatic index rebuild on startup with persistent storage
        from app.services.vector_store import VectorStore
        from app.models import DocumentChunk
        from app.services.ai_service import AIService
        from app.routes import sync_storage
        
        # Initialize vector store with Supabase persistent storage
        vector_store = VectorStore.get_instance()  # Use singleton instance
        index_name = 'vector_index'
        
        # Try to load existing index first from Supabase storage
        logging.info("Attempting to load existing vector index from Supabase storage...")
        if vector_store.index_exists(index_name):
            logging.info("Vector index files found in Supabase storage, attempting to load...")
            try:
                vector_store.load_index(index_name)
                print("Loaded existing vector index from Supabase storage")
                logging.info(f"Successfully loaded vector index. Stats: {vector_store.get_stats()}")
            except Exception as e:
                logging.error(f"Could not load existing index from Supabase: {e}")
                print(f"Could not load existing index from Supabase, will rebuild: {e}")
        else:
            logging.info("No vector index files found in Supabase storage")
        
        # Rebuild index if AUTO_REBUILD_INDEX is true and API token is available
        if Config.AUTO_REBUILD_INDEX and Config.HUGGINGFACE_API_TOKEN:
            logging.info("Starting index rebuild process...")
            try:
                # Optional: sync storage into DB before rebuilding index
                try:
                    sync_storage()
                except Exception as e:
                    logging.warning(f"Storage sync skipped: {e}")
                    print(f"Storage sync skipped: {e}")
                chunks = DocumentChunk.query.all()
                logging.info(f"Found {len(chunks)} document chunks to index")
                if chunks:
                    # Clear existing index and rebuild
                    vector_store.clear()
                    texts = [c.chunk_text for c in chunks]
                    logging.info(f"Generating embeddings for {len(texts)} text chunks...")
                    embeddings = AIService.get_embeddings(texts)
                    metadata = [{'text': c.chunk_text, 'doc_id': c.document_id} for c in chunks]
                    vector_store.add_documents(embeddings, metadata)
                    
                    # Save the rebuilt index to Supabase storage
                    vector_store.save_index(index_name)
                    print(f"Rebuilt and saved vector index to Supabase storage with {len(chunks)} chunks")
                    logging.info(f"Vector store stats after rebuild: {vector_store.get_stats()}")
            except Exception as e:
                logging.error(f"Index rebuild failed: {e}")
                print(f"Index rebuild skipped: {e}")
        
        # Background storage auto-sync (disabled in containerized environments like Render)
        # This feature can cause issues in containerized environments where long-running threads
        # are not supported or managed properly
        # if Config.AUTO_SYNC_STORAGE:
        #     def _sync_loop():
        #         while True:
        #             try:
        #                 with app.app_context():
        #                     from app.routes import sync_storage
        #                     sync_storage()
        #             except Exception as e:
        #                 print(f"Storage auto-sync error: {e}")
        #             time.sleep(Config.SYNC_STORAGE_INTERVAL)
        #     try:
        #         t = threading.Thread(target=_sync_loop, daemon=True)
        #         t.start()
        #     except Exception as e:
        #         print(f"Failed to start storage auto-sync thread: {e}"))
        
    return app
