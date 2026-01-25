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
        from app.services.index_rebuilder import rebuild_index_from_db  # Import the new rebuilder
        
        # Initialize vector store with Supabase persistent storage
        vector_store = VectorStore.get_instance()  # Use singleton instance
        index_name = 'vector_index'
        
        # Rebuild index from database on startup (this handles Render's ephemeral filesystem)
        print("🔄 Starting vector index rebuild from database...")
        logging.info("🔄 Starting vector index rebuild from database...")
        try:
            rebuild_index_from_db()
            print("✅ Vector index rebuild completed successfully")
            logging.info("✅ Vector index rebuild completed successfully")
        except Exception as e:
            print(f"❌ Vector index rebuild failed: {e}")
            logging.error(f"❌ Vector index rebuild failed: {e}", exc_info=True)
            # Continue anyway - the vector store will be empty but the app should still start
        
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
