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

        # Robust Database Migrations and Indexing
        try:
            from sqlalchemy import text
            eng = db.session.get_bind()
            dialect = eng.dialect.name
            
            # 1. Performance Indexes (CRITICAL)
            if 'postgresql' in dialect:
                print("🚀 Ensuring database indexes for high performance...")
                try:
                    # Index for fast retrieval from document chunks
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON public.document_chunks (document_id)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_document_chunks_comp_search ON public.document_chunks (document_id, chunk_index)"))
                    
                    # Indexes for user chat history and session management
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_user_id ON public.chat_messages (user_id)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON public.chat_messages (created_at DESC)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON public.chat_sessions (user_id)"))
                    
                    # Indexes for filtered document lookups
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_filters ON public.documents (course, semester, subject)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON public.documents (uploaded_by)"))
                    
                    db.session.commit()
                    print("✅ Performance indexes verified")
                except Exception as ex:
                    db.session.rollback()
                    print(f"⚠️ Indexing warning: {ex}")

            # 2. Schema Migrations (Conditional to avoid locks)
            if 'sqlite' in dialect:
                # SQLite fallback
                res = db.session.execute(text("PRAGMA table_info(users)")).fetchall()
                cols = [r[1] for r in res]
                if 'pref_course' not in cols:
                    db.session.execute(text("ALTER TABLE users ADD COLUMN pref_course TEXT"))
                    db.session.execute(text("ALTER TABLE users ADD COLUMN pref_semester TEXT"))
                    db.session.execute(text("ALTER TABLE users ADD COLUMN pref_subject TEXT"))
                
                res = db.session.execute(text("PRAGMA table_info(documents)")).fetchall()
                if 'doc_type' not in [r[1] for r in res]:
                    db.session.execute(text("ALTER TABLE documents ADD COLUMN doc_type TEXT DEFAULT 'syllabus'"))
                db.session.commit()
            elif 'postgresql' in dialect:
                print("🔄 Verifying PostgreSQL schema...")
                # Check for column existence first to avoid expensive/blocking ALTER TABLE locks
                check_sql = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'users' 
                    AND column_name IN ('pref_course', 'pref_semester', 'pref_subject')
                """)
                existing_users_cols = [r[0] for r in db.session.execute(check_sql).fetchall()]
                
                # Only run ALTER if columns are missing
                if len(existing_users_cols) < 3:
                    print("🛠 Adding missing user preference columns...")
                    db.session.execute(text("SET LOCAL statement_timeout = 60000")) # 60s timeout for migration
                    if 'pref_course' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS pref_course VARCHAR(100)"))
                    if 'pref_semester' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS pref_semester VARCHAR(20)"))
                    if 'pref_subject' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS pref_subject VARCHAR(100)"))
                    db.session.commit()
                
                # Check for doc_type
                check_doc_sql = text("""
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'documents' AND column_name = 'doc_type'
                """)
                if not db.session.execute(check_doc_sql).first():
                    print("🛠 Adding missing doc_type column...")
                    db.session.execute(text("ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS doc_type VARCHAR(50) DEFAULT 'syllabus'"))
                    db.session.commit()

                # Check for parent_id in filter_options
                check_filter_sql = text("""
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'filter_options' AND column_name = 'parent_id'
                """)
                if not db.session.execute(check_filter_sql).first():
                    print("🛠 Adding missing parent_id column...")
                    db.session.execute(text("ALTER TABLE public.filter_options ADD COLUMN IF NOT EXISTS parent_id INTEGER REFERENCES public.filter_options(id)"))
                    db.session.commit()
                
                print("✅ Schema verification complete")

        except Exception as e:
            db.session.rollback()
            print(f"⚠️ Startup maintenance warning: {e}")
            logging.warning(f"Startup maintenance warning: {e}")

        admin = User.query.filter_by(email=Config.ADMIN_EMAIL).first()
        if not admin:
            db.session.add(User(email=Config.ADMIN_EMAIL, password_hash=generate_password_hash(Config.ADMIN_PASSWORD), role='admin'))
            db.session.commit()
        
        # Register Blueprints
        print("🔄 Registering blueprints...")
        app.register_blueprint(routes.bp)
        print("✅ Blueprints registered")
        
        # Automatic index rebuild on startup with persistent storage
        print("🔄 Initializing vector store...")
        from app.services.vector_store import VectorStore
        from app.models import DocumentChunk
        from app.services.ai_service import AIService
        from app.routes import sync_storage
        from app.services.index_rebuilder import rebuild_index_from_db  # Import the new rebuilder
        
        # Initialize vector store with Supabase persistent storage
        print("🔄 Getting vector store instance...")
        vector_store = VectorStore.get_instance()  # Use singleton instance
        print("✅ Vector store instance ready")
        index_name = 'vector_index'
        
        # Check if we need to rebuild (only if index is empty)
        print("🔄 Checking vector store stats...")
        current_stats = vector_store.get_stats()
        print(f"📊 Current stats: {current_stats}")
        if current_stats['total_vectors'] == 0:
            # Rebuild index from database on startup (this handles Render's ephemeral filesystem)
            print("🔄 Starting vector index rebuild from database...")
            logging.info("🔄 Starting vector index rebuild from database...")
            try:
                # First try to load existing index from Supabase storage
                from app.services.vector_store import VectorStore
                vector_store = VectorStore.get_instance()
                # Skip loading from Supabase since we're rebuilding in-memory on each startup for Render compatibility
                print("🔄 Rebuilding vector index from database for Render compatibility...")
                logging.info("Rebuilding vector index from database for Render compatibility...")
                print("🔄 Calling rebuild_index_from_db...")
                rebuild_index_from_db()
                print("✅ rebuild_index_from_db completed")
                
                # Final validation
                final_stats = vector_store.get_stats()
                print(f"📊 Final vector store stats: {final_stats}")
                logging.info(f"📊 Final vector store stats: {final_stats}")
                
                # Log for debugging purposes
                if final_stats['total_vectors'] == 0:
                    print("❌ CRITICAL: Vector store has 0 vectors after rebuild - chat will not work!")
                    logging.critical("CRITICAL: Vector store has 0 vectors after rebuild - chat will not work!")
                else:
                    print(f"✅ Vector store is ready with {final_stats['total_vectors']} vectors")
                    logging.info(f"✅ Vector store is ready with {final_stats['total_vectors']} vectors")
                    
            except Exception as e:
                print(f"❌ Vector index rebuild failed: {e}")
                logging.error(f"❌ Vector index rebuild failed: {e}", exc_info=True)
                # Continue anyway - the vector store will be empty but the app should still start
        else:
            print(f"✅ Vector store already has {current_stats['total_vectors']} vectors, skipping rebuild")
            logging.info(f"✅ Vector store already has {current_stats['total_vectors']} vectors, skipping rebuild")
        
        # Start background workers (Web Source Auto-Refresh) - delayed start to not block app startup
        try:
            from app.services.web_source_refresher import WebSourceRefresher
            # Start with longer initial delay to let main app initialize
            import threading
            import time
            
            def delayed_worker_start():
                time.sleep(10)  # Wait 10 seconds for app to fully start
                try:
                    WebSourceRefresher.start_worker(app)
                    print("🚀 Web Source Auto-Refresher started.")
                    logging.info("Web Source Auto-Refresher started.")
                except Exception as e:
                    print(f"❌ Failed to start WebSourceRefresher: {e}")
                    logging.error(f"❌ Failed to start WebSourceRefresher: {e}")
            
            # Start in background thread
            worker_thread = threading.Thread(target=delayed_worker_start, daemon=True)
            worker_thread.start()
            print("⏰ Web Source Auto-Refresher scheduled to start in 10 seconds...")
            logging.info("Web Source Auto-Refresher scheduled to start in 10 seconds...")
            
        except Exception as e:
            print(f"❌ Failed to schedule WebSourceRefresher: {e}")
            logging.error(f"❌ Failed to schedule WebSourceRefresher: {e}")

    return app