from flask import Flask
from flask_cors import CORS
from flask_compress import Compress
from flask_sqlalchemy import SQLAlchemy
from config import Config
import threading
import time
import os
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
from prometheus_flask_exporter import PrometheusMetrics

db = SQLAlchemy()
limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
metrics = PrometheusMetrics.for_app_factory()

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    CORS(app)
    Compress(app) # Gzip compression for all JSON responses
    
    # Initialize Prometheus Metrics
    metrics.init_app(app)
    
    # Initialize Rate Limiter with Redis if available
    if app.config.get('REDIS_URL'):
        app.config['RATELIMIT_STORAGE_URI'] = app.config['REDIS_URL']
    limiter.init_app(app)
    
    # Initialize Sentry if DSN is provided
    if app.config.get('SENTRY_DSN'):
        sentry_sdk.init(
            dsn=app.config['SENTRY_DSN'],
            integrations=[FlaskIntegration()],
            traces_sample_rate=1.0,
            profiles_sample_rate=1.0,
        )
    
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
                if 'show_tour' not in cols:
                    db.session.execute(text("ALTER TABLE users ADD COLUMN show_tour INTEGER DEFAULT 1"))
                if 'is_active' not in cols:
                    db.session.execute(text("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"))
                
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
                    AND column_name IN ('pref_course', 'pref_semester', 'pref_subject', 'show_tour', 'is_active')
                """)
                existing_users_cols = [r[0] for r in db.session.execute(check_sql).fetchall()]
                
                # Only run ALTER if columns are missing
                if len(existing_users_cols) < 5:
                    print("🛠 Adding missing user preference columns...")
                    db.session.execute(text("SET LOCAL statement_timeout = 60000")) # 60s timeout for migration
                    if 'pref_course' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS pref_course VARCHAR(100)"))
                    if 'pref_semester' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS pref_semester VARCHAR(20)"))
                    if 'pref_subject' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS pref_subject VARCHAR(100)"))
                    if 'show_tour' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS show_tour BOOLEAN DEFAULT TRUE"))
                    if 'is_active' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE"))
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
                
                # Check for feedback column in chat_messages
                check_msg_sql = text("""
                    SELECT 1 FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'chat_messages' AND column_name = 'feedback'
                """)
                if not db.session.execute(check_msg_sql).first():
                    print("🛠 Adding missing feedback column to chat_messages...")
                    db.session.execute(text("ALTER TABLE public.chat_messages ADD COLUMN IF NOT EXISTS feedback VARCHAR(20)"))
                    db.session.commit()
                
                print("✅ Schema verification complete")
                
                # --- Supabase pgvector Setup (Must happen before vector store init) ---
                try:
                    print("🚀 Checking Supabase pgvector extension and tables...")
                    # 1. Enable extension
                    db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
                    
                    # 2. Create embeddings table
                    db.session.execute(text('''
                        CREATE TABLE IF NOT EXISTS public.embeddings (
                            id BIGSERIAL PRIMARY KEY,
                            content TEXT,
                            metadata JSONB,
                            embedding VECTOR(384)
                        )
                    '''))
                    
                    # 3. Create search function (idempotent CREATE OR REPLACE)
                    db.session.execute(text('''
                        CREATE OR REPLACE FUNCTION match_documents (
                            query_embedding VECTOR(384),
                            match_threshold FLOAT,
                            match_count INT
                        )
                        RETURNS TABLE (
                            id BIGINT,
                            content TEXT,
                            metadata JSONB,
                            similarity FLOAT
                        )
                        LANGUAGE plpgsql
                        AS $$
                        BEGIN
                            RETURN QUERY
                            SELECT
                                embeddings.id,
                                embeddings.content,
                                embeddings.metadata,
                                1 - (embeddings.embedding <=> query_embedding) AS similarity
                            FROM embeddings
                            WHERE 1 - (embeddings.embedding <=> query_embedding) > match_threshold
                            ORDER BY embeddings.embedding <=> query_embedding
                            LIMIT match_count;
                        END;
                        $$;
                    '''))
                    
                    # 4. Create Index
                    db.session.execute(text('CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx ON public.embeddings USING hnsw (embedding vector_cosine_ops)'))
                    
                    # 5. 🔥 CRITICAL: Refresh PostgREST schema cache so Supabase client sees the new table immediately
                    try:
                        db.session.execute(text("NOTIFY pgrst, 'reload_schema'"))
                    except Exception:
                        pass
                        
                    db.session.commit()
                    print("✅ Supabase pgvector setup complete")
                except Exception as se:
                    db.session.rollback()
                    print(f"❌ Supabase setup error: {se}")

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
        
        # --- Persistent Vector Store Setup ---
        # With Supabase, we DON'T need to rebuild on every startup as it's persistently stored in the DB.
        # Clearing and rebuilding on startup is slow, expensive (API calls), and can cause data loss.
        print("🔄 Checking vector store status...")
        from app.services.vector_store import VectorStore
        vector_store = VectorStore.get_instance()
        
        try:
            current_stats = vector_store.get_stats()
            print(f"📊 Vector store stats: {current_stats}")
            if current_stats['total_vectors'] == 0:
                print("⚠️ WARNING: Vector store is empty! Use Admin panel to sync documents.")
                logging.warning("Vector store is empty on startup.")
            else:
                print(f"✅ Vector store ready with {current_stats['total_vectors']} vectors")
        except Exception as e:
            print(f"⚠️ Vector store check failed: {e}")
            logging.warning(f"Vector store check failed: {e}")
        
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

# Expose app for Gunicorn/Render compatibility with 'app:app' entry point
app = create_app()
