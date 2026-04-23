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
from flask import session, make_response
from flask_wtf.csrf import CSRFProtect, generate_csrf
from flask_talisman import Talisman

db = SQLAlchemy()
csrf = CSRFProtect()
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://", # Default to memory for now to avoid complexity with DB drivers
    strategy="fixed-window"
)

# Admin Exemption: Admins are never rate limited
@limiter.request_filter
def admin_whitelist():
    return session.get('role') == 'admin'

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    CORS(app)
    Compress(app) # Gzip compression for all JSON responses
    db.init_app(app)
    
    # Security Middleware
    csrf.init_app(app)
    
    # Configure Talisman for HSTS and Force HTTPS
    # content_security_policy=None to avoid breaking dynamic AI content for now
    Talisman(app, 
             force_https=app.config.get('FORCE_HTTPS', False), 
             strict_transport_security=True,
             session_cookie_secure=app.config.get('SESSION_COOKIE_SECURE', False),
             content_security_policy=None)
    
    # Synchronize CSRF token to a cookie for frontend Fetch/XHR requests
    @app.after_request
    def set_csrf_cookie(response):
        response.set_cookie('csrf_token', generate_csrf(), 
                          samesite=app.config.get('SESSION_COOKIE_SAMESITE', 'Lax'),
                          secure=app.config.get('SESSION_COOKIE_SECURE', False))
        return response

    # Configure Limiter with App Config
    limiter_defaults = app.config.get('RATELIMIT_DEFAULT', "1000 per day; 200 per hour")
    limiter.init_app(app)
    
    # We set default limits dynamically if specified in config
    if limiter_defaults:
        app.config.setdefault("RATELIMIT_DEFAULTS", limiter_defaults)
    if app.config.get('RATELIMIT_STORAGE_URL'):
        # For production use with gunicorn, memory storage isn't shared. 
        # For simplicity in this environment, memory is fine, but we'll try to use DB if it's there.
        try:
            # Simple check to see if we should try DB storage
            if 'postgresql' in app.config['RATELIMIT_STORAGE_URL']:
                # Flask-Limiter uses different URI format for sqlalchemy
                uri = app.config['RATELIMIT_STORAGE_URL'].replace('postgresql://', 'sqlalchemy+postgresql://')
                # However, sqlalchemy storage in flask-limiter can be tricky with some drivers.
                # Sticking to memory for now as requested 'fully proper' might mean reliability first.
                pass 
        except Exception:
            pass

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
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON public.chat_messages (session_id)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_messages_created_at ON public.chat_messages (created_at DESC)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON public.chat_sessions (user_id)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON public.chat_sessions (updated_at DESC)"))
                    
                    # Indexes for dropdowns and filters
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_filter_options_category ON public.filter_options (category)"))
                    db.session.execute(text("CREATE INDEX IF NOT EXISTS idx_filter_options_parent_id ON public.filter_options (parent_id)"))
                    
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
                    db.session.execute(text("ALTER TABLE users ADD COLUMN preferred_name TEXT"))
                    db.session.execute(text("ALTER TABLE users ADD COLUMN pref_course TEXT"))
                    db.session.execute(text("ALTER TABLE users ADD COLUMN pref_semester TEXT"))
                    db.session.execute(text("ALTER TABLE users ADD COLUMN pref_subject TEXT"))
                if 'show_tour' not in cols:
                    db.session.execute(text("ALTER TABLE users ADD COLUMN show_tour INTEGER DEFAULT 1"))
                if 'is_active' not in cols:
                    db.session.execute(text("ALTER TABLE users ADD COLUMN is_active INTEGER DEFAULT 1"))
                
                res = db.session.execute(text("PRAGMA table_info(documents)")).fetchall()
                doc_cols = [r[1] for r in res]
                if 'doc_type' not in doc_cols:
                    db.session.execute(text("ALTER TABLE documents ADD COLUMN doc_type TEXT DEFAULT 'syllabus'"))
                if 'structure_json' not in doc_cols:
                    db.session.execute(text("ALTER TABLE documents ADD COLUMN structure_json TEXT"))
                db.session.commit()
            elif 'postgresql' in dialect:
                print("🔄 Verifying PostgreSQL schema...")
                # Check for column existence first to avoid expensive/blocking ALTER TABLE locks
                check_sql = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'users' 
                    AND column_name IN ('preferred_name', 'pref_course', 'pref_semester', 'pref_subject', 'show_tour', 'is_active')
                """)
                existing_users_cols = [r[0] for r in db.session.execute(check_sql).fetchall()]
                
                # Ensure password_hash is nullable (for social logins)
                db.session.execute(text("ALTER TABLE public.users ALTER COLUMN password_hash DROP NOT NULL"))
                db.session.commit()

                # Only run ALTER if columns are missing
                if len(existing_users_cols) < 6:
                    print("🛠 Adding missing user preference columns...")
                    db.session.execute(text("SET LOCAL statement_timeout = 60000")) # 60s timeout for migration
                    
                    if 'preferred_name' not in existing_users_cols:
                        db.session.execute(text("ALTER TABLE public.users ADD COLUMN IF NOT EXISTS preferred_name VARCHAR(100)"))
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
                
                # Check for doc_type and structure_json
                check_doc_sql = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_schema = 'public' AND table_name = 'documents' AND column_name IN ('doc_type', 'structure_json')
                """)
                existing_doc_cols = [r[0] for r in db.session.execute(check_doc_sql).fetchall()]
                if 'doc_type' not in existing_doc_cols:
                    print("🛠 Adding missing doc_type column...")
                    db.session.execute(text("ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS doc_type VARCHAR(50) DEFAULT 'syllabus'"))
                if 'structure_json' not in existing_doc_cols:
                    print("🛠 Adding missing structure_json column...")
                    db.session.execute(text("ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS structure_json TEXT"))
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

    # Global Rate Limit Error Handler
    from flask import jsonify
    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify({
            "status": "error",
            "error": "Rate limit exceeded",
            "message": str(e.description) if hasattr(e, 'description') else "Too many requests. Please try again later."
        }), 429

    return app

# Expose app for Gunicorn/Render compatibility with 'app:app' entry point
app = create_app()