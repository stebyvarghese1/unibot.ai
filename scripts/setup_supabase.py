import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_supabase_vector():
    db_url = os.getenv('SUPABASE_DB_URL') or os.getenv('DATABASE_URL')
    
    if not db_url:
        print("❌ Error: SUPABASE_DB_URL or DATABASE_URL not found in environment.")
        return

    # Normalize DATABASE_URL for psycopg2 (if it uses the 'postgresql://' scheme)
    if db_url.startswith('postgresql://') and 'sslmode=' not in db_url:
        db_url += '?sslmode=require'

    try:
        print(f"🔗 Connecting to database...")
        conn = psycopg2.connect(db_url)
        conn.autocommit = True
        cur = conn.cursor()

        print("🚀 Enabling pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        print("📁 Creating 'embeddings' table...")
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id BIGSERIAL PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding VECTOR(384)
            );
        """)

        print("🔍 Creating 'match_documents' function...")
        cur.execute("""
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
        """)

        print("⚡ Creating HNSW index for better performance...")
        # Note: ivfflat is also good, but hnsw is generally faster for search
        cur.execute("CREATE INDEX ON embeddings USING hnsw (embedding vector_cosine_ops);")

        print("✅ Supabase pgvector setup completed successfully!")
        
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error setting up database: {e}")

if __name__ == "__main__":
    setup_supabase_vector()
