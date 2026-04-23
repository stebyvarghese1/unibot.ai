import os
import sys
import psycopg2
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

def get_connection():
    db_url = os.getenv('SUPABASE_DB_URL') or os.getenv('DATABASE_URL')
    if not db_url:
        raise ValueError("SUPABASE_DB_URL or DATABASE_URL not found in .env")
        
    if db_url.startswith('postgresql://') and 'sslmode=' not in db_url:
        db_url += '?sslmode=require'
        
    return psycopg2.connect(db_url)

def run_sql(sql_file_or_query):
    """Run a SQL query or a SQL file."""
    conn = None
    try:
        conn = get_connection()
        conn.autocommit = True
        cur = conn.cursor()
        
        if os.path.exists(sql_file_or_query):
            with open(sql_file_or_query, 'r') as f:
                query = f.read()
            print(f"Running SQL from file: {sql_file_or_query}")
        else:
            query = sql_file_or_query
            print(f"Running SQL query...")
            
        cur.execute(query)
        
        if cur.description:
            colnames = [desc[0] for desc in cur.description]
            print(f"Results ({len(colnames)} columns):")
            print(" | ".join(colnames))
            rows = cur.fetchall()
            for row in rows:
                print(" | ".join(map(str, row)))
            print(f"Total rows: {len(rows)}")
        else:
            print("SUCCESS: Query executed successfully (no results).")
            
        cur.close()
    except Exception as e:
        print(f"ERROR: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/sql_tool.py <query_or_file_path>")
    else:
        run_sql(sys.argv[1])
