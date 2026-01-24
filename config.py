import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    _DB_URL = os.getenv('SUPABASE_DB_URL', os.getenv('DATABASE_URL', 'sqlite:///app.db'))
    if _DB_URL.startswith('postgresql://') and 'supabase.co' in _DB_URL and 'sslmode=' not in _DB_URL:
        _DB_URL = _DB_URL + ('&sslmode=require' if '?' in _DB_URL else '?sslmode=require')
    SQLALCHEMY_DATABASE_URI = _DB_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True
    }
    
    # Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET', 'documents')
    SUPABASE_SERVICE_ROLE = os.getenv('SUPABASE_SERVICE_ROLE')
    
    # Hugging Face
    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
    HF_EMBEDDING_MODEL = os.getenv('HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    HF_LLM_MODEL = os.getenv('HF_LLM_MODEL', 'HuggingFaceH4/zephyr-7b-beta')
    HF_SMALLTALK_MODEL = os.getenv('HF_SMALLTALK_MODEL', 'google/flan-t5-small')
    
    # Uploads
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx'}

    # Admin
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL', 'admin@university.edu')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD', 'admin')

    # Startup behavior
    AUTO_REBUILD_INDEX = os.getenv('AUTO_REBUILD_INDEX', 'true').lower() == 'true'
    AUTO_SYNC_STORAGE = os.getenv('AUTO_SYNC_STORAGE', 'true').lower() == 'true'
    SYNC_STORAGE_INTERVAL = int(os.getenv('SYNC_STORAGE_INTERVAL', '120'))
    
    # Retrieval tuning
    VECTOR_MAX_DISTANCE = float(os.getenv('VECTOR_MAX_DISTANCE', '1.2'))

# Ensure upload directory exists
if not os.path.exists(Config.UPLOAD_FOLDER):
    os.makedirs(Config.UPLOAD_FOLDER)
