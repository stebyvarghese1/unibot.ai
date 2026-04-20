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
    
    # Supabase
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_KEY = os.getenv('SUPABASE_KEY')
    SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET', 'documents')
    SUPABASE_SERVICE_ROLE = os.getenv('SUPABASE_SERVICE_ROLE')
    
    # Hugging Face
    HUGGINGFACE_API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
    HF_EMBEDDING_MODEL = os.getenv('HF_EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')
    HF_LLM_MODEL = os.getenv('HF_LLM_MODEL', 'google/gemma-2b-it')
    HF_SMALLTALK_MODEL = os.getenv('HF_SMALLTALK_MODEL', 'google/gemma-2b-it')
    HF_IMAGE_CAPTION_MODEL = os.getenv('HF_IMAGE_CAPTION_MODEL', 'Salesforce/blip-image-captioning-large')
    
    # Uploads (using Supabase storage only, no local storage)
    UPLOAD_FOLDER = '/tmp/uploads'  # Temporary folder that gets cleaned up
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx'}

    # Admin
    ADMIN_EMAIL = os.getenv('ADMIN_EMAIL')
    ADMIN_PASSWORD = os.getenv('ADMIN_PASSWORD')

    # Startup behavior
    AUTO_REBUILD_INDEX = os.getenv('AUTO_REBUILD_INDEX', 'true').lower() == 'true'
    AUTO_SYNC_STORAGE = os.getenv('AUTO_SYNC_STORAGE', 'true').lower() == 'true'
    SYNC_STORAGE_INTERVAL = int(os.getenv('SYNC_STORAGE_INTERVAL', '120'))
    
    # Retrieval tuning
    VECTOR_MAX_DISTANCE = float(os.getenv('VECTOR_MAX_DISTANCE', '3.0'))  # Permissive threshold for better recall

    # Rate Limiting & Stability
    RATELIMIT_DEFAULT = os.getenv('RATELIMIT_DEFAULT', '200 per day; 50 per hour')
    REDIS_URL = os.getenv('REDIS_URL', os.getenv('REDIS_EXTERNAL_URL'))
    SENTRY_DSN = os.getenv('SENTRY_DSN')
    
    # Connection Pooling
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
        'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
        'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
        'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '1800')),
        'pool_pre_ping': True
    }

# No local upload directory needed - using Supabase storage only
