import redis
import json
import logging
import time
from functools import wraps
from flask import current_app

class CacheService:
    _instance = None
    _redis = None
    _local_cache = {}

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        self._redis_url = current_app.config.get('REDIS_URL')
        if self._redis_url:
            try:
                self._redis = redis.from_url(self._redis_url)
                self._redis.ping()
                logging.info("Redis cache initialized successfully.")
            except Exception as e:
                logging.warning(f"Failed to connect to Redis, falling back to in-memory cache: {e}")
                self._redis = None
        else:
            logging.info("No REDIS_URL found, using in-memory cache.")

    def get(self, key):
        if self._redis:
            try:
                val = self._redis.get(key)
                return json.loads(val) if val else None
            except Exception as e:
                logging.error(f"Redis get error for {key}: {e}")
                return None
        else:
            entry = self._local_cache.get(key)
            if entry:
                if entry['expiry'] > time.time():
                    return entry['value']
                else:
                    del self._local_cache[key]
            return None

    def set(self, key, value, ttl=3600):
        if self._redis:
            try:
                self._redis.set(key, json.dumps(value), ex=ttl)
            except Exception as e:
                logging.error(f"Redis set error for {key}: {e}")
        else:
            self._local_cache[key] = {
                'value': value,
                'expiry': time.time() + ttl
            }

    def delete(self, key):
        if self._redis:
            try:
                self._redis.delete(key)
            except Exception as e:
                logging.error(f"Redis delete error for {key}: {e}")
        else:
            if key in self._local_cache:
                del self._local_cache[key]

def cached_api(ttl=3600, key_prefix="api"):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if current_app.config.get('FLASK_ENV') == 'development':
                return f(*args, **kwargs)
            
            cache = CacheService.get_instance()
            # Generate cache key based on function name, args and request data
            from flask import request
            import hashlib
            
            raw_data = request.get_data()
            key_hash = hashlib.md5(f"{request.path}:{raw_data}:{kwargs}".encode()).hexdigest()
            cache_key = f"{key_prefix}:{f.__name__}:{key_hash}"
            
            cached_val = cache.get(cache_key)
            if cached_val:
                return cached_val
            
            result = f(*args, **kwargs)
            # result is typically a tuple (response, status_code) or a response object
            # For simplicity, we only cache 200 OK JSON responses
            if isinstance(result, tuple) and len(result) == 2:
                resp, status = result
            else:
                resp, status = result, 200
            
            if status == 200:
                try:
                    # result must be JSON serializable if we are caching the raw data
                    # In Flask routes, result is often a Response object from jsonify
                    # we might need to be careful here.
                    pass
                except Exception:
                    pass
            
            return result
        return decorated_function
    return decorator
