import requests
from config import Config
from flask import current_app
from supabase import create_client, Client


class SupabaseService:
    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SupabaseService, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        # Only initialize once
        if self._client is not None:
            return

        # Prefer runtime overrides from Flask config when available
        supa_url = None
        supa_key = None
        supa_service_role = None
        supa_bucket = None
        try:
            if current_app:
                supa_url = (current_app.config.get("SUPABASE_URL") or "").strip()
                supa_key = (current_app.config.get("SUPABASE_KEY") or "").strip()
                supa_service_role = (current_app.config.get("SUPABASE_SERVICE_ROLE") or "").strip()
                supa_bucket = (current_app.config.get("SUPABASE_BUCKET") or "").strip()
        except Exception:
            pass
            
        if not supa_url:
            supa_url = Config.SUPABASE_URL
        if not supa_key:
            supa_key = Config.SUPABASE_KEY
        if not supa_service_role:
            supa_service_role = Config.SUPABASE_SERVICE_ROLE or supa_key
        if not supa_bucket:
            supa_bucket = Config.SUPABASE_BUCKET
            
        if not supa_url or not supa_key:
            raise RuntimeError("Supabase configuration missing")
            
        self.url = supa_url.rstrip("/")
        self.key = supa_key
        self.service_role = supa_service_role or self.key
        self.bucket = supa_bucket
        self.base = f"{self.url}/storage/v1/object"
        self.headers_base = {
            "Authorization": f"Bearer {self.service_role}",
            "apikey": self.key,
        }
        
        # Official client for DB operations (pgvector)
        # Reused across all requests
        self._client = create_client(self.url, self.service_role)

    @property
    def client(self) -> Client:
        return self._client

    def upload_file(self, file_bytes: bytes, path: str, content_type: str = "application/octet-stream") -> str:
        try:
            # Use official client storage for better stability
            self.client.storage.from_(self.bucket).upload(
                path=path,
                file=file_bytes,
                file_options={"content-type": content_type, "x-upsert": "true"}
            )
            return path
        except Exception as e:
            # Fallback if upload fails (e.g. file already exists despite upsert, or SSL error)
            logging.error(f"Supabase Storage Upload Error: {e}")
            raise RuntimeError(f"Storage upload failed: {e}")

    def download_file(self, path: str) -> bytes:
        try:
            # Use official client storage for better stability
            res = self.client.storage.from_(self.bucket).download(path)
            return res
        except Exception as e:
            logging.error(f"Supabase Storage Download Error: {e}")
            raise RuntimeError(f"Storage download failed: {e}")

    def get_signed_url(self, path: str, expires_in: int = 3600) -> str:
        """Generate a short-lived signed URL for private document access (defaults to 1 hour)."""
        try:
            res = self.client.storage.from_(self.bucket).create_signed_url(path, expires_in)
            if isinstance(res, dict) and 'signedURL' in res:
                return res['signedURL']
            return str(res)
        except Exception as e:
            # Fallback to public URL if signed URL generation fails (e.g. bucket doesn't support it)
            return self.client.storage.from_(self.bucket).get_public_url(path)

    def delete_file(self, path: str):
        try:
            self.client.storage.from_(self.bucket).remove([path])
            return True
        except Exception as e:
            logging.warning(f"Storage delete warning (file might not exist): {e}")
            return False

    def list_files(self, prefix: str = "", limit: int = 100, offset: int = 0):
        try:
            res = self.client.storage.from_(self.bucket).list(
                path=prefix,
                options={"limit": limit, "offset": offset}
            )
            return res
        except Exception as e:
            raise RuntimeError(f"Storage list failed: {e}")

    def delete_user_by_email(self, email: str):
        """
        Deletes a user from Supabase Auth by their email address.
        Requires Service Role Key.
        """
        try:
            # 1. Find user by email
            # Note: auth.admin.list_users() is available in newer versions of supabase-py
            # If not, we can use the direct GoTrue API
            users_res = self.client.auth.admin.list_users()
            target_user = None
            
            # The response structure might vary slightly depending on version
            users = getattr(users_res, 'users', users_res if isinstance(users_res, list) else [])
            
            for u in users:
                if u.email.lower() == email.lower():
                    target_user = u
                    break
            
            if target_user:
                self.client.auth.admin.delete_user(target_user.id)
                return True
            return False
        except Exception as e:
            import logging
            logging.error(f"Failed to delete user from Supabase Auth: {e}")
            return False


