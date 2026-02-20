import requests
from config import Config
from flask import current_app


class SupabaseService:
    def __init__(self):
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

    def upload_file(self, file_bytes: bytes, path: str, content_type: str = "application/octet-stream") -> str:
        headers = {
            **self.headers_base,
            "Content-Type": content_type,
            "x-upsert": "true",
        }
        resp = requests.post(f"{self.base}/{self.bucket}/{path}", headers=headers, data=file_bytes)
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"Storage upload failed: {resp.status_code} {resp.text}")
        return path

    def download_file(self, path: str) -> bytes:
        headers = {**self.headers_base}
        resp = requests.get(f"{self.base}/{self.bucket}/{path}", headers=headers)
        if resp.status_code != 200:
            raise RuntimeError(f"Storage download failed: {resp.status_code} {resp.text}")
        return resp.content

    def get_public_url(self, path: str) -> str:
        # Requires bucket to be public or signed URL mechanism (not implemented here)
        return f"{self.base}/public/{self.bucket}/{path}"

    def delete_file(self, path: str):
        headers = {**self.headers_base}
        resp = requests.delete(f"{self.base}/{self.bucket}/{path}", headers=headers)
        if resp.status_code not in (200, 204):
            raise RuntimeError(f"Storage delete failed: {resp.status_code} {resp.text}")
        return True

    def list_files(self, prefix: str = "", limit: int = 100, offset: int = 0):
        url = f"{self.url}/storage/v1/object/list/{self.bucket}"
        headers = {**self.headers_base, "Content-Type": "application/json"}
        payload = {"prefix": prefix, "limit": limit, "offset": offset}
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Storage list failed: {resp.status_code} {resp.text}")
        return resp.json()
