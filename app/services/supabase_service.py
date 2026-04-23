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
        
        try:
            with current_app.app_context():
                supa_url = current_app.config.get('SUPABASE_URL')
                supa_key = current_app.config.get('SUPABASE_SERVICE_ROLE') or current_app.config.get('SUPABASE_KEY')
        except (RuntimeError, AttributeError):
            # Fallback to direct config if not in app context
            supa_url = Config.SUPABASE_URL
            supa_key = Config.SUPABASE_SERVICE_ROLE or Config.SUPABASE_KEY

        if not supa_url or not supa_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be configured")

        self.url = supa_url
        self.key = supa_key
        self._client = create_client(supa_url, supa_key)

    @property
    def client(self) -> Client:
        return self._client

    def upload_file(self, bucket_name: str, path: str, file_data: bytes, content_type: str = None):
        """Uploads a file to Supabase Storage"""
        options = {}
        if content_type:
            options['content-type'] = content_type
        
        # Use upsert=True to allow overwriting files with same name
        return self.client.storage.from_(bucket_name).upload(
            path=path,
            file=file_data,
            file_options={"content-type": content_type, "upsert": "true"}
        )

    def delete_file(self, path: str, bucket_name: str = None):
        """Deletes a file from Supabase Storage"""
        if not bucket_name:
            bucket_name = Config.SUPABASE_BUCKET
        return self.client.storage.from_(bucket_name).remove([path])

    def list_files(self, bucket_name: str, path: str = ""):
        """Lists files in a storage bucket path"""
        return self.client.storage.from_(bucket_name).list(path)

    def get_public_url(self, bucket_name: str, path: str):
        """Gets a public URL for a storage object"""
        return self.client.storage.from_(bucket_name).get_public_url(path)
        
    def get_signed_url(self, bucket_name: str, path: str, expires_in: int = 3600):
        """
        Creates a signed URL for a private storage object.
        Default expiration is 1 hour (3600 seconds).
        """
        try:
            res = self.client.storage.from_(bucket_name).create_signed_url(path, expires_in)
            if isinstance(res, dict) and 'signedURL' in res:
                return res['signedURL']
            elif hasattr(res, 'signed_url'):
                return res.signed_url
            return None
        except Exception as e:
            import logging
            logging.error(f"Error generating signed URL for {path}: {e}")
            return None

    def delete_user_by_email(self, email: str):
        """
        Deletes a user from Supabase Auth by their email address.
        Requires Service Role Key.
        """
        import logging
        try:
            target_user_id = None
            page = 1
            per_page = 100
            search_email = email.strip().lower()
            
            logging.info(f"🔍 Searching Supabase for user: {search_email}")
            
            while True:
                users_res = self.client.auth.admin.list_users(page=page, per_page=per_page)
                
                # Try multiple common response structures for compatibility
                users = []
                if hasattr(users_res, 'users'):
                    users = users_res.users
                elif hasattr(users_res, 'data') and isinstance(users_res.data, list):
                    users = users_res.data
                elif isinstance(users_res, list):
                    users = users_res
                
                if not users:
                    logging.info("ℹ️ No more users returned from Supabase list.")
                    break
                    
                for u in users:
                    u_email = (getattr(u, 'email', None) or (u.get('email') if isinstance(u, dict) else None) or '').strip().lower()
                    if u_email == search_email:
                        target_user_id = getattr(u, 'id', None) or (u.get('id') if isinstance(u, dict) else None)
                        logging.info(f"✅ Found user in Supabase! ID: {target_user_id}")
                        break
                
                if target_user_id or len(users) < per_page:
                    break
                page += 1
            
            if target_user_id:
                logging.info(f"🗑 Calling Supabase admin.delete_user for {target_user_id}")
                self.client.auth.admin.delete_user(target_user_id)
                return True
            
            logging.warning(f"⚠️ User {search_email} not found in Supabase Auth list.")
            return False
        except Exception as e:
            logging.error(f"❌ Failed to delete user from Supabase Auth: {e}")
            return False

    def send_otp(self, email: str):
        """Sends an OTP to the user's email using Supabase Auth."""
        try:
            # This triggers the 'Magic Link' email template in Supabase,
            # which the user has now configured to show the {{ .Token }} as a numeric code.
            self.client.auth.sign_in_with_otp({"email": email})
            return True
        except Exception as e:
            import logging
            logging.error(f"Failed to send OTP via Supabase: {e}")
            return False

    def verify_otp(self, email: str, token: str):
        """Verifies the OTP token for the given email."""
        try:
            # type="magiclink" is the default for email-based OTP in Supabase Auth
            res = self.client.auth.verify_otp({"email": email, "token": token, "type": "magiclink"})
            return res is not None
        except Exception as e:
            import logging
            logging.error(f"Failed to verify Supabase OTP: {e}")
            return False
