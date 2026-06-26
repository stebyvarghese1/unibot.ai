# Gunicorn configuration for production deployment on Render.com
import os

# Port to bind
port = os.environ.get("PORT", "5000")
bind = f"0.0.0.0:{port}"

# Worker configuration
# 'gthread' is recommended for handling slow requests/uploads without deadlocking the worker heartbeat.
worker_class = "gthread"
threads = 4
workers = int(os.environ.get("WEB_CONCURRENCY", "2"))

# Timeout configuration
# Extend timeout to 180 seconds to allow large uploads (up to 100MB) to complete over typical connections.
timeout = 180

# Keep-alive connection timeout
keepalive = 5
