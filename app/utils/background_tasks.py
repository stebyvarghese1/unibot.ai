import threading
import logging
from flask import current_app
from app import db

def run_background_task(task_func, *args, **kwargs):
    """
    Runs a function in a background thread with its own application context 
    and proper database session management.
    """
    app = current_app._get_current_object()
    
    def wrapper():
        with app.app_context():
            try:
                task_func(*args, **kwargs)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                logging.error(f"Background task failed: {e}", exc_info=True)
            finally:
                db.session.remove()
                
    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    return thread
