import threading
import logging
import time
from flask import current_app
from app import db

class TaskTracker:
    _tasks = {}
    _lock = threading.Lock()

    @classmethod
    def update_progress(cls, task_name, current, total, message=""):
        with cls._lock:
            cls._tasks[task_name] = {
                'is_running': True,
                'current': current,
                'total': total,
                'message': message,
                'last_update': time.time()
            }

    @classmethod
    def complete_task(cls, task_name, message="Completed"):
        with cls._lock:
            if task_name in cls._tasks:
                cls._tasks[task_name].update({
                    'is_running': False,
                    'message': message,
                    'last_update': time.time()
                })
            else:
                cls._tasks[task_name] = {
                    'is_running': False,
                    'current': 0,
                    'total': 0,
                    'message': message,
                    'last_update': time.time()
                }

    @classmethod
    def get_status(cls, task_name):
        with cls._lock:
            return cls._tasks.get(task_name, {
                'is_running': False,
                'current': 0,
                'total': 0,
                'message': 'No task active',
                'last_update': 0
            })

def run_background_task(task_func, *args, app=None, **kwargs):
    """
    Runs a function in a background thread with its own application context 
    and proper database session management.
    """
    if app is None:
        try:
            app = current_app._get_current_object()
        except RuntimeError:
            logging.error("run_background_task called without app object and outside request context")
            raise

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
