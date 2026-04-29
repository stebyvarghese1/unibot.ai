from flask import Blueprint, render_template, session, redirect
from app.routes.auth import page_login_required

main_bp = Blueprint('main', __name__)

def page_admin_required(f):
    from functools import wraps
    @wraps(f)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        if session.get('role') != 'admin':
            return redirect('/chat')
        return f(*args, **kwargs)
    return wrapped

@main_bp.route('/')
def index():
    if 'user_id' in session:
        return redirect('/admin' if session.get('role') == 'admin' else '/chat')
    return render_template('user/signin.html')

@main_bp.route('/admin')
@page_admin_required
def admin_panel():
    from app.services.vector_store import VectorStore
    try:
        vs = VectorStore.get_instance()
        stats = vs.get_stats()
    except Exception:
        stats = {'total_vectors': 0, 'dimension': 1536}
        
    return render_template('admin/admin.html', active_page='dashboard', stats=stats)

@main_bp.route('/admin/documents')
@page_admin_required
def admin_documents():
    return render_template('admin/documents.html', active_page='documents')

@main_bp.route('/admin/chunks')
@page_admin_required
def admin_chunks():
    return render_template('admin/chunks.html', active_page='chunks')

@main_bp.route('/admin/general-mode')
@page_admin_required
def admin_general_mode():
    return render_template('admin/general_mode.html', active_page='general_mode')

@main_bp.route('/admin/users')
@page_admin_required
def admin_users():
    return render_template('admin/users.html', active_page='users')

@main_bp.route('/login')
def login_page():
    if 'user_id' in session:
        return redirect('/admin' if session.get('role') == 'admin' else '/chat')
    return render_template('user/signin.html')

@main_bp.route('/signup')
def signup_page():
    if 'user_id' in session:
        return redirect('/admin' if session.get('role') == 'admin' else '/chat')
    return render_template('user/signup.html')

@main_bp.route('/profile')
@page_login_required
def profile_page():
    return render_template('user/profile.html')

@main_bp.route('/chat')
@page_login_required
def chat_page():
    return render_template('user/chat.html')

@main_bp.route('/logout')
def logout_page():
    session.clear()
    return redirect('/login')

