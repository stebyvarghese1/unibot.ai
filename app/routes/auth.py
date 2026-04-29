from flask import Blueprint, request, jsonify, session, redirect, current_app
from app import db, limiter, csrf
from app.models import User
from app.services.supabase_service import SupabaseService
from werkzeug.security import check_password_hash, generate_password_hash
import functools
import logging
import requests

auth_bp = Blueprint('auth', __name__)

# --- Auth Decorators ---
def login_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            return jsonify({'error': 'Unauthorized'}), 401
        return f(*args, **kwargs)
    return wrapped

def admin_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            return jsonify({'error': 'Forbidden'}), 403
        return f(*args, **kwargs)
    return wrapped

def page_login_required(f):
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        if 'user_id' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return wrapped

# --- API Auth ---

@auth_bp.route('/api/login', methods=['POST'])
@csrf.exempt
@limiter.limit("5 per minute", error_message="Too many login attempts. Please try again in a minute.")
def login():
    data = request.json
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    if user and check_password_hash(user.password_hash, password):
        if not user.is_active:
            return jsonify({'error': 'Your account has been deactivated. Please contact support.'}), 403
            
        session.permanent = True
        session['user_id'] = user.id
        session['role'] = user.role
        session['pref_course'] = user.pref_course
        session['pref_semester'] = user.pref_semester
        session['pref_subject'] = user.pref_subject
        return jsonify({
            'message': 'Logged in successfully', 
            'role': user.role,
            'show_tour': user.show_tour
        })
    
    return jsonify({'error': 'Invalid credentials'}), 401

@auth_bp.route('/api/logout', methods=['POST', 'GET'])
def logout():
    session.clear()
    if request.method == 'POST':
        return jsonify({'message': 'Logged out'})
    return redirect('/login')

@auth_bp.route('/api/signup', methods=['POST'])
@csrf.exempt
@limiter.limit("5 per hour", error_message="Too many accounts created. Please try again later.")
def signup():
    data = request.json
    email = (data.get('email') or '').strip().lower()
    password = data.get('password')
    if not email or not password:
        return jsonify({'error': 'Email and password required'}), 400
    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({'error': 'Email already registered'}), 400
    pwd_hash = generate_password_hash(password)
    user = User(email=email, password_hash=pwd_hash, role='student')
    db.session.add(user)
    db.session.commit()
    session.permanent = True
    session['user_id'] = user.id
    session['role'] = user.role
    return jsonify({'message': 'Signed up', 'role': user.role})

@auth_bp.route('/api/auth/verify-supabase', methods=['POST'])
@csrf.exempt
def verify_supabase():
    token = request.json.get('access_token')
    if not token:
        return jsonify({'error': 'Token missing'}), 400
    
    try:
        supa = SupabaseService()
        headers = {
            "Authorization": f"Bearer {token}",
            "apikey": supa.key
        }
        resp = requests.get(f"{supa.url}/auth/v1/user", headers=headers)
        if resp.status_code != 200:
            return jsonify({'error': 'Invalid authentication session'}), 401
            
        user_data = resp.json()
        email = (user_data.get('email') or '').strip().lower()
        if not email:
            return jsonify({'error': 'Email not found in Google account'}), 400
            
        user = User.query.filter_by(email=email).first()
        supabase_uid = user_data.get('id')
        
        if not user:
            user = User(email=email, supabase_uid=supabase_uid, role='student', is_active=True)
            db.session.add(user)
            db.session.commit()
        elif not user.supabase_uid and supabase_uid:
            user.supabase_uid = supabase_uid
            db.session.commit()
            
        if not user.is_active:
            return jsonify({'error': 'Your account has been deactivated. Please contact support.'}), 403
            
        session.permanent = True
        session['user_id'] = user.id
        session['role'] = user.role
        session['pref_course'] = user.pref_course
        session['pref_semester'] = user.pref_semester
        session['pref_subject'] = user.pref_subject
        
        return jsonify({
            'success': True, 
            'role': user.role,
            'show_tour': user.show_tour
        })
        
    except Exception as e:
        logging.error(f"Supabase verification error: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error during verification'}), 500

@auth_bp.route('/api/auth/callback', methods=['GET'])
def auth_callback():
    from flask import render_template, current_app
    return render_template('user/callback.html', config=current_app.config)

@auth_bp.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'role': session.get('role')})
    return jsonify({'authenticated': False})

@auth_bp.route('/api/profile', methods=['GET'])
@login_required
def get_profile():
    u = User.query.get(session['user_id'])
    if not u:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(u.to_dict())

@auth_bp.route('/api/change-password', methods=['POST'])
@login_required
def change_password():
    data = request.json
    current = data.get('current_password')
    newpwd = data.get('new_password')
    if not current or not newpwd:
        return jsonify({'error': 'Current and new password required'}), 400
    u = User.query.get(session['user_id'])
    if not u or not check_password_hash(u.password_hash, current):
        return jsonify({'error': 'Invalid current password'}), 400
    u.password_hash = generate_password_hash(newpwd)
    db.session.commit()
    return jsonify({'message': 'Password updated'})

@auth_bp.route('/api/prefs', methods=['GET', 'POST'])
@login_required
def prefs():
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404

    if request.method == 'GET':
        return jsonify({
            'name': user.preferred_name,
            'course': user.pref_course,
            'semester': user.pref_semester,
            'subject': user.pref_subject
        })
    
    data = request.json or {}
    if 'name' in data:
        user.preferred_name = (data.get('name') or '').strip() or None
    if 'course' in data:
        user.pref_course = (data.get('course') or '').strip() or None
    if 'semester' in data:
        user.pref_semester = (data.get('semester') or '').strip() or None
    if 'subject' in data:
        user.pref_subject = (data.get('subject') or '').strip() or None
    
    db.session.commit()
    # Update session cache
    session['pref_course'] = user.pref_course
    session['pref_semester'] = user.pref_semester
    session['pref_subject'] = user.pref_subject
    
    return jsonify({'message': 'Preferences saved'})

@auth_bp.route('/api/tour/complete', methods=['POST'])
@login_required
def complete_tour():
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    user.show_tour = False
    db.session.commit()
    return jsonify({'message': 'Tour completed'})

@auth_bp.route('/api/profile/delete-otp-request', methods=['POST'])
@login_required
def request_delete_otp():
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    supa = SupabaseService()
    success = supa.send_otp(user.email)
    if success:
        return jsonify({'message': 'Verification code sent'})
    return jsonify({'error': 'Failed to send verification code'}), 500

@auth_bp.route('/api/profile', methods=['DELETE'])
@login_required
def delete_profile():
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404
        
    data = request.json or {}
    
    if user.password_hash:
        password = data.get('password')
        if not password or not check_password_hash(user.password_hash, password):
            return jsonify({'error': 'Invalid password'}), 400
    else:
        otp = data.get('otp')
        if not otp:
            return jsonify({'error': 'Verification code required'}), 400
            
        supa = SupabaseService()
        if not supa.verify_otp(user.email, otp):
            return jsonify({'error': 'Invalid verification code'}), 400
            
    # Delete from Supabase Auth if applicable
    if user.supabase_uid or not user.password_hash:
        supa = SupabaseService()
        supa.delete_user_by_email(user.email)
            
    # Delete user from database
    db.session.delete(user)
    db.session.commit()
    session.clear()
    
    return jsonify({'message': 'Account deleted successfully'})
