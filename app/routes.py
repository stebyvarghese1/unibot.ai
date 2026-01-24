from flask import Blueprint, request, jsonify, render_template, session, current_app, redirect
from app import db
from app.models import User, Document, DocumentChunk, ChatMessage
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.ai_service import AIService
from app.services.supabase_service import SupabaseService
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import os
import functools
import json
from config import Config
from sqlalchemy.exc import ProgrammingError

bp = Blueprint('main', __name__)

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

# --- Routes ---

@bp.route('/')
def index():
    return render_template('user/signin.html')

@bp.route('/admin')
def admin_panel():
    return render_template('admin/admin.html', active_page='dashboard')

@bp.route('/admin/documents')
def admin_documents():
    return render_template('admin/documents.html', active_page='documents')

@bp.route('/admin/chunks')
def admin_chunks():
    return render_template('admin/chunks.html', active_page='chunks')

@bp.route('/admin/users')
def admin_users():
    return render_template('admin/users.html', active_page='users')


@bp.route('/login')
def login_page():
    return render_template('user/signin.html')
@bp.route('/signup')
def signup_page():
    return render_template('user/signup.html')
@bp.route('/profile')
def profile_page():
    return render_template('user/profile.html')
@bp.route('/chat')
def chat_page():
    return render_template('user/chat.html')

# --- API Auth ---

@bp.route('/api/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    
    user = User.query.filter_by(email=email).first()
    
    if user and check_password_hash(user.password_hash, password):
        session['user_id'] = user.id
        session['role'] = user.role
        return jsonify({'message': 'Logged in successfully', 'role': user.role})
    
    return jsonify({'error': 'Invalid credentials'}), 401

@bp.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({'message': 'Logged out'})
@bp.route('/api/signup', methods=['POST'])
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
    session['user_id'] = user.id
    session['role'] = user.role
    return jsonify({'message': 'Signed up', 'role': user.role})
@bp.route('/api/profile', methods=['GET'])
def get_profile():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
    u = User.query.get(session['user_id'])
    if not u:
        return jsonify({'error': 'Unauthorized'}), 401
    return jsonify(u.to_dict())
@bp.route('/api/change-password', methods=['POST'])
def change_password():
    if 'user_id' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
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

@bp.route('/api/check-auth', methods=['GET'])
def check_auth():
    if 'user_id' in session:
        return jsonify({'authenticated': True, 'role': session.get('role')})
    return jsonify({'authenticated': False})
@bp.route('/api/prefs', methods=['GET', 'POST'])
@login_required
def prefs():
    if request.method == 'GET':
        return jsonify({
            'course': session.get('pref_course'),
            'semester': session.get('pref_semester'),
            'subject': session.get('pref_subject')
        })
    data = request.json or {}
    session['pref_course'] = (data.get('course') or '').strip() or None
    session['pref_semester'] = (data.get('semester') or '').strip() or None
    session['pref_subject'] = (data.get('subject') or '').strip() or None
    return jsonify({'message': 'Preferences saved'})
@bp.route('/api/filters', methods=['GET'])
@login_required
def list_filters():
    try:
        docs = Document.query.all()
        courses = sorted(list({(d.course or '').strip() for d in docs if d.course}))
        semesters = sorted(list({(d.semester or '').strip() for d in docs if d.semester}))
        subjects = sorted(list({(d.subject or '').strip() for d in docs if d.subject}))
        return jsonify({'courses': courses, 'semesters': semesters, 'subjects': subjects})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API Admin ---

@bp.route('/api/admin/upload', methods=['POST'])
@admin_required
def upload_document():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        course = request.form.get('course')
        semester = request.form.get('semester')
        subject = request.form.get('subject')
        if not course or not semester or not subject:
            return jsonify({'error': 'Course, semester, and subject are required'}), 400
        file_bytes = file.read()
        # Upload to Supabase Storage
        try:
            supa = SupabaseService()
            storage_path = supa.upload_file(file_bytes, filename, content_type=file.mimetype)
        except Exception as e:
            return jsonify({'error': f'Storage error: {e}'}), 500
        
        # Save to DB
        new_doc = Document(
            filename=filename,
            file_path=storage_path,
            uploaded_by=session['user_id'],
            status='pending',
            course=course,
            semester=semester,
            subject=subject
        )
        db.session.add(new_doc)
        db.session.commit()
        
        # Trigger processing (async in real world, sync here for MVP)
        try:
            process_document(new_doc.id)
            return jsonify({'message': 'File uploaded and processed successfully'})
        except Exception as e:
            new_doc.status = 'error'
            db.session.commit()
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

@bp.route('/api/admin/documents', methods=['GET'])
@admin_required
def list_documents():
    try:
        docs = Document.query.all()
        return jsonify([d.to_dict() for d in docs])
    except Exception as e:
        try:
            from sqlalchemy import text
            eng = None
            try:
                eng = db.session.get_bind()
            except Exception:
                eng = None
            if not eng:
                try:
                    eng = db.engine
                except Exception:
                    eng = None
            dialect = (eng.dialect.name if eng and eng.dialect else 'sqlite')
            if dialect == 'sqlite':
                stmts = [
                    "ALTER TABLE documents ADD COLUMN course VARCHAR(100)",
                    "ALTER TABLE documents ADD COLUMN semester VARCHAR(20)",
                    "ALTER TABLE documents ADD COLUMN subject VARCHAR(100)",
                ]
            else:
                stmts = [
                    "ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS course VARCHAR(100)",
                    "ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS semester VARCHAR(20)",
                    "ALTER TABLE public.documents ADD COLUMN IF NOT EXISTS subject VARCHAR(100)",
                ]
            for s in stmts:
                try:
                    db.session.execute(text(s))
                    db.session.commit()
                except Exception:
                    db.session.rollback()
            docs = Document.query.all()
            return jsonify([d.to_dict() for d in docs])
        except Exception as fix_error:
            return jsonify({'error': f'schema fix failed: {str(fix_error)}'}), 500

@bp.route('/api/admin/documents/<int:doc_id>', methods=['DELETE'])
@admin_required
def delete_document(doc_id):
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
            
        # Delete from Supabase (optional, but good practice)
        try:
            supa = SupabaseService()
            supa.delete_file(doc.file_path)
            # Also delete the chunks JSON dump if it exists
            try:
                supa.delete_file(f"chunks/{doc.id}.json")
            except Exception:
                pass
        except Exception:
            pass # Ignore if fails
            
        # Delete from Vector Store
        try:
            vector_store = VectorStore()
            vector_store.remove_document(doc_id)
        except Exception:
            pass
        
        # Delete chunks from DB
        try:
            DocumentChunk.query.filter_by(document_id=doc.id).delete()
        except Exception:
            pass
            
        db.session.delete(doc)
        db.session.commit()
        
        return jsonify({'message': 'Document deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/documents/<int:doc_id>/chunks', methods=['GET'])
@admin_required
def get_document_chunks(doc_id):
    try:
        chunks = DocumentChunk.query.filter_by(document_id=doc_id).order_by(DocumentChunk.chunk_index).all()
        return jsonify([c.to_dict() for c in chunks])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/sync-storage', methods=['POST'])
@admin_required
def sync_storage_route():
    try:
        count = sync_storage()
        return jsonify({'message': f'Storage synced. {count} new documents processed.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/rebuild-index', methods=['POST'])
@admin_required
def rebuild_index():
    try:
        vector_store = VectorStore()
        vector_store.clear()
        
        chunks = DocumentChunk.query.all()
        if not chunks:
            return jsonify({'message': 'Index cleared. No chunks to index.'})
            
        texts = [c.chunk_text for c in chunks]
        # In production, batch this!
        embeddings = AIService.get_embeddings(texts) 
        
        # Prepare metadata
        # include filename and public URL
        doc_map = {d.id: d for d in Document.query.all()}
        supa = SupabaseService()
        metadata = [{
            'text': c.chunk_text,
            'doc_id': c.document_id,
            'filename': doc_map[c.document_id].filename if c.document_id in doc_map else None,
            'url': supa.get_public_url(doc_map[c.document_id].file_path) if c.document_id in doc_map else None
        } for c in chunks]
        
        vector_store.add_documents(embeddings, metadata)
        
        return jsonify({'message': f'Index rebuilt with {len(chunks)} chunks.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_stats():
    vector_store = VectorStore()
    return jsonify(vector_store.get_stats())

@bp.route('/api/admin/chunks', methods=['GET'])
@admin_required
def list_chunks():
    try:
        # Limit to 100 for now to avoid overload
        chunks = DocumentChunk.query.order_by(DocumentChunk.id.desc()).limit(100).all()
        # Join with Document to get filename
        result = []
        for c in chunks:
            doc = Document.query.get(c.document_id)
            c_dict = c.to_dict()
            c_dict['document_filename'] = doc.filename if doc else "Unknown"
            c_dict['full_text'] = c.chunk_text # Send full text for inspection
            result.append(c_dict)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/users', methods=['GET'])
@admin_required
def list_users():
    try:
        users = User.query.all()
        return jsonify([u.to_dict() for u in users])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@bp.route('/api/admin/db-status', methods=['GET'])
@admin_required
def db_status():
    try:
        bind = None
        try:
            bind = db.session.get_bind()
        except Exception:
            bind = db.engine
        dialect = (bind.dialect.name if bind and bind.dialect else 'unknown')
        url_str = ''
        try:
            url_str = str(bind.url)
        except Exception:
            url_str = ''
        is_supabase = ('supabase.co' in url_str.lower())
        counts = {
            'users': User.query.count(),
            'documents': Document.query.count(),
            'chunks': DocumentChunk.query.count(),
        }
        try:
            from app.models import ChatMessage
            counts['chat_messages'] = ChatMessage.query.count()
        except Exception:
            counts['chat_messages'] = None
        return jsonify({
            'dialect': dialect,
            'supabase_db': is_supabase,
            'counts': counts
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/admin-account', methods=['GET', 'POST'])
@admin_required
def admin_account():
    try:
        u = User.query.get(session['user_id'])
        if not u or u.role != 'admin':
            return jsonify({'error': 'Forbidden'}), 403
        if request.method == 'GET':
            return jsonify({'email': u.email})
        data = request.json or {}
        new_email = (data.get('email') or '').strip().lower()
        current_pwd = data.get('current_password')
        new_pwd = data.get('new_password')
        if new_email:
            exists = User.query.filter(User.email == new_email, User.id != u.id).first()
            if exists:
                return jsonify({'error': 'Email already in use'}), 400
            u.email = new_email
        if new_pwd:
            if not current_pwd or not check_password_hash(u.password_hash, current_pwd):
                return jsonify({'error': 'Invalid current password'}), 400
            u.password_hash = generate_password_hash(new_pwd)
        if not new_email and not new_pwd:
            return jsonify({'error': 'No changes provided'}), 400
        db.session.commit()
        return jsonify({'message': 'Admin account updated'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
# --- API Student ---

@bp.route('/api/query', methods=['POST'])
@login_required
def query():
    data = request.json
    question = data.get('question')
    course = (data.get('course') or session.get('pref_course') or '').strip()
    semester = (data.get('semester') or session.get('pref_semester') or '').strip()
    subject = (data.get('subject') or session.get('pref_subject') or '').strip()
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
        
    try:
        from app.services.ai_service import AIService
        if AIService.is_smalltalk(question):
            answer = AIService.generate_smalltalk(question)
            return jsonify({'answer': answer, 'sources': []})
        # 1. Embed question
        q_embedding = AIService.get_embeddings([question])
        # get_embeddings returns list of list (batch), we need the first one if it's a list
        if isinstance(q_embedding, list) and len(q_embedding) > 0:
             # Handle varying return types from HF API (sometimes list of float, sometimes list of list)
             if isinstance(q_embedding[0], list):
                 q_vec = q_embedding[0]
             else:
                 q_vec = q_embedding
        else:
             return jsonify({'error': 'Failed to embed question'}), 500

        # 2. Search
        vector_store = VectorStore()
        results = vector_store.search(q_vec, k=8)
        # Filter low-confidence matches by distance threshold; if empty, fallback to top results
        filtered = [r for r in results if r.get('distance') is not None and r['distance'] <= Config.VECTOR_MAX_DISTANCE]
        if not filtered and results:
            filtered = results
        # Apply course/semester/subject filters if provided
        doc_map = {d.id: d for d in Document.query.all()}
        if course or semester or subject:
            def match_cat(r):
                did = r.get('doc_id')
                d = doc_map.get(did)
                if not d:
                    return False
                ok_course = True if not course else ((d.course or '').strip().lower() == course.lower())
                ok_sem = True if not semester else ((d.semester or '').strip().lower() == semester.lower())
                ok_subj = True if not subject else ((d.subject or '').strip().lower() == subject.lower())
                return ok_course and ok_sem and ok_subj
            filtered = [r for r in filtered if match_cat(r)]
            if not filtered:
                return jsonify({'answer': 'Not available in selected category (No context found).', 'sources': []})
        
        if not filtered:
            return jsonify({'answer': 'Not available in uploaded documents (No context found).', 'sources': []})
            
        # 3. Generate Answer
        context = "\n\n".join([r['text'] for r in filtered])
        answer = AIService.generate_answer(question, context)
        # Deduplicate sources by doc_id
        unique = {}
        for r in filtered:
            key = r.get('doc_id')
            if key is None:
                key = f"unknown-{id(r)}"
            if key not in unique:
                # Prefer filename from metadata; if missing, pull from DB
                fn = r.get('filename')
                if not fn and isinstance(key, int) and key in doc_map:
                    fn = doc_map[key].filename
                url = r.get('url')
                if not url and isinstance(key, int) and key in doc_map:
                    # Compute public URL from stored path
                    try:
                        supa = SupabaseService()
                        url = supa.get_public_url(doc_map[key].file_path)
                    except Exception:
                        url = None
                unique[key] = {
                    'doc_id': r.get('doc_id'),
                    'filename': fn,
                    'url': url
                }
        sources = list(unique.values())
        try:
            msg = ChatMessage(
                user_id=session['user_id'],
                question=question,
                answer=answer,
                course=course or None,
                semester=semester or None,
                subject=subject or None,
                sources_json=json.dumps(sources)
            )
            db.session.add(msg)
            db.session.commit()
        except Exception:
            db.session.rollback()
        return jsonify({
            'answer': answer,
            'sources': sources
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@bp.route('/api/chat/history', methods=['GET'])
@login_required
def chat_history():
    try:
        limit = int(request.args.get('limit', '50'))
        offset = int(request.args.get('offset', '0'))
        q = ChatMessage.query.filter_by(user_id=session['user_id']).order_by(ChatMessage.created_at.desc())
        msgs = q.offset(offset).limit(limit).all()
        def parse_sources(s):
            try:
                return json.loads(s) if s else []
            except Exception:
                return []
        return jsonify([{
            'id': m.id,
            'question': m.question,
            'answer': m.answer,
            'course': m.course,
            'semester': m.semester,
            'subject': m.subject,
            'sources': parse_sources(m.sources_json),
            'created_at': m.created_at.isoformat()
        } for m in msgs])
    except Exception as e:
        return jsonify({'error': str(e)}), 500
@bp.route('/api/chat/history/<int:msg_id>', methods=['DELETE'])
@login_required
def delete_chat_message(msg_id):
    try:
        m = ChatMessage.query.get(msg_id)
        if not m or m.user_id != session['user_id']:
            return jsonify({'error': 'Not found'}), 404
        db.session.delete(m)
        db.session.commit()
        return jsonify({'message': 'Deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
@bp.route('/api/chat/history', methods=['DELETE'])
@login_required
def clear_chat_history():
    try:
        ChatMessage.query.filter_by(user_id=session['user_id']).delete()
        db.session.commit()
        return jsonify({'message': 'Cleared'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# --- Helpers ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS']

def process_document(doc_id):
    doc = Document.query.get(doc_id)
    if not doc:
        return
        
    try:
        # Download from Supabase Storage
        supa = SupabaseService()
        file_bytes = supa.download_file(doc.file_path)
        text = DocumentProcessor.extract_text_from_bytes(file_bytes, doc.filename)
        chunks = DocumentProcessor.chunk_text(text)
        
        for i, chunk_text in enumerate(chunks):
            new_chunk = DocumentChunk(
                document_id=doc.id,
                chunk_text=chunk_text,
                chunk_index=i
            )
            db.session.add(new_chunk)
            
        doc.status = 'processed'
        db.session.commit()
        
        # Store chunks as JSON in Supabase Storage for audit/export
        try:
            chunks_payload = json.dumps([{'chunk_index': i, 'text': t} for i, t in enumerate(chunks)]).encode('utf-8')
            supa.upload_file(chunks_payload, f"chunks/{doc.id}.json", content_type="application/json")
        except Exception as e:
            # Non-fatal: continue even if chunk JSON upload fails
            pass
        
        # Auto-update index (optional, or wait for manual rebuild)
        # For MVP, let's try to update immediately if small
        vector_store = VectorStore()
        # Need to re-embed just this doc's chunks
        # But for simplicity/consistency with "rebuild" logic, maybe just leave it for manual or background job
        # Or just do it:
        chunk_texts = [c for c in chunks]
        embeddings = AIService.get_embeddings(chunk_texts)
        metadata = [{'text': c, 'doc_id': doc.id, 'filename': doc.filename, 'url': supa.get_public_url(doc.file_path)} for c in chunks]
        vector_store.add_documents(embeddings, metadata)
        
    except Exception as e:
        doc.status = 'error'
        db.session.commit()
        raise e

def sync_storage():
    # Clear any previous failed transaction state
    try:
        db.session.rollback()
    except Exception:
        pass
    supa = SupabaseService()
    items = supa.list_files(prefix="")
    added = 0
    # Resolve uploader without using session/request context
    from app.models import User
    uploader_id = None
    try:
        admin = User.query.filter_by(email=Config.ADMIN_EMAIL).first()
        if admin:
            uploader_id = admin.id
    except Exception:
        uploader_id = None
    for it in items:
        try:
            name = it.get('name') or it.get('Key') or ''
            if not name or name.startswith('chunks/'):
                continue
            if not allowed_file(name):
                continue
            exists = Document.query.filter_by(file_path=name).first()
            if exists:
                continue
            filename = os.path.basename(name)
            new_doc = Document(
                filename=filename,
                file_path=name,
                uploaded_by=uploader_id or 1,
                status='pending'
            )
            db.session.add(new_doc)
            db.session.commit()
            process_document(new_doc.id)
            added += 1
        except Exception as e:
            try:
                db.session.rollback()
            except Exception:
                pass
            continue
    # Ensure background thread does not hold onto a stale session
    try:
        db.session.remove()
    except Exception:
        pass
    return added
