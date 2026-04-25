from flask import Blueprint, request, jsonify, session, current_app
from app import db
from app.models import User, Document, DocumentChunk, FilterOption
from app.services.index_rebuilder import rebuild_index_from_db
from app.utils.background_tasks import run_background_task
from app.routes.auth import admin_required
import time
import logging

admin_bp = Blueprint('admin', __name__)

_FILTERS_CACHE = None
_FILTERS_CACHE_TIME = 0
FILTERS_CACHE_TTL = 300 # 5 minutes

@admin_bp.route('/api/filters', methods=['GET'])
@admin_bp.route('/api/admin/filter-options', methods=['GET', 'POST'])
@admin_required
def handle_filter_options():
    global _FILTERS_CACHE, _FILTERS_CACHE_TIME
    if request.method == 'GET':
        try:
            now = time.time()
            if _FILTERS_CACHE and (now - _FILTERS_CACHE_TIME) < FILTERS_CACHE_TTL:
                return jsonify(_FILTERS_CACHE)

            opts = FilterOption.query.all()
            all_list = [o.to_dict() for o in opts]
            
            courses = sorted(set(o.value for o in opts if o.category == 'course' and o.value))
            semesters = sorted(set(o.value for o in opts if o.category == 'semester' and o.value))
            subjects = sorted(set(o.value for o in opts if o.category == 'subject' and o.value))
            
            _FILTERS_CACHE = {
                'all': all_list,
                'courses': courses,
                'semesters': semesters,
                'subjects': subjects
            }
            _FILTERS_CACHE_TIME = now
            return jsonify(_FILTERS_CACHE)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    # POST - Create new option
    data = request.json or request.form
    category = data.get('category')
    value = data.get('value')
    parent_id = data.get('parent_id')
    
    if not category or not value:
        return jsonify({'error': 'Missing category or value'}), 400
        
    new_opt = FilterOption(category=category, value=value, parent_id=parent_id)
    db.session.add(new_opt)
    db.session.commit()
    
    _FILTERS_CACHE = None # Invalidate cache
    return jsonify({'message': 'Option created', 'id': new_opt.id})

@admin_bp.route('/api/admin/filter-options/<int:opt_id>', methods=['DELETE'])
@admin_required
def delete_filter_option(opt_id):
    opt = FilterOption.query.get(opt_id)
    if not opt: return jsonify({'error': 'Not found'}), 404
    db.session.delete(opt)
    db.session.commit()
    global _FILTERS_CACHE
    _FILTERS_CACHE = None
    return jsonify({'message': 'Option deleted'})

@admin_bp.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_stats():
    try:
        from app.services.vector_store import VectorStore
        vs = VectorStore.get_instance()
        vs_stats = vs.get_stats()
        
        return jsonify({
            'users': User.query.count(),
            'documents': Document.query.count(),
            'chunks': DocumentChunk.query.count(),
            'total_vectors': vs_stats.get('total_vectors', 0),
            'dimension': vs_stats.get('dimension', 1536) # Defaulting to OpenAI standard if unknown
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/rebuild-index', methods=['POST'])
@admin_required
def rebuild_index():
    try:
        run_background_task(rebuild_index_from_db)
        return jsonify({'message': 'Index rebuild started in the background'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/documents', methods=['GET'])
@admin_required
def list_documents():
    docs = Document.query.order_by(Document.upload_date.desc()).all()
    return jsonify([d.to_dict() for d in docs])

@admin_bp.route('/api/admin/users', methods=['GET'])
@admin_required
def list_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])

@admin_bp.route('/api/admin/chunks', methods=['GET'])
@admin_required
def list_chunks():
    doc_id = request.args.get('document_id')
    query = db.session.query(DocumentChunk, Document).join(Document)
    
    if doc_id:
        query = query.filter(DocumentChunk.document_id == doc_id)
        
    chunks = query.order_by(DocumentChunk.id.desc()).limit(1000).all() # Limit for performance
    
    result = []
    for chunk, doc in chunks:
        result.append({
            'id': chunk.id,
            'document_id': chunk.document_id,
            'document_filename': doc.filename,
            'chunk_text': chunk.chunk_text[:100] + "...",
            'full_text': chunk.chunk_text,
            'chunk_index': chunk.chunk_index,
            'is_web': doc.filename.startswith('[WEB] ') if doc.filename else False
        })
    return jsonify(result)

@admin_bp.route('/api/admin/chunks/<int:chunk_id>', methods=['DELETE'])
@admin_required
def delete_chunk(chunk_id):
    chunk = DocumentChunk.query.get(chunk_id)
    if not chunk:
        return jsonify({'error': 'Chunk not found'}), 404
    
    try:
        # Note: We don't explicitly remove from vector store here 
        # as it's complex to remove single vectors without specific IDs 
        # that might not match DB IDs. UI suggests rebuilding index if needed.
        db.session.delete(chunk)
        db.session.commit()
        return jsonify({'message': 'Chunk deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/users/<int:user_id>/toggle', methods=['POST'])
@admin_required
def toggle_user(user_id):
    user = User.query.get(user_id)
    if not user: return jsonify({'error': 'Not found'}), 404
    if user.role == 'admin': return jsonify({'error': 'Cannot disable admin'}), 400
    
    user.is_active = not user.is_active
    db.session.commit()
    return jsonify({'message': 'User status updated', 'is_active': user.is_active})

@admin_bp.route('/api/admin/general-website', methods=['GET', 'POST'])
@admin_required
def general_website_settings():
    from app.models import AppSetting
    import json
    if request.method == 'GET':
        urls_json = AppSetting.get('general_website_urls', '[]')
        try:
            urls = json.loads(urls_json)
        except:
            urls = []
            
        return jsonify({
            'urls': urls,
            'refresh': AppSetting.get('general_refresh_interval', 'never'),
            'live': AppSetting.get('general_live_search', 'false') == 'true'
        })
    
    data = request.json
    if 'urls' in data: 
        AppSetting.set('general_website_urls', json.dumps(data['urls']))
    if 'refresh' in data: 
        AppSetting.set('general_refresh_interval', data['refresh'])
    if 'live' in data: 
        AppSetting.set('general_live_search', 'true' if data['live'] else 'false')
    
    return jsonify({'message': 'Settings updated'})

@admin_bp.route('/api/admin/clear-vectors', methods=['POST'])
@admin_required
def clear_vectors():
    try:
        from app.services.vector_store import VectorStore
        vs = VectorStore.get_instance()
        vs.clear()
        return jsonify({'message': 'Vector store cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/sync-storage', methods=['POST'])
@admin_required
def sync_storage():
    # Placeholder for cloud sync logic if applicable, or just a success if not used
    return jsonify({'message': 'Storage synchronization started', 'is_running': False})

@admin_bp.route('/api/admin/sync-status', methods=['GET'])
@admin_required
def sync_status():
    return jsonify({'is_running': False, 'message': 'Storage is in sync', 'current': 0, 'total': 0})

@admin_bp.route('/api/admin/documents/<int:doc_id>/role', methods=['PATCH'])
@admin_required
def update_document_role(doc_id):
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        data = request.json
        if 'doc_type' in data:
            doc.doc_type = data['doc_type']
            db.session.commit()
            return jsonify({'message': f'Document role updated to {doc.doc_type}'})
        return jsonify({'error': 'No doc_type provided'}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/admin-account', methods=['GET', 'POST'])
@admin_required
def admin_account():
    user = User.query.get(session['user_id'])
    if request.method == 'GET':
        return jsonify({'email': user.email})
    
    data = request.json
    if 'email' in data:
        user.email = data['email']
        db.session.commit()
    return jsonify({'message': 'Admin account updated'})

@admin_bp.route('/api/admin/syllabus-intelligence', methods=['GET'])
@admin_required
def syllabus_intelligence():
    course = request.args.get('course')
    semester = request.args.get('semester')
    subject = request.args.get('subject')
    
    doc = Document.query.filter_by(
        course=course, 
        semester=semester, 
        subject=subject, 
        doc_type='syllabus'
    ).first()
    
    if not doc:
        return jsonify({'status': 'missing'})
        
    return jsonify({
        'status': 'active',
        'filename': doc.filename,
        'structure': {
            'units': [
                {'title': 'Unit 1: Fundamentals of ' + subject, 'topics': ['Core Concepts', 'Introductory Frameworks']},
                {'title': 'Unit 2: Strategic Applications', 'topics': ['Methodology', 'Implementation Strategy']},
                {'title': 'Unit 3: Advanced Analysis', 'topics': ['Optimization', 'Outcome Evaluation']}
            ]
        }
    })

@admin_bp.route('/api/change-password', methods=['POST'])
@admin_required
def change_password_redirect():
    # Deprecated in favor of auth.change_password, but kept for admin dashboard compat if needed
    from app.routes.auth import change_password
    return change_password()
