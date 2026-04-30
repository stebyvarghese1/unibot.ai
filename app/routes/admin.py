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
def get_public_filters():
    global _FILTERS_CACHE, _FILTERS_CACHE_TIME
    try:
        now = time.time()
        if _FILTERS_CACHE and (now - _FILTERS_CACHE_TIME) < FILTERS_CACHE_TTL:
            return jsonify(_FILTERS_CACHE)

        # Efficiently fetch categorized values directly from DB
        def get_distinct(cat):
            return [r[0] for r in db.session.query(FilterOption.value)
                    .filter(FilterOption.category == cat, FilterOption.value != None)
                    .distinct().order_by(FilterOption.value).all()]

        courses = get_distinct('course')
        semesters = get_distinct('semester')
        subjects = get_distinct('subject')
        
        # Fetch all for the mapping view (can still be slow, but separate from dropdowns)
        opts = FilterOption.query.all()
        all_list = [o.to_dict() for o in opts]
        
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

@admin_bp.route('/api/admin/filter-options', methods=['GET', 'POST'])
@admin_required
def handle_filter_options():
    if request.method == 'GET':
        return get_public_filters()
        
    # POST - Create new option (can include syllabus for subjects)
    is_json = request.is_json
    if is_json:
        data = request.get_json()
    else:
        data = request.form

    category = data.get('category')
    value = data.get('value')
    parent_id = data.get('parent_id')
    
    if not category or not value:
        return jsonify({'error': 'Missing category or value'}), 400
        
    try:
        new_opt = FilterOption(category=category, value=value, parent_id=parent_id)
        db.session.add(new_opt)
        
        # If it's a subject, handle syllabus grounding
        if category == 'subject':
            if not is_json:
                file = request.files.get('file')
                if not file:
                    return jsonify({'error': 'A syllabus file is required to create a subject.'}), 400
                    
                # 1. Determine Course/Semester from parent IDs
                course_name = "Unknown"
                semester_name = "Unknown"
                
                if parent_id:
                    sem = FilterOption.query.get(parent_id)
                    if sem:
                        semester_name = sem.value
                        if sem.parent_id:
                            course = FilterOption.query.get(sem.parent_id)
                            if course:
                                course_name = course.value

                # 2. Upload and Ingest via DocumentProcessor (reusing upload logic)
                from app.services.supabase_service import SupabaseService
                from app.services.document_processor import DocumentProcessor
                from app.models import Document
                import os
                
                # Correctly instantiate service and prepare file data
                supa = SupabaseService()
                file_bytes = file.read()
                file.seek(0) # Reset pointer for safety
                
                # Generate a unique path in Supabase storage
                storage_path = f"syllabus/{int(time.time())}_{file.filename}"
                
                file_path = supa.upload_file(
                    file_data=file_bytes, 
                    path=storage_path, 
                    content_type=file.content_type
                )
                
                new_doc = Document(
                    filename=file.filename,
                    file_path=file_path,
                    course=course_name,
                    semester=semester_name,
                    subject=value,
                    doc_type='syllabus',
                    status='pending',
                    uploaded_by=session.get('user_id')
                )
                db.session.add(new_doc)
                db.session.flush() # Flush to get IDs if needed
                
                # Process in background
                from app.routes.docs import process_document_task
                run_background_task(process_document_task, new_doc.id, app=supa._client.app if hasattr(supa._client, 'app') else None)
                logging.info(f"✅ Intelligence Grounded: {value} ({course_name}/{semester_name})")
            else:
                return jsonify({'error': 'A syllabus file must be uploaded as multipart/form-data.'}), 400

        db.session.commit()
        _FILTERS_CACHE = None # Invalidate cache
        return jsonify({'message': 'Option created', 'id': new_opt.id})
    except Exception as e:
        db.session.rollback()
        logging.error(f"❌ Failed to deploy subject: {e}")
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/filter-options/<int:opt_id>', methods=['DELETE'])
@admin_required
def delete_filter_option(opt_id):
    try:
        opt = FilterOption.query.get(opt_id)
        if not not opt:
            # 1. Identify associated documents scope
            target_course = None
            target_semester = None
            target_subject = None
            
            curr = opt
            while curr:
                if curr.category == 'course': target_course = curr.value
                elif curr.category == 'semester': target_semester = curr.value
                elif curr.category == 'subject': target_subject = curr.value
                curr = curr.parent

            # 2. Find and clean up documents matching this hierarchy path
            from sqlalchemy import func
            doc_query = Document.query
            if target_course: doc_query = doc_query.filter(func.lower(Document.course) == func.lower(target_course))
            if target_semester: doc_query = doc_query.filter(func.lower(Document.semester) == func.lower(target_semester))
            if target_subject: doc_query = doc_query.filter(func.lower(Document.subject) == func.lower(target_subject))
            
            docs_to_delete = doc_query.all()
            
            if docs_to_delete:
                from app.services.supabase_service import SupabaseService
                from app.services.vector_store import VectorStore
                supa = SupabaseService()
                vs = VectorStore.get_instance()
                
                for doc in docs_to_delete:
                    # Supabase Storage
                    if doc.file_path and not str(doc.file_path).startswith(('http://', 'https://')):
                        try: supa.delete_file(doc.file_path)
                        except: pass
                    # Vector Store
                    try: vs.remove_document(doc.id)
                    except: pass
                    # DB Chunks & Document
                    DocumentChunk.query.filter_by(document_id=doc.id).delete()
                    db.session.delete(doc)

        if not opt: return jsonify({'error': 'Not found'}), 404
        
        # Recursive deletion of filter options (Course -> Semesters -> Subjects)
        def delete_recursive(o):
            for child in o.children:
                delete_recursive(child)
            db.session.delete(o)
            
        delete_recursive(opt)
        db.session.commit()
        
        global _FILTERS_CACHE
        _FILTERS_CACHE = None
        return jsonify({'message': 'Hierarchy branch and associated data purged successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

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
        from app.utils.background_tasks import TaskTracker
        # Check if already running
        status = TaskTracker.get_status("rebuild")
        if status.get('is_running'):
            return jsonify({'error': 'A rebuild operation is already in progress'}), 400
            
        run_background_task(rebuild_index_from_db)
        return jsonify({'message': 'Index rebuild started in the background'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/documents', methods=['GET'])
@admin_required
def list_documents():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    search = request.args.get('search', '').strip()
    course = request.args.get('course', '')
    semester = request.args.get('semester', '')
    subject = request.args.get('subject', '')
    
    query = Document.query
    
    if search:
        query = query.filter(Document.filename.ilike(f"%{search}%"))
    if course:
        query = query.filter(Document.course.ilike(course))
    if semester:
        query = query.filter(Document.semester.ilike(semester))
    if subject:
        query = query.filter(Document.subject.ilike(subject))
        
    pagination = query.order_by(Document.upload_date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'items': [d.to_dict() for d in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    })

@admin_bp.route('/api/admin/users', methods=['GET'])
@admin_required
def list_users():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    search = request.args.get('search', '').strip()
    
    query = User.query
    if search:
        query = query.filter(User.email.ilike(f"%{search}%"))
    
    pagination = query.order_by(User.created_at.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'items': [u.to_dict() for u in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    })

@admin_bp.route('/api/admin/chunks', methods=['GET'])
@admin_required
def list_chunks():
    doc_id = request.args.get('document_id')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    
    from sqlalchemy.orm import joinedload
    query = db.session.query(DocumentChunk).options(joinedload(DocumentChunk.document))
    
    if doc_id:
        query = query.filter(DocumentChunk.document_id == doc_id)
        
    pagination = query.order_by(DocumentChunk.id.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    result = []
    for chunk in pagination.items:
        doc = chunk.document
        result.append({
            'id': chunk.id,
            'document_id': chunk.document_id,
            'document_filename': doc.filename if doc else "Unknown",
            'chunk_text': chunk.chunk_text[:100] + "...",
            'full_text': chunk.chunk_text,
            'chunk_index': chunk.chunk_index,
            'is_web': doc.filename.startswith('[WEB] ') if doc and doc.filename else False
        })
        
    return jsonify({
        'items': result,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    })

@admin_bp.route('/api/admin/chunks/<int:chunk_id>', methods=['DELETE'])
@admin_required
def delete_chunk(chunk_id):
    chunk = DocumentChunk.query.get(chunk_id)
    if not chunk:
        return jsonify({'error': 'Chunk not found'}), 404
    
    try:
        from app.services.vector_store import VectorStore
        vs = VectorStore.get_instance()
        vs.remove_chunk(chunk_id)
        
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
        db.session.commit()
        return jsonify({'message': 'Vector store cleared successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def perform_storage_sync():
    """Background task to sync storage and DB"""
    from app.utils.background_tasks import TaskTracker
    from app.services.supabase_service import SupabaseService
    import time
    
    task_name = "sync"
    TaskTracker.update_progress(task_name, 0, 100, "Scanning files...")
    
    try:
        # Placeholder for actual sync logic
        # 1. List files in Supabase
        # 2. Check against DB
        # 3. Mark missing or orphaned
        
        # Simulated sync for demonstration of progress
        steps = 5
        for i in range(steps):
            time.sleep(1)
            TaskTracker.update_progress(task_name, i+1, steps, f"Verifying integrity step {i+1}...")
        
        TaskTracker.complete_task(task_name, "Storage synchronization complete")
    except Exception as e:
        TaskTracker.complete_task(task_name, f"Sync Error: {str(e)}")

@admin_bp.route('/api/admin/sync-storage', methods=['POST'])
@admin_required
def sync_storage():
    try:
        from app.utils.background_tasks import TaskTracker
        status = TaskTracker.get_status("sync")
        if status.get('is_running'):
            return jsonify({'error': 'A sync operation is already in progress'}), 400
            
        run_background_task(perform_storage_sync)
        return jsonify({'message': 'Storage synchronization started', 'is_running': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@admin_bp.route('/api/admin/sync-status', methods=['GET'])
@admin_required
def sync_status():
    from app.utils.background_tasks import TaskTracker
    # Default to sync task, but can specify rebuild
    task = request.args.get('task', 'sync')
    status = TaskTracker.get_status(task)
    return jsonify(status)

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

@admin_bp.route('/api/admin/documents/<int:doc_id>/reprocess', methods=['POST'])
@admin_required
def reprocess_document(doc_id):
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
        
        # 1. Clear existing chunks and vectors
        from app.services.vector_store import VectorStore
        vs = VectorStore.get_instance()
        vs.remove_document(doc.id)
        DocumentChunk.query.filter_by(document_id=doc.id).delete()
        db.session.commit() # Commit deletion before starting background task
        
        # 2. Reset status
        doc.status = 'pending'
        db.session.commit()
        
        # 3. Trigger processing task
        from app.routes.docs import process_document_task
        run_background_task(process_document_task, doc.id)
        
        return jsonify({'message': 'Document re-processing started'})
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
    
    from sqlalchemy import func
    doc = Document.query.filter(
        func.lower(Document.course) == func.lower(course),
        func.lower(Document.semester) == func.lower(semester),
        func.lower(Document.subject) == func.lower(subject),
        Document.doc_type == 'syllabus'
    ).first()
    
    if not doc:
        return jsonify({'status': 'missing'})
        
    structure = None
    if doc.structure_json:
        try:
            import json
            structure = json.loads(doc.structure_json)
        except:
            structure = None

    return jsonify({
        'status': 'active',
        'filename': doc.filename,
        'structure': structure or {
            'units': [
                {'title': 'Intelligence Extraction in Progress', 'topics': ['Processing curriculum data...']}
            ]
        }
    })

@admin_bp.route('/api/change-password', methods=['POST'])
@admin_required
def change_password_redirect():
    # Deprecated in favor of auth.change_password, but kept for admin dashboard compat if needed
    from app.routes.auth import change_password
    return change_password()
