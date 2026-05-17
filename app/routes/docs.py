from flask import Blueprint, request, jsonify, session, current_app
from app import db, limiter
from app.models import Document, DocumentChunk
from app.services.supabase_service import SupabaseService
from app.services.vector_store import VectorStore
from app.services.document_processor import DocumentProcessor
from app.utils.background_tasks import run_background_task
from app.routes.auth import admin_required
from werkzeug.utils import secure_filename
import logging
import os

docs_bp = Blueprint('docs', __name__)

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'pptx', 'txt'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_document_task(doc_id):
    """Internal task for processing a document in the background"""
    import json
    
    doc = Document.query.get(doc_id)
    if not doc:
        logging.error(f"Document {doc_id} not found for processing")
        return

    try:
        doc.status = 'processing'
        db.session.commit()
        
        # 1. Download file
        supa = SupabaseService()
        file_bytes = supa.download_file(doc.file_path)
        
        # 2. Extract text
        text = DocumentProcessor.extract_text_from_bytes(file_bytes, doc.filename)
        
        if not text:
            raise ValueError("No text extracted from document")

        # 3. Analyze Syllabus Structure (Intelligence Grounding)
        if doc.doc_type == 'syllabus':
            logging.info(f"🧠 Extracting Intelligence Schema for: {doc.filename}")
            # Initialize with empty structure to prevent UI hanging
            doc.structure_json = json.dumps({"units": []})
            db.session.commit()
            
            try:
                # Use the AI Service to find Units and Topics
                structure_data_raw = DocumentProcessor.analyze_syllabus_structure(text)
                if structure_data_raw:
                    # Validate JSON before saving
                    try:
                        structure_data = json.loads(structure_data_raw)
                        if 'units' in structure_data:
                            doc.structure_json = structure_data_raw
                            db.session.commit()
                            
                            # 3.1 Embed Units/Modules specifically (Intelligence Grounding)
                            vector_store = VectorStore.get_instance()
                            unit_texts = []
                            unit_metas = []
                            
                            for unit in structure_data['units']:
                                title = unit.get('title', 'Unknown Unit')
                                topics_list = unit.get('topics', [])
                                if not isinstance(topics_list, list): topics_list = []
                                topics = ", ".join(topics_list)
                                unit_summary = f"UNIT SYLLABUS: {title}\nTOPICS: {topics}"
                                
                                unit_texts.append(unit_summary)
                                unit_metas.append({
                                    'text': unit_summary,
                                    'doc_id': doc_id,
                                    'filename': doc.filename,
                                    'doc_type': 'unit_summary',
                                    'course': doc.course,
                                    'semester': doc.semester,
                                    'subject': doc.subject,
                                    'unit_title': title
                                })
                            
                            if unit_texts:
                                logging.info(f"📡 Embedding {len(unit_texts)} units for structural grounding.")
                                vector_store.add_texts(unit_texts, unit_metas)
                            
                        logging.info(f"✅ Intelligence Grounded for {doc.subject}")
                    except Exception as je:
                        logging.error(f"Failed to parse or embed unit structure: {je}")
            except Exception as e:
                logging.error(f"Failed to analyze syllabus structure: {e}")

        # 4. Chunk and Embed Raw Text (Standard Deep Search)
        chunks = DocumentProcessor.chunk_text(text)
        vector_store = VectorStore.get_instance()
        
        chunk_objects = []
        chunk_texts = []
        chunk_metas = []
        
        for i, chunk_text in enumerate(chunks):
            chunk_obj = DocumentChunk(
                document_id=doc_id,
                chunk_text=chunk_text,
                chunk_index=i
            )
            db.session.add(chunk_obj)
            chunk_objects.append(chunk_obj)
            
        db.session.commit()
        
        for i, chunk_obj in enumerate(chunk_objects):
            chunk_texts.append(chunk_obj.chunk_text)
            chunk_metas.append({
                'text': chunk_obj.chunk_text,
                'doc_id': doc_id,
                'chunk_id': chunk_obj.id,
                'filename': doc.filename,
                'doc_type': doc.doc_type,
                'course': doc.course,
                'semester': doc.semester,
                'subject': doc.subject
            })
            
        vector_store.add_texts(chunk_texts, chunk_metas)
        
        doc.status = 'processed'
        db.session.commit()
        logging.info(f"Successfully processed document {doc.filename}")
        
    except Exception as e:
        logging.error(f"Processing failed for document {doc_id}: {e}", exc_info=True)
        doc.status = 'error'
        db.session.commit()

@docs_bp.route('/api/admin/upload', methods=['POST'])
@limiter.limit("20 per hour")
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
        doc_type = request.form.get('doc_type', 'syllabus')
        
        file_bytes = file.read()
        try:
            supa = SupabaseService()
            storage_path = supa.upload_file(file_bytes, filename, content_type=file.mimetype)
            
            new_doc = Document(
                filename=filename,
                file_path=storage_path,
                uploaded_by=session['user_id'],
                status='pending',
                course=course,
                semester=semester,
                subject=subject,
                doc_type=doc_type
            )
            db.session.add(new_doc)
            db.session.commit()
            
            run_background_task(process_document_task, new_doc.id)
            
            return jsonify({
                'message': 'File uploaded and is being processed.',
                'document_id': new_doc.id,
                'status': 'processing'
            })
        except Exception as e:
            db.session.rollback()
            logging.error(f"Upload failed: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400

@docs_bp.route('/api/admin/documents/<int:doc_id>', methods=['DELETE'])
@admin_required
def delete_document(doc_id):
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
            
        # 1. Delete from Supabase Storage
        supa = SupabaseService()
        if doc.file_path and not str(doc.file_path).startswith(('http://', 'https://')):
            try: supa.delete_file(doc.file_path)
            except Exception: pass
            
        # 2. Delete from Vector Store
        try:
            vector_store = VectorStore.get_instance()
            vector_store.remove_document(doc_id)
        except Exception as e:
            logging.error(f"Vector deletion failed: {e}")
        
        # 3. Delete from DB (cascades will handle chunks if configured, but we'll be explicit)
        DocumentChunk.query.filter_by(document_id=doc_id).delete()
        db.session.delete(doc)
        db.session.commit()
        
        return jsonify({'message': 'Document deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@docs_bp.route('/api/admin/documents/<int:doc_id>/reprocess', methods=['POST'])
@admin_required
def reprocess_document(doc_id):
    """Manually re-trigger the intelligence extraction and vector indexing for a document."""
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
            
        # Reset status
        doc.status = 'pending'
        db.session.commit()
        
        # Start background task
        run_background_task(process_document_task, doc_id)
        
        return jsonify({'message': 'Reprocessing started', 'status': 'processing'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@docs_bp.route('/api/admin/documents/<int:doc_id>/role', methods=['PATCH'])
@admin_required
def update_document_role(doc_id):
    """Switch document between 'syllabus' and 'general' types."""
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
            
        data = request.json
        new_type = data.get('doc_type')
        if new_type not in ['syllabus', 'general', 'system_info']:
            return jsonify({'error': 'Invalid document type'}), 400
            
        doc.doc_type = new_type
        db.session.commit()
        
        # If it was promoted to syllabus, trigger reprocessing to get units
        if new_type == 'syllabus':
            run_background_task(process_document_task, doc.id)
            
        return jsonify({'message': f'Document role updated to {new_type}'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@docs_bp.route('/api/admin/documents', methods=['GET'])
@admin_required
def list_documents():
    """List documents with pagination and filtering."""
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 25, type=int)
    search = request.args.get('search', '').strip()
    course = request.args.get('course', '').strip()
    semester = request.args.get('semester', '').strip()
    subject = request.args.get('subject', '').strip()
    
    query = Document.query
    
    if search:
        query = query.filter(Document.filename.ilike(f"%{search}%"))
    if course:
        query = query.filter(Document.course == course)
    if semester:
        query = query.filter(Document.semester == semester)
    if subject:
        query = query.filter(Document.subject == subject)
        
    pagination = query.order_by(Document.upload_date.desc()).paginate(page=page, per_page=per_page, error_out=False)
    
    return jsonify({
        'items': [d.to_dict() for d in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': pagination.page
    })
