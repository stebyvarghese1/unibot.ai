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
    from app.services.document_processor import DocumentProcessor
    from app.services.vector_store import VectorStore
    
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
        
        # 2. Extract text and metadata
        # We need to save temporary file to disk for some processors
        temp_path = f"temp_{doc_id}_{doc.filename}"
        with open(temp_path, "wb") as f:
            f.write(file_bytes)
            
        try:
            # Process based on extension
            ext = doc.filename.rsplit('.', 1)[1].lower()
            text = ""
            if ext == 'pdf':
                text = DocumentProcessor.extract_text_from_pdf(temp_path)
            elif ext == 'docx':
                text = DocumentProcessor.extract_text_from_docx(temp_path)
            elif ext == 'pptx':
                text = DocumentProcessor.extract_text_from_pptx(temp_path)
            elif ext == 'txt':
                with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

        if not text:
            raise ValueError("No text extracted from document")

        # 3. Chunk and Embed
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
            
        # We need to commit to get chunk IDs for metadata
        db.session.commit()
        
        for i, chunk_obj in enumerate(chunk_objects):
            chunk_texts.append(chunk_obj.chunk_text)
            chunk_metas.append({
                'text': chunk_obj.chunk_text,
                'doc_id': doc_id,
                'chunk_id': chunk_obj.id,
                'filename': doc.filename,
                'doc_type': doc.doc_type
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
        except Exception: pass
        
        # 3. Delete from DB (cascades will handle chunks if configured, but we'll be explicit)
        DocumentChunk.query.filter_by(document_id=doc.id).delete()
        db.session.delete(doc)
        db.session.commit()
        
        return jsonify({'message': 'Document deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
