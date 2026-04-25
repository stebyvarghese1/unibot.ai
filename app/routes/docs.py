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
            try:
                # Use the AI Service to find Units and Topics
                structure_data_raw = DocumentProcessor.analyze_syllabus_structure(text)
                if structure_data_raw:
                    doc.structure_json = structure_data_raw
                    
                    # 3.1 Embed Units/Modules specifically (Intelligence Grounding)
                    try:
                        structure_data = json.loads(structure_data_raw)
                        if 'units' in structure_data:
                            vector_store = VectorStore.get_instance()
                            unit_texts = []
                            unit_metas = []
                            
                            for unit in structure_data['units']:
                                title = unit.get('title', 'Unknown Unit')
                                topics = ", ".join(unit.get('topics', []))
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
                                
                    except Exception as je:
                        logging.error(f"Failed to parse or embed unit structure: {je}")
                    
                    logging.info(f"✅ Intelligence Grounded for {doc.subject}")
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
        except Exception: pass
        
        # 3. Delete from DB (cascades will handle chunks if configured, but we'll be explicit)
        DocumentChunk.query.filter_by(document_id=doc.id).delete()
        db.session.delete(doc)
        db.session.commit()
        
        return jsonify({'message': 'Document deleted successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500
