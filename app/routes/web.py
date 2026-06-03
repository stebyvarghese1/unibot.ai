from flask import Blueprint, request, jsonify, session, current_app
from app import db
from app.models import Document, DocumentChunk
from app.services.web_scraper import WebScraper
from app.services.vector_store import VectorStore
from app.services.document_processor import DocumentProcessor
from app.utils.background_tasks import run_background_task
from app.routes.auth import admin_required
from urllib.parse import urlparse
import logging

web_bp = Blueprint('web', __name__)

def process_website_task(doc_id, url, filename):
    """Internal task for crawling a website in the background"""
    from app.services.web_scraper import WebScraper
    from app.services.vector_store import VectorStore
    from app.services.document_processor import DocumentProcessor
    
    try:
        # 1. Scrape
        ok, pages = WebScraper.crawl_website(url, max_pages_override=10000, time_cap_override=10800)
        
        doc = Document.query.get(doc_id)
        if not ok or not pages:
            if doc:
                doc.status = 'error'
                db.session.commit()
            return

        # 2. Process & Chunk
        total_chunks = 0
        chunks_to_add = []
        all_chunk_texts = []
        all_chunk_metas = []
        
        for page_url, raw_text in pages:
            text = DocumentProcessor._sanitize_text(raw_text)
            from app.services.web_scraper import GENERAL_MODE_CHUNK_WORDS, GENERAL_MODE_CHUNK_OVERLAP
            chunks = DocumentProcessor.chunk_text(text, chunk_size=GENERAL_MODE_CHUNK_WORDS, overlap=GENERAL_MODE_CHUNK_OVERLAP)
            for i, chunk_text in enumerate(chunks):
                final_text = f"[Source: {page_url}]\n{chunk_text}"
                chunk_obj = DocumentChunk(
                    document_id=doc_id,
                    chunk_text=final_text,
                    chunk_index=total_chunks
                )
                db.session.add(chunk_obj)
                chunks_to_add.append((chunk_obj, page_url))
                total_chunks += 1
            
        # Commit to get chunk IDs
        db.session.commit()
        
        # 3. Vectorize
        for c, page_url in chunks_to_add:
            all_chunk_texts.append(c.chunk_text)
            all_chunk_metas.append({
                'text': c.chunk_text,
                'doc_id': doc_id,
                'chunk_id': c.id,
                'url': page_url,
                'filename': filename,
                'doc_type': doc.doc_type if doc else 'general',
                'course': doc.course.strip().upper() if (doc and doc.course) else None,
                'semester': doc.semester.strip().upper() if (doc and doc.semester) else None,
                'subject': doc.subject.strip().upper() if (doc and doc.subject) else None
            })
            
        if all_chunk_texts:
            vector_store = VectorStore.get_instance()
            vector_store.add_texts(all_chunk_texts, all_chunk_metas)
        
        doc.status = 'processed'
        db.session.commit()
        logging.info(f"Successfully processed website {url}")
        
    except Exception as e:
        logging.error(f"Website processing failed for {doc_id}: {e}", exc_info=True)
        doc = Document.query.get(doc_id)
        if doc:
            doc.status = 'error'
            db.session.commit()

@web_bp.route('/api/admin/add-website', methods=['POST'])
@admin_required
def add_website():
    try:
        data = request.json
        url = (data.get('url') or '').strip()
        course = data.get('course')
        semester = data.get('semester')
        subject = data.get('subject')

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        domain = urlparse(url).netloc or 'unknown'
        filename = f"[WEB] {domain} - {url}"[:250]
        
        new_doc = Document(
            filename=filename,
            file_path=url,
            uploaded_by=session['user_id'],
            status='processing',
            course=course,
            semester=semester,
            subject=subject,
            doc_type='general' if course == 'General Mode' else 'syllabus'
        )
        db.session.add(new_doc)
        db.session.commit()
        
        run_background_task(process_website_task, new_doc.id, url, new_doc.filename)

        return jsonify({
            'message': 'Website scraping started.',
            'document_id': new_doc.id,
            'status': 'processing'
        })
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Add website failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500
