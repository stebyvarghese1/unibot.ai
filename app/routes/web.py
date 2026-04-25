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
        ok, pages = WebScraper.crawl_website(url, max_pages_override=30, time_cap_override=60)
        
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
                chunks_to_add.append(chunk_obj)
                total_chunks += 1
            
        # Commit to get chunk IDs
        db.session.commit()
        
        # 3. Vectorize
        for c in chunks_to_add:
            all_chunk_texts.append(c.chunk_text)
            # Find matching page_url for metadata
            # (In a real scenario, we might want better mapping, but this is consistent with original)
            all_chunk_metas.append({
                'text': c.chunk_text,
                'doc_id': doc_id,
                'chunk_id': c.id,
                'filename': filename
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
            doc_type='syllabus'
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
