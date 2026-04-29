from flask import Blueprint, request, jsonify, session, current_app
from app import db, limiter
from app.models import ChatMessage, ChatSession, User
from app.services.ai_service import AIService
from app.services.vector_store import VectorStore
from app.routes.auth import login_required
import uuid
import json
import logging
from datetime import datetime

chat_bp = Blueprint('chat', __name__)

# Cache for live scraping results to avoid hitting the network on every message
_SCRAPE_CACHE = {}
_SCRAPE_CACHE_TIME = {}
SCRAPE_CACHE_TTL = 3600 # 1 hour

@chat_bp.route('/api/query', methods=['POST'])
@limiter.limit("30 per hour")
@login_required
def chat():
    data = request.json
    question = (data.get('question') or '').strip()
    session_id = data.get('session_id')
    mode = data.get('mode', 'syllabus') # 'syllabus' or 'general'
    
    if not question:
        return jsonify({'error': 'Question is required'}), 400
        
    user = User.query.get(session['user_id'])
    
    # 1. Handle Session
    if not session_id:
        session_id = str(uuid.uuid4())
        new_session = ChatSession(id=session_id, user_id=user.id, title=question[:50])
        db.session.add(new_session)
    else:
        chat_session = ChatSession.query.get(session_id)
        if not chat_session or chat_session.user_id != user.id:
            session_id = str(uuid.uuid4())
            new_session = ChatSession(id=session_id, user_id=user.id, title=question[:50])
            db.session.add(new_session)

    try:
        # 2. Retrieval
        vector_store = VectorStore.get_instance()
        ai_service = AIService()
        
        # Get embeddings for query
        query_emb = ai_service.get_embeddings([question])[0]
        
        # Search static vectors with subject-level grounding
        search_filter = {}
        if mode == 'syllabus':
            if user.pref_course: search_filter['course'] = user.pref_course
            if user.pref_semester: search_filter['semester'] = user.pref_semester
            if user.pref_subject: search_filter['subject'] = user.pref_subject
            
        results = vector_store.search(query_emb, k=10, filter=search_filter if search_filter else None)
        
        # 2.5 Live Search (General Mode Only)
        live_context = ""
        if mode == 'general':
            from app.models import AppSetting
            is_live = AppSetting.get('general_live_search', 'false') == 'true'
            if is_live:
                urls_json = AppSetting.get('general_website_urls', '[]')
                try:
                    urls = json.loads(urls_json)
                    if urls:
                        # Perform a quick, targeted fetch of the primary configured URLs
                        import time
                        now = time.time()
                        for url in urls[:2]:
                            # Check cache first
                            if url in _SCRAPE_CACHE and (now - _SCRAPE_CACHE_TIME.get(url, 0)) < SCRAPE_CACHE_TTL:
                                pages = _SCRAPE_CACHE[url]
                                ok = True
                            else:
                                from app.services.web_scraper import WebScraper
                                ok, pages = WebScraper.crawl_website(url, max_pages_override=1, time_cap_override=5)
                                if ok:
                                    _SCRAPE_CACHE[url] = pages
                                    _SCRAPE_CACHE_TIME[url] = now
                                    
                            if ok and pages:
                                for page_url, text in pages:
                                    # Add a snippet of live content to context
                                    live_context += f"\n[LIVE DATA FROM {page_url}]:\n{text[:2000]}\n"
                except Exception as le:
                    logging.warning(f"Live search failed: {le}")

        # 2.7 Fetch Master Syllabus Structure for Grounding
        syllabus_structure = None
        if mode == 'syllabus' and user.pref_course and user.pref_semester and user.pref_subject:
            from app.models import Document
            master_doc = Document.query.filter_by(
                course=user.pref_course,
                semester=user.pref_semester,
                subject=user.pref_subject,
                doc_type='syllabus'
            ).first()
            if master_doc and master_doc.structure_json:
                syllabus_structure = master_doc.structure_json

        # 3. Generation
        context = "\n\n".join([r['text'] for r in results])
        if live_context:
            context = f"--- LIVE WEB DATA ---\n{live_context}\n\n--- STATIC KNOWLEDGE ---\n{context}"

        answer = ai_service.generate_answer(
            question, 
            context,
            mode=mode,
            user_preferred_name=user.preferred_name,
            course=user.pref_course,
            semester=user.pref_semester,
            subject=user.pref_subject,
            syllabus_context=syllabus_structure
        )
        
        # 4. Save Message
        new_msg = ChatMessage(
            user_id=user.id,
            session_id=session_id,
            question=question,
            answer=answer,
            course=user.pref_course,
            semester=user.pref_semester,
            subject=user.pref_subject,
            sources_json=json.dumps([{'filename': r.get('filename'), 'url': r.get('url')} for r in results])
        )
        db.session.add(new_msg)
        db.session.commit()
        
        return jsonify({
            'answer': answer,
            'session_id': session_id,
            'message_id': new_msg.id,
            'sources': results
        })
        
    except Exception as e:
        logging.error(f"Chat error: {e}", exc_info=True)
        return jsonify({'error': 'Failed to generate answer'}), 500

@chat_bp.route('/api/chat/sessions', methods=['GET'])
@login_required
def list_sessions():
    sessions = ChatSession.query.filter_by(user_id=session['user_id']).order_by(ChatSession.updated_at.desc()).all()
    return jsonify([s.to_dict() for s in sessions])

@chat_bp.route('/api/chat/sessions/<string:session_id>', methods=['GET', 'DELETE'])
@login_required
def handle_session(session_id):
    chat_session = ChatSession.query.get(session_id)
    if not chat_session or chat_session.user_id != session['user_id']:
        return jsonify({'error': 'Not found'}), 404
        
    if request.method == 'GET':
        messages = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.asc()).all()
        return jsonify({
            'session': chat_session.to_dict(),
            'messages': [m.to_dict() for m in messages]
        })
    else:
        db.session.delete(chat_session)
        db.session.commit()
        return jsonify({'message': 'Session deleted'})

@chat_bp.route('/api/chat/sessions/<string:session_id>/rename', methods=['POST'])
@login_required
def rename_session(session_id):
    chat_session = ChatSession.query.get(session_id)
    if not chat_session or chat_session.user_id != session['user_id']:
        return jsonify({'error': 'Not found'}), 404
    
    data = request.json
    title = data.get('title')
    if not title: return jsonify({'error': 'Title required'}), 400
    
    chat_session.title = title
    db.session.commit()
    return jsonify({'message': 'Session renamed'})

@chat_bp.route('/api/chat/message/<int:message_id>/feedback', methods=['POST'])
@login_required
def message_feedback(message_id):
    msg = ChatMessage.query.get(message_id)
    if not msg or msg.user_id != session['user_id']:
        return jsonify({'error': 'Not found'}), 404
    
    data = request.json
    feedback = data.get('feedback') # 'like', 'dislike', or null
    msg.feedback = feedback
    db.session.commit()
    return jsonify({'message': 'Feedback saved'})
