from flask import Blueprint, request, jsonify, session, current_app
from app import db, limiter
from app.models import ChatMessage, ChatSession, User
# Lazy imports to avoid circular dependency on startup
# from app.services.ai_service import AIService
# from app.services.vector_store import VectorStore
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
        # 1.5 Fetch History
        history = []
        if session_id:
            past_msgs = ChatMessage.query.filter_by(session_id=session_id).order_by(ChatMessage.created_at.desc()).limit(6).all()
            # Reverse to get chronological order for the AI
            for m in reversed(past_msgs):
                history.append({"role": "user", "content": m.question})
                history.append({"role": "assistant", "content": m.answer})

        # 2. Retrieval & AI Service Init (Lazy Loaded)
        from app.services.ai_service import AIService
        from app.services.vector_store import VectorStore
        
        vector_store = VectorStore.get_instance()
        ai_service = AIService()
        
        # 2.1 Handle Smalltalk Shortcut
        if ai_service.is_smalltalk(question):
            answer = ai_service.generate_smalltalk(
                question, 
                user_preferred_name=user.preferred_name,
                course=user.pref_course,
                semester=user.pref_semester,
                subject=user.pref_subject
            )
            
            # Save smalltalk message without sources
            new_msg = ChatMessage(
                user_id=user.id,
                session_id=session_id,
                question=question,
                answer=answer,
                course=user.pref_course,
                semester=user.pref_semester,
                subject=user.pref_subject,
                sources_json="[]"
            )
            db.session.add(new_msg)
            db.session.commit()
            
            return jsonify({
                'answer': answer,
                'session_id': session_id,
                'message_id': new_msg.id,
                'sources': []
            })

        # 2.2 Query Parameters & Rewriting
        course = (data.get('course') or user.pref_course or '').strip()
        semester = (data.get('semester') or user.pref_semester or '').strip()
        subject = (data.get('subject') or user.pref_subject or '').strip()

        search_query = ai_service.rewrite_query(question, history) if history else question
        
        # Get embeddings for search query
        query_emb = ai_service.get_embeddings([search_query])[0]
        
        # 2.5 Live Search (General Mode Only)
        live_context = ""
        # Live search removed. Relying entirely on daily automated background scraper (WebSourceAutoRefresher)
        # for maximum chat speed and accuracy.
        
        # 2.7 Fetch Master Syllabus Structure for Grounding
        syllabus_structure = None
        if mode == 'syllabus':
            from app.models import Document
            from sqlalchemy import func
            
            # Use stripped and lowercase versions for better matching
            c_low = (course or "").strip().lower()
            s_low = (semester or "").strip().lower()
            sub_low = (subject or "").strip().lower()
            
            # Try 1: Strict Match (Course + Semester + Subject)
            if c_low and s_low and sub_low:
                master_doc = Document.query.filter(
                    func.lower(Document.course) == c_low,
                    func.lower(Document.semester) == s_low,
                    func.lower(Document.subject) == sub_low,
                    Document.doc_type == 'syllabus',
                    Document.status == 'processed'
                ).order_by(Document.created_at.desc()).first()
                
                if master_doc and master_doc.structure_json:
                    syllabus_structure = master_doc.structure_json
                    course, semester, subject = master_doc.course, master_doc.semester, master_doc.subject
            
            # Try 2: Loose Match (Subject + Course) if strict failed or filters incomplete
            if not syllabus_structure and sub_low:
                query = Document.query.filter(
                    func.lower(Document.subject) == sub_low,
                    Document.doc_type == 'syllabus',
                    Document.status == 'processed'
                )
                if c_low:
                    query = query.filter(func.lower(Document.course) == c_low)
                
                master_doc = query.order_by(Document.created_at.desc()).first()
                if master_doc and master_doc.structure_json:
                    syllabus_structure = master_doc.structure_json
                    course, semester, subject = master_doc.course, master_doc.semester, master_doc.subject
            
            # Try 3: Very Loose Match (Subject only) as last resort
            if not syllabus_structure and sub_low:
                master_doc = Document.query.filter(
                    Document.subject.ilike(f"%{subject.strip()}%"),
                    Document.doc_type == 'syllabus',
                    Document.status == 'processed'
                ).order_by(Document.created_at.desc()).first()
                if master_doc and master_doc.structure_json:
                    syllabus_structure = master_doc.structure_json
                    course, semester, subject = master_doc.course, master_doc.semester, master_doc.subject

        # Update search filter with finalized values for vector retrieval
        search_filter = {}
        if mode == 'syllabus':
            if course: search_filter['course'] = course.strip().upper()
            if semester: search_filter['semester'] = semester.strip().upper()
            if subject: search_filter['subject'] = subject.strip().upper()
        elif mode == 'general':
            search_filter['doc_type'] = 'general'

        # 3. Retrieval
        # Increase k to 30 to ensure we capture specific facts (like a name) that might be buried 
        # among dozens of other semantically similar pages (e.g. "former vice chancellors").
        results = vector_store.search(query_emb, k=30, filter=search_filter if search_filter else None)
        
        # 4. Generation
        context = "\n\n".join([r['text'] for r in results])
        if live_context:
            context = f"--- LIVE WEB DATA ---\n{live_context}\n\n--- STATIC KNOWLEDGE ---\n{context}"

        answer = ai_service.generate_answer(
            question, 
            context,
            mode=mode,
            history=history,
            user_preferred_name=user.preferred_name,
            course=course,
            semester=semester,
            subject=subject,
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
