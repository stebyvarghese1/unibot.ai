from flask import Blueprint, request, jsonify, render_template, session, current_app, redirect
from app import db
from app.models import User, Document, DocumentChunk, ChatMessage, ChatSession, FilterOption, AppSetting
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.services.ai_service import AIService
from app.services.supabase_service import SupabaseService
from app.services.web_scraper import WebScraper
from werkzeug.utils import secure_filename
from werkzeug.security import check_password_hash, generate_password_hash
import os
import functools
import json
import logging
import time
from config import Config
from sqlalchemy.exc import ProgrammingError
_GENERAL_INDEX_CACHE = {}

bp = Blueprint('main', __name__)

# Constants for General Mode
GENERAL_MODE_CHUNK_WORDS = 150
GENERAL_MODE_CHUNK_OVERLAP = 30
GENERAL_MODE_MAX_CHUNKS = 1000
GENERAL_MODE_QUICK_MAX_CHUNKS = 50
GENERAL_MODE_EMBED_BATCH = 8
GENERAL_MODE_CACHE_TTL = 3600
GENERAL_MODE_TOP_K = 5


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

@bp.route('/admin/general-mode')
def admin_general_mode():
    return render_template('admin/general_mode.html', active_page='general_mode')

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
        # Load prefs into session
        session['pref_course'] = user.pref_course
        session['pref_semester'] = user.pref_semester
        session['pref_subject'] = user.pref_subject
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
    user = User.query.get(session['user_id'])
    if not user:
        return jsonify({'error': 'User not found'}), 404

    if request.method == 'GET':
        return jsonify({
            'course': user.pref_course,
            'semester': user.pref_semester,
            'subject': user.pref_subject
        })
    
    data = request.json or {}
    user.pref_course = (data.get('course') or '').strip() or None
    user.pref_semester = (data.get('semester') or '').strip() or None
    user.pref_subject = (data.get('subject') or '').strip() or None
    
    # Also update session for backward compatibility if any other code uses it
    session['pref_course'] = user.pref_course
    session['pref_semester'] = user.pref_semester
    session['pref_subject'] = user.pref_subject
    
    db.session.commit()
    return jsonify({'message': 'Preferences saved'})
@bp.route('/api/filters', methods=['GET'])
@login_required
def list_filters():
    try:
        # Use declared filter options to support hierarchy
        opts = FilterOption.query.all()
        all_list = [o.to_dict() for o in opts]
        # Also return flat arrays for profile and simple dropdowns
        courses = sorted(set(o.value for o in opts if o.category == 'course' and o.value))
        semesters = sorted(set(o.value for o in opts if o.category == 'semester' and o.value))
        subjects = sorted(set(o.value for o in opts if o.category == 'subject' and o.value))
        return jsonify({
            'all': all_list,
            'courses': courses,
            'semesters': semesters,
            'subjects': subjects
        })
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
        doc_type = request.form.get('doc_type', 'syllabus')
        
        if doc_type == 'syllabus':
            if not course or not semester or not subject:
                return jsonify({'error': 'Course, semester, and subject are required for syllabus documents'}), 400
        
        file_bytes = file.read()
        try:
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
                course=course if doc_type == 'syllabus' else None,
                semester=semester if doc_type == 'syllabus' else None,
                subject=subject if doc_type == 'syllabus' else None,
                doc_type=doc_type
            )
            db.session.add(new_doc)
            db.session.commit()
            
            # Trigger processing (async in real world, sync here for MVP)
            try:
                process_document(new_doc.id)
                return jsonify({'message': 'File uploaded and processed successfully'})
            except Exception as e:
                # If processing fails, we need to rollback to ensure the session is clean
                # Then we commit the error status change in a fresh transaction
                db.session.rollback()
                
                # Fetch doc again in fresh session/transaction attempt
                try:
                    failed_doc = Document.query.get(new_doc.id)
                    if failed_doc:
                        failed_doc.status = 'error'
                        db.session.commit()
                except Exception:
                    db.session.rollback()
                    
                logging.error(f"Processing failed for document {filename}: {e}", exc_info=True)
                return jsonify({'error': f'Processing failed: {str(e)}'}), 500
                
        except Exception as e:
            db.session.rollback()
            logging.error(f"Upload failed for document {filename}: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'File type not allowed'}), 400


@bp.route('/api/admin/add-website', methods=['POST'])
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

        # 1. Scrape
        # Limit to 30 pages and reasonable char limit for "Real Time" responsiveness
        # Import inside function to avoid circular imports if any
        from app.services.web_scraper import WebScraper
        ok, pages = WebScraper.crawl_website(url, max_pages_override=30, time_cap_override=60)
        
        if not ok:
            return jsonify({'error': f'Failed to scrape: {pages}'}), 400
        
        if not pages:
             return jsonify({'error': 'No content found on the website.'}), 400

        # 2. Create Document entry
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            filename = f"[WEB] {domain} - {url}"
            if len(filename) > 250:
                filename = filename[:247] + "..."
        except Exception:
            filename = f"[WEB] {url}"[:250]
        
        new_doc = Document(
            filename=filename,
            file_path=url, # Store URL in file_path
            uploaded_by=session['user_id'],
            status='processing',
            course=course,
            semester=semester,
            subject=subject
        )
        db.session.add(new_doc)
        db.session.commit()
        
        # 3. Process & Chunk
        total_chunks = 0
        from app.services.vector_store import VectorStore
        vector_store = VectorStore.get_instance()
        
        chunks_to_add = []
        
        # Pre-process chunks
        all_chunk_texts = []
        all_chunk_metas = []
        
        for page_url, raw_text in pages:
            # Sanitize text to remove NUL characters before processing
            text = DocumentProcessor._sanitize_text(raw_text)
            
            # Chunking
            chunks = DocumentProcessor.chunk_text(text)
            for i, chunk_text in enumerate(chunks):
                # Prepend source URL for citation
                final_text = f"[Source: {page_url}]\n{chunk_text}"
                
                chunk_obj = DocumentChunk(
                    document_id=new_doc.id,
                    chunk_text=final_text,
                    chunk_index=total_chunks
                )
                db.session.add(chunk_obj)
                chunks_to_add.append(chunk_obj)
                
                all_chunk_texts.append(final_text)
                all_chunk_metas.append({
                    'text': final_text,
                    'doc_id': new_doc.id,
                    'chunk_id': None, # We don't have ID yet
                    'url': page_url
                })
                
                total_chunks += 1
            
        new_doc.status = 'processed'
        db.session.commit()
        
        # 4. Update Index (Real-time)
        try:
            # Update IDs in metadata now that we've committed
            for i, c in enumerate(chunks_to_add):
                all_chunk_metas[i]['chunk_id'] = c.id
                
            if all_chunk_texts:
                vector_store.add_texts(all_chunk_texts, all_chunk_metas)
        except Exception as e:
            logging.error(f"Vector store update failed: {e}")
            # Non-fatal, can rebuild index later
        
        return jsonify({'message': f'Website scraped ({len(pages)} pages) and indexed ({total_chunks} chunks).'})
        
    except Exception as e:
        db.session.rollback()
        logging.error(f"Add website failed: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


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
        from app.services.vector_store import VectorStore
        vector_store = VectorStore.get_instance()
        vector_store.clear()
        
        chunks = DocumentChunk.query.all()
        if not chunks:
            return jsonify({'message': 'Index cleared. No chunks to index.'})
            
        texts = [c.chunk_text for c in chunks]
        # Use the new add_texts method which handles embedding internally
        # Prepare metadata
        # include filename and public URL
        doc_map = {d.id: d for d in Document.query.all()}
        supa = SupabaseService()
        metadata = [{
            'text': c.chunk_text,
            'doc_id': c.document_id,
            'doc_type': doc_map[c.document_id].doc_type if c.document_id in doc_map else 'syllabus',
            'filename': doc_map[c.document_id].filename if c.document_id in doc_map else None,
            'url': supa.get_public_url(doc_map[c.document_id].file_path) if c.document_id in doc_map else None
        } for c in chunks]
        
        vector_store.add_texts(texts, metadata)
        
        # REMOVED FOR RENDER COMPATIBILITY - each worker maintains its own in-memory index
        # vector_store.save_index('vector_index')
        
        return jsonify({'message': f'Index rebuilt with {len(chunks)} chunks.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/stats', methods=['GET'])
@admin_required
def get_stats():
    from app.services.vector_store import VectorStore
    vector_store = VectorStore.get_instance()
    return jsonify(vector_store.get_stats())

@bp.route('/api/admin/chunks', methods=['GET'])
@admin_required
def list_chunks():
    try:
        document_id = request.args.get('document_id', type=int)
        
        # Use a join to fetch the document name in the same query (Fixes N+1 problem)
        query = db.session.query(DocumentChunk, Document.filename)\
            .join(Document, DocumentChunk.document_id == Document.id)\
            .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
            
        if document_id is not None:
            query = query.filter(DocumentChunk.document_id == document_id)
            
        results = query.limit(500).all()
        
        output = []
        for chunk, filename in results:
            c_dict = chunk.to_dict()
            c_dict['document_filename'] = filename or "Unknown"
            c_dict['document_id'] = chunk.document_id
            c_dict['full_text'] = chunk.chunk_text
            c_dict['is_web'] = filename.startswith('[WEB]') if filename else False
            output.append(c_dict)
            
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/admin/chunks/<int:chunk_id>', methods=['DELETE'])
@admin_required
def delete_chunk(chunk_id):
    try:
        chunk = DocumentChunk.query.get(chunk_id)
        if not chunk:
            return jsonify({'error': 'Chunk not found'}), 404
        db.session.delete(chunk)
        db.session.commit()
        return jsonify({'message': 'Chunk deleted'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


def _normalize_website_url(url):
    """Ensure URL has scheme and is a string we can parse."""
    if not url or not isinstance(url, str):
        return ''
    s = url.strip().replace('\n', '').replace('\r', '')
    if not s:
        return ''
    if not s.startswith(('http://', 'https://')):
        s = 'https://' + s
    return s


# General mode: crawl → temporary embedding index → retrieve at query time (Perplexity-style).
GENERAL_MODE_MAX_PAGES = 25
GENERAL_MODE_MAX_TOTAL_CHARS = 200_000
GENERAL_MODE_MAX_CHUNKS = 200
GENERAL_MODE_EMBED_BATCH = 20
GENERAL_MODE_TOP_K = 8
GENERAL_MODE_CHUNK_WORDS = 350
GENERAL_MODE_CHUNK_OVERLAP = 50
GENERAL_MODE_CACHE_TTL = 300
GENERAL_MODE_QUICK_MAX_CHUNKS = 80

def _get_limits_for_url(url):
    from urllib.parse import urlparse
    try:
        netloc = (urlparse(_normalize_website_url(url)).netloc or '').lower()
    except Exception:
        netloc = ''
    if netloc.endswith('uoc.ac.in'):
        return {'max_pages': 120, 'max_chars': 1_200_000, 'time_cap': 35}
    return {'max_pages': GENERAL_MODE_MAX_PAGES, 'max_chars': GENERAL_MODE_MAX_TOTAL_CHARS, 'time_cap': 20}

def _fetch_sitemap_urls(base_url):
    try:
        import requests, xml.etree.ElementTree as ET, gzip, io
        from urllib.parse import urlparse
        b = _normalize_website_url(base_url).rstrip('/')
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0'}
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        base_netloc = (urlparse(b).netloc or '').lower()
        base_root = _domain_root(base_netloc)
        candidates = [b + '/sitemap.xml']
        try:
            rb = requests.get(b + '/robots.txt', headers=headers, timeout=5, verify=False)
            if rb.status_code == 200:
                for line in rb.text.splitlines():
                    line = (line or '').strip()
                    if not line:
                        continue
                    if line.lower().startswith('sitemap:'):
                        sm = line.split(':', 1)[1].strip()
                        if sm:
                            candidates.append(sm)
        except Exception:
            pass
        urls = set()
        nested = []
        def parse_xml(text):
            try:
                root = ET.fromstring(text)
                ns = '{http://www.sitemaps.org/schemas/sitemap/0.9}'
                for loc in root.iter(f'{ns}loc'):
                    u = _normalize_crawl_url((loc.text or '').strip())
                    if not u:
                        continue
                    p = urlparse(u)
                    nl = (p.netloc or '').lower()
                    if p.scheme in ('http', 'https') and (_domain_root(nl) == base_root):
                        if u.lower().endswith('.xml') or u.lower().endswith('.xml.gz'):
                            nested.append(u)
                        else:
                            urls.add(u)
            except Exception:
                pass
        for sm in candidates:
            try:
                r = requests.get(sm, headers=headers, timeout=5, verify=False)
                if r.status_code != 200:
                    continue
                content = r.content
                if sm.lower().endswith('.gz'):
                    try:
                        content = gzip.decompress(content)
                    except Exception:
                        try:
                            with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
                                content = gz.read()
                        except Exception:
                            content = r.text.encode('utf-8', 'ignore')
                parse_xml(content.decode('utf-8', 'ignore'))
            except Exception:
                continue
        for sm in nested[:5]:
            try:
                r = requests.get(sm, headers=headers, timeout=5, verify=False)
                if r.status_code != 200:
                    continue
                content = r.content
                if sm.lower().endswith('.gz'):
                    try:
                        content = gzip.decompress(content)
                    except Exception:
                        try:
                            with gzip.GzipFile(fileobj=io.BytesIO(content)) as gz:
                                content = gz.read()
                        except Exception:
                            content = r.text.encode('utf-8', 'ignore')
                parse_xml(content.decode('utf-8', 'ignore'))
            except Exception:
                continue
        if len(urls) > 100:
            urls = set(list(urls)[:100])
        return urls
    except Exception:
        return set()


def _extract_text_from_html(html, base_url):
    """Clean and extract main content from HTML. Returns (soup, text)."""
    from bs4 import BeautifulSoup
    soup_all = BeautifulSoup(html, 'html.parser')
    soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'header', 'footer', 'aside', 'iframe']):
        tag.decompose()
    body = soup.find('body') or soup
    text = (body.get_text(separator='\n', strip=True) if body else '') or soup.get_text(separator='\n', strip=True)
    text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    return soup_all, text


def _fetch_one_page_requests(url):
    """Fetch one URL with requests (no JS). Returns (True, soup, text) or (False, None, error_message)."""
    try:
        import requests
        url = _normalize_website_url(url)
        if not url:
            return False, None, 'Invalid URL'
        try:
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        except Exception:
            pass
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0'}
        r = requests.get(url, headers=headers, timeout=5, verify=False)
        r.raise_for_status()
        r.encoding = r.apparent_encoding or 'utf-8'
        soup, text = _extract_text_from_html(r.text, url)
        return True, soup, text
    except requests.RequestException as e:
        return False, None, str(e) or 'Could not fetch the page'
    except Exception as e:
        logging.exception('Single page fetch failed')
        return False, None, str(e)


def _fetch_one_page_playwright(url, page):
    """Fetch one URL with Playwright (renders JS). page is a Playwright page. Returns (True, soup, text) or (False, None, error_message)."""
    try:
        from bs4 import BeautifulSoup
        url = _normalize_website_url(url)
        if not url:
            return False, None, 'Invalid URL'
        page.goto(url, wait_until='domcontentloaded', timeout=15000)
        html = page.content()
        soup, text = _extract_text_from_html(html, url)
        return True, soup, text
    except Exception as e:
        logging.warning('Playwright fetch failed for %s: %s', url, e)
        return False, None, str(e)


def _fetch_one_page(url, playwright_page=None):
    """Fetch one URL; use Playwright if page given, else requests. Returns (True, soup, text) or (False, None, error_message)."""
    if playwright_page:
        return _fetch_one_page_playwright(url, playwright_page)
    return _fetch_one_page_requests(url)


def _normalize_crawl_url(u):
    """One canonical form for crawl dedup (strip fragment, trailing slash)."""
    from urllib.parse import urlparse
    try:
        p = urlparse(u)
        scheme = (p.scheme or 'https').lower()
        netloc = (p.netloc or '').lower()
        path = (p.path or '/').rstrip('/') or '/'
        return f"{scheme}://{netloc}{path}"
    except Exception:
        return u


def _domain_root(netloc):
    try:
        import importlib
        tld = importlib.import_module('tldextract')
        ext = tld.extract(netloc)
        rd = ext.registered_domain
        if rd:
            return rd.lower()
    except Exception:
        pass
    parts = (netloc or '').split('.')
    if len(parts) >= 3:
        sfx = parts[-2] + '.' + parts[-1]
        if sfx in ('ac.in', 'co.in', 'org.in', 'edu.in', 'gov.in', 'nic.in'):
            return '.'.join(parts[-3:]).lower()
        if sfx in ('co.uk', 'org.uk', 'gov.uk', 'ac.uk'):
            return '.'.join(parts[-3:]).lower()
        if sfx in ('com.au', 'org.au', 'net.au'):
            return '.'.join(parts[-3:]).lower()
    if len(parts) >= 2:
        return '.'.join(parts[-2:]).lower()
    return netloc.lower()


def _same_domain_links(soup, base_url):
    """Return set of absolute same-domain http(s) URLs from soup, given base_url."""
    from urllib.parse import urljoin, urlparse
    if not soup:
        return set()
    base = base_url.strip().rstrip('/') or base_url
    try:
        base_netloc = (urlparse(base).netloc or '').lower()
        base_root = _domain_root(base_netloc)
    except Exception:
        return set()
    out = set()
    for a in soup.find_all('a', href=True):
        href = (a.get('href') or '').strip()
        if not href or href.startswith('#') or href.startswith('mailto:') or href.startswith('javascript:'):
            continue
        try:
            absolute = urljoin(base, href)
            parsed = urlparse(absolute)
            if parsed.scheme not in ('http', 'https'):
                continue
            netloc = (parsed.netloc or '').lower()
            cand_root = _domain_root(netloc)
            if cand_root != base_root:
                continue
            out.add(_normalize_crawl_url(absolute))
        except Exception:
            continue
    return out


def _run_crawl_loop(queue, seen, playwright_page, max_pages, max_total_chars, time_cap_s):
    """One BFS crawl loop. Fills pages_list and total_chars; mutates queue and seen."""
    from collections import deque
    pages_list = []
    total_chars = 0
    pages_done = 0
    start_time = time.time()
    while queue and pages_done < max_pages and total_chars < max_total_chars and (time.time() - start_time) < time_cap_s:
        if playwright_page:
            current = queue.popleft()
            ok, soup, text = _fetch_one_page(current, playwright_page=playwright_page)
            if ok:
                if text and len(text) >= 15:
                    pages_list.append((current, text))
                    total_chars += len(text)
                    if total_chars > max_total_chars:
                        break
                pages_done += 1
                if soup and pages_done < max_pages:
                    for link in _same_domain_links(soup, current):
                        if link not in seen:
                            seen.add(link)
                            queue.append(link)
        else:
            batch = []
            while queue and len(batch) < 8 and pages_done + len(batch) < max_pages:
                batch.append(queue.popleft())
            if not batch:
                break
            try:
                from concurrent.futures import ThreadPoolExecutor
                def _task(u):
                    ok, soup, text = _fetch_one_page_requests(u)
                    return u, ok, soup, text
                with ThreadPoolExecutor(max_workers=6) as ex:
                    results = list(ex.map(_task, batch))
                for u, ok, soup, text in results:
                    if ok and text and len(text) >= 15:
                        pages_list.append((u, text))
                        total_chars += len(text)
                        if total_chars > max_total_chars:
                            break
                    if soup and pages_done < max_pages:
                        for link in _same_domain_links(soup, u):
                            if link not in seen:
                                seen.add(link)
                                queue.append(link)
                pages_done += len(batch)
            except Exception:
                for u in batch:
                    ok, soup, text = _fetch_one_page_requests(u)
                    if ok and text and len(text) >= 15:
                        pages_list.append((u, text))
                        total_chars += len(text)
                        if total_chars > max_total_chars:
                            break
                    if soup and pages_done < max_pages:
                        for link in _same_domain_links(soup, u):
                            if link not in seen:
                                seen.add(link)
                                queue.append(link)
                pages_done += len(batch)
    return pages_list, total_chars


def _fetch_website_pages(url, max_pages_override=None, max_chars_override=None, time_cap_override=None):
    """Recursively crawl same-domain site (BFS). Use Playwright for JS rendering when available.
    Returns (True, [(url, text), ...]) or (False, error_message)."""
    try:
        from collections import deque
        url = _normalize_website_url(url)
        if not url:
            return False, 'Invalid URL'
        url = _normalize_crawl_url(url)
        limits = _get_limits_for_url(url)
        if max_pages_override is not None:
            limits['max_pages'] = max_pages_override
        if max_chars_override is not None:
            limits['max_chars'] = max_chars_override
        if time_cap_override is not None:
            limits['time_cap'] = time_cap_override
        seen = {url}
        seeds = list(_fetch_sitemap_urls(url))
        if seeds:
            if len(seeds) > 30:
                seeds = seeds[:30]
            queue = deque([url] + seeds)
        else:
            queue = deque([url])
        pages_list = []
        total_chars = 0
        try:
            import importlib
            pwa = importlib.import_module('playwright.sync_api')
            with pwa.sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(ignore_https_errors=True)
                page = context.new_page()
                page.set_extra_http_headers({
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0'
                })
                pages_list, total_chars = _run_crawl_loop(queue, seen, page, limits['max_pages'], limits['max_chars'], limits['time_cap'])
                browser.close()
        except ImportError:
            pass
        except Exception as e:
            logging.warning('Playwright crawl failed, using requests: %s', e)
        if not pages_list:
            seen = {url}
            if seeds:
                queue = deque([url] + seeds)
            else:
                queue = deque([url])
            pages_list, total_chars = _run_crawl_loop(queue, seen, None, limits['max_pages'], limits['max_chars'], limits['time_cap'])
        if not pages_list:
            return False, 'No text content found on the site'
        logging.info('General mode crawl: %d pages, %d chars', len(pages_list), total_chars)
        return True, pages_list
    except Exception as e:
        logging.exception('Website crawl failed')
        return False, str(e)


def _tokenize_query(text):
    try:
        t = (text or '').lower()
        out = []
        cur = []
        for ch in t:
            if ch.isalnum():
                cur.append(ch)
            else:
                if cur:
                    w = ''.join(cur)
                    if len(w) >= 2:
                        out.append(w)
                    cur = []
        if cur:
            w = ''.join(cur)
            if len(w) >= 2:
                out.append(w)
        return list(dict.fromkeys(out))
    except Exception:
        return []

def _score_url_for_tokens(u, tokens):
    try:
        from urllib.parse import urlparse
        p = urlparse(u)
        path = (p.path or '').lower()
        score = 0
        for tok in tokens:
            if tok and tok in path:
                score += 3
        hints = ('result', 'exam', 'notification', 'student', 'admission', 'schedule', 'timetable')
        for h in hints:
            if h in path:
                score += 2
        return score
    except Exception:
        return 0

def _site_search_candidates(base_url, tokens):
    try:
        b = _normalize_website_url(base_url).rstrip('/')
        q = '+'.join(tokens[:4]) if tokens else ''
        cands = set()
        if q:
            cands.add(f"{b}/search?q={q}")
            cands.add(f"{b}/?s={q}")
            cands.add(f"{b}/search/?q={q}")
            cands.add(f"{b}/?q={q}")
        return cands
    except Exception:
        return set()

def _targeted_fetch_for_question(url, question):
    try:
        url = _normalize_website_url(url)
        tokens = _tokenize_query(question)
        ok, soup, text = _fetch_one_page_requests(url)
        same_links = set()
        if ok and soup:
            same_links = _same_domain_links(soup, url)
        sitemap_links = _fetch_sitemap_urls(url)
        search_links = _site_search_candidates(url, tokens)
        cands = set()
        for s in [same_links, sitemap_links, search_links]:
            for u in s:
                cands.add(_normalize_crawl_url(u))
        scored = []
        for u in cands:
            scored.append((u, _score_url_for_tokens(u, tokens)))
        scored.sort(key=lambda x: x[1], reverse=True)
        top = [u for u, _ in scored[:30]] or list(cands)[:20]
        pages_list = []
        if ok and text and len(text) >= 15:
            pages_list.append((url, text))
        if not top:
            ok2, pl = _fetch_website_pages(url)
            if ok2 and pl:
                return pl
            return pages_list
        try:
            from concurrent.futures import ThreadPoolExecutor
            def _task(u):
                ou, ok1, soup1, text1 = u, *_fetch_one_page_requests(u)
                return ou, ok1, text1
            with ThreadPoolExecutor(max_workers=6) as ex:
                for ou, ok1, text1 in ex.map(_task, top):
                    if ok1 and text1 and len(text1) >= 15:
                        pages_list.append((ou, text1))
                        if len(pages_list) >= 60:
                            break
        except Exception:
            for ou in top:
                ok1, soup1, text1 = _fetch_one_page_requests(ou)
                if ok1 and text1 and len(text1) >= 15:
                    pages_list.append((ou, text1))
                    if len(pages_list) >= 60:
                        break
        return pages_list
    except Exception:
        return []


def _build_general_index(pages_list):
    """Chunk all pages with source URL, embed in batches, build temporary index.
    Returns list of (embedding, text, source_url)."""
    chunks_with_sources = []  # (text, url)
    for page_url, text in pages_list:
        if not (text or text.strip()):
            continue
        chunks = DocumentProcessor.chunk_text(
            text.strip(),
            chunk_size=GENERAL_MODE_CHUNK_WORDS,
            overlap=GENERAL_MODE_CHUNK_OVERLAP
        )
        for c in chunks:
            if c and c.strip():
                chunks_with_sources.append((c.strip(), page_url))
    if len(chunks_with_sources) > GENERAL_MODE_MAX_CHUNKS:
        chunks_with_sources = chunks_with_sources[:GENERAL_MODE_MAX_CHUNKS]
    if not chunks_with_sources:
        return []
    index = []
    texts_only = [t for t, _ in chunks_with_sources]
    for i in range(0, len(texts_only), GENERAL_MODE_EMBED_BATCH):
        batch = texts_only[i:i + GENERAL_MODE_EMBED_BATCH]
        try:
            embs = AIService.get_embeddings(batch)
            for j, vec in enumerate(embs):
                idx = i + j
                if idx < len(chunks_with_sources):
                    text, url = chunks_with_sources[idx]
                    index.append((vec, text, url))
        except Exception as e:
            logging.warning('Embedding batch failed: %s', e)
    return index

def _get_general_index(url):
    """Retrieve or build a vector index for a specific website URL.
    Prioritizes pre-scraped data from the database if available.
    Returns (ok, index, error) where index is a list of (embedding, text, source_url).
    """
    now = time.time()
    
    # Check cache first
    c = _GENERAL_INDEX_CACHE.get(url)
    if c and now - c.get('ts', 0) < GENERAL_MODE_CACHE_TTL and c.get('index'):
        return True, c.get('index'), None

    # 1. Try to find pre-scraped chunks in the Database first
    try:
        # Search for documents that match this web URL
        # Filename starts with '[WEB] ' followed by the URL
        web_docs = Document.query.filter(Document.filename.like(f"[WEB] {url}%")).all()
        if web_docs:
            doc_ids = [d.id for d in web_docs]
            db_chunks = DocumentChunk.query.filter(DocumentChunk.document_id.in_(doc_ids)).all()
            if db_chunks:
                logging.info(f"Found {len(db_chunks)} pre-scraped chunks in DB for {url}")
                pages_list = []
                # Reconstruct pages_list or just build index directly
                # For simplicity, we'll treat chunks as a single page or grouped by doc
                texts_only = [c.chunk_text for c in db_chunks]
                index = []
                # Embed chunks in batches
                for i in range(0, len(texts_only), 32): # Larger batch since DB data is trusted
                    batch = texts_only[i:i + 32]
                    try:
                        embs = AIService.get_embeddings(batch)
                        for j, vec in enumerate(embs):
                            idx = i + j
                            if idx < len(texts_only):
                                # Try to match back to a URL if stored in metadata (MVP: just use the base url)
                                index.append((vec, texts_only[idx], url))
                    except Exception as e:
                        logging.warning(f"DB chunks embedding failed: {e}")
                
                if index:
                    _GENERAL_INDEX_CACHE[url] = {'ts': now, 'index': index}
                    return True, index, None

    except Exception as e:
        logging.error(f"Error checking DB for general index: {e}")

    # 2. Fallback to real-time scraping if not in DB or DB empty
    logging.info(f"No DB data for {url}, performing real-time scrape...")
    
    # Try fetching home page first for quick response
    ok, soup, text = WebScraper.fetch_one_page(url)
    if not ok or not text or len(text) < 50:
        # Fallback to crawl if home page empty/failed
        ok2, pages_list = WebScraper.crawl_website(url, max_pages_override=10, time_cap_override=10)
        if not ok2:
            return False, None, pages_list
        idx2 = _build_general_index(pages_list)
        _GENERAL_INDEX_CACHE[url] = {'ts': now, 'index': idx2}
        return True, idx2, None

    # We have text from home page, build quick index
    chunks = DocumentProcessor.chunk_text(text.strip(), chunk_size=GENERAL_MODE_CHUNK_WORDS, overlap=GENERAL_MODE_CHUNK_OVERLAP)
    if len(chunks) > GENERAL_MODE_QUICK_MAX_CHUNKS:
        chunks = chunks[:GENERAL_MODE_QUICK_MAX_CHUNKS]
    texts_only = [c for c in chunks if c and c.strip()]
    index = []
    
    # Embed chunks
    for i in range(0, len(texts_only), GENERAL_MODE_EMBED_BATCH):
        batch = texts_only[i:i + GENERAL_MODE_EMBED_BATCH]
        try:
            embs = AIService.get_embeddings(batch)
            for j, vec in enumerate(embs):
                idx = i + j
                if idx < len(texts_only):
                    index.append((vec, texts_only[idx], url))
        except Exception as e:
            logging.warning('Quick index embedding failed: %s', e)
            
    _GENERAL_INDEX_CACHE[url] = {'ts': now, 'index': index}
    
    # Background enrich
    try:
        import threading
        def enrich():
            # Deeper crawl in background
            ok2, pages_list = WebScraper.crawl_website(url, max_pages_override=50, max_chars_override=500_000, time_cap_override=60)
            if ok2 and isinstance(pages_list, list) and pages_list:
                idx2 = _build_general_index(pages_list)
                _GENERAL_INDEX_CACHE[url] = {'ts': time.time(), 'index': idx2}
        t = threading.Thread(target=enrich, daemon=True)
        t.start()
    except Exception:
        pass
    return True, index, None


def _general_retrieve(index, question, top_k=None):
    """Embed question, search temporary index, return top_k (text, source_url)."""
    if not index:
        return []
    top_k = top_k or GENERAL_MODE_TOP_K
    import numpy as np
    q_emb = AIService.get_embeddings([question])
    q_vec = np.array(q_emb[0] if isinstance(q_emb[0], list) else q_emb, dtype=np.float32)
    texts = []
    urls = []
    vecs = []
    for vec, text, url in index:
        vecs.append(np.array(vec if isinstance(vec, list) else vec, dtype=np.float32))
        texts.append(text)
        urls.append(url)
    q_norm = np.linalg.norm(q_vec) + 1e-9
    scores = []
    for v in vecs:
        vn = np.linalg.norm(v) + 1e-9
        scores.append(float(np.dot(q_vec, v) / (q_norm * vn)))
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(texts[i], urls[i]) for i in top_idx]


def _general_context_and_sources(retrieved):
    """Build context string (with source labels) and list of unique source URLs for display."""
    from urllib.parse import urlparse
    parts = []
    seen_urls = set()
    for text, url in retrieved:
        parts.append(f'[Source: {url}]\n{text}')
        seen_urls.add(url)
    context = '\n\n---\n\n'.join(parts)
    sources = []
    for u in sorted(seen_urls):
        try:
            path = (urlparse(u).path or '/').strip().rstrip('/') or '/'
            label = path if path != '/' else u
        except Exception:
            label = 'Website'
        sources.append({'url': u, 'filename': label})
    return context, sources


@bp.route('/api/admin/general-website', methods=['GET'])
@admin_required
def get_general_website():
    try:
        import json as _json
        url = AppSetting.get('general_chat_url') or ''
        urls_raw = AppSetting.get('general_chat_urls') or ''
        live_raw = (AppSetting.get('general_live_mode') or '').strip().lower()
        live = live_raw in ('1', 'true', 'yes', 'on')
        refresh_val = AppSetting.get('general_refresh_interval', 'never')
        urls = []
        if urls_raw:
            try:
                val = _json.loads(urls_raw)
                if isinstance(val, list):
                    urls = [(_normalize_website_url(u) or '').strip() for u in val if isinstance(u, str)]
                    urls = [u for u in urls if u]
            except Exception:
                urls = []
        if not urls and url:
            urls = [(_normalize_website_url(url) or '').strip()]
        return jsonify({'url': url, 'urls': urls, 'live': live, 'refresh': refresh_val})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/admin/general-website', methods=['POST'])
@admin_required
def set_general_website():
    try:
        import json as _json
        data = request.get_json() or {}
        urls = data.get('urls')
        url = (data.get('url') or '').strip()
        live_flag = data.get('live')
        refresh_interval = (data.get('refresh') or 'never').strip().lower()
        
        # Save Auto-Refresh setting
        if refresh_interval in ('never', '1', '7', '30'):
            AppSetting.set('general_refresh_interval', refresh_interval)

        if isinstance(live_flag, bool):
            AppSetting.set('general_live_mode', 'true' if live_flag else 'false')
        elif isinstance(live_flag, str):
            s = live_flag.strip().lower()
            AppSetting.set('general_live_mode', 'true' if s in ('1', 'true', 'yes', 'on') else 'false')
        if isinstance(urls, list):
            cleaned = []
            for u in urls:
                if not isinstance(u, str):
                    continue
                u2 = (_normalize_website_url(u) or '').strip()
                if u2 and u2.startswith(('http://', 'https://')):
                    cleaned.append(u2)
            
            # Allow empty list to effectively clear the config
            AppSetting.set('general_chat_urls', _json.dumps(cleaned))
            if cleaned:
                AppSetting.set('general_chat_url', cleaned[0])
            else:
                AppSetting.set('general_chat_url', '')
                
            AppSetting.set('general_chat_content', '')  # no cache
            return jsonify({'message': f'Saved {len(cleaned)} URL(s) and settings.'})
        else:
            if not url:
                return jsonify({'error': 'URL is required'}), 400
            url = _normalize_website_url(url)
            if not url.startswith(('http://', 'https://')):
                return jsonify({'error': 'URL must start with http:// or https://'}), 400
            AppSetting.set('general_chat_url', url)
            AppSetting.set('general_chat_urls', _json.dumps([url]))
            AppSetting.set('general_chat_content', '')  # no cache
            return jsonify({'message': 'Website and settings saved.'})
    except Exception as e:
        logging.exception('set_general_website failed')
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

@bp.route('/api/admin/documents', methods=['GET'])
@admin_required
def list_documents():
    try:
        from sqlalchemy import func
        # Optimized query using joins and aggregation to fetch everything in one go
        results = db.session.query(
            Document, 
            func.count(DocumentChunk.id).label('chunk_count'),
            User.email.label('uploaded_by_email')
        ).outerjoin(DocumentChunk, Document.id == DocumentChunk.document_id)\
         .outerjoin(User, Document.uploaded_by == User.id)\
         .group_by(Document.id, User.email)\
         .order_by(Document.upload_date.desc())\
         .all()
        
        output = []
        for doc, chunk_count, email in results:
            d_dict = doc.to_dict()
            d_dict['chunk_count'] = chunk_count
            d_dict['uploaded_by_email'] = email or "Unknown"
            output.append(d_dict)
            
        return jsonify(output)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/documents/<int:doc_id>/chunks', methods=['GET'])
@admin_required
def get_document_chunks(doc_id):
    try:
        doc = Document.query.get(doc_id)
        if not doc:
            return jsonify({'error': 'Document not found'}), 404
            
        chunks = DocumentChunk.query.filter_by(document_id=doc_id).order_by(DocumentChunk.chunk_index).all()
        return jsonify({
            'document': doc.to_dict(),
            'chunks': [c.to_dict() for c in chunks]
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- API Student ---
@bp.route('/api/admin/filter-options', methods=['GET'])
@admin_required
def list_filter_options():
    try:
        # Schema migration for parent_id (hack for MVP)
        try:
            from sqlalchemy import text
            eng = db.session.get_bind()
            # SQLite check
            if 'sqlite' in eng.dialect.name:
                 # Check if column exists
                 res = db.session.execute(text("PRAGMA table_info(filter_options)")).fetchall()
                 cols = [r[1] for r in res]
                 if 'parent_id' not in cols:
                     db.session.execute(text("ALTER TABLE filter_options ADD COLUMN parent_id INTEGER REFERENCES filter_options(id)"))
                     db.session.commit()
            else:
                 # Postgres
                 db.session.execute(text("ALTER TABLE public.filter_options ADD COLUMN IF NOT EXISTS parent_id INTEGER REFERENCES public.filter_options(id)"))
                 db.session.commit()
        except Exception as e:
            db.session.rollback()
            # logging.error(f"Schema migration failed: {e}")

        try:
            opts = FilterOption.query.all()
            return jsonify({'courses': sorted(list({o.value for o in opts if o.category == 'course'})),
                            'semesters': sorted(list({o.value for o in opts if o.category == 'semester'})),
                            'subjects': sorted(list({o.value for o in opts if o.category == 'subject'})),
                            'all': [o.to_dict() for o in opts]})
        except Exception as e:
            # Fallback if table/column issue
            print(f"FilterOption query failed: {e}")
            return jsonify({'courses': [], 'semesters': [], 'subjects': [], 'all': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@bp.route('/api/admin/filter-options', methods=['POST'])
@admin_required
def add_filter_option():
    try:
        data = request.json
        category = data.get('category')
        value = (data.get('value') or '').strip()
        parent_id = data.get('parent_id')
        
        if not category or not value:
            return jsonify({'error': 'Category and value required'}), 400
            
        if category not in ['course', 'semester', 'subject']:
            return jsonify({'error': 'Invalid category'}), 400
            
        # Check uniqueness within the same parent context
        query = FilterOption.query.filter_by(category=category, value=value)
        if parent_id:
            query = query.filter_by(parent_id=parent_id)
        else:
            query = query.filter(FilterOption.parent_id.is_(None))
            
        exists = query.first()
        if exists:
            return jsonify({'message': 'Option already exists'})
            
        opt = FilterOption(category=category, value=value, parent_id=parent_id)
        db.session.add(opt)
        db.session.commit()
        return jsonify({'message': 'Option added', 'option': opt.to_dict()})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

def _delete_filter_option_and_descendants(opt):
    """Delete this filter option and all descendants (children, grandchildren, etc.) from the DB."""
    if not opt:
        return
    children = FilterOption.query.filter_by(parent_id=opt.id).all()
    for child in children:
        _delete_filter_option_and_descendants(child)
    db.session.delete(opt)


@bp.route('/api/admin/filter-options/<int:id>', methods=['DELETE'])
@admin_required
def delete_filter_option(id):
    try:
        opt = FilterOption.query.get(id)
        if not opt:
            return jsonify({'message': 'Option deleted'})
        _delete_filter_option_and_descendants(opt)
        db.session.commit()
        return jsonify({'message': 'Option deleted'})
    except Exception as e:
        db.session.rollback()
        logging.exception('delete_filter_option failed')
        return jsonify({'error': str(e)}), 500

# --- API Student ---

@bp.route('/api/chat/sessions', methods=['GET'])
@login_required
def list_chat_sessions():
    sessions = ChatSession.query.filter_by(user_id=session['user_id']).order_by(ChatSession.updated_at.desc()).all()
    return jsonify([s.to_dict() for s in sessions])

@bp.route('/api/chat/sessions/<session_id>', methods=['GET'])
@login_required
def get_chat_history(session_id):
    session_id = (session_id or '').strip()
    if not session_id:
        return jsonify([])
        
    uid = session.get('user_id')
    msgs = ChatMessage.query.filter_by(session_id=session_id, user_id=uid).order_by(ChatMessage.created_at.asc()).all()
    
    result = []
    for m in msgs:
        sources = []
        if m.sources_json:
            try:
                sources = json.loads(m.sources_json)
            except Exception:
                sources = []
        
        result.append({
            'id': m.id,
            'question': m.question,
            'answer': m.answer,
            'sources': sources,
            'created_at': m.created_at.isoformat() + 'Z'
        })
    return jsonify(result)

@bp.route('/api/chat/sessions/<session_id>', methods=['DELETE'])
@login_required
def delete_chat_session(session_id):
    sess = ChatSession.query.filter_by(id=session_id, user_id=session['user_id']).first()
    if sess:
        db.session.delete(sess)
        db.session.commit()
    return jsonify({'message': 'Session deleted'})

@bp.route('/api/chat/sessions/<session_id>/rename', methods=['POST'])
@login_required
def rename_chat_session(session_id):
    data = request.json
    new_title = (data.get('title') or '').strip()
    if not new_title:
        return jsonify({'error': 'Title required'}), 400
    sess = ChatSession.query.filter_by(id=session_id, user_id=session['user_id']).first()
    if sess:
        sess.title = new_title
        db.session.commit()
        return jsonify({'message': 'Renamed'})
    return jsonify({'error': 'Session not found'}), 404

@bp.route('/api/query', methods=['POST'])
@login_required
def query():
    try:
        data = request.json
        question = (data.get('question') or '').strip()
        session_id = (data.get('session_id') or '').strip() or None
        mode = (data.get('mode') or 'studies').strip().lower()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400

        uid = session.get('user_id')
        import uuid
        if not session_id:
            session_id = str(uuid.uuid4())
            curr_sess = ChatSession(id=session_id, user_id=uid, title='New Chat')
            db.session.add(curr_sess)
            logging.info(f"Created new session: {session_id}")
        else:
            curr_sess = db.session.get(ChatSession, session_id)
            if not curr_sess:
                curr_sess = ChatSession(id=session_id, user_id=uid, title='New Chat')
                db.session.add(curr_sess)
                logging.info(f"Created session from provided ID: {session_id}")
            elif curr_sess.user_id != uid:
                session_id = str(uuid.uuid4())
                curr_sess = ChatSession(id=session_id, user_id=uid, title='New Chat')
                db.session.add(curr_sess)
                logging.info(f"Created new session due to user mismatch: {session_id}")
            
        # Auto-title if it's the first message
        if curr_sess.title == 'New Chat' or not curr_sess.title:
            title = question[:30] + ('...' if len(question) > 30 else '')
            curr_sess.title = title
        
        from datetime import datetime
        curr_sess.updated_at = datetime.utcnow()
        try:
            db.session.commit()
            session_title = curr_sess.title
        except Exception as se:
            db.session.rollback()
            logging.error(f"Failed to commit session update: {se}", exc_info=True)
            session_title = "New Chat"
            
        # Sanitize question to prevent database errors (NUL characters)
        question = DocumentProcessor._sanitize_text(question)
            
        # Fetch current user to get latest prefs from DB (handles cross-device sync)
        user = User.query.get(session['user_id'])
        pref_c = user.pref_course if user else None
        pref_s = user.pref_semester if user else None
        pref_sub = user.pref_subject if user else None

        course = (data.get('course') or pref_c or '').strip()
        semester = (data.get('semester') or pref_s or '').strip()
        subject = (data.get('subject') or pref_sub or '').strip()

        logging.info(f"Processing query: '{question}' mode={mode} - Course: {course}, Semester: {semester}, Subject: {subject}, Session: {session_id}")
        
        try:
            from app.services.ai_service import AIService
            if AIService.is_smalltalk(question):
                answer = AIService.generate_smalltalk(question)
                try:
                    msg = ChatMessage(
                        user_id=session['user_id'],
                        question=question,
                        answer=answer,
                        sources_json='[]',
                        session_id=session_id
                    )
                    db.session.add(msg)
                    db.session.commit()
                except Exception as e: 
                    db.session.rollback()
                    logging.error(f"Failed to save smalltalk message: {e}", exc_info=True)
                logging.info("Returning smalltalk response")
                return jsonify({'answer': answer, 'sources': []})

            if mode == 'general':
                from app.models import AppSetting
                
                # Get configured URLs (support for single or multiple)
                urls_raw = AppSetting.get('general_chat_urls')
                primary_url = AppSetting.get('general_chat_url')
                
                target_urls = []
                if urls_raw:
                    try:
                        target_urls = json.loads(urls_raw)
                    except Exception:
                        if primary_url: target_urls = [primary_url]
                elif primary_url:
                    target_urls = [primary_url]
                
                if not target_urls:
                     return jsonify({'answer': 'General mode is not configured. Please ask the admin to set a website URL in the admin dashboard.', 'sources': []})
                
                # Combine indices for all configured URLs
                all_index = []
                for url in target_urls:
                    ok, index, err = _get_general_index(url)
                    if ok and index:
                        all_index.extend(index)
                    else:
                        logging.warning(f"Could not build index for {url}: {err}")
                
                if not all_index:
                     return jsonify({'answer': 'I could not retrieve any content from the configured website(s). Please check if the URL is correct and accessible.', 'sources': []})
                
                # Retrieve from combined index
                retrieved = _general_retrieve(all_index, question)
                context, sources = _general_context_and_sources(retrieved)
                
                if not context:
                     return jsonify({'answer': 'I processed the website but found no content relevant to your question.', 'sources': []})
                
                # Generate
                answer = AIService.generate_answer_from_website(question, context, source_url=target_urls[0])
                
                # Save
                try:
                    msg = ChatMessage(
                        user_id=session['user_id'],
                        question=question,
                        answer=answer,
                        sources_json=json.dumps(sources),
                        session_id=session_id,
                        course='General',
                        semester='General',
                        subject='General'
                    )
                    db.session.add(msg)
                    db.session.commit()
                except Exception as e:
                    db.session.rollback()
                    logging.error(f"Failed to save general chat message: {e}")
                    
                return jsonify({
                    'answer': answer, 
                    'sources': sources, 
                    'session_id': session_id,
                    'session_title': session_title
                })

            # 1. Embed question (Studies mode)
            q_embedding = AIService.get_embeddings([question])
            # get_embeddings returns list of list (batch), we need the first one if it's a list
            if isinstance(q_embedding, list) and len(q_embedding) > 0:
                 # Handle varying return types from HF API (sometimes list of float, sometimes list of list)
                 if isinstance(q_embedding[0], list):
                     q_vec = q_embedding[0]
                 else:
                     q_vec = q_embedding
                 logging.info(f"Successfully embedded question. Vector length: {len(q_vec)}")
                 logging.info(f"Question embedding sample: {q_vec[:5] if len(q_vec) >= 5 else q_vec}")
            else:
                 logging.error(f"Failed to embed question. Response: {q_embedding}")
                 return jsonify({'error': 'Failed to embed question'}), 500

            # --- Intelligence Tier: Identity Intent Detection ---
            id_keywords = [
                'who are you', 'who you are', "who you're", 'what are you', 'your name', 
                'created you', 'developer', 'about yourself', 'your purpose', 
                'what can you do', 'how you work', 'about this software', 'about the bot'
            ]
            identity_intent = any(k in question.lower() for k in id_keywords)
            
            # 2. Search
            from app.services.vector_store import VectorStore
            vector_store = VectorStore.get_instance()
            
            # Dynamic K: Increase depth if we suspect an identity query to ensure system info is found
            search_k = 40 if identity_intent else 25
            results = vector_store.search(q_vec, k=search_k)
            logging.info(f"Initial vector search (Intent: {'Identity' if identity_intent else 'General'}) returned {len(results)} results")
            
            # Check if the index has vectors before applying distance filtering
            stats = vector_store.get_stats()
            if stats['total_vectors'] == 0:
                logging.warning("Vector store has 0 vectors - no documents indexed")
                return jsonify({'answer': 'Vector index empty — rebuild failed or no documents processed', 'sources': []})
            
            # 1. Fetch documents for metadata mapping fallback
            doc_map = {d.id: d for d in Document.query.all()}
            logging.info(f"Total documents in system: {len(doc_map)}")
            
            # PHASE 1: Confidence Filtering with Identity Bypass
            # Syllabus docs must be close (distance threshold), but Identity docs should be more resilient
            filtered = []
            for r in results:
                dist = r.get('distance')
                if dist is None: continue
                
                # Check type to see if we apply threshold
                did = r.get('doc_id') or r.get('document_id')
                dtype = r.get('doc_type')
                if not dtype or dtype == 'syllabus':
                     if did and did in doc_map:
                         dtype = getattr(doc_map[did], 'doc_type', 'syllabus')
                
                # Bypass threshold for system info or if distance is low enough
                if dtype == 'system_info' or dist <= Config.VECTOR_MAX_DISTANCE:
                    filtered.append(r)
            
            logging.info(f"Initial filtering: {len(results)} -> {len(filtered)} results (Threshold: {Config.VECTOR_MAX_DISTANCE}, Identities Bypassed)")
            
            # If nothing survived, fallback to best matches to avoid 0-context silence
            if not filtered and results:
                logging.info(f"0 results within threshold. Falling back to top {min(3, len(results))} raw results.")
                filtered = results[:3]
            
            # If we still don't have filtered results but have search results, take the closest one as final fallback
            if not filtered and results:
                closest_result = min(results, key=lambda x: x.get('distance', float('inf')))
                if closest_result:
                    logging.info(f"Taking closest result with distance: {closest_result.get('distance')}")
                    filtered = [closest_result]
            
            if course or semester or subject:
                def match_cat(r):
                    # Robust ID resolution (handle stale metadata keys)
                    did = r.get('doc_id') or r.get('document_id')
                    
                    # Try metadata type first (fast)
                    dtype = r.get('doc_type')
                    
                    # If metadata is missing or says syllabus, we MUST verify with DB for system_info
                    # because system info must ignore all filters
                    if not dtype or dtype == 'syllabus':
                        d = doc_map.get(did)
                        if d:
                            dtype = getattr(d, 'doc_type', 'syllabus')
                    
                    if dtype == 'system_info':
                        return True
                        
                    # Now handle syllabus filtering
                    if not did: return False
                    d = doc_map.get(did)
                    if not d: return False
                        
                    ok_course = True if not course else ((d.course or '').strip().lower() == course.lower())
                    ok_sem = True if not semester else ((d.semester or '').strip().lower() == semester.lower())
                    ok_subj = True if not subject else ((d.subject or '').strip().lower() == subject.lower())
                    return ok_course and ok_sem and ok_subj
                
                before_count = len(filtered)
                filtered = [r for r in filtered if match_cat(r)]
                logging.info(f"Category filtering: {before_count} -> {len(filtered)} results (Course: {course}, Semester: {semester}, Subject: {subject})")
                
            # PHASE 2: Intelligence Mixing & Primacy Protection
            # We must ensure System Identity bits aren't drowned out by syllabus bits
            system_bits = []
            academic_bits = []
            
            for r in filtered:
                did = r.get('doc_id') or r.get('document_id')
                dtype = r.get('doc_type')
                
                # Verify type if meta is stale
                if not dtype or dtype == 'syllabus':
                    if did and did in doc_map:
                        dtype = getattr(doc_map[did], 'doc_type', 'syllabus')
                
                if dtype == 'system_info':
                    system_bits.append(r)
                else:
                    academic_bits.append(r)
            
            # --- URGENT IDENTITY RECOVERY ---
            # If user asked about identity but vector search missed it, force load from DB
            if identity_intent and not system_bits:
                sys_docs = Document.query.filter_by(doc_type='system_info', status='processed').all()
                if sys_docs:
                    logging.info(f"Identity recovery: Manually loading chunks for {len(sys_docs)} system docs")
                    for sd in sys_docs:
                        # Grab first few chunks as identifying info
                        recovery_chunks = DocumentChunk.query.filter_by(document_id=sd.id).limit(3).all()
                        for rc in recovery_chunks:
                            system_bits.append({
                                'text': rc.chunk_text,
                                'doc_id': sd.id,
                                'doc_type': 'system_info',
                                'filename': sd.filename,
                                'distance': 0.0 # Force priority
                            })
            
            # IDENTITY PRIMACY: If we detected an identity intent, we prioritize identity docs even more
            # We take a higher ratio of system info bits when identity is the likely intent
            sys_limit = 8 if identity_intent else 4
            acad_limit = 3 if identity_intent else 6
            
            final_filtered = system_bits[:sys_limit] + academic_bits[:acad_limit]
            
            if not final_filtered:
                # Diagnostics: why was it empty?
                all_docs = Document.query.all()
                if not all_docs:
                    return jsonify({'answer': 'No documents have been uploaded yet.', 'sources': []})
                
                sys_docs = [d for d in all_docs if d.doc_type == 'system_info']
                logging.warning(f"Query '{question}' returned 0 survivors. Initial search: {len(results)}, System docs in DB: {len(sys_docs)}")
                
                return jsonify({
                    'answer': 'Not available in selected category (No context found).', 
                    'sources': [],
                    'debug_info': {
                        'total_docs': len(all_docs),
                        'system_docs': len(sys_docs),
                        'initial_matches': len(results)
                    }
                })
            
            # Overwrite filtered with our prioritized list
            filtered = final_filtered
                
            # 3. Generate Answer
            context = "\n\n".join([r['text'] for r in filtered])
            logging.info(f"Context Construction: {len(system_bits)} identity bits found, {len(academic_bits)} academic bits. Selected: {len(filtered)}")
            answer = AIService.generate_answer(question, context)
            
            # Deduplicate sources by doc_id
            unique = {}
            for r in filtered:
                did = r.get('doc_id') or r.get('document_id')
                
                # RECOGNITION: Check if this is a system/identity document
                dtype = r.get('doc_type')
                if not dtype or dtype == 'syllabus':
                    # Verify with DB if metadata is uncertain
                    if did and did in doc_map:
                        dtype = getattr(doc_map[did], 'doc_type', 'syllabus')
                
                # USER RULE: Don't show sources for 'About the Software' or Identity documents
                if dtype == 'system_info':
                    continue
                    
                key = did
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
                        'doc_id': did,
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
                    sources_json=json.dumps(sources),
                    session_id=session_id
                )
                db.session.add(msg)
                db.session.commit()
            except Exception as e:
                db.session.rollback()
                logging.error(f"Failed to save studies chat message: {e}", exc_info=True)
            
            logging.info(f"Query processed successfully. Answer length: {len(answer)}, Sources: {len(sources)}")
            return jsonify({
                'answer': answer,
                'sources': sources,
                'session_id': session_id,
                'session_title': session_title
            })
            
        except Exception as e:
            logging.error(f"Error in query processing: {str(e)}", exc_info=True)
            return jsonify({'error': f'Query processing failed: {str(e)}'}), 500
            
    except Exception as e:
        logging.error(f"General error in query endpoint: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


# --- Health Check ---

@bp.route('/health', methods=['GET'])
def health_check():
    try:
        # Test database connectivity
        from app.models import User
        db.session.query(User).first()  # Simple query to test connection
        
        # Test environment variables
        required_vars = ['DATABASE_URL', 'HUGGINGFACE_API_TOKEN']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            return jsonify({
                'status': 'error',
                'message': f'Missing required environment variables: {missing_vars}'
            }), 500
            
        return jsonify({
            'status': 'healthy',
            'database': 'connected',
            'environment': 'configured'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

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
        try:
            from app.services.vector_store import VectorStore
            vector_store = VectorStore.get_instance()
            # Need to re-embed just this doc's chunks
            # But for simplicity/consistency with "rebuild" logic, maybe just leave it for manual or background job
            # Or just do it:
            chunk_texts = [c for c in chunks]
            # Use the new add_texts method which handles embedding internally
            metadata = [{
                'text': c, 
                'doc_id': doc.id, 
                'document_id': doc.id, # Double mapping for compatibility
                'doc_type': doc.doc_type or 'syllabus',
                'filename': doc.filename, 
                'url': supa.get_public_url(doc.file_path)
            } for c in chunks]
            vector_store.add_texts(chunk_texts, metadata)
            
            # REMOVED FOR RENDER COMPATIBILITY - each worker maintains its own in-memory index
            # vector_store.save_index('vector_index')
            logging.info(f"Added document {doc.filename} to vector store")
            logging.info(f"Vector store now has {vector_store.get_stats()['total_vectors']} vectors")
        except Exception as e:
            logging.error(f"Failed to update vector store with new document: {e}")
            # Continue anyway, user can manually rebuild index later
        
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
    try:
        items = supa.list_files(prefix="")
    except Exception:
        items = [] # Fallback if list fails
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
