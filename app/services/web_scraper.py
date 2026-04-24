import logging
import time
import requests
import gzip
import io
import functools
from urllib.parse import urlparse, urljoin
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from collections import deque

# Configuration constants
GENERAL_MODE_MAX_PAGES = 25
GENERAL_MODE_MAX_TOTAL_CHARS = 200_000
GENERAL_MODE_TIME_CAP = 20

# Jina Reader is used as a fallback for JS-heavy pages
_JINA_ENABLED = True

class WebScraper:
    @staticmethod
    def normalize_url(url):
        """Ensure URL has scheme and is a string we can parse."""
        if not url or not isinstance(url, str):
            return ''
        s = url.strip().replace('\n', '').replace('\r', '')
        if not s:
            return ''
        if not s.startswith(('http://', 'https://')):
            s = 'https://' + s
        return s

    @staticmethod
    def get_limits_for_url(url):
        try:
            netloc = (urlparse(WebScraper.normalize_url(url)).netloc or '').lower()
        except Exception:
            netloc = ''
        if netloc.endswith('uoc.ac.in'):
            return {'max_pages': 120, 'max_chars': 1_200_000, 'time_cap': 120}
        return {'max_pages': GENERAL_MODE_MAX_PAGES, 'max_chars': GENERAL_MODE_MAX_TOTAL_CHARS, 'time_cap': GENERAL_MODE_TIME_CAP}

    @staticmethod
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

    @staticmethod
    def normalize_crawl_url(u):
        """One canonical form for crawl dedup (strip fragment, trailing slash)."""
        try:
            p = urlparse(u)
            scheme = (p.scheme or 'https').lower()
            netloc = (p.netloc or '').lower()
            path = (p.path or '/').rstrip('/') or '/'
            return f"{scheme}://{netloc}{path}"
        except Exception:
            return u

    @staticmethod
    def fetch_sitemap_urls(base_url):
        try:
            b = WebScraper.normalize_url(base_url).rstrip('/')
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0'}
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass
            
            base_netloc = (urlparse(b).netloc or '').lower()
            base_root = WebScraper._domain_root(base_netloc)
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
                        u = WebScraper.normalize_crawl_url((loc.text or '').strip())
                        if not u:
                            continue
                        p = urlparse(u)
                        nl = (p.netloc or '').lower()
                        if p.scheme in ('http', 'https') and (WebScraper._domain_root(nl) == base_root):
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
            
            if len(urls) > 100:
                urls = set(list(urls)[:100])
            return urls
        except Exception:
            return set()

    @staticmethod
    def extract_text_from_html(html, base_url):
        if not html:
            return None, ""
        # Use only one soup instance for performance and memory
        soup = BeautifulSoup(html, 'lxml' if 'lxml' in sys.modules else 'html.parser')
        soup_all = soup # Keep reference if needed, but avoid re-parsing
        
        # Remove unwanted tags, but keep noscript as it may contain fallback text
        for tag in soup(['script', 'style', 'header', 'footer', 'aside', 'iframe', 'nav', 'svg']):
            tag.decompose()
            
        # Extract text from body or whole soup
        body = soup.find('body') or soup
        text = (body.get_text(separator='\n', strip=True) if body else '') or soup.get_text(separator='\n', strip=True)
        
        # Clean up lines
        lines = []
        for line in text.splitlines():
            s = line.strip()
            if s:
                lines.append(s)
        text = '\n'.join(lines)
        
        # Remove NUL characters to prevent database errors
        text = text.replace('\x00', '')
        
        # Limit total text length to avoid memory bloat
        if len(text) > 500000:
            text = text[:500000] + "... [Content Truncated]"
            
        return soup_all, text

    @staticmethod
    def fetch_one_page_requests(url):
        try:
            url = WebScraper.normalize_url(url)
            if not url:
                return False, None, 'Invalid URL'
            try:
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
            except Exception:
                pass
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
            # Use stream=True to check content length before downloading everything
            with requests.get(url, headers=headers, timeout=8, verify=False, stream=True) as r:
                r.raise_for_status()
                
                # Check content length (if available) - limit to 5MB
                content_length = r.headers.get('Content-Length')
                if content_length and int(content_length) > 5 * 1024 * 1024:
                    return False, None, 'Page too large (>5MB)'
                
                # Download content in chunks with a cap
                html_content = ""
                total_size = 0
                for chunk in r.iter_content(chunk_size=8192, decode_unicode=True):
                    if chunk:
                        html_content += chunk
                        total_size += len(chunk)
                        if total_size > 5 * 1024 * 1024: # Hard cap 5MB
                            break
                
                soup, text = WebScraper.extract_text_from_html(html_content, url)
                return True, soup, text
        except requests.RequestException as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def fetch_one_page_jina(url):
        """Fetch page content using Jina Reader API (perfect for AI/RAG)."""
        try:
            from config import Config
            headers = {
                'X-Return-Format': 'markdown',
                'X-No-Cache': 'true'
            }
            if Config.JINA_API_KEY:
                headers['Authorization'] = f"Bearer {Config.JINA_API_KEY}"
            
            # r.jina.ai prepends to the URL to get the clean markdown content
            jina_url = f"https://r.jina.ai/{url}"
            response = requests.get(jina_url, headers=headers, timeout=20)
            response.raise_for_status()
            
            text = response.text
            # Jina returns clean text/markdown, so we don't need soup for extraction, 
            # but we return None for soup as it's primarily used for link discovery in crawls.
            return True, None, text
        except Exception as e:
            logging.warning(f"Jina Reader fetch failed for {url}: {e}")
            return False, None, str(e)

    @staticmethod
    def fetch_one_page(url, use_jina=False):
        if use_jina:
            return WebScraper.fetch_one_page_jina(url)
        return WebScraper.fetch_one_page_requests(url)

    @staticmethod
    def same_domain_links(soup, base_url):
        if not soup:
            return set()
        base = base_url.strip().rstrip('/') or base_url
        try:
            base_netloc = (urlparse(base).netloc or '').lower()
            base_root = WebScraper._domain_root(base_netloc)
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
                cand_root = WebScraper._domain_root(netloc)
                if cand_root != base_root:
                    continue
                out.add(WebScraper.normalize_crawl_url(absolute))
            except Exception:
                continue
        return out

    @staticmethod
    def run_crawl_loop(queue, seen, max_pages, max_total_chars, time_cap_s):
        pages_list = []
        total_chars = 0
        pages_done = 0
        start_time = time.time()
        
        while queue and pages_done < max_pages and total_chars < max_total_chars and (time.time() - start_time) < time_cap_s:
            batch = []
            while queue and len(batch) < 8 and pages_done + len(batch) < max_pages:
                batch.append(queue.popleft())
            if not batch:
                break
            
            try:
                def _task(u):
                    # Try standard requests first
                    ok, soup, text = WebScraper.fetch_one_page_requests(u)
                    
                    # If requests returned very little text, it might be a JS-heavy page
                    # Try Jina Reader as a fallback for high-potential pages
                    if ok and (not text or len(text) < 300):
                        ok_j, _, text_j = WebScraper.fetch_one_page_jina(u)
                        if ok_j and text_j and len(text_j) > (len(text) if text else 0):
                            return u, True, None, text_j
                            
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
                        for link in WebScraper.same_domain_links(soup, u):
                            if link not in seen:
                                seen.add(link)
                                queue.append(link)
                pages_done += len(batch)
            except Exception:
                # Fallback to serial if thread pool fails
                for u in batch:
                    ok, soup, text = WebScraper.fetch_one_page_requests(u)
                    if ok and text and len(text) >= 15:
                        pages_list.append((u, text))
                        total_chars += len(text)
                        if total_chars > max_total_chars:
                            break
                    if soup and pages_done < max_pages:
                        for link in WebScraper.same_domain_links(soup, u):
                            if link not in seen:
                                seen.add(link)
                                queue.append(link)
                pages_done += len(batch)
                    
        return pages_list, total_chars

    @staticmethod
    def crawl_website(url, max_pages_override=None, max_chars_override=None, time_cap_override=None):
        """Recursively crawl same-domain site (BFS). Returns (True, [(url, text), ...]) or (False, error_message)."""
        try:
            url = WebScraper.normalize_url(url)
            if not url:
                return False, 'Invalid URL'
            url = WebScraper.normalize_crawl_url(url)
            
            limits = WebScraper.get_limits_for_url(url)
            if max_pages_override is not None:
                limits['max_pages'] = max_pages_override
            if max_chars_override is not None:
                limits['max_chars'] = max_chars_override
            if time_cap_override is not None:
                limits['time_cap'] = time_cap_override
                
            seen = {url}
            seeds = list(WebScraper.fetch_sitemap_urls(url))
            if seeds:
                if len(seeds) > 30:
                    seeds = seeds[:30]
                queue = deque([url] + seeds)
            else:
                queue = deque([url])
                
            pages_list = []
            total_chars = 0
            
            # Standard crawl loop (Uses Requests with Jina fallback for problematic pages)
            pages_list, total_chars = WebScraper.run_crawl_loop(queue, seen, limits['max_pages'], limits['max_chars'], limits['time_cap'])
                
            if not pages_list:
                return False, 'No text content found on the site'
                
            logging.info('Crawl result: %d pages, %d chars', len(pages_list), total_chars)
            return True, pages_list
            
        except Exception as e:
            logging.exception('Website crawl failed')
            return False, str(e)
            
    @staticmethod
    def _site_search_candidates(base_url, query):
        try:
            b = WebScraper.normalize_url(base_url).rstrip('/')
            tokens = query.split() if query else []
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
            
    @staticmethod
    def fetch_targeted_pages(url, question, max_pages=15):
        """Fetch pages from a site relevant to a question (Home + Sitemap + Search + scored links).
           Uses Requests for speed, falls back to Playwright for top candidates if texts are suspiciously short."""
        try:
            url = WebScraper.normalize_url(url)
            tokens = [t.lower() for t in question.split() if len(t) > 3]
            
            # 1. Discovery Phase (Home + Sitemap + Search)
            # Use requests for discovery to be fast
            ok, soup, text = WebScraper.fetch_one_page_requests(url)
            same_links = set()
            if ok and soup:
                same_links = WebScraper.same_domain_links(soup, url)
                
            sitemap_links = WebScraper.fetch_sitemap_urls(url)
            search_links = WebScraper._site_search_candidates(url, question)
            
            cands = set()
            for s in [same_links, sitemap_links, search_links]:
                for u in s:
                    cands.add(WebScraper.normalize_crawl_url(u))
                    
            # 2. Scoring
            scored = []
            for u in cands:
                score = 0
                u_lower = u.lower()
                for tok in tokens:
                    if tok in u_lower:
                        score += 3
                hints = ('result', 'exam', 'notification', 'student', 'admission', 'schedule', 'timetable', 'syllabus')
                for h in hints:
                    if h in u_lower:
                        score += 2
                scored.append((u, score))
                
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [u for u, _ in scored[:max_pages]]
            if not top:
                 # Fallback to basic crawl if no relevant links found
                 return WebScraper.crawl_website(url, max_pages_override=max_pages, time_cap_override=10)
                 
            # 3. Fast Fetch (Requests)
            pages_list = []
            if ok and text and len(text) >= 100:
                pages_list.append((url, text))
                
            failed_or_empty_candidates = []
                
            try:
                def _task(u):
                    # verify=False is critical for many uni sites
                    ou, ok1, soup1, text1 = u, *WebScraper.fetch_one_page_requests(u)
                    return ou, ok1, text1
                    
                with ThreadPoolExecutor(max_workers=8) as ex:
                    results = list(ex.map(_task, top))
                    
                for ou, ok1, text1 in results:
                    # If text is substantial, keep it
                    if ok1 and text1 and len(text1) >= 200:
                        pages_list.append((ou, text1))
                    elif ok1:
                        # Success but little text -> likely JS rendered
                        failed_or_empty_candidates.append(ou)
                    else:
                        # Connection failed -> retry might help or might not
                        failed_or_empty_candidates.append(ou)
                        
            except Exception:
                # Serial fallback
                for ou in top:
                    ok1, soup1, text1 = WebScraper.fetch_one_page_requests(ou)
                    if ok1 and text1 and len(text1) >= 200:
                        pages_list.append((ou, text1))
                    else:
                        failed_or_empty_candidates.append(ou)

            # 4. Jina Reader Fallback (for JS-heavy or problematic pages)
            # Only pick the top 5 scored candidates that failed with requests
            suspicious_high_value = [u for u in top if u in failed_or_empty_candidates][:5]
            
            if suspicious_high_value:
                for u in suspicious_high_value:
                    ok_j, _, text_j = WebScraper.fetch_one_page_jina(u)
                    if ok_j and text_j and len(text_j) > 100:
                        pages_list.append((u, text_j))
                        
            return True, pages_list
            
        except Exception as e:
            logging.error(f"Targeted fetch failed: {e}")
            return False, []
