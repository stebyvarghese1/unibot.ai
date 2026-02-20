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
        soup_all = BeautifulSoup(html, 'html.parser')
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['script', 'style', 'header', 'footer', 'aside', 'iframe', 'nav']):
            tag.decompose()
        body = soup.find('body') or soup
        text = (body.get_text(separator='\n', strip=True) if body else '') or soup.get_text(separator='\n', strip=True)
        text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
        # Remove NUL characters to prevent database errors
        text = text.replace('\x00', '')
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
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0'}
            # Add strict timeout to prevent hangs
            r = requests.get(url, headers=headers, timeout=5, verify=False)
            r.raise_for_status()
            r.encoding = r.apparent_encoding or 'utf-8'
            soup, text = WebScraper.extract_text_from_html(r.text, url)
            return True, soup, text
        except requests.RequestException as e:
            return False, None, str(e)
        except Exception as e:
            return False, None, str(e)

    @staticmethod
    def fetch_one_page_playwright(url, page):
        try:
            url = WebScraper.normalize_url(url)
            if not url:
                return False, None, 'Invalid URL'
            page.goto(url, wait_until='domcontentloaded', timeout=15000)
            html = page.content()
            soup, text = WebScraper.extract_text_from_html(html, url)
            return True, soup, text
        except Exception as e:
            logging.warning('Playwright fetch failed for %s: %s', url, e)
            return False, None, str(e)

    @staticmethod
    def fetch_one_page(url, playwright_page=None):
        if playwright_page:
            return WebScraper.fetch_one_page_playwright(url, playwright_page)
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
    def run_crawl_loop(queue, seen, playwright_page, max_pages, max_total_chars, time_cap_s):
        pages_list = []
        total_chars = 0
        pages_done = 0
        start_time = time.time()
        
        while queue and pages_done < max_pages and total_chars < max_total_chars and (time.time() - start_time) < time_cap_s:
            if playwright_page:
                current = queue.popleft()
                ok, soup, text = WebScraper.fetch_one_page(current, playwright_page=playwright_page)
                if ok:
                    if text and len(text) >= 15:
                        pages_list.append((current, text))
                        total_chars += len(text)
                        if total_chars > max_total_chars:
                            break
                    pages_done += 1
                    if soup and pages_done < max_pages:
                        for link in WebScraper.same_domain_links(soup, current):
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
                    def _task(u):
                        # Enforce individual page timeout logic in fetch_one_page_requests
                        ok, soup, text = WebScraper.fetch_one_page_requests(u)
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
            
            # Try Playwright first
            try:
                import importlib
                pwa = importlib.import_module('playwright.sync_api')
                with pwa.sync_playwright() as p:
                    # Added timeout to launch
                    browser = p.chromium.launch(headless=True, timeout=10000)
                    context = browser.new_context(ignore_https_errors=True)
                    page = context.new_page()
                    page.set_extra_http_headers({
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; rv:109.0) Gecko/20100101 Firefox/115.0'
                    })
                    pages_list, total_chars = WebScraper.run_crawl_loop(queue, seen, page, limits['max_pages'], limits['max_chars'], limits['time_cap'])
                    browser.close()
            except Exception as e:
                logging.warning('Playwright crawl failed or not installed, using requests: %s', e)
                
            # If Playwright failed or didn't run, or returned nothing but we have seeds, try requests fallback for what's left
            if not pages_list:
                seen = {url}
                if seeds:
                    queue = deque([url] + seeds)
                else:
                    queue = deque([url])
                pages_list, total_chars = WebScraper.run_crawl_loop(queue, seen, None, limits['max_pages'], limits['max_chars'], limits['time_cap'])
                
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

            # 4. Playwright Fallback (for stuck/JS pages)
            # Only pick the top 3 scored candidates that failed with requests to save time
            suspicious_high_value = [u for u in top if u in failed_or_empty_candidates][:3]
            
            if suspicious_high_value:
                try:
                    import importlib
                    pwa = importlib.import_module('playwright.sync_api')
                    with pwa.sync_playwright() as p:
                        browser = p.chromium.launch(headless=True, timeout=15000)
                        context = browser.new_context(ignore_https_errors=True, user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36')
                        page = context.new_page()
                        
                        for u in suspicious_high_value:
                            try:
                                # Short page load timeout
                                page.goto(u, wait_until='domcontentloaded', timeout=10000)
                                # Wait a sec for hydration
                                page.wait_for_timeout(1000)
                                content = page.content()
                                _, p_text = WebScraper.extract_text_from_html(content, u)
                                if len(p_text) > 50:
                                    pages_list.append((u, p_text))
                            except Exception as e:
                                logging.warning(f"PW fallback failed for {u}: {e}")
                                
                        browser.close()
                except Exception as e:
                     logging.warning(f"Playwright fallback unavailable: {e}")
                        
            return True, pages_list
            
        except Exception as e:
            logging.error(f"Targeted fetch failed: {e}")
            return False, []
