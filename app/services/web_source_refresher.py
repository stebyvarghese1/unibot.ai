import logging
import time
import threading
from datetime import datetime, timezone, timedelta
from app import db
from app.models import Document, DocumentChunk, AppSetting
from app.services.web_scraper import WebScraper
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
from app.utils.background_tasks import run_background_task

class WebSourceRefresher:
    @staticmethod
    def refresh_stale_sources():
        """
        Background logic that periodically checks for stale web sources
        and updates them automatically if auto-refresh is enabled.
        """
        try:
            # 1. Check if auto-refresh is enabled
            interval_str = AppSetting.get('general_refresh_interval', 'never')
            if interval_str == 'never':
                return

            try:
                days = int(interval_str)
            except ValueError:
                return

            threshold_date = datetime.now(timezone.utc) - timedelta(days=days)
            
            # 2. Find web-sourced documents older than the threshold
            # Documents are identified by [WEB] prefix in filename
            stale_docs = Document.query.filter(
                Document.filename.like('[WEB]%'),
                Document.upload_date < threshold_date
            ).all()

            if not stale_docs:
                return

            logging.info(f"🔄 Found {len(stale_docs)} stale web sources. Starting auto-refresh...")

            vector_store = VectorStore.get_instance()

            for doc in stale_docs:
                url = doc.file_path
                logging.info(f"🌐 Auto-refreshing: {url}")

                try:
                    # 3. Scrape new content
                    ok, pages = WebScraper.crawl_website(url, max_pages_override=30, time_cap_override=60)
                    if not ok or not pages:
                        logging.warning(f"⚠️ Failed to re-scrape {url}")
                        # Update date anyway to avoid infinite retries on failure
                        doc.upload_date = datetime.now(timezone.utc)
                        db.session.commit()
                        continue

                    # 4. Clear old data from Vector Store (using unified key doc_id)
                    try:
                        vector_store.remove_document(doc.id)
                    except Exception as e:
                        logging.error(f"Error removing doc {doc.id} from vector store: {e}")

                    # 5. Delete old chunks from DB
                    DocumentChunk.query.filter_by(document_id=doc.id).delete()
                    
                    # 6. Process & Add new chunks
                    total_chunks = 0
                    all_chunk_texts = []
                    all_chunk_metas = []
                    
                    for page_url, raw_text in pages:
                        text = DocumentProcessor._sanitize_text(raw_text)
                        from app.services.web_scraper import GENERAL_MODE_CHUNK_WORDS, GENERAL_MODE_CHUNK_OVERLAP
                        chunks = DocumentProcessor.chunk_text(text, chunk_size=GENERAL_MODE_CHUNK_WORDS, overlap=GENERAL_MODE_CHUNK_OVERLAP)
                        
                        for chunk_text in chunks:
                            final_text = f"[Source: {page_url}]\n{chunk_text}"
                            
                            chunk_obj = DocumentChunk(
                                document_id=doc.id,
                                chunk_text=final_text,
                                chunk_index=total_chunks
                            )
                            db.session.add(chunk_obj)
                            
                            all_chunk_texts.append(final_text)
                            all_chunk_metas.append({
                                'text': final_text,
                                'doc_id': doc.id,
                                'chunk_id': None, # Updated after commit
                                'url': page_url,
                                'filename': doc.filename
                            })
                            total_chunks += 1

                    # Update doc metadata
                    doc.upload_date = datetime.now(timezone.utc)
                    doc.status = 'processed'
                    db.session.commit()

                    # 7. Update Vector Store index
                    new_chunks = DocumentChunk.query.filter_by(document_id=doc.id).order_by(DocumentChunk.chunk_index).all()
                    for i, c in enumerate(new_chunks):
                        if i < len(all_chunk_metas):
                            all_chunk_metas[i]['chunk_id'] = c.id
                    
                    if all_chunk_texts:
                        vector_store.add_texts(all_chunk_texts, all_chunk_metas)

                    logging.info(f"✅ Successfully auto-refreshed {url} ({total_chunks} chunks)")

                except Exception as e:
                    db.session.rollback()
                    logging.error(f"❌ Failed auto-refresh for {url}: {e}", exc_info=True)

        except Exception as e:
            logging.error(f"❌ WebSourceRefresher logic error: {e}")

    @staticmethod
    def start_worker(app):
        """
        Starts the background worker thread using the robust background task utility.
        """
        def run_loop():
            time.sleep(30) # Initial wait
            while True:
                try:
                    # Use the application context utility to run the task
                    run_background_task(WebSourceRefresher.refresh_stale_sources)
                except Exception as e:
                    logging.error(f"Refresher thread error: {e}")
                
                # Check every 6 hours
                time.sleep(6 * 3600)

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        return thread
