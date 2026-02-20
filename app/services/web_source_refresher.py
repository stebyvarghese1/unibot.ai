import logging
import time
import threading
from datetime import datetime, timedelta
from app import db
from app.models import Document, DocumentChunk, AppSetting
from app.services.web_scraper import WebScraper
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore

class WebSourceRefresher:
    @staticmethod
    def refresh_stale_sources(app):
        """
        Background worker that periodically checks for stale web sources
        and updates them automatically if auto-refresh is enabled.
        """
        with app.app_context():
            try:
                # 1. Check if auto-refresh is enabled
                interval_str = AppSetting.get('general_refresh_interval', 'never')
                if interval_str == 'never':
                    return

                try:
                    days = int(interval_str)
                except ValueError:
                    return

                threshold_date = datetime.utcnow() - timedelta(days=days)
                
                # 2. Find web-sourced documents older than the threshold
                stale_docs = Document.query.filter(
                    Document.filename.like('[WEB]%'),
                    Document.upload_date < threshold_date
                ).all()

                if not stale_docs:
                    return

                logging.info(f"🔄 Found {len(stale_docs)} stale web sources. Starting auto-refresh...")

                vector_store = VectorStore.get_instance()

                for doc in stale_docs:
                    url = doc.file_path # We stored the URL in file_path for web docs
                    logging.info(f"🌐 Auto-refreshing: {url}")

                    try:
                        # 3. Scrape new content
                        ok, pages = WebScraper.crawl_website(url, max_pages_override=30, time_cap_override=60)
                        if not ok or not pages:
                            logging.warning(f"⚠️ Failed to re-scrape {url}: {pages}")
                            # Update date anyway to avoid infinite retries on failure, or just skip?
                            # For now, we update the date to "now" so we don't try again immediately
                            doc.upload_date = datetime.utcnow()
                            db.session.commit()
                            continue

                        # 4. Clear old data from Vector Store
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
                            chunks = DocumentProcessor.chunk_text(text)
                            
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
                                    'chunk_id': None, # Will be updated after commit if needed
                                    'url': page_url
                                })
                                total_chunks += 1

                        # Update doc metadata
                        doc.upload_date = datetime.utcnow()
                        doc.status = 'processed'
                        db.session.commit()

                        # 7. Update Vector Store index
                        # Since we committed, we can get the new IDs
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
                logging.error(f"❌ WebSourceRefresher loop error: {e}")

    @staticmethod
    def start_worker(app):
        """
        Starts the background worker thread.
        Checks every 6 hours to be polite to the server.
        """
        def run_loop():
            # Initial wait to let app start fully
            time.sleep(30)
            while True:
                try:
                    WebSourceRefresher.refresh_stale_sources(app)
                except Exception as e:
                    logging.error(f"Refresher thread error: {e}")
                
                # Check every 6 hours
                time.sleep(6 * 3600)

        thread = threading.Thread(target=run_loop, daemon=True)
        thread.start()
        return thread
