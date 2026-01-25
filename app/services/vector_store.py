import faiss
import numpy as np
import pickle
import os
import logging

class VectorStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance.index = None
            cls._instance.chunks = [] # Store metadata/text mapping
            cls._instance.dimension = 384 # Default for all-MiniLM-L6-v2
            # Ensure index is initialized
            cls._instance.initialize_index(cls._instance.dimension)
        return cls._instance

    def initialize_index(self, dimension=384):
        self.dimension = dimension
        # IndexFlatIP is good for cosine similarity if vectors are normalized
        # IndexFlatL2 is standard Euclidean
        self.index = faiss.IndexFlatL2(dimension)
        self.chunks = []

    def add_documents(self, embeddings, chunks_metadata):
        """
        embeddings: list of floats or numpy array
        chunks_metadata: list of dicts containing text and other info
        """
        # Ensure index is initialized
        if self.index is None:
            if embeddings and len(embeddings) > 0:
                self.initialize_index(len(embeddings[0]))
            else:
                self.initialize_index(self.dimension)
        
        vectors = np.array(embeddings).astype('float32')
        self.index.add(vectors)
        self.chunks.extend(chunks_metadata)

    def remove_document(self, doc_id):
        if self.index is None or not self.chunks:
            return

        # Identify indices to keep
        keep_indices = []
        new_chunks = []
        
        for i, chunk in enumerate(self.chunks):
            # Check if chunk belongs to doc_id
            # We assume chunk metadata has 'doc_id' or 'document_id'
            c_doc_id = chunk.get('doc_id') or chunk.get('document_id')
            if c_doc_id != doc_id:
                keep_indices.append(i)
                new_chunks.append(chunk)

        # If nothing to remove, return
        if len(keep_indices) == len(self.chunks):
            return

        # Create new index
        new_index = faiss.IndexFlatL2(self.dimension)
        
        # Transfer vectors for kept chunks
        # We can't batch add easily without collecting all vectors first
        if keep_indices:
            vectors = []
            for i in keep_indices:
                try:
                    vec = self.index.reconstruct(i)
                    vectors.append(vec)
                except:
                    # If reconstruct fails (some indices don't support it), we might be in trouble
                    # But IndexFlatL2 supports it.
                    pass
            
            if vectors:
                vectors_np = np.array(vectors).astype('float32')
                new_index.add(vectors_np)

        self.index = new_index
        self.chunks = new_chunks

    def search(self, query_vector, k=5):
        # Ensure index is initialized
        if self.index is None:
            self.initialize_index(self.dimension)
        
        if self.index.ntotal == 0:
            return []
            
        vector = np.array([query_vector]).astype('float32')
        distances, indices = self.index.search(vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['distance'] = float(distances[0][i])
                results.append(result)
                
        return results

    def clear(self):
        self.chunks = []
        self.initialize_index(self.dimension)

    def get_stats(self):
        total_vectors = 0
        if self.index is not None:
            total_vectors = self.index.ntotal if hasattr(self.index, 'ntotal') else 0
        return {
            'total_vectors': total_vectors,
            'dimension': self.dimension
        }
    
    def save_index(self, index_name='vector_index'):
        """Save the FAISS index and metadata to Supabase storage"""
        try:
            if self.index is not None:
                # Save index using temporary file
                import tempfile
                import os
                
                # Create temporary file for index
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                
                try:
                    # Write index to temporary file
                    faiss.write_index(self.index, tmp_path)
                    
                    # Read the file content
                    with open(tmp_path, 'rb') as f:
                        index_bytes = f.read()
                finally:
                    # Clean up temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                
                # Save metadata
                meta_data = {
                    'chunks': self.chunks,
                    'dimension': self.dimension
                }
                meta_bytes = pickle.dumps(meta_data)
                
                # Try to upload to Supabase Storage
                try:
                    from app.services.supabase_service import SupabaseService
                    supa = SupabaseService()
                    
                    # Upload index file
                    supa.upload_file(index_bytes, f"indexes/{index_name}.faiss", "application/octet-stream")
                    # Upload metadata file
                    supa.upload_file(meta_bytes, f"indexes/{index_name}.meta", "application/octet-stream")
                    
                    logging.info(f"Vector index saved to Supabase storage as {index_name}")
                    return True
                except Exception as supa_error:
                    logging.warning(f"Failed to save index to Supabase, falling back to in-memory only: {supa_error}")
                    return True  # Return True to continue operation
        except Exception as e:
            logging.error(f"Critical error saving index: {e}")
            return False
    
    def load_index(self, index_name='vector_index'):
        """Load the FAISS index and metadata from Supabase storage"""
        try:
            from app.services.supabase_service import SupabaseService
            import tempfile
            import os
            supa = SupabaseService()
            
            # Try to download index file
            try:
                index_bytes = supa.download_file(f"indexes/{index_name}.faiss")
                
                # Create temporary file for index
                with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                    tmp_path = tmp_file.name
                    tmp_file.write(index_bytes)
                
                # Load index from temporary file
                try:
                    self.index = faiss.read_index(tmp_path)
                except Exception as read_error:
                    # Clean up and re-raise
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    raise read_error
                
                # Load metadata
                meta_bytes = supa.download_file(f"indexes/{index_name}.meta")
                meta_data = pickle.loads(meta_bytes)
                self.chunks = meta_data.get('chunks', [])
                self.dimension = meta_data.get('dimension', 384)
                
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
                logging.info(f"Vector index loaded from Supabase storage as {index_name}")
                logging.info(f"Loaded index with {len(self.chunks)} chunks and {self.index.ntotal if self.index else 0} vectors")
                return True
            except Exception as download_error:
                logging.info(f"No existing index found in Supabase storage: {download_error}")
                return False
                
        except Exception as e:
            logging.error(f"Error loading index from Supabase: {e}")
            return False
    
    def index_exists(self, index_name='vector_index'):
        """Check if index files exist in Supabase storage"""
        try:
            from app.services.supabase_service import SupabaseService
            supa = SupabaseService()
            
            # Try to download both files to check existence
            try:
                # Try to download index file
                supa.download_file(f"indexes/{index_name}.faiss")
                # If successful, try to download metadata file
                supa.download_file(f"indexes/{index_name}.meta")
                return True
            except Exception:
                return False
        except Exception:
            return False

    @classmethod
    def get_instance(cls):
        """Get the singleton instance of VectorStore"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
