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
        # Check dimensionality from input
        new_dim = self.dimension
        if embeddings and len(embeddings) > 0:
             first_emb = embeddings[0]
             if isinstance(first_emb, list):
                 new_dim = len(first_emb)
             elif hasattr(first_emb, '__len__') and not isinstance(first_emb, str):
                 new_dim = len(first_emb)
             else:
                 new_dim = 1
        
        # Initialize or Re-initialize if dimension mismatch and empty
        if self.index is None:
             self.initialize_index(new_dim)
        elif self.index.d != new_dim:
             if self.index.ntotal == 0:
                 logging.info(f"Re-initializing index dimension from {self.index.d} to {new_dim} based on input embeddings")
                 self.initialize_index(new_dim)
             else:
                 raise ValueError(f"Embedding dimension mismatch: Index has {self.index.d}, new embeddings have {new_dim}. Clear index first.")
        
        # 🔥 NORMALIZE embedding shape for FAISS
        if isinstance(embeddings, list) and len(embeddings) > 0:
            if isinstance(embeddings[0], list):
                # Batch case: [[384], [384], ...] -> shape (N, 384)
                vectors = np.array(embeddings, dtype="float32")
            elif hasattr(embeddings[0], '__iter__') and not isinstance(embeddings[0], str):
                # Handle numpy arrays or other iterables
                vectors = np.array(embeddings, dtype="float32")
            else:
                # Single embedding case: [384] -> shape (1, 384)
                vectors = np.array([embeddings], dtype="float32")
        else:
            raise ValueError(f"Invalid embeddings format: {type(embeddings)}, {embeddings}")
        
        # Validate that vectors is 2D
        if vectors.ndim == 1:
            # If 1D, reshape to (1, N) assuming it's a single vector
            vectors = vectors.reshape(1, -1)
        elif vectors.ndim != 2:
            raise ValueError(f"Invalid vector shape: {vectors.shape}, must be 2D")
        
        self.index.add(vectors)
        self.chunks.extend(chunks_metadata)

    def add_texts(self, texts, metadata_list=None):
        """
        Add raw texts to the vector store by converting them to embeddings
        texts: list of text strings
        metadata_list: optional list of metadata dicts (same length as texts)
        """
        if not texts:
            return
            
        # Generate embeddings for the texts
        from app.services.ai_service import AIService
        embeddings = AIService.get_embeddings(texts)
        
        if not embeddings or len(embeddings) == 0:
            logging.error("Failed to generate embeddings for texts")
            return
            
        # Prepare metadata
        if metadata_list is None:
            metadata_list = [{'text': text} for text in texts]
        elif len(metadata_list) != len(texts):
            logging.warning(f"Metadata count ({len(metadata_list)}) doesn't match text count ({len(texts)}), padding with defaults")
            while len(metadata_list) < len(texts):
                idx = len(metadata_list)
                metadata_list.append({'text': texts[idx]})
        
        # Add the computed embeddings and metadata using the normalized method
        self.add_documents(embeddings, metadata_list)

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
        # NOTE: This method is not suitable for Render multi-worker environments
        # Each worker would have its own in-memory state, making file-based persistence unreliable
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
        # NOTE: This method is not suitable for Render multi-worker environments
        # Each worker would have its own in-memory state, making file-based persistence unreliable
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
