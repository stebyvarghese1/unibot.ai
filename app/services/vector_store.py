import faiss
import numpy as np
import pickle
import os

class VectorStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance.index = None
            cls._instance.chunks = [] # Store metadata/text mapping
            cls._instance.dimension = 384 # Default for all-MiniLM-L6-v2
        return cls._instance

    def initialize_index(self, dimension=384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.chunks = []

    def add_documents(self, embeddings, chunks_metadata):
        """
        embeddings: list of floats or numpy array
        chunks_metadata: list of dicts containing text and other info
        """
        if self.index is None:
            self.initialize_index(len(embeddings[0]))
            
        vectors = np.array(embeddings).astype('float32')
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = vectors / norms
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
        new_index = faiss.IndexFlatIP(self.dimension)
        
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
        if self.index is None or self.index.ntotal == 0:
            return []
            
        vector = np.array([query_vector]).astype('float32')
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        vector = vector / norm
        distances, indices = self.index.search(vector, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1 and idx < len(self.chunks):
                result = self.chunks[idx].copy()
                result['distance'] = float(distances[0][i])
                results.append(result)
                
        return results

    def clear(self):
        self.index = None
        self.chunks = []
        self.initialize_index(self.dimension)

    def get_stats(self):
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension
        }
