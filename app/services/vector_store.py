import numpy as np
import logging
from app.services.supabase_service import SupabaseService

class VectorStore:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(VectorStore, cls).__new__(cls)
            cls._instance.dimension = 384 # Default for all-MiniLM-L6-v2
            cls._instance.supabase = SupabaseService()
        return cls._instance

    def initialize_index(self, dimension=384):
        """
        In Supabase, the 'index' is managed by the database table.
        We ensure the dimension matches what the database expects.
        """
        self.dimension = dimension
        logging.info(f"Supabase VectorStore initialized with dimension {dimension}")

    def add_documents(self, embeddings, chunks_metadata):
        """
        embeddings: list of floats or numpy array
        chunks_metadata: list of dicts containing text and other info
        """
        if not embeddings or len(embeddings) == 0:
            return

        # 🔥 Normalize to list of floats for Supabase
        records = []
        for i, emb in enumerate(embeddings):
            # Ensure embedding is a list of floats
            if hasattr(emb, 'tolist'):
                vector = emb.tolist()
            else:
                vector = list(emb)
            
            metadata = chunks_metadata[i].copy()
            content = metadata.pop('text', '')
            
            records.append({
                'content': content,
                'metadata': metadata,
                'embedding': vector
            })
        
        try:
            # Upsert into 'embeddings' table
            result = self.supabase.client.table('embeddings').insert(records).execute()
            logging.info(f"Successfully added {len(records)} documents to Supabase pgvector")
            return result
        except Exception as e:
            logging.error(f"Error adding documents to Supabase: {e}")
            raise

    def add_texts(self, texts, metadata_list=None):
        """
        Add raw texts to the vector store by converting them to embeddings
        """
        if not texts:
            return
            
        from app.services.ai_service import AIService
        embeddings = AIService.get_embeddings(texts)
        
        if not embeddings or len(embeddings) == 0:
            logging.error("Failed to generate embeddings for texts")
            return
            
        if metadata_list is None:
            metadata_list = [{'text': text} for text in texts]
        elif len(metadata_list) != len(texts):
            while len(metadata_list) < len(texts):
                idx = len(metadata_list)
                metadata_list.append({'text': texts[idx]})
        else:
            # Ensure 'text' is in metadata for record creation
            for i, meta in enumerate(metadata_list):
                if 'text' not in meta:
                    meta['text'] = texts[i]
        
        self.add_documents(embeddings, metadata_list)

    def remove_document(self, doc_id):
        """
        Remove documents by their document_id from the metadata JSONB column
        """
        try:
            # Supabase allows filtering on JSONB columns using arrow operators
            # We delete records where metadata->>doc_id or metadata->>document_id matches
            self.supabase.client.table('embeddings').delete().eq('metadata->>doc_id', str(doc_id)).execute()
            self.supabase.client.table('embeddings').delete().eq('metadata->>document_id', str(doc_id)).execute()
            logging.info(f"Removed documents with doc_id {doc_id} from Supabase")
        except Exception as e:
            logging.error(f"Error removing document from Supabase: {e}")

    def search(self, query_vector, k=5):
        """
        Search for similar documents using match_documents RPC
        """
        try:
            # Ensure query_vector is a list
            if hasattr(query_vector, 'tolist'):
                vector = query_vector.tolist()
            else:
                vector = list(query_vector)

            # Call the match_documents stored procedure
            # This function needs to be defined in Supabase SQL
            rpc_params = {
                'query_embedding': vector,
                'match_threshold': 0.1, # Significantly lowered to avoid missing relevant content
                'match_count': k
            }
            
            response = self.supabase.client.rpc('match_documents', rpc_params).execute()
            
            results = []
            for item in response.data:
                res = item.get('metadata', {}).copy()
                res['text'] = item.get('content', '')
                res['distance'] = 1.0 - item.get('similarity', 0) # Convert similarity to distance
                results.append(res)
                
            return results
        except Exception as e:
            logging.error(f"Error searching in Supabase: {e}")
            return []

    def clear(self):
        """
        Clear all records from the embeddings table
        """
        try:
            # Simple way to clear table in Supabase (requires permissions)
            self.supabase.client.table('embeddings').delete().neq('content', '___NEVER_MATCH___').execute()
            logging.info("Cleared all embeddings from Supabase")
        except Exception as e:
            logging.error(f"Error clearing Supabase embeddings: {e}")

    def get_stats(self):
        try:
            # Count records
            response = self.supabase.client.table('embeddings').select('id', count='exact').execute()
            count = response.count if hasattr(response, 'count') else 0
            return {
                'total_vectors': count,
                'dimension': self.dimension
            }
        except Exception:
            return {'total_vectors': 0, 'dimension': self.dimension}
    
    def save_index(self, index_name='vector_index'):
        """No-op for Supabase as it's persistently stored in the DB"""
        return True
    
    def load_index(self, index_name='vector_index'):
        """No-op for Supabase as it's managed by the DB"""
        return True
    
    def index_exists(self, index_name='vector_index'):
        """Always returns True for Supabase if the table exists"""
        return True

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
