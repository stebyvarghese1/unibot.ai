import unittest
from app import create_app, db
from app.models import User, Document
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import VectorStore
import os

class BasicTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        
        with self.app.app_context():
            db.create_all()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

    def test_user_creation(self):
        with self.app.app_context():
            user = User(email='test@test.com', password_hash='hash', role='student')
            db.session.add(user)
            db.session.commit()
            self.assertIsNotNone(user.id)

    def test_vector_store_initialization(self):
        store = VectorStore()
        store.initialize_index(384)
        stats = store.get_stats()
        self.assertEqual(stats['dimension'], 384)
        self.assertEqual(stats['total_vectors'], 0)

    def test_chunking(self):
        text = "word " * 1000
        chunks = DocumentProcessor.chunk_text(text, chunk_size=100, overlap=0)
        self.assertTrue(len(chunks) > 0)

if __name__ == '__main__':
    unittest.main()
