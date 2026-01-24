import unittest
from app import create_app, db
from app.models import User, ChatMessage
from werkzeug.security import generate_password_hash
from config import Config
import json


class ChatHistoryTestCase(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.app.config['TESTING'] = True
        self.app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
        self.client = self.app.test_client()
        with self.app.app_context():
            db.create_all()
            admin = User(
                email=Config.ADMIN_EMAIL,
                password_hash=generate_password_hash(Config.ADMIN_PASSWORD),
                role='admin',
            )
            student = User(
                email='student@test.com',
                password_hash=generate_password_hash('student'),
                role='student',
            )
            db.session.add(admin)
            db.session.add(student)
            db.session.commit()
            self.student_id = student.id

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def login_student(self):
        return self.client.post(
            '/api/login',
            json={'email': 'student@test.com', 'password': 'student'},
        )

    def test_history_list_and_delete(self):
        r = self.login_student()
        self.assertEqual(r.status_code, 200)
        with self.app.app_context():
            m1 = ChatMessage(
                user_id=self.student_id,
                question='Q1',
                answer='A1',
                subject='math',
                sources_json=json.dumps([]),
            )
            m2 = ChatMessage(
                user_id=self.student_id,
                question='Q2',
                answer='A2',
                subject='cs',
                sources_json=json.dumps([]),
            )
            db.session.add(m1)
            db.session.add(m2)
            db.session.commit()
        res = self.client.get('/api/chat/history')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(isinstance(data, list))
        self.assertGreaterEqual(len(data), 2)
        msg_id = data[0]['id']
        del_res = self.client.delete(f'/api/chat/history/{msg_id}')
        self.assertEqual(del_res.status_code, 200)
        clear_res = self.client.delete('/api/chat/history')
        self.assertEqual(clear_res.status_code, 200)
        res2 = self.client.get('/api/chat/history')
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(len(res2.get_json()), 0)


if __name__ == '__main__':
    unittest.main()
