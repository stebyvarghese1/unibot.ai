import unittest
from app import create_app, db
from app.models import User
from werkzeug.security import generate_password_hash
from config import Config


class AdminAccountTestCase(unittest.TestCase):
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
            db.session.add(admin)
            db.session.commit()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def login_admin(self):
        return self.client.post(
            '/api/login',
            json={'email': Config.ADMIN_EMAIL, 'password': Config.ADMIN_PASSWORD},
        )

    def test_get_admin_account(self):
        r = self.login_admin()
        self.assertEqual(r.status_code, 200)
        res = self.client.get('/api/admin/admin-account')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('email', data)
        self.assertEqual(data['email'], Config.ADMIN_EMAIL)

    def test_update_admin_email_and_password(self):
        r = self.login_admin()
        self.assertEqual(r.status_code, 200)
        new_email = 'admin2@university.edu'
        new_password = 'new_admin_password'
        res = self.client.post(
            '/api/admin/admin-account',
            json={
                'email': new_email,
                'current_password': Config.ADMIN_PASSWORD,
                'new_password': new_password,
            },
        )
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn('message', data)
        self.assertEqual(data['message'], 'Admin account updated')
        r2 = self.client.post(
            '/api/login',
            json={'email': new_email, 'password': new_password},
        )
        self.assertEqual(r2.status_code, 200)


if __name__ == '__main__':
    unittest.main()
