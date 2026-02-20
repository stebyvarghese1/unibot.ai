from app import create_app, db
from app.models import User
import os

app = create_app()

def init_db():
    with app.app_context():
        db.create_all()
        # Create admin user if not exists
        from config import Config
        from werkzeug.security import generate_password_hash
        
        admin = User.query.filter_by(email=Config.ADMIN_EMAIL).first()
        if not admin:
            print(f"Creating admin user: {Config.ADMIN_EMAIL}")
            hashed_pw = generate_password_hash(Config.ADMIN_PASSWORD)
            new_admin = User(email=Config.ADMIN_EMAIL, password_hash=hashed_pw, role='admin')
            db.session.add(new_admin)
            db.session.commit()

if __name__ == '__main__':
    # Initialize DB on start (for MVP simplicity)
    # Check if we can connect to DB first to avoid crash if env vars are missing
    if os.getenv('DATABASE_URL'):
        try:
            init_db()
        except Exception as e:
            print(f"Database initialization failed: {e}")
            print("Running without database connection (features will be limited)")
    
    port = int(os.environ.get('PORT', 5000))
    
    # Reduce logging verbosity in production
    if os.getenv('FLASK_ENV') != 'development':
        import logging
        logging.getLogger('werkzeug').setLevel(logging.WARNING)
        logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_ENV') == 'development')
