from app import create_app, db
from sqlalchemy import text

app = create_app()
with app.app_context():
    with db.engine.connect() as conn:
        print("Starting migration on Postgres...")
        
        try:
            conn.execute(text("ALTER TABLE documents ADD COLUMN course VARCHAR(100)"))
            print("Added course column")
        except Exception as e:
            print(f"Course column might exist or error: {e}")
        
        try:
            conn.execute(text("ALTER TABLE documents ADD COLUMN semester VARCHAR(20)"))
            print("Added semester column")
        except Exception as e:
            print(f"Semester column might exist or error: {e}")
            
        conn.commit()
        print("Migration finished.")
