
import sys
import os
# Add the project root to sys.path
sys.path.append(os.getcwd())

from app import create_app, db
from app.models import Document

app = create_app()
with app.app_context():
    docs = Document.query.filter_by(doc_type='syllabus').order_by(Document.created_at.desc()).limit(5).all()
    print(f"Checking {len(docs)} syllabus documents:")
    for d in docs:
        has_json = bool(d.structure_json)
        json_len = len(d.structure_json) if d.structure_json else 0
        print(f"ID: {d.id} | Subject: {d.subject} | Status: {d.status} | JSON: {has_json} ({json_len} chars)")
        if has_json:
            print(f"  Snippet: {d.structure_json[:100]}...")
