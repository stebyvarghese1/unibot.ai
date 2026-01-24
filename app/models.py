from app import db
from datetime import datetime

class User(db.Model):
    __tablename__ = 'users'
    __table_args__ = {'schema': 'public'}
    
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    role = db.Column(db.String(20), nullable=False, default='student') # 'admin' or 'student'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat()
        }

class Document(db.Model):
    __tablename__ = 'documents'
    __table_args__ = {'schema': 'public'}
    
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    file_path = db.Column(db.String(512), nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('public.users.id'), nullable=False)
    status = db.Column(db.String(50), default='pending') # pending, processed, error
    course = db.Column(db.String(100), nullable=True)
    semester = db.Column(db.String(20), nullable=True)
    subject = db.Column(db.String(100), nullable=True)
    
    chunks = db.relationship('DocumentChunk', backref='document', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'upload_date': self.upload_date.isoformat(),
            'status': self.status,
            'course': self.course,
            'semester': self.semester,
            'subject': self.subject
        }

class DocumentChunk(db.Model):
    __tablename__ = 'document_chunks'
    __table_args__ = {'schema': 'public'}
    
    id = db.Column(db.Integer, primary_key=True)
    document_id = db.Column(db.Integer, db.ForeignKey('public.documents.id'), nullable=False)
    chunk_text = db.Column(db.Text, nullable=False)
    chunk_index = db.Column(db.Integer, nullable=False)
    # Metadata can be stored as JSON if needed, or simple columns
    # For now, we'll keep it simple
    
    def to_dict(self):
        return {
            'id': self.id,
            'document_id': self.document_id,
            'chunk_text': self.chunk_text[:50] + "...",
            'chunk_index': self.chunk_index
        }

class ChatMessage(db.Model):
    __tablename__ = 'chat_messages'
    __table_args__ = {'schema': 'public'}
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('public.users.id'), nullable=False)
    question = db.Column(db.Text, nullable=False)
    answer = db.Column(db.Text, nullable=False)
    course = db.Column(db.String(100), nullable=True)
    semester = db.Column(db.String(20), nullable=True)
    subject = db.Column(db.String(100), nullable=True)
    sources_json = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'user_id': self.user_id,
            'question': self.question,
            'answer': (self.answer[:200] + '...') if self.answer and len(self.answer) > 200 else self.answer,
            'course': self.course,
            'semester': self.semester,
            'subject': self.subject,
            'sources': self.sources_json,
            'created_at': self.created_at.isoformat()
        }
