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
    
    # User Preferences
    pref_course = db.Column(db.String(100), nullable=True)
    pref_semester = db.Column(db.String(20), nullable=True)
    pref_subject = db.Column(db.String(100), nullable=True)

    def to_dict(self):
        return {
            'id': self.id,
            'email': self.email,
            'role': self.role,
            'created_at': self.created_at.isoformat() + 'Z',
            'pref_course': self.pref_course,
            'pref_semester': self.pref_semester,
            'pref_subject': self.pref_subject
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
    doc_type = db.Column(db.String(50), default='syllabus', nullable=True) # syllabus, system_info, general
    
    chunks = db.relationship('DocumentChunk', backref='document', lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.filename,
            'upload_date': self.upload_date.isoformat(),
            'status': self.status,
            'course': self.course,
            'semester': self.semester,
            'subject': self.subject,
            'doc_type': self.doc_type
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
    session_id = db.Column(db.String(36), db.ForeignKey('public.chat_sessions.id'), nullable=True, index=True) # UUID for chat session
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
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat() + 'Z'
        }

class ChatSession(db.Model):
    __tablename__ = 'chat_sessions'
    __table_args__ = {'schema': 'public'}
    
    id = db.Column(db.String(36), primary_key=True) # UUID
    user_id = db.Column(db.Integer, db.ForeignKey('public.users.id'), nullable=False)
    title = db.Column(db.String(255), nullable=False, default='New Chat')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationship to messages
    messages = db.relationship('ChatMessage', backref='session', lazy=True, cascade="all, delete-orphan",
                               primaryjoin="ChatMessage.session_id == ChatSession.id",
                               foreign_keys="ChatMessage.session_id")

    def to_dict(self):
        return {
            'id': self.id,
            'title': self.title,
            'created_at': self.created_at.isoformat() + 'Z',
            'updated_at': self.updated_at.isoformat() + 'Z'
        }

class AppSetting(db.Model):
    __tablename__ = 'app_settings'
    __table_args__ = {'schema': 'public'}

    id = db.Column(db.Integer, primary_key=True)
    key = db.Column(db.String(128), unique=True, nullable=False)
    value = db.Column(db.Text, nullable=True)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    @staticmethod
    def get(key, default=None):
        row = AppSetting.query.filter_by(key=key).first()
        return row.value if row else default

    @staticmethod
    def set(key, value):
        row = AppSetting.query.filter_by(key=key).first()
        if row:
            row.value = value
            row.updated_at = datetime.utcnow()
        else:
            row = AppSetting(key=key, value=value)
            db.session.add(row)
        db.session.commit()


class FilterOption(db.Model):
    __tablename__ = 'filter_options'
    __table_args__ = {'schema': 'public'}
    
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String(50), nullable=False) # 'course', 'semester', 'subject'
    value = db.Column(db.String(100), nullable=False)
    parent_id = db.Column(db.Integer, db.ForeignKey('public.filter_options.id'), nullable=True)
    
    children = db.relationship('FilterOption', backref=db.backref('parent', remote_side=[id]), lazy=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'category': self.category,
            'value': self.value,
            'parent_id': self.parent_id
        }
