# backend/app/models/user.py
from sqlalchemy import Column, String, DateTime, Boolean, Enum as SQLEnum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
import uuid
from app.core.database import Base

class UserRole(str, enum.Enum):
    ADMIN = "admin"
    USER = "user"
    API_USER = "api_user"

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    role = Column(SQLEnum(UserRole), default=UserRole.USER)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    api_key = Column(String, unique=True, index=True)
    tenant_id = Column(String, nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    documents = relationship("Document", back_populates="owner", cascade="all, delete-orphan")
    queries = relationship("Query", back_populates="user", cascade="all, delete-orphan")

# backend/app/models/document.py
class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    original_filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_size = Column(Integer)
    mime_type = Column(String)
    
    status = Column(SQLEnum(DocumentStatus), default=DocumentStatus.PENDING)
    error_message = Column(String, nullable=True)
    
    # Metadata
    num_chunks = Column(Integer, default=0)
    num_tokens = Column(Integer, default=0)
    language = Column(String, nullable=True)
    
    # Indexing
    index_id = Column(String, nullable=True)
    vector_count = Column(Integer, default=0)
    
    # Ownership
    user_id = Column(String, nullable=False, index=True)
    tenant_id = Column(String, nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    indexed_at = Column(DateTime, nullable=True)
    
    # Relationships
    owner = relationship("User", back_populates="documents")
    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")

# backend/app/models/chunk.py
class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String, nullable=False, index=True)
    
    chunk_index = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    encrypted_text = Column(String, nullable=True)  # For sensitive data
    
    # Metadata
    start_char = Column(Integer)
    end_char = Column(Integer)
    num_tokens = Column(Integer)
    
    # Vector reference
    vector_id = Column(String, nullable=True, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    document = relationship("Document", back_populates="chunks")

# backend/app/models/query.py
class Query(Base):
    __tablename__ = "queries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    tenant_id = Column(String, nullable=True, index=True)
    
    query_text = Column(String, nullable=False)
    response_text = Column(String, nullable=False)
    
    # Performance metrics
    retrieval_time_ms = Column(Integer)
    generation_time_ms = Column(Integer)
    total_time_ms = Column(Integer)
    
    # Retrieved sources
    num_sources = Column(Integer)
    source_ids = Column(String)  # JSON array of document IDs
    
    # Quality feedback
    user_rating = Column(Integer, nullable=True)  # 1-5
    user_feedback = Column(String, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Relationships
    user = relationship("User", back_populates="queries")

# backend/app/models/tenant.py
class Tenant(Base):
    __tablename__ = "tenants"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    
    # Quotas
    max_documents = Column(Integer, default=1000)
    max_queries_per_day = Column(Integer, default=10000)
    max_storage_mb = Column(Integer, default=5000)
    
    # Usage
    current_documents = Column(Integer, default=0)
    current_storage_mb = Column(Integer, default=0)
    queries_today = Column(Integer, default=0)
    
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)