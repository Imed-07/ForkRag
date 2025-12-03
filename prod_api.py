# backend/app/api/v1/endpoints/query.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from app.core.security import get_current_user, RoleChecker
from app.api.middleware.rate_limit import rate_limit
from app.services.rag.pipeline import RAGPipeline
from app.dependencies import get_rag_pipeline
import logging

logger = logging.getLogger(__name__)
router = APIRouter()

# Request/Response Models
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500, description="User query")
    top_k: int = Field(5, ge=1, le=20, description="Number of sources to retrieve")
    enable_reranking: bool = Field(True, description="Use reranker for precision")
    stream: bool = Field(False, description="Stream response tokens")
    filters: Optional[Dict] = Field(None, description="Metadata filters")

class Source(BaseModel):
    text: str
    source: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    confidence: float
    metrics: Dict
    query_id: str

# Endpoints
@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
@rate_limit(max_calls=60, period=60)  # 60 req/min
async def query_documents(
    request: QueryRequest,
    user: Dict = Depends(get_current_user),
    rag_pipeline: RAGPipeline = Depends(get_rag_pipeline)
):
    """
    Query the RAG system
    
    **Rate Limit**: 60 requests/minute per user
    
    **Returns**: Answer with sources and confidence score
    """
    try:
        tenant_id = user.get('tenant_id')
        
        if request.stream:
            # Return streaming response
            async def generate():
                async for chunk in rag_pipeline.run(
                    query=request.query,
                    tenant_id=tenant_id,
                    top_k=request.top_k,
                    stream=True,
                    enable_reranking=request.enable_reranking
                ):
                    yield f"data: {json.dumps(chunk)}\n\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        
        # Standard response
        result = await rag_pipeline.run(
            query=request.query,
            tenant_id=tenant_id,
            top_k=request.top_k,
            enable_reranking=request.enable_reranking
        )
        
        return result
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}"
        )

@router.post("/feedback")
async def submit_feedback(
    query_id: str,
    rating: int = Field(..., ge=1, le=5),
    feedback: Optional[str] = None,
    user: Dict = Depends(get_current_user)
):
    """Submit feedback for a query"""
    # TODO: Store feedback in database
    return {"status": "feedback_received", "query_id": query_id}

# backend/app/api/v1/endpoints/documents.py
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, status, BackgroundTasks
from typing import List
from pydantic import BaseModel
from app.core.security import get_current_user
from app.services.ingestion.ingest_service import IngestionService
from app.tasks.ingestion_tasks import process_document_task
import os
import uuid

router = APIRouter()

class DocumentResponse(BaseModel):
    id: str
    filename: str
    status: str
    message: str

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: Dict = Depends(get_current_user),
    ingestion_service: IngestionService = Depends()
):
    """
    Upload and index a document
    
    **Supported formats**: PDF, DOCX, TXT, HTML, MD
    
    **Max size**: 50MB
    
    **Processing**: Asynchronous (check status endpoint)
    """
    # Validate file
    file_ext = os.path.splitext(file.filename)[1].lower()
    allowed_exts = {'.pdf', '.docx', '.txt', '.html', '.md'}
    
    if file_ext not in allowed_exts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed: {', '.join(allowed_exts)}"
        )
    
    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large. Maximum size: 50MB"
        )
    
    # Generate document ID
    doc_id = str(uuid.uuid4())
    tenant_id = user.get('tenant_id')
    user_id = user['id']
    
    # Save file
    upload_dir = f"data/uploads/{tenant_id}"
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, f"{doc_id}{file_ext}")
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    # Queue processing task
    background_tasks.add_task(
        process_document_task.delay,
        doc_id=doc_id,
        file_path=file_path,
        filename=file.filename,
        user_id=user_id,
        tenant_id=tenant_id
    )
    
    return DocumentResponse(
        id=doc_id,
        filename=file.filename,
        status="queued",
        message="Document queued for processing"
    )

@router.get("/status/{document_id}")
async def get_document_status(
    document_id: str,
    user: Dict = Depends(get_current_user)
):
    """Check document processing status"""
    # TODO: Query database
    return {
        "id": document_id,
        "status": "processing",
        "progress": 75
    }

@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    user: Dict = Depends(RoleChecker(['admin', 'user']))
):
    """Delete a document and its vectors"""
    # TODO: Delete from vector store and database
    return {"status": "deleted", "id": document_id}

@router.get("/list")
async def list_documents(
    user: Dict = Depends(get_current_user),
    skip: int = 0,
    limit: int = 100
):
    """List user's documents"""
    # TODO: Query database with pagination
    return {
        "documents": [],
        "total": 0,
        "skip": skip,
        "limit": limit
    }

# backend/app/api/v1/endpoints/admin.py
from fastapi import APIRouter, Depends, HTTPException
from app.core.security import RoleChecker
from app.services.evaluation.metrics import RAGMetrics
from app.dependencies import get_metrics

router = APIRouter()

@router.get("/metrics", dependencies=[Depends(RoleChecker(['admin']))])
async def get_system_metrics(
    metrics: RAGMetrics = Depends(get_metrics)
):
    """Get system performance metrics (admin only)"""
    return metrics.get_recent_stats(hours=24)

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow()
    }

@router.post("/rebuild-index", dependencies=[Depends(RoleChecker(['admin']))])
async def rebuild_index():
    """Rebuild vector index (admin only)"""
    # TODO: Trigger index rebuild
    return {"status": "rebuilding"}

# backend/app/api/middleware/rate_limit.py
from fastapi import Request, HTTPException
from functools import wraps
import redis
import time
from app.config import get_settings

settings = get_settings()
redis_client = redis.from_url(settings.REDIS_URL)

def rate_limit(max_calls: int = 60, period: int = 60):
    """
    Rate limiting decorator
    
    Args:
        max_calls: Maximum calls allowed
        period: Time period in seconds
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            request: Request = kwargs.get('request') or args[0]
            
            # Get user ID or IP
            user = kwargs.get('user', {})
            user_id = user.get('id', request.client.host)
            
            # Rate limit key
            key = f"rate_limit:{func.__name__}:{user_id}"
            
            # Check current count
            current = redis_client.get(key)
            
            if current and int(current) >= max_calls:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Max {max_calls} calls per {period}s"
                )
            
            # Increment
            pipe = redis_client.pipeline()
            pipe.incr(key)
            pipe.expire(key, period)
            pipe.execute()
            
            return await func(*args, **kwargs)
        
        return wrapper
    return decorator