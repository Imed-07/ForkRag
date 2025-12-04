from typing import Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from app.dependencies import get_current_user
from app.schemas.query import QueryRequest, QueryResponse
from app.core.rate_limiter import limiter
from app.core.cache import cache
from app.core.audit_logger import audit
from app.services.ml_gateway import ml_query
router = APIRouter()
class RAGRequest(BaseModel):
    question: str
    top_k: int = 5
    filters: dict | None = None
@router.post("/", response_model=QueryResponse)
@limiter.limit("10/minute")
async def rag_query(
    payload: RAGRequest,
    user=Depends(get_current_user),
) -> Any:
    cache_key = f"q:{hash(payload.question + str(user.tenant_id))}"
    cached = await cache.get(cache_key)
    if cached:
        return QueryResponse(**cached)
    answer = await ml_query(
        question=payload.question,
        tenant_id=user.tenant_id,
        top_k=payload.top_k,
        filters=payload.filters,
    )
    await cache.set(cache_key, answer.dict(), expire=300)
    await audit.log(
        user_id=user.id,
        action="rag_query",
        details={"question": payload.question, "answer": answer.answer},
    )
    return answer
