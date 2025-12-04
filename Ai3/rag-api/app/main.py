import logging
from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from sqlmodel import SQLModel
from app.config import get_settings
from app.api.v1.router import api_router
from app.core.audit_logger import setup_audit_logger
from app.core.cache import init_cache
from app.core.observability import setup_otlp, setup_sentry
settings = get_settings()
logger = logging.getLogger(__name__)
@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_otlp()
    setup_sentry()
    setup_audit_logger()
    await init_cache()
    yield
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)
# Middlewares
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"] if settings.DEBUG else [".yourdomain.com"],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Routes
app.include_router(api_router, prefix="/api/v1")
# Health
@app.get("/healthz")
async def k8s_health():
    return {"status": "ok"}
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
