# backend/app/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional
import secrets

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "RAG System Pro"
    APP_VERSION: str = "2.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"
    
    # API
    API_V1_PREFIX: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # Celery
    CELERY_BROKER_URL: str = "amqp://guest:guest@localhost:5672//"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/1"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # File Upload
    MAX_UPLOAD_SIZE: int = 50 * 1024 * 1024  # 50MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".txt", ".html", ".md"}
    UPLOAD_DIR: str = "data/uploads"
    
    # Vector Store
    FAISS_INDEX_DIR: str = "data/indexes"
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    
    # LLM
    LLM_MODEL_PATH: str = "data/models/mistral-7b-instruct-v0.3.Q4_K_M.gguf"
    LLM_CONTEXT_LENGTH: int = 32768
    LLM_MAX_TOKENS: int = 1024
    LLM_TEMPERATURE: float = 0.1
    LLM_THREADS: int = 8
    
    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 128
    MIN_CHUNK_SIZE: int = 50
    
    # Retrieval
    TOP_K_RETRIEVAL: int = 20
    TOP_K_RERANK: int = 5
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Reranking
    RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"
    USE_RERANKING: bool = True
    
    # Monitoring
    PROMETHEUS_PORT: int = 9090
    ENABLE_METRICS: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_DIR: str = "logs"
    
    # Security
    ENCRYPTION_KEY: str = secrets.token_urlsafe(32)
    BCRYPT_ROUNDS: int = 12
    CORS_ORIGINS: list = ["http://localhost:3000", "https://yourdomain.com"]
    
    # S3 Backup
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = None
    S3_REGION: str = "us-east-1"
    
    # Feature Flags
    ENABLE_FINE_TUNING: bool = True
    ENABLE_AB_TESTING: bool = True
    ENABLE_TELEMETRY: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    return Settings()