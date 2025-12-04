import os
from functools import lru_cache
from pydantic import BaseSettings, PostgresDsn, RedisDsn, AnyUrl
class Settings(BaseSettings):
    # API
    PROJECT_NAME: str = "rag-api"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    ENV: str = os.getenv("ENV", "prod")
    # Security
    JWT_PUBLIC_KEY: str
    JWT_PRIVATE_KEY: str
    JWT_ALGORITHM: str = "RS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_CONTEXT_SCHEME: str = "bcrypt"
    # Postgres
    DATABASE_URL: PostgresDsn
    # Redis
    REDIS_URL: RedisDsn
    # ML-Worker gRPC
    ML_WORKER_GRPC_TARGET: str
    # Observability
    OTLP_ENDPOINT: str | None
    SENTRY_DSN: str | None
    # Feature-flags JSON path
    FEATURE_FLAGS_PATH: str = "/etc/rag-api/feature-flags.json"
    class Config:
        env_file = ".env", ".env.prod"
        case_sensitive = True
@lru_cache
def get_settings() -> Settings:
    return Settings()
