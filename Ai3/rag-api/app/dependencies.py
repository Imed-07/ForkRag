from typing import AsyncGenerator
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from app.config import get_settings
from app.models.user import User
from app.core.security import decode_jwt
from app.core.cache import get_redis
settings = get_settings()
async_engine = create_async_engine(
    str(settings.DATABASE_URL), pool_pre_ping=True, future=True
)
AsyncSessionLocal = sessionmaker(
    bind=async_engine, class_=AsyncSession, expire_on_commit=False
)
security = HTTPBearer()
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session
async def get_current_user(
    creds: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    token = creds.credentials
    try:
        payload = decode_jwt(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await db.get(User, int(user_id))
    if user is None or not user.is_active:
        raise HTTPException(status_code=404, detail="User inactive")
    return user
TenantHeader = Depends(lambda x: x.headers.get("x-tenant-id", "default"))
