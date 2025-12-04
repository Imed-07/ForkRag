from datetime import timedelta
from typing import Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.config import get_settings
from app.dependencies import get_db
from app.models.user import User
from app.schemas.auth import LoginRequest, TokenPair
from app.core.security import (
    verify_password,
    create_jwt,
    hash_password,
    generate_secure_random_bytes,
)
router = APIRouter()
settings = get_settings()
@router.post("/login", response_model=TokenPair)
async def login(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> Any:
    user = await User.authenticate(db, body.email, body.password)
    if not user:
        raise HTTPException(status_code=401, detail="Bad credentials")
    access = create_jwt(
        {"sub": str(user.id), "tenant": user.tenant_id},
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    refresh = create_jwt(
        {"sub": str(user.id), "type": "refresh"},
        timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
    )
    return TokenPair(access_token=access, refresh_token=refresh)
@router.post("/refresh", response_model=TokenPair)
async def refresh(refresh_token: str, db: AsyncSession = Depends(get_db)) -> Any:
    from app.core.security import decode_jwt
    payload = decode_jwt(refresh_token)
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await db.get(User, int(payload["sub"]))
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User inactive")
    access = create_jwt(
        {"sub": str(user.id), "tenant": user.tenant_id},
        timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return TokenPair(access_token=access, refresh_token=refresh_token)
@router.post("/register", status_code=201)
async def register(body: LoginRequest, db: AsyncSession = Depends(get_db)) -> Any:
    exists = await User.get_by_email(db, body.email)
    if exists:
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        tenant_id="default",
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return {"id": user.id, "email": user.email}
