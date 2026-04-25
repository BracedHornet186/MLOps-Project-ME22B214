"""
api/auth.py
─────────────────────────────────────────────────────────────────────────────
JWT bearer-token authentication for the Scene Reconstruction API.

Secrets are loaded from Docker Secrets (``/run/secrets/jwt_secret``) with a
fallback to the ``JWT_SECRET`` environment variable for local development.

Usage in serve_app.py:
    from api.auth import get_current_user, auth_router
    fastapi_app.include_router(auth_router)
    # Then add Depends(get_current_user) to protected endpoints.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

log = logging.getLogger("auth")

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

TOKEN_EXPIRY_SECONDS = int(os.environ.get("JWT_EXPIRY_SECONDS", "900"))  # 15 min
AUTH_ENABLED = os.environ.get("AUTH_ENABLED", "true").lower() in ("1", "true", "yes")

# Admin credentials — loaded from Docker Secrets or env vars
_API_USERNAME = os.environ.get("API_USERNAME", "admin")


def _load_secret(name: str, env_fallback: str, default: str = "") -> str:
    """Read a Docker Secret file, falling back to an env var."""
    secret_path = Path(f"/run/secrets/{name}")
    if secret_path.exists():
        return secret_path.read_text().strip()
    return os.environ.get(env_fallback, default)


JWT_SECRET = _load_secret("jwt_secret", "JWT_SECRET", "dev-only-insecure-key")
_API_PASSWORD_HASH = _load_secret("api_password_hash", "API_PASSWORD_HASH", "")

# If no password hash configured, use a default (hash of "admin") for dev.
if not _API_PASSWORD_HASH:
    _API_PASSWORD_HASH = hashlib.sha256(b"admin").hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# Minimal JWT implementation (HS256) — no external dependency needed
# ─────────────────────────────────────────────────────────────────────────────

def _b64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    import base64
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def create_access_token(subject: str, expires_delta: int = TOKEN_EXPIRY_SECONDS) -> str:
    """Create an HS256-signed JWT token."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": subject,
        "iat": int(time.time()),
        "exp": int(time.time()) + expires_delta,
    }
    segments = [
        _b64url_encode(json.dumps(header).encode()),
        _b64url_encode(json.dumps(payload).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def verify_token(token: str) -> dict:
    """Verify and decode an HS256 JWT. Raises ValueError on failure."""
    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed token")

    signing_input = f"{parts[0]}.{parts[1]}".encode()
    expected_sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    actual_sig = _b64url_decode(parts[2])

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("Invalid signature")

    payload = json.loads(_b64url_decode(parts[1]))

    if payload.get("exp", 0) < time.time():
        raise ValueError("Token expired")

    return payload


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI dependency
# ─────────────────────────────────────────────────────────────────────────────

_bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> Optional[str]:
    """
    FastAPI dependency that validates JWT bearer tokens.
    If AUTH_ENABLED is False, returns "anonymous" for all requests.
    """
    if not AUTH_ENABLED:
        return "anonymous"

    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = verify_token(credentials.credentials)
        return payload.get("sub", "unknown")
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Auth router
# ─────────────────────────────────────────────────────────────────────────────

auth_router = APIRouter(tags=["auth"])


class TokenRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = TOKEN_EXPIRY_SECONDS


@auth_router.post("/auth/token", response_model=TokenResponse)
async def login(body: TokenRequest):
    """
    Authenticate with username/password and receive a short-lived JWT.
    Default credentials: admin / admin (override via Docker Secrets).
    """
    password_hash = hashlib.sha256(body.password.encode()).hexdigest()

    if body.username != _API_USERNAME or not hmac.compare_digest(password_hash, _API_PASSWORD_HASH):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token(subject=body.username)
    log.info("Token issued for user=%s", body.username)
    return TokenResponse(access_token=token)
