import pytest
import time
import os
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

from api.auth import (
    create_access_token,
    verify_token,
    get_current_user
)

def test_create_and_verify_token_success():
    subject = "test_user"
    token = create_access_token(subject)
    
    payload = verify_token(token)
    assert payload["sub"] == subject
    assert "exp" in payload
    assert "iat" in payload

def test_verify_token_expired():
    token = create_access_token("test_user", expires_delta=-10)
    
    with pytest.raises(ValueError, match="Token expired"):
        verify_token(token)

def test_verify_token_invalid_signature():
    token = create_access_token("test_user")
    
    parts = token.split(".")
    import base64
    tampered_payload = base64.urlsafe_b64encode(b'{"sub": "hacker"}').rstrip(b"=").decode()
    tampered_token = f"{parts[0]}.{tampered_payload}.{parts[2]}"
    
    with pytest.raises(ValueError, match="Invalid signature"):
        verify_token(tampered_token)

def test_verify_token_malformed():
    with pytest.raises(ValueError, match="Malformed token"):
        verify_token("not.a.valid.token.format")

@pytest.mark.asyncio
async def test_get_current_user_valid_token(monkeypatch):
    monkeypatch.setattr("api.auth.AUTH_ENABLED", True)
    token = create_access_token("admin")
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    user = await get_current_user(creds)
    assert user == "admin"

@pytest.mark.asyncio
async def test_get_current_user_missing_credentials(monkeypatch):
    monkeypatch.setattr("api.auth.AUTH_ENABLED", True)
    # If credentials object is not provided, it should raise a 401
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(None)
    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == "Missing Authorization header"

@pytest.mark.asyncio
async def test_get_current_user_invalid_token(monkeypatch):
    monkeypatch.setattr("api.auth.AUTH_ENABLED", True)
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="invalid.token.str")
    with pytest.raises(HTTPException) as excinfo:
        await get_current_user(creds)
    assert excinfo.value.status_code == 401
    assert "Invalid token" in excinfo.value.detail
