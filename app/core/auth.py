# Authentication — same contract as arkgpt's requireSocialListeningUser.
#
# src/lib/community-mapper/auth.ts:
#   - Authorization: Bearer <supabase access_token> is required
#   - Cookie-only auth is rejected
#   - No developer-flag check (social-listening console is for signed-in users)
#
# This service verifies the JWT with GET {SUPABASE_URL}/auth/v1/user using the
# anon/public key as `apikey`, matching callerUserId() in arkgpt.

import logging
from dataclasses import dataclass
from typing import Optional

import httpx
from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.errors import raise_http

logger = logging.getLogger(__name__)

LOCAL_USER_ID = "local-dev"


@dataclass
class AuthUser:
    id: str
    email: Optional[str] = None
    jwt: Optional[str] = None


def bearer_token(request: Request) -> Optional[str]:
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not header or not header.startswith("Bearer "):
        return None
    token = header[7:].strip()
    return token or None


def _verify_supabase_jwt(settings: Settings, token: str) -> Optional[AuthUser]:
    url = (settings.next_public_supabase_url or "").rstrip("/")
    anon = settings.supabase_anon_key
    if not url or not anon:
        logger.warning("Auth required but Supabase public key/url missing")
        return None

    try:
        response = httpx.get(
            f"{url}/auth/v1/user",
            headers={
                "apikey": anon,
                "Authorization": f"Bearer {token}",
            },
            timeout=8.0,
        )
    except httpx.HTTPError as exc:
        logger.error("Supabase auth lookup failed: %s", exc)
        raise_http(502, "Auth provider unavailable", "auth_upstream")

    if response.status_code != 200:
        return None

    data = response.json()
    user_id = data.get("id")
    if not user_id:
        return None
    return AuthUser(id=str(user_id), email=data.get("email"), jwt=token)


def get_current_user(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> AuthUser:
    """FastAPI dependency. Anonymous local-dev user when AUTH_REQUIRED=false."""

    if not settings.auth_required:
        token = bearer_token(request)
        if token and settings.supabase_anon_key and settings.next_public_supabase_url:
            user = _verify_supabase_jwt(settings, token)
            if user:
                return user
        return AuthUser(id=LOCAL_USER_ID)

    token = bearer_token(request)
    if not token:
        raise_http(401, "auth required", "auth_required")

    user = _verify_supabase_jwt(settings, token)
    if not user:
        raise_http(401, "auth required", "invalid_token")
    return user
