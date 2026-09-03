import logging
from typing import Optional

from supabase import create_client, Client

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_client: Optional[Client] = None


def get_supabase_admin(settings: Optional[Settings] = None) -> Optional[Client]:
    """Create or return the Supabase admin client (service-role).

    Mirrors arkgpt's getSupabaseAdmin() — uses SUPABASE_SECRET_KEY to
    bypass RLS. Returns None when Supabase isn't configured so callers
    can fall back to hardcoded data."""

    global _client

    if _client is not None:
        return _client

    if settings is None:
        settings = get_settings()

    if not settings.supabase_configured:
        logger.info("Supabase not configured — client unavailable")
        return None

    try:
        _client = create_client(
            settings.next_public_supabase_url,
            settings.supabase_secret_key,
        )
        logger.info("Supabase admin client created")
        return _client
    except Exception as exc:
        logger.warning("Failed to create Supabase client: %s", exc)
        return None
