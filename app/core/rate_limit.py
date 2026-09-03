# In-process sliding-window limiter for POST /ask.
# Keyed by authenticated user id, or by client IP when the caller is anonymous.

import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from fastapi import Depends, Request

from app.core.auth import AuthUser, get_current_user
from app.core.config import Settings, get_settings
from app.core.errors import raise_http


class SlidingWindowLimiter:
    def __init__(self) -> None:
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()

    def check(self, key: str, limit: int, window_seconds: int) -> Tuple[bool, int, int]:
        """Returns (allowed, remaining, retry_after_seconds)."""
        now = time.monotonic()
        cutoff = now - window_seconds
        with self._lock:
            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= limit:
                retry = int(bucket[0] + window_seconds - now) + 1
                return False, 0, max(retry, 1)
            bucket.append(now)
            remaining = limit - len(bucket)
            return True, remaining, 0


limiter = SlidingWindowLimiter()


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def enforce_rate_limit(
    request: Request,
    user: AuthUser = Depends(get_current_user),
    settings: Settings = Depends(get_settings),
) -> AuthUser:
    key = user.id if user.id != "local-dev" else f"ip:{client_ip(request)}"
    allowed, remaining, retry_after = limiter.check(
        key,
        settings.rate_limit_requests,
        settings.rate_limit_window_seconds,
    )
    request.state.rate_limit_remaining = remaining
    request.state.rate_limit_limit = settings.rate_limit_requests
    if not allowed:
        raise_http(
            429,
            "Rate limit exceeded",
            "rate_limited",
            headers={"Retry-After": str(retry_after)},
        )
    return user
