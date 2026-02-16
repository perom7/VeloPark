import time
import redis
from flask import current_app

_redis = None
_revoked_memory: dict[str, float | None] = {}
_revoked_token_strings: set[str] = set()


def init_app(app):
    global _redis
    url = app.config.get('REDIS_URL')
    try:
        _redis = redis.from_url(url, decode_responses=True)
    except Exception:
        _redis = None


def add(jti: str, expires_seconds: int | None = None):
    """Add token jti to blocklist with optional TTL."""
    if _redis is None:
        # In-memory fallback for tests/dev when Redis is unavailable
        expires_at = (time.time() + expires_seconds) if expires_seconds else None
        _revoked_memory[jti] = expires_at
        return True
    key = f"revoked_token:{jti}"
    if expires_seconds:
        _redis.setex(key, expires_seconds, '1')
    else:
        _redis.set(key, '1')
    return True

def add_token_string(token: str):
    """Fallback: store raw token string as revoked."""
    if token:
        _revoked_token_strings.add(token)


def is_revoked(jwt_header, jwt_payload):
    """Return True if token is revoked (used by flask_jwt_extended token_in_blocklist_loader)."""
    global _redis
    if _redis is None:
        # In-memory check
        jti = jwt_payload.get('jti')
        if not jti:
            return False
        expires_at = _revoked_memory.get(jti)
        if expires_at is None:
            # present and no expiry -> revoked
            return jti in _revoked_memory
        # remove expired entries lazily
        if expires_at and expires_at < time.time():
            _revoked_memory.pop(jti, None)
            return False
        return jti in _revoked_memory
    jti = jwt_payload.get('jti')
    if not jti:
        return False
    key = f"revoked_token:{jti}"
    return _redis.exists(key) == 1

def is_token_string_revoked(token: str) -> bool:
    return token in _revoked_token_strings
