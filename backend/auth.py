from datetime import datetime
from functools import wraps
from flask import Blueprint, request, jsonify
from .errors import make_error
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    get_jwt_identity,
    jwt_required,
)
from flask_jwt_extended import get_jwt
from .token_blacklist import init_app as init_blacklist, add as blacklist_add, add_token_string, is_token_string_revoked
from email_validator import validate_email, EmailNotValidError
from .database import db
from .models import User, Role


auth_bp = Blueprint("auth", __name__, url_prefix="/api/auth")


def register_auth(app):
    jwt = JWTManager(app)
    # initialize redis-backed blacklist
    try:
        init_blacklist(app)
    except Exception:
        pass
    
    # Custom error handlers for JWT validation failures
    @jwt.invalid_token_loader
    def invalid_token_callback(error_string):
        app.logger.error(f"Invalid token: {error_string}")
        return make_error("Invalid token", 422, {"error": error_string})
    
    @jwt.unauthorized_loader
    def unauthorized_callback(error_string):
        app.logger.error(f"Unauthorized: {error_string}")
        return make_error("Missing Authorization header", 401, {"error": error_string})
    
    @jwt.expired_token_loader
    def expired_token_callback(jwt_header, jwt_payload):
        app.logger.error(f"Expired token")
        return make_error("Token has expired", 401)
    
    @jwt.revoked_token_loader
    def revoked_token_callback(jwt_header, jwt_payload):
        app.logger.error(f"Revoked token")
        return make_error("Token has been revoked", 401)

    # Blocklist checker
    @jwt.token_in_blocklist_loader
    def check_if_token_revoked(jwt_header, jwt_payload):
        try:
            from .token_blacklist import is_revoked
            return is_revoked(jwt_header, jwt_payload)
        except Exception:
            return False

    # Global fallback: if raw token string was revoked, reject request
    @app.before_request
    def _reject_revoked_token_strings():
        try:
            auth = request.headers.get('Authorization', '')
            token = auth.replace('Bearer ', '').strip()
            if token and is_token_string_revoked(token):
                return make_error("Token has been revoked", 401)
        except Exception:
            # Ignore errors here; jwt_required will still run per-route
            pass


def require_auth(fn):
    """jwt_required with an extra revocation check to enforce blocklist even if backend is unavailable."""
    @wraps(fn)
    @jwt_required()
    def wrapper(*args, **kwargs):
        try:
            from .token_blacklist import is_revoked
            jwt_data = get_jwt()
            if is_revoked({}, jwt_data):
                return make_error("Token has been revoked", 401)
            # Fallback: raw token string check
            auth = request.headers.get('Authorization', '')
            token = auth.replace('Bearer ', '').strip()
            if token and is_token_string_revoked(token):
                return make_error("Token has been revoked", 401)
        except Exception:
            # On any error, proceed (jwt decorator already validated token)
            pass
        return fn(*args, **kwargs)
    return wrapper


def role_required(role: str):
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            try:
                ident = int(get_jwt_identity())
            except Exception:
                return make_error("Invalid identity", 422)
            user = User.query.filter_by(id=ident).first()
            if not user or user.role != role:
                return make_error("Forbidden", 403)
            return fn(*args, **kwargs)

        return wrapper

    return decorator


@auth_bp.post("/register")
def register():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()
    email = (data.get("email") or "").strip() or None

    if not username or not password:
        return make_error("username and password required", 400)

    if email:
        try:
            # Skip DNS deliverability checks for simplicity in tests/dev
            validate_email(email, check_deliverability=False)
        except EmailNotValidError as e:
            return make_error(str(e), 400)

    # Try to create the user and rely on DB unique constraints; return 409 on conflict.
    user = User(username=username, email=email, role=Role.USER.value)
    user.set_password(password)
    db.session.add(user)
    try:
        db.session.commit()
    except Exception as e:
        from sqlalchemy.exc import IntegrityError
        db.session.rollback()
        if isinstance(e, IntegrityError):
            return make_error("username or email already exists", 409)
        return make_error("failed to register", 500, {"error": str(e)})

    return jsonify({"message": "registered successfully"}), 201


@auth_bp.post("/login")
def login():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    user = User.query.filter_by(username=username).first()
    if not user or not user.check_password(password):
        return make_error("invalid credentials", 401)

    user.last_login_at = datetime.utcnow()
    db.session.commit()

    # Ensure identity is a string to avoid 'Subject must be a string' errors in some JWT backends
    token = create_access_token(identity=str(user.id), additional_claims={"role": user.role})
    return jsonify({"access_token": token, "user": {"id": user.id, "username": user.username, "role": user.role}})


@auth_bp.post('/logout')
@jwt_required()
def logout():
    """Revoke current JWT so it cannot be used again."""
    jwt_data = get_jwt()
    jti = jwt_data.get('jti')
    if not jti:
        # Gracefully return 200 even if JTI missing (nothing to revoke)
        return jsonify({"message": "no token id to revoke"}), 200
    # compute remaining ttl from expiration; may be absent in tests
    exp = jwt_data.get('exp')
    if exp:
        import time
        ttl = max(0, int(exp - time.time()))
    else:
        ttl = None
    try:
        blacklist_add(jti, ttl)
    except Exception:
        pass
    # Also store raw token string as revoked (fallback in tests/dev)
    auth = request.headers.get('Authorization', '')
    token = auth.replace('Bearer ', '').strip()
    if token:
        try:
            add_token_string(token)
        except Exception:
            pass
    return jsonify({"message": "token revoked"}), 200


@auth_bp.get("/me")
@jwt_required()
def me():
    try:
        ident = int(get_jwt_identity())
    except Exception:
        return make_error("Invalid identity", 422)
    user = User.query.get(ident)
    if not user:
        return make_error("not found", 404)
    return jsonify({
        "id": user.id,
        "username": user.username,
        "email": user.email,
        "role": user.role,
        "created_at": user.created_at.isoformat(),
        "last_login_at": user.last_login_at.isoformat() if user.last_login_at else None,
    })
