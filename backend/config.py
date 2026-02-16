import os
from datetime import timedelta


class Config:
    # Flask
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change")
    ENV = "development"

    # Database (SQLite only)
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", f"sqlite:///{os.path.join(os.path.dirname(__file__), 'app.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # JWT
    JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-jwt-secret-change")
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=12)
    # Enable blocklist callbacks for revoked tokens
    JWT_BLACKLIST_ENABLED = True
    # Ensure access tokens are checked against the blocklist
    JWT_BLOCKLIST_TOKEN_CHECKS = "access"

    # Caching – use SimpleCache (in-memory); no Redis required
    CACHE_TYPE = os.getenv("CACHE_TYPE", "SimpleCache")
    CACHE_DEFAULT_TIMEOUT = int(os.getenv("CACHE_DEFAULT_TIMEOUT", "60"))

    # Mail (simple SMTP settings; for dev you can use a dummy or print to console)
    MAIL_SERVER = os.getenv("MAIL_SERVER", "localhost")
    MAIL_PORT = int(os.getenv("MAIL_PORT", "25"))
    MAIL_USERNAME = os.getenv("MAIL_USERNAME")
    MAIL_PASSWORD = os.getenv("MAIL_PASSWORD")
    MAIL_USE_TLS = os.getenv("MAIL_USE_TLS", "false").lower() == "true"
    MAIL_USE_SSL = os.getenv("MAIL_USE_SSL", "false").lower() == "true"
    MAIL_DEFAULT_SENDER = os.getenv("MAIL_DEFAULT_SENDER", "noreply@parking-app.local")

    # Notifications
    CHAT_WEBHOOK_URL = os.getenv("CHAT_WEBHOOK_URL")  # Google Chat Incoming Webhook (optional)

    # Billing policy: options: 'proportional' (default), 'per_15_min', 'per_hour', 'minimum_1_hour'
    BILLING_POLICY = os.getenv("BILLING_POLICY", "proportional")
    BILLING_MINUTES = int(os.getenv("BILLING_MINUTES", "15"))


def get_config():
    return Config()
