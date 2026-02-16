import os
import sys
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Support running both as a module (python -m backend.app) and as a script (python backend/app.py)
try:
    from .config import get_config
    from .database import db
    from .cache import cache
    from .auth import auth_bp, register_auth
    from .routes_admin import admin_bp
    from .routes_reservations import res_bp
    from .routes_metrics import metrics_bp
    from .routes_jobs import jobs_bp
    from .routes_ml import ml_bp
    from .seed import seed_admin
except ImportError:  # pragma: no cover - fallback for direct script execution
    # Add the project root to sys.path and import via absolute package name
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    from backend.config import get_config
    from backend.database import db
    from backend.cache import cache
    from backend.auth import auth_bp, register_auth
    from backend.routes_admin import admin_bp
    from backend.routes_reservations import res_bp
    from backend.routes_metrics import metrics_bp
    from backend.routes_jobs import jobs_bp
    from backend.routes_ml import ml_bp
    from backend.seed import seed_admin


def create_app():
    app = Flask(__name__, static_folder=None)
    app.config.from_object(get_config())

    # Database
    db.init_app(app)

    # CORS for local dev
    CORS(app)

    # Cache (SimpleCache – no Redis needed)
    cache.init_app(app)

    # JWT
    register_auth(app)

    # JWT error handlers for debugging
    from flask_jwt_extended.exceptions import JWTExtendedException
    @app.errorhandler(JWTExtendedException)
    def handle_jwt_error(e):
        app.logger.error(f"JWT error: {e}")
        return jsonify({"message": str(e)}), 422

    @app.errorhandler(422)
    def handle_unprocessable(e):
        app.logger.error(f"422 Unprocessable: {e}")
        return jsonify({"message": "Unprocessable Entity", "error": str(e)}), 422

    # Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_bp)
    app.register_blueprint(res_bp)
    app.register_blueprint(metrics_bp)
    app.register_blueprint(jobs_bp)
    app.register_blueprint(ml_bp)

    # Static frontend: serve `frontend/index.html` and assets
    @app.route("/")
    def index():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        frontend = os.path.join(root, "frontend")
        resp = send_from_directory(frontend, "index.html")
        # Dev: force fresh HTML to avoid stale SW cached shell
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        return resp

    @app.route("/assets/<path:path>")
    def assets(path):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        frontend = os.path.join(root, "frontend")
        resp = send_from_directory(frontend, path)
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    # PWA assets
    @app.route("/manifest.webmanifest")
    def manifest():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        frontend = os.path.join(root, "frontend")
        resp = send_from_directory(frontend, "manifest.webmanifest")
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    @app.route("/service-worker.js")
    def service_worker():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        frontend = os.path.join(root, "frontend")
        resp = send_from_directory(frontend, "service-worker.js")
        # Bump a manual version header to help force update
        resp.headers['Cache-Control'] = 'no-store'
        resp.headers['X-SW-Version'] = 'dev-refresh-1'
        return resp

    @app.route("/icons/<path:path>")
    def icons(path):
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        frontend = os.path.join(root, "frontend")
        resp = send_from_directory(os.path.join(frontend, "icons"), path)
        resp.headers['Cache-Control'] = 'no-store'
        return resp

    # Favicon
    @app.route("/favicon.ico")
    def favicon():
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        frontend = os.path.join(root, "frontend")
        # Serve our SVG icon as favicon
        return send_from_directory(frontend, "icons/icon-192.svg")

    @app.get("/api/health")
    def api_health():
        return jsonify({"status": "ok"})

    @app.get("/api/version")
    def api_version():
        return jsonify({"name": "parking-app", "version": "1.0.0"})

    # CLI command: init-db
    @app.cli.command("init-db")
    def init_db_cmd():
        with app.app_context():
            db.create_all()
            seed_admin()
            print("Initialized the database and ensured admin user exists.")

    return app


if __name__ == "__main__":
    app = create_app()
    with app.app_context():
        db.create_all()
        seed_admin()
    app.run(host="0.0.0.0", port=5000, debug=True)
