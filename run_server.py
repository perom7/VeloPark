"""Start server without reloader to prevent training interruptions."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backend.app import create_app
from backend.database import db
from backend.seed import seed_admin

app = create_app()
with app.app_context():
    db.create_all()
    seed_admin()
app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
