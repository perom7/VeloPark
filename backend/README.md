# Backend (Flask)

Requirements:
- Python 3.10+
- Redis server (for caching and Celery)

Setup (Windows PowerShell):

1. Create and activate a virtual environment
   - python -m venv .venv
   - .\.venv\Scripts\Activate.ps1

2. Install dependencies
   - pip install -r backend/requirements.txt

3. Configure environment (optional for dev; defaults are provided)
   - Copy `.env.example` to `.env` and adjust values

4. Initialize the database (SQLite)
   - python -m backend.app init-db
   or
   - flask --app backend.app init-db

5. Run the API
   - Recommended: `python -m backend.app`
   - Also supported: `python backend/app.py` (script mode enabled)

6. Run Celery (separate terminals)
   - celery -A backend.tasks.celery worker -l info
   - celery -A backend.tasks.celery beat -l info

Redis without Docker (Windows options):

- Memurai (Windows service, Redis-compatible):
   1. Download and install Memurai (Community Edition).
   2. Ensure the Memurai service is running (starts on boot by default).
   3. Use REDIS_URL=redis://localhost:6379/0.

- WSL2 + Ubuntu:
   1. Install WSL2 and Ubuntu from Microsoft Store.
   2. In Ubuntu: `sudo apt update && sudo apt install -y redis-server && sudo systemctl enable redis-server && sudo systemctl start redis-server`.
   3. Use REDIS_URL=redis://localhost:6379/0 (Windows can reach WSL on localhost).

To verify connectivity (VS Code): run the task "Redis: Verify (Local)".

Notes:
- Admin is auto-created on first run: username `admin`, password `admin123`.
- API root: http://localhost:5000
- Health check: GET /api/health

Notifications & schedules:
- Email: configure MAIL_* settings in `.env` (server, port, username/password, TLS/SSL). If not set, messages are printed to console.
- Google Chat: set `CHAT_WEBHOOK_URL` for an incoming webhook; used as fallback if email is not configured.
- Celery Beat schedules read cron strings from `DAILY_REMINDER_CRON` and `MONTHLY_REPORT_CRON`.
