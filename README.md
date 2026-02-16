# Vehicle Parking App V2

Multi-user parking management application (4-wheeler) with Admin and User roles.

## Tech Stack
Backend: Flask, SQLite (SQLAlchemy), JWT (flask-jwt-extended), Redis (caching + Celery broker), Celery (async + scheduled jobs)
Frontend: Vue 3 (CDN), Bootstrap 5, Chart.js, Service Worker (PWA)
PDF: xhtml2pdf (monthly reports) | CSV export via Celery

## Features (Core)
Admin:
- Auto-created superuser (admin/admin123)
- Create / update / delete parking lots (bulk spot creation, dynamic resizing with safety checks)
- View per-lot occupancy and current vehicle/reservation details
- View registered users
- Trigger monthly reports & daily reminders manually (demo endpoints)
- Bar chart of lot occupancy

User:
- Register & login (JWT)
- View public lot availability (cached)
- Reserve first available spot in chosen lot (auto-allocation)
- Release spot (cost computed using billing policy)
- View reservation history & user metrics (favorite lot, spend, active count)
- Trigger CSV export job and download result

Async Jobs (Celery):
- Daily reminders (inactive users) — scheduled by beat
- Monthly activity reports (HTML + optional PDF attachment via email/chat)
- CSV export (user-triggered)
- New lot notification broadcast

Performance & Caching:
- Redis (or SimpleCache fallback during tests) for lots, admin lots, spot listings.
- Explicit cache invalidation on mutations.

Security:
- JWT access tokens, logout token revocation (Redis blocklist with in-memory fallback)
- Validation (server-side + client-side) for lot and reservation payloads.

PWA:
- Manifest + service worker for offline shell and “Add to desktop”.

## Quick Start (Windows PowerShell)
```powershell
# 1. Clone (private repo recommended)
git clone <your-private-repo-url> parking-app
cd parking-app

# 2. Python virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r backend\requirements.txt

# 4. Start Redis (Memurai or native redis-server)
# Memurai service typically auto-starts; otherwise start manually:
# Start-Service MemuraiServer

# 5. (Optional) Set environment variables
$env:REDIS_URL='redis://localhost:6379/0'
$env:BILLING_POLICY='proportional'

# 6. Initialize DB & seed admin
python -m backend.app init-db

# 7. Start Celery worker & beat (two separate terminals)
python -m celery -A backend.tasks.celery worker -l info -P solo
python -m celery -A backend.tasks.celery beat -l info

# 8. Run API
python backend\app.py

# 9. Open browser
Start-Process http://localhost:5000/
```

## VS Code Tasks (Optional)
Tasks are configured for: API (Flask), Celery worker (solo), Celery beat, Redis verify script.

## Environment Variables (Selected)
| Var | Purpose | Default |
|-----|---------|---------|
| REDIS_URL | Redis connection | redis://localhost:6379/0 |
| BILLING_POLICY | Cost calc mode (proportional | per_15_min | per_hour | minimum_1_hour) | proportional |
| DAILY_REMINDER_CRON | Celery beat schedule for reminders | 0 18 * * * |
| MONTHLY_REPORT_CRON | Celery beat schedule for reports | 0 9 1 * * |
| CHAT_WEBHOOK_URL | Google Chat webhook (optional) | (unset) |
| MAIL_* | SMTP config for emails | dev defaults |

## API Overview (Selected)
Auth: POST /api/auth/register | POST /api/auth/login | GET /api/auth/me | POST /api/auth/logout
Admin Lots: GET /api/admin/lots | POST /api/admin/lots | PUT /api/admin/lots/<id> | DELETE /api/admin/lots/<id>
Admin Reports: POST /api/admin/reports/monthly/generate | GET /api/admin/reports/monthly | GET /api/admin/reports/monthly/download?filename=...
Admin Reminders: POST /api/admin/reminders/daily/run
Public Lots: GET /api/lots
Reservations: POST /api/reservations | POST /api/reservations/<id>/release | GET /api/reservations/me
Metrics: GET /api/metrics/admin | GET /api/metrics/user
CSV Export: POST /api/export/csv | GET /api/export/csv/<task_id>/status | GET /api/export/csv/<task_id>/download

## Testing
```powershell
pytest -q
```
Celery tasks run eagerly in tests (no Redis required). CSV export/status/download are patched to fallback when backend results store is absent.

## Monthly Reports (PDF)
`send_monthly_reports` generates styled HTML + PDF (if xhtml2pdf installed) per user, stored under `backend/reports/` and emailed/notified.

## CSV Export Flow
1. User triggers export → Celery task writes file to `backend/exports/`
2. Poll status endpoint until state=SUCCESS
3. Download using task id

## Token Revocation
Logout endpoint stores token JTI (and raw string fallback) in Redis / in-memory dict; before-request hook rejects revoked tokens uniformly.

## Validation
Server: `validators.py` (lot create/update, reservation, vehicle fields). Errors standardized via `make_error` (`message`, `status`, `errors`).
Client: Basic constraints and user feedback toasts in `frontend/js/app.js`.

## PWA
- `manifest.webmanifest` + `service-worker.js` for basic offline and install prompt.

## Generating ER Diagram
See `backend/ER.md` for schema description. Mermaid source & SVG in `docs/` folder.

## Building Report (Submission)
`REPORT.md` will be converted to PDF (external tool like pandoc or VS Code export). Keep under 5 pages.

## Security Notes
- Use HTTPS & secure cookie/headers in production.
- Rotate JWT secret for revocation hygiene; implement refresh tokens only if session longevity needed.
- Limit exposure of admin endpoints; deploy behind reverse proxy with rate limiting.

## Plagiarism & Privacy
Repository should remain PRIVATE until grading completes. No third-party code beyond framework/library defaults.

## AI / LLM Usage Statement (To Customize)
Development assistance included code review suggestions, refactoring automation, and documentation drafting. All business logic authored and understood by the developer.

## Next Tasks
See remaining TODO in this repository for: accessibility polish, user metrics chart, ER diagram refinement, security doc, Postman improvements.

---
For deeper backend specifics, see `backend/README.md`.
