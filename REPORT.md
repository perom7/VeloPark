# Vehicle Parking System V2 – Project Report

Author: Param Desai  
Email: 23f3001882@ds.study.iitm.ac.in  
Date: 2025-11-29  
Repository: https://github.com/23f3001882/vehicle-parking-system-v2

---
## 1. Problem Statement
Design and implement a multi-user parking management web application supporting:
- Admin management of parking lots & dynamic spot allocation.
- User reservation lifecycle (book / release) with billing.
- Real-time occupancy & user metrics.
- Asynchronous background jobs (reminders, monthly reports, CSV export, notifications).
- Caching & performance optimizations.
- JWT authentication with token revocation.
- Progressive Web App (PWA) capability.

## 2. High-Level Architecture
```
[Browser (Vue + Bootstrap + Chart.js + Service Worker)]
        | REST + JSON
[Flask API Layer]
        | SQLAlchemy ORM
[SQLite DB] <-> [Redis Cache]
        | Celery (Worker + Beat)
[Async Tasks: reminders, reports (PDF), CSV export, notifications]
```
- Flask provides REST endpoints under /api.
- SQLite used for simplicity; easily swappable with Postgres.
- Redis delivers caching + Celery broker + token revocation storage.
- Celery beat schedules periodic tasks; worker executes them.

## 3. Data Model Overview
Entities: User, ParkingLot, ParkingSpot, Reservation.  
Relationships:  
- ParkingLot 1..N ParkingSpot  
- ParkingSpot 1..N Reservation  
- User 1..N Reservation  

Key Rules:
- `Reservation.end_time = NULL` => active.
- Cost computed on release via policy (proportional by default).  
- Lot resizing enforces only AVAILABLE spots may be removed.

## 4. Core Features Implemented
| Area | Summary |
|------|---------|
| Auth | JWT login/register, /auth/me, logout revocation (Redis fallback to memory). |
| Admin | Create/update/delete lots; view occupancy; manual trigger reminders/reports. |
| User | Book first available spot, release with cost calculation, view history & metrics. |
| Metrics | User spend & favorite lot; Admin aggregated occupancy & counts. |
| Async | Daily inactivity reminders, monthly per-user PDF report, CSV export, new lot broadcast. |
| Caching | Redis-backed endpoints using Flask-Caching; automatic invalidation on lot changes. |
| PWA | Manifest + service worker for install & offline shell. |
| Validation | Server-side schema checks (lots, reservations, vehicles) + consistent errors. |
| UX | Animated charts, auto-refresh dashboards, gradient theme, accessibility improvements. |

## 5. Asynchronous Task Design
Tasks declared in `backend/tasks.py` with Flask app context wrapping:
- `send_daily_reminders`: Cron (default 18:00 UTC) – notifies inactive users.
- `send_monthly_reports`: Cron (1st day 09:00 UTC) – generates HTML + optional PDF using `xhtml2pdf` (saved & emailed).
- `export_csv(user_id)`: User-triggered; produces reservation history file under `backend/exports/`.
- `notify_new_lot_created(lot_id)`: Broadcast to non-admin users when new lot is added.
Fallback: If Redis broker unreachable at startup, Celery runs in eager (synchronous) mode to preserve functionality.

## 6. Security Approach
- JWT access tokens (12h expiry). Logout pushes token JTI into Redis blocklist.
- Role-based route protection via decorator (`@role_required`).
- Validation layer prevents malformed or unsafe input (spot count changes, negative price, etc.).
- Planned improvement: Refresh token flow + HTTPS + rate limiting + secret rotation guidance.

## 7. Testing Summary
Pytest suites cover:
- Core flows: registration, login, reservation creation/release.
- Validation failures (lot resizing, invalid payload).
- Metrics endpoints (user/admin).
- Async tasks (CSV export, daily reminders, monthly report scheduling) executed eagerly in test mode.
All tests pass under Windows (solo Celery pool). Edge cases (eager fallback, empty reservations) validated.

## 8. Performance & Caching
- Caching of admin lots, public lot listings reduces DB round-trips.
- SimpleCache fallback if Redis unavailable (transparent).  
- Auto-refresh (frontend timers) provides perceived real-time feel without websockets.

## 9. API Surface (Abbreviated)
Auth: `POST /api/auth/register`, `POST /api/auth/login`, `GET /api/auth/me`, `POST /api/auth/logout`  
Lots (Admin): `GET /api/admin/lots`, `POST /api/admin/lots`, `PUT /api/admin/lots/<id>`, `DELETE /api/admin/lots/<id>`  
Reports: `POST /api/admin/reports/monthly/generate`, `GET /api/admin/reports/monthly`, `GET /api/admin/reports/monthly/download?filename=`  
Reminders: `POST /api/admin/reminders/daily/run`  
Reservations: `POST /api/reservations`, `POST /api/reservations/<id>/release`, `GET /api/reservations/me`  
Metrics: `GET /api/metrics/user`, `GET /api/metrics/admin`  
CSV Export: `POST /api/export/csv`, `GET /api/export/csv/<task_id>/status`, `GET /api/export/csv/<task_id>/download`

## 10. AI / LLM Assistance Statement
AI tools assisted with: error handling standardization, fallback strategies (eager Celery), documentation scaffolding, and UI modernization. All business logic, data modeling, and architectural decisions were understood and verified manually.

## 11. Known Limitations / Future Work
- No rate limiting or refresh tokens yet.
- SQLite storage; migrate to Postgres for production concurrency.
- Monthly report styling could adopt a shared template engine.
- Expand metrics (time-series occupancy, user spend trends).
- Security doc & REPORT PDF automation.

## 12. How to Generate This PDF (Three Methods)
1. **GitHub Download**: Use a Markdown-to-PDF extension locally.
2. **Pandoc (Recommended)**:
   ```powershell
   pandoc REPORT.md -o REPORT.pdf
   ```
3. **Python Script (provided)**: `python scripts/generate_report.py` (uses xhtml2pdf).

---
*End of Report*
