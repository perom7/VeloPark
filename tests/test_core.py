import os
import json
import pytest
from backend.app import create_app
from backend.database import db
from backend.seed import seed_admin
from backend.models import User, Role


@pytest.fixture
def app():
    # Ensure test-friendly settings before app is created
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["CACHE_TYPE"] = "SimpleCache"
    os.environ["REDIS_URL"] = "memory://"
    app = create_app()
    app.config.update({
        "TESTING": True,
        "JWT_ACCESS_TOKEN_EXPIRES": False,
        "ENV": "development",
        "CACHE_DEFAULT_TIMEOUT": 60,
    })
    with app.app_context():
        db.drop_all()
        db.create_all()
        seed_admin()
        # Run Celery tasks eagerly during tests so .delay() executes inline
        from backend.tasks import celery
        celery.conf.task_always_eager = True
        celery.conf.task_eager_propagates = True
        yield app


@pytest.fixture
def client(app):
    return app.test_client()


def get_token(client, username, password):
    res = client.post('/api/auth/login', json={"username": username, "password": password})
    assert res.status_code == 200
    data = res.get_json()
    return data['access_token']


def test_register_login_and_reserve_release(client):
    # Register a user
    r = client.post('/api/auth/register', json={"username": "u1", "password": "pass123", "email": "u1@example.com"})
    assert r.status_code == 201

    # Login as user
    user_token = get_token(client, 'u1', 'pass123')

    # Login as admin (seed_admin creates admin/admin123)
    admin_token = get_token(client, 'admin', 'admin123')

    # Admin: create a lot
    lot_payload = {"prime_location_name": "Test Lot", "price_per_hour": 10, "number_of_spots": 2}
    r = client.post('/api/admin/lots', json=lot_payload, headers={"Authorization": f"Bearer {admin_token}"})
    assert r.status_code == 201
    lot_id = r.get_json()['id']

    # User: list lots (should be accessible)
    r = client.get('/api/lots')
    assert r.status_code == 200

    # User: reserve a spot
    r = client.post('/api/reservations', json={"lot_id": lot_id}, headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code == 201
    res_id = r.get_json()['reservation_id']

    # Release
    r = client.post(f'/api/reservations/{res_id}/release', headers={"Authorization": f"Bearer {user_token}"})
    assert r.status_code == 200
    data = r.get_json()
    assert 'parking_cost' in data


def test_logout_revokes_token(client):
    # register and login
    r = client.post('/api/auth/register', json={"username": "u2", "password": "pass123", "email": "u2@example.com"})
    assert r.status_code == 201
    token = get_token(client, 'u2', 'pass123')
    # hit a protected endpoint should work
    r = client.get('/api/reservations/me', headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (200, 404)
    # logout
    r = client.post('/api/auth/logout', headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    # subsequent call should be rejected (401 or 422 depending on loader)
    r = client.get('/api/reservations/me', headers={"Authorization": f"Bearer {token}"})
    assert r.status_code in (401, 422)


def test_user_cannot_create_lot(client):
    # register + login user
    r = client.post('/api/auth/register', json={"username": "u3", "password": "pass123", "email": "u3@example.com"})
    assert r.status_code == 201
    token = get_token(client, 'u3', 'pass123')
    # try to create lot
    lot_payload = {"prime_location_name": "Bad Lot", "price_per_hour": 10, "number_of_spots": 2}
    r = client.post('/api/admin/lots', json=lot_payload, headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 403
