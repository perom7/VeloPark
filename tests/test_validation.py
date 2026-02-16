import os
from backend.app import create_app
from backend.database import db
from backend.seed import seed_admin
import pytest

@pytest.fixture
def app():
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    os.environ['CACHE_TYPE'] = 'SimpleCache'
    os.environ['REDIS_URL'] = 'memory://'
    app = create_app()
    app.config.update({'TESTING': True})
    with app.app_context():
        db.drop_all(); db.create_all(); seed_admin()
        from backend.tasks import celery
        celery.conf.task_always_eager = True
        celery.conf.task_eager_propagates = True
        yield app

@pytest.fixture
def client(app):
    return app.test_client()


def get_token(client, username, password):
    r = client.post('/api/auth/login', json={'username': username, 'password': password})
    assert r.status_code == 200
    return r.get_json()['access_token']


def test_create_lot_validation_errors(client):
    # login admin
    token = get_token(client, 'admin', 'admin123')
    # missing name
    r = client.post('/api/admin/lots', json={'price_per_hour': 10, 'number_of_spots': 2}, headers={'Authorization': f'Bearer {token}'})
    assert r.status_code == 400
    data = r.get_json(); assert data['message'] == 'validation failed'
    assert 'prime_location_name' in data.get('errors', {})
    # negative price
    r = client.post('/api/admin/lots', json={'prime_location_name': 'X', 'price_per_hour': -1, 'number_of_spots': 2}, headers={'Authorization': f'Bearer {token}'})
    assert r.status_code == 400
    assert 'price_per_hour' in r.get_json()['errors']


def test_reservation_validation_missing_lot(client):
    # register user
    r = client.post('/api/auth/register', json={'username': 'uval', 'password': 'pass123'})
    assert r.status_code == 201
    token = get_token(client, 'uval', 'pass123')
    # missing lot_id
    r = client.post('/api/reservations', json={}, headers={'Authorization': f'Bearer {token}'})
    assert r.status_code == 400
    data = r.get_json(); assert 'lot_id' in data.get('errors', {})
