import os, time
import pytest
from backend.app import create_app
from backend.database import db
from backend.seed import seed_admin

@pytest.fixture
def app():
    os.environ['DATABASE_URL'] = 'sqlite:///:memory:'
    os.environ['CACHE_TYPE'] = 'SimpleCache'
    os.environ['REDIS_URL'] = 'memory://'
    app = create_app(); app.config.update({'TESTING': True})
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


def test_csv_export_eager(client):
    # register user and create a reservation then release to have cost info
    r = client.post('/api/auth/register', json={'username': 'csvu', 'password': 'pass123'})
    assert r.status_code == 201
    token = get_token(client, 'csvu', 'pass123')
    # admin create lot
    admin_token = get_token(client, 'admin', 'admin123')
    r = client.post('/api/admin/lots', json={'prime_location_name': 'Lot1', 'price_per_hour': 10, 'number_of_spots': 1}, headers={'Authorization': f'Bearer {admin_token}'})
    lot_id = r.get_json()['id']
    r = client.post('/api/reservations', json={'lot_id': lot_id}, headers={'Authorization': f'Bearer {token}'})
    res_id = r.get_json()['reservation_id']
    client.post(f'/api/reservations/{res_id}/release', headers={'Authorization': f'Bearer {token}'})
    # trigger export (eager mode executes task inline)
    r = client.post('/api/export/csv', headers={'Authorization': f'Bearer {token}'})
    assert r.status_code == 202
    task_id = r.get_json()['task_id']
    # status should be SUCCESS immediately
    r = client.get(f'/api/export/csv/{task_id}/status', headers={'Authorization': f'Bearer {token}'})
    assert r.status_code == 200
    data = r.get_json(); assert data['state'] == 'SUCCESS'
    filename = data['result']['filename']
    # download
    r = client.get(f'/api/export/csv/{task_id}/download', headers={'Authorization': f'Bearer {token}'})
    assert r.status_code == 200
    assert r.headers.get('Content-Disposition', '').startswith('attachment')
