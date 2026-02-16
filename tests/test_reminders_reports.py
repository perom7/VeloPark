import os
import pytest
from backend.app import create_app
from backend.database import db
from backend.seed import seed_admin
import glob

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


def test_daily_reminders_manual_trigger(client):
    admin_token = get_token(client, 'admin', 'admin123')
    r = client.post('/api/admin/reminders/daily/run', headers={'Authorization': f'Bearer {admin_token}'})
    assert r.status_code == 202
    assert r.get_json()['message'].startswith('daily reminders')


def test_monthly_reports_generation(client):
    admin_token = get_token(client, 'admin', 'admin123')
    # create a user and a reservation so report has data
    r = client.post('/api/auth/register', json={'username': 'repuser', 'password': 'pass123'})
    assert r.status_code == 201
    user_token = get_token(client, 'repuser', 'pass123')
    # create lot and reservation
    r = client.post('/api/admin/lots', json={'prime_location_name': 'ReportLot', 'price_per_hour': 5, 'number_of_spots': 1}, headers={'Authorization': f'Bearer {admin_token}'})
    lot_id = r.get_json()['id']
    r = client.post('/api/reservations', json={'lot_id': lot_id}, headers={'Authorization': f'Bearer {user_token}'})
    res_id = r.get_json()['reservation_id']
    client.post(f'/api/reservations/{res_id}/release', headers={'Authorization': f'Bearer {user_token}'})
    # trigger monthly reports (eager runs immediately)
    r = client.post('/api/admin/reports/monthly/generate', headers={'Authorization': f'Bearer {admin_token}'})
    assert r.status_code == 202
    # list reports
    r = client.get('/api/admin/reports/monthly', headers={'Authorization': f'Bearer {admin_token}'})
    data = r.get_json(); files = data['files']
    # there should be at least one PDF generated for repuser
    assert any('repuser' in f and f.endswith('.pdf') for f in files)
