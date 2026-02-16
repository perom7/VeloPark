import os
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


def test_metrics_user_and_admin(client):
    # register user
    r = client.post('/api/auth/register', json={'username': 'metu', 'password': 'pass123'})
    assert r.status_code == 201
    user_token = get_token(client, 'metu', 'pass123')
    admin_token = get_token(client, 'admin', 'admin123')
    # user metrics
    r = client.get('/api/metrics/user', headers={'Authorization': f'Bearer {user_token}'})
    assert r.status_code == 200
    data = r.get_json(); assert 'total_reservations' in data
    # admin metrics
    r = client.get('/api/metrics/admin', headers={'Authorization': f'Bearer {admin_token}'})
    assert r.status_code == 200
    adm = r.get_json(); assert 'totals' in adm and 'per_lot' in adm
