from .database import db
from .models import User, Role


def seed_admin(username: str = "admin", password: str = "param2412", email: str | None = None):
    exists = User.query.filter_by(role=Role.ADMIN.value).first()
    if exists:
        return exists
    admin = User(username=username, email=email, role=Role.ADMIN.value)
    admin.set_password(password)
    db.session.add(admin)
    db.session.commit()
    return admin
