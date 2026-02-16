from datetime import datetime
from enum import Enum
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import UniqueConstraint
from .database import db


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"


class SpotStatus(str, Enum):
    AVAILABLE = "A"
    OCCUPIED = "O"


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    password_hash = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(20), default=Role.USER.value, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    last_login_at = db.Column(db.DateTime)

    reservations = db.relationship("Reservation", back_populates="user")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class ParkingLot(db.Model):
    __tablename__ = "parking_lots"

    id = db.Column(db.Integer, primary_key=True)
    prime_location_name = db.Column(db.String(120), nullable=False)
    price_per_hour = db.Column(db.Float, nullable=False)
    address = db.Column(db.String(250))
    pincode = db.Column(db.String(12))
    number_of_spots = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    spots = db.relationship("ParkingSpot", back_populates="lot", cascade="all, delete-orphan")


class ParkingSpot(db.Model):
    __tablename__ = "parking_spots"

    id = db.Column(db.Integer, primary_key=True)
    lot_id = db.Column(db.Integer, db.ForeignKey("parking_lots.id", ondelete="CASCADE"), nullable=False)
    index_number = db.Column(db.Integer, nullable=False)
    status = db.Column(db.String(1), nullable=False, default=SpotStatus.AVAILABLE.value)

    lot = db.relationship("ParkingLot", back_populates="spots")
    reservations = db.relationship("Reservation", back_populates="spot")

    __table_args__ = (UniqueConstraint("lot_id", "index_number", name="uq_lot_index"),)


class Reservation(db.Model):
    __tablename__ = "reservations"

    id = db.Column(db.Integer, primary_key=True)
    spot_id = db.Column(db.Integer, db.ForeignKey("parking_spots.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    start_time = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    end_time = db.Column(db.DateTime)
    parking_cost = db.Column(db.Float)
    remarks = db.Column(db.String(250))
    # Optional vehicle details
    vehicle_number = db.Column(db.String(32))  # license plate / registration number
    vehicle_model = db.Column(db.String(100))

    spot = db.relationship("ParkingSpot", back_populates="reservations")
    user = db.relationship("User", back_populates="reservations")
