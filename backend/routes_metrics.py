from datetime import datetime
from collections import Counter
from flask import Blueprint, jsonify
from .errors import make_error
from flask_jwt_extended import jwt_required, get_jwt_identity
from sqlalchemy import func
from .database import db
from .models import Reservation, ParkingLot, ParkingSpot, SpotStatus, User, Role
from .auth import role_required


metrics_bp = Blueprint("metrics", __name__, url_prefix="/api/metrics")


@metrics_bp.get("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.utcnow().isoformat()})


@metrics_bp.get("/admin")
@role_required(Role.ADMIN.value)
def admin_metrics():
    total_users = db.session.scalar(db.select(func.count(User.id))) or 0
    total_lots = db.session.scalar(db.select(func.count(ParkingLot.id))) or 0
    total_spots = db.session.scalar(db.select(func.count(ParkingSpot.id))) or 0
    occupied = db.session.scalar(db.select(func.count(ParkingSpot.id)).filter(ParkingSpot.status == SpotStatus.OCCUPIED.value)) or 0

    # per-lot occupancy
    lots = ParkingLot.query.all()
    per_lot = []
    for lot in lots:
        total = len(lot.spots)
        occ = sum(1 for s in lot.spots if s.status == SpotStatus.OCCUPIED.value)
        per_lot.append({
            "lot": lot.prime_location_name,
            "occupied": occ,
            "available": total - occ,
        })

    # Total revenue
    total_revenue = db.session.scalar(
        db.select(func.coalesce(func.sum(Reservation.parking_cost), 0.0))
    ) or 0.0

    return jsonify({
        "totals": {
            "users": total_users,
            "lots": total_lots,
            "spots": total_spots,
            "occupied": occupied,
            "total_revenue": round(float(total_revenue), 2),
        },
        "per_lot": per_lot,
    })


@metrics_bp.get("/admin/revenue")
@role_required(Role.ADMIN.value)
def admin_revenue():
    """Per-user revenue breakdown for the admin pie chart."""
    # Total revenue across all completed reservations
    total_revenue = db.session.scalar(
        db.select(func.coalesce(func.sum(Reservation.parking_cost), 0.0))
    ) or 0.0

    # Total completed reservations
    total_completed = db.session.scalar(
        db.select(func.count(Reservation.id)).filter(Reservation.end_time.isnot(None))
    ) or 0

    # Per-user breakdown
    per_user_rows = (
        db.session.query(
            User.username,
            func.coalesce(func.sum(Reservation.parking_cost), 0.0).label("total_spent")
        )
        .join(Reservation, Reservation.user_id == User.id)
        .filter(Reservation.parking_cost.isnot(None))
        .filter(User.role != Role.ADMIN.value)
        .group_by(User.id, User.username)
        .order_by(func.sum(Reservation.parking_cost).desc())
        .all()
    )

    per_user = [
        {"username": row.username, "total_spent": round(float(row.total_spent), 2)}
        for row in per_user_rows
    ]

    return jsonify({
        "total_revenue": round(float(total_revenue), 2),
        "total_completed_reservations": total_completed,
        "per_user": per_user,
    })


@metrics_bp.get("/admin/most-booked")
@role_required(Role.ADMIN.value)
def admin_most_booked():
    """Return total bookings per lot for the Most Booked Lots chart."""
    # Join Reservation -> ParkingSpot -> ParkingLot to count bookings per lot
    rows = (
        db.session.query(
            ParkingLot.prime_location_name,
            func.count(Reservation.id).label("bookings")
        )
        .join(ParkingSpot, ParkingSpot.lot_id == ParkingLot.id)
        .join(Reservation, Reservation.spot_id == ParkingSpot.id)
        .group_by(ParkingLot.id, ParkingLot.prime_location_name)
        .order_by(func.count(Reservation.id).desc())
        .all()
    )

    lots = [
        {"lot": row.prime_location_name, "bookings": row.bookings}
        for row in rows
    ]

    return jsonify({"lots": lots})


@metrics_bp.get("/user")
@jwt_required()
def user_metrics():
    ident = int(get_jwt_identity())
    # total reservations, active, last month spend
    q = Reservation.query.filter_by(user_id=ident)
    total = q.count()
    active = q.filter(Reservation.end_time.is_(None)).count()
    last_30 = db.session.execute(
        db.select(func.coalesce(func.sum(Reservation.parking_cost), 0.0))
        .filter(Reservation.user_id == ident)
        .filter(Reservation.start_time >= func.datetime("now", "-30 days"))
    ).scalar_one()

    # favorite lot by count
    lots = [r.spot.lot.prime_location_name for r in q.all()]
    fav = None
    if lots:
        fav = Counter(lots).most_common(1)[0][0]

    return jsonify({
        "total_reservations": total,
        "active_reservations": active,
        "amount_spent_30_days": float(last_30 or 0),
        "favorite_lot": fav,
    })
