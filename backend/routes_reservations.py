from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from .auth import require_auth
from sqlalchemy import asc
from .database import db
from .models import ParkingLot, ParkingSpot, Reservation, SpotStatus, User
from .cache import cache
from .validators import validate_reservation_payload, validate_vehicle_fields
from .errors import make_error
from .billing import compute_parking_cost


res_bp = Blueprint("res", __name__, url_prefix="/api")


@res_bp.get("/lots")
@jwt_required(optional=True)
@cache.cached(timeout=60, key_prefix="public_lots")
def list_public_lots():
    lots = ParkingLot.query.all()
    payload = []
    for lot in lots:
        total = len(lot.spots)
        occ = sum(1 for s in lot.spots if s.status == SpotStatus.OCCUPIED.value)
        payload.append({
            "id": lot.id,
            "prime_location_name": lot.prime_location_name,
            "price_per_hour": lot.price_per_hour,
            "available": total - occ,
            "total": total,
        })
    return jsonify(payload)


@res_bp.get("/lots/<int:lot_id>/spots")
@jwt_required()
@cache.cached(timeout=30, query_string=True)
def list_spots(lot_id: int):
    lot = ParkingLot.query.get_or_404(lot_id)
    return jsonify([
        {"id": s.id, "index_number": s.index_number, "status": s.status}
        for s in sorted(lot.spots, key=lambda x: x.index_number)
    ])


@res_bp.post("/reservations")
@jwt_required()
def reserve_spot():
    ident = int(get_jwt_identity())
    user = User.query.get_or_404(ident)
    data = request.get_json() or {}
    cleaned, errors = validate_reservation_payload(data)
    if errors:
        return make_error("validation failed", 400, errors)
    lot_id = cleaned["lot_id"]
    vehicle_number = cleaned.get("vehicle_number")
    vehicle_model = cleaned.get("vehicle_model")
    lot = ParkingLot.query.get_or_404(lot_id)

    # first available spot by index
    spot = (
        ParkingSpot.query.filter_by(lot_id=lot.id, status=SpotStatus.AVAILABLE.value)
        .order_by(asc(ParkingSpot.index_number))
        .first()
    )
    if not spot:
        return make_error("no available spots in selected lot", 409)

    spot.status = SpotStatus.OCCUPIED.value
    res = Reservation(
        spot=spot,
        user=user,
        start_time=datetime.now(),
        vehicle_number=vehicle_number,
        vehicle_model=vehicle_model,
    )
    db.session.add(res)
    db.session.commit()
    cache.delete("public_lots")
    return jsonify({
        "reservation_id": res.id,
        "spot_id": spot.id,
        "lot_id": lot.id,
        "index_number": spot.index_number,
        "start_time": res.start_time.isoformat(),
        "vehicle_number": res.vehicle_number,
        "vehicle_model": res.vehicle_model,
    }), 201


@res_bp.post("/reservations/<int:res_id>/release")
@jwt_required()
def release_spot(res_id: int):
    ident = int(get_jwt_identity())
    res = Reservation.query.get_or_404(res_id)
    if res.user_id != ident:
        return make_error("forbidden", 403)
    if res.end_time:
        return make_error("already released", 400)
    res.end_time = datetime.now()
    # compute cost using billing policy from config
    policy = current_app.config.get('BILLING_POLICY', 'proportional')
    minutes_unit = int(current_app.config.get('BILLING_MINUTES', 15))
    price = res.spot.lot.price_per_hour
    res.parking_cost = compute_parking_cost(res.start_time, res.end_time, price, policy=policy, minutes_unit=minutes_unit)
    res.spot.status = SpotStatus.AVAILABLE.value
    db.session.commit()
    cache.delete("public_lots")
    return jsonify({
        "reservation_id": res.id,
        "spot_id": res.spot_id,
        "lot_id": res.spot.lot_id,
        "start_time": res.start_time.isoformat(),
        "end_time": res.end_time.isoformat(),
        "parking_cost": res.parking_cost,
        "vehicle_number": res.vehicle_number,
        "vehicle_model": res.vehicle_model,
    })


@res_bp.get("/reservations/me")
@require_auth
def my_reservations():
    ident = int(get_jwt_identity())
    items = (
        Reservation.query.filter_by(user_id=ident)
        .order_by(Reservation.start_time.desc())
        .all()
    )
    return jsonify([
        {
            "id": r.id,
            "spot_id": r.spot_id,
            "lot_id": r.spot.lot_id,
            "index_number": r.spot.index_number,
            "start_time": r.start_time.isoformat(),
            "end_time": r.end_time.isoformat() if r.end_time else None,
            "parking_cost": r.parking_cost,
            "vehicle_number": r.vehicle_number,
            "vehicle_model": r.vehicle_model,
            "price_per_hour": r.spot.lot.price_per_hour,
        }
        for r in items
    ])
