from flask import Blueprint, request, jsonify, send_file, current_app
import os
from sqlalchemy import func
from .database import db
from .models import ParkingLot, ParkingSpot, SpotStatus, User, Role, Reservation
from .auth import role_required
from .validators import validate_lot_payload, validate_lot_update_payload
from .errors import make_error
from .cache import cache
from .tasks import notify_new_lot_created


admin_bp = Blueprint("admin", __name__, url_prefix="/api/admin")


def _create_spots(lot: ParkingLot, to_count: int):
    # Create spots up to to_count (1-based index)
    existing = {s.index_number for s in lot.spots}
    new = []
    for idx in range(1, to_count + 1):
        if idx not in existing:
            new.append(ParkingSpot(lot=lot, index_number=idx, status=SpotStatus.AVAILABLE.value))
    db.session.add_all(new)


@admin_bp.get("/users")
@role_required(Role.ADMIN.value)
def list_users():
    users = User.query.order_by(User.created_at.desc()).all()
    return jsonify([
        {"id": u.id, "username": u.username, "email": u.email, "role": u.role, "created_at": u.created_at.isoformat()} for u in users
    ])


@admin_bp.get("/lots")
@role_required(Role.ADMIN.value)
@cache.cached(timeout=60, key_prefix="admin_lots")
def list_lots():
    lots = ParkingLot.query.order_by(ParkingLot.created_at.desc()).all()
    payload = []
    for lot in lots:
        total = len(lot.spots)
        occ = sum(1 for s in lot.spots if s.status == SpotStatus.OCCUPIED.value)
        payload.append({
            "id": lot.id,
            "prime_location_name": lot.prime_location_name,
            "price_per_hour": lot.price_per_hour,
            "address": lot.address,
            "pincode": lot.pincode,
            "number_of_spots": lot.number_of_spots,
            "occupied": occ,
            "available": total - occ,
        })
    return jsonify(payload)


@admin_bp.post("/lots")
@role_required(Role.ADMIN.value)
def create_lot():
    data = request.get_json() or {}
    cleaned, errors = validate_lot_payload(data)
    if errors:
        return make_error("validation failed", 400, errors)
    name = cleaned["prime_location_name"]
    price = cleaned["price_per_hour"]
    address = cleaned.get("address")
    pincode = cleaned.get("pincode")
    count = cleaned["number_of_spots"]
    lot = ParkingLot(
        prime_location_name=name,
        price_per_hour=price,
        address=address,
        pincode=pincode,
        number_of_spots=count,
    )
    db.session.add(lot)
    db.session.flush()  # get id
    _create_spots(lot, count)
    db.session.commit()
    cache.delete("admin_lots")
    # Fire notification to users about the new lot
    try:
        notify_new_lot_created(lot.id)
    except Exception:
        pass
    return jsonify({"id": lot.id}), 201


@admin_bp.get("/lots/<int:lot_id>")
@role_required(Role.ADMIN.value)
def get_lot(lot_id: int):
    lot = ParkingLot.query.get_or_404(lot_id)
    spots = []
    for s in sorted(lot.spots, key=lambda x: x.index_number):
        current_res = None
        if s.status == SpotStatus.OCCUPIED.value:
            # Find active reservation for this spot
            r = next((r for r in s.reservations if r.end_time is None), None)
            if r:
                current_res = {
                    "reservation_id": r.id,
                    "user_id": r.user_id,
                    "username": r.user.username if r.user else None,
                    "start_time": r.start_time.isoformat(),
                    "vehicle_number": r.vehicle_number,
                    "vehicle_model": r.vehicle_model,
                }
        spots.append({
            "id": s.id,
            "index_number": s.index_number,
            "status": s.status,
            "current_reservation": current_res,
        })
    return jsonify({
        "id": lot.id,
        "prime_location_name": lot.prime_location_name,
        "price_per_hour": lot.price_per_hour,
        "address": lot.address,
        "pincode": lot.pincode,
        "number_of_spots": lot.number_of_spots,
        "spots": spots,
    })


@admin_bp.put("/lots/<int:lot_id>")
@role_required(Role.ADMIN.value)
def update_lot(lot_id: int):
    lot = ParkingLot.query.get_or_404(lot_id)
    data = request.get_json() or {}
    current = {
        "prime_location_name": lot.prime_location_name,
        "price_per_hour": lot.price_per_hour,
        "address": lot.address,
        "pincode": lot.pincode,
        "number_of_spots": lot.number_of_spots,
    }
    cleaned, errors = validate_lot_update_payload(current, data)
    if errors:
        return make_error("validation failed", 400, errors)
    if "prime_location_name" in data:
        lot.prime_location_name = (data["prime_location_name"] or lot.prime_location_name).strip()
    if "price_per_hour" in data:
        price = cleaned["price_per_hour"]
        lot.price_per_hour = price
    if "address" in data:
        lot.address = data["address"]
    if "pincode" in data:
        lot.pincode = data["pincode"]

    if "number_of_spots" in data:
        new_count = int(data["number_of_spots"]) or 0
        if new_count <= 0:
            return make_error("number_of_spots must be > 0", 400)

        current = lot.number_of_spots
        if new_count > current:
            _create_spots(lot, new_count)
            lot.number_of_spots = new_count
        elif new_count < current:
            # remove highest index available spots only
            diff = current - new_count
            removable = [
                s for s in sorted(lot.spots, key=lambda x: x.index_number, reverse=True)
                if s.status == SpotStatus.AVAILABLE.value
            ]
            if len(removable) < diff:
                return make_error("not enough free spots to reduce", 400)
            for s in removable[:diff]:
                db.session.delete(s)
            lot.number_of_spots = new_count

    db.session.commit()
    cache.delete("admin_lots")
    return jsonify({"message": "updated"})


@admin_bp.post('/reports/generate')
@role_required(Role.ADMIN.value)
def generate_report():
    """Generate an admin report PDF synchronously."""
    try:
        from .tasks import generate_admin_report_pdf
        fpath = generate_admin_report_pdf()
        if fpath and os.path.exists(fpath):
            fname = os.path.basename(fpath)
            return jsonify({"message": "report generated", "filename": fname}), 201
        return make_error("PDF generation failed – reportlab may not be installed", 500)
    except Exception as e:
        current_app.logger.error(f"Failed to generate report: {e}")
        return make_error("failed", 500, {"error": str(e)})


@admin_bp.delete("/lots/<int:lot_id>")
@role_required(Role.ADMIN.value)
def delete_lot(lot_id: int):
    lot = ParkingLot.query.get_or_404(lot_id)
    
    # Check if any spots have active reservations (where end_time is NULL)
    has_active = db.session.query(Reservation).join(ParkingSpot).filter(
        ParkingSpot.lot_id == lot_id,
        Reservation.end_time == None
    ).first() is not None
    
    if has_active:
        return make_error("cannot delete lot with active reservations", 400)
    
    # Delete the lot - cascade will handle spots, but we need to manually delete reservations
    # Get all spot IDs for this lot
    spot_ids = [spot.id for spot in lot.spots]
    
    # Delete all reservations for these spots
    if spot_ids:
        Reservation.query.filter(Reservation.spot_id.in_(spot_ids)).delete(synchronize_session=False)
    
    # Now delete the lot (spots will cascade)
    db.session.delete(lot)
    db.session.commit()
    cache.delete("admin_lots")
    return jsonify({"message": "deleted"})


@admin_bp.get("/reports")
@role_required(Role.ADMIN.value)
def list_reports():
    """List generated PDF reports (filenames)."""
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    os.makedirs(reports_dir, exist_ok=True)
    files = [f for f in os.listdir(reports_dir) if f.endswith('.pdf')]
    return jsonify({"files": sorted(files, reverse=True)})


@admin_bp.get("/reports/download")
@role_required(Role.ADMIN.value)
def download_report():
    """Download a specific report by filename."""
    filename = request.args.get('filename')
    if not filename:
        return make_error("filename query param required", 400)
    reports_dir = os.path.join(os.path.dirname(__file__), "reports")
    path = os.path.join(reports_dir, filename)
    if not os.path.exists(path):
        return make_error("file not found", 404)
    return send_file(path, as_attachment=True)
