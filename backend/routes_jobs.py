import csv
from io import StringIO
from flask import Blueprint, jsonify, make_response
from flask_jwt_extended import jwt_required, get_jwt_identity
from .errors import make_error
from .models import Reservation, ParkingSpot
from .database import db

jobs_bp = Blueprint("jobs", __name__, url_prefix="/api/export")


def _generate_csv_for_user(user_id: int) -> str:
    """Generate CSV content for a user's reservations (synchronous)."""
    rows = Reservation.query.filter_by(user_id=user_id).all()
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["reservation_id", "spot_id", "lot_id", "lot_name", "index_number",
                      "start_time", "end_time", "cost"])
    for r in rows:
        writer.writerow([
            r.id,
            r.spot_id,
            r.spot.lot_id,
            r.spot.lot.prime_location_name,
            r.spot.index_number,
            r.start_time.isoformat(),
            r.end_time.isoformat() if r.end_time else "",
            r.parking_cost or 0,
        ])
    return buf.getvalue()


@jobs_bp.get("/csv/download")
@jwt_required()
def direct_csv_download():
    """Direct synchronous CSV download."""
    user_id = int(get_jwt_identity())
    csv_content = _generate_csv_for_user(user_id)
    response = make_response(csv_content)
    response.headers["Content-Disposition"] = f"attachment; filename=reservations-export-{user_id}.csv"
    response.headers["Content-Type"] = "text/csv"
    return response
