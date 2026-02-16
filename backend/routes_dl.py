"""
VeloPark – Deep Learning API Routes
====================================
Endpoints:
  GET   /api/dl/status              – DL model status
  POST  /api/dl/train               – Train all DL models  (admin)
  GET   /api/dl/forecast            – 24h demand forecast
  GET   /api/dl/anomalies           – Detect anomalies
  GET   /api/dl/pricing             – Dynamic pricing suggestions
"""

from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required
from .auth import role_required
from .models import Role

dl_bp = Blueprint("dl", __name__, url_prefix="/api/dl")


@dl_bp.get("/status")
@jwt_required()
def dl_status():
    """Return training status / metrics for all DL models."""
    from .dl_engine import dl_store
    return jsonify(dl_store.status_dict())


@dl_bp.post("/train")
@role_required(Role.ADMIN.value)
def dl_train():
    """Train all deep learning models (admin only)."""
    from .dl_engine import train_all_dl_models
    result = train_all_dl_models()
    return jsonify(result), 200


@dl_bp.get("/forecast")
@jwt_required()
def dl_forecast():
    """Get 24-hour demand forecast per lot."""
    from .dl_engine import get_demand_forecast
    forecasts = get_demand_forecast()
    if not forecasts:
        return jsonify({
            "available": False,
            "message": "Demand Forecaster not trained yet.",
            "forecasts": [],
        })
    return jsonify({
        "available": True,
        "forecasts": forecasts,
    })


@dl_bp.get("/anomalies")
@role_required(Role.ADMIN.value)
def dl_anomalies():
    """Detect anomalous reservations."""
    from .dl_engine import detect_anomalies
    anomalies = detect_anomalies()
    return jsonify({
        "available": True,
        "count": len(anomalies),
        "anomalies": anomalies,
    })


@dl_bp.get("/pricing")
@role_required(Role.ADMIN.value)
def dl_pricing():
    """Get dynamic pricing suggestions."""
    from .dl_engine import get_pricing_suggestions
    suggestions = get_pricing_suggestions()
    if not suggestions:
        return jsonify({
            "available": False,
            "message": "Pricing Engine not trained yet.",
            "suggestions": [],
        })
    return jsonify({
        "available": True,
        "suggestions": suggestions,
    })
