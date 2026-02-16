"""
VeloPark – ML API Routes
=========================
Endpoints for AI/ML features:
  GET   /api/ml/status              – model training status
  POST  /api/ml/train               – retrain all models  (admin)
  GET   /api/ml/predict-duration    – predict parking duration for a user+lot
  GET   /api/ml/recommend           – get smart lot recommendations
"""

from flask import Blueprint, jsonify, request
from flask_jwt_extended import jwt_required, get_jwt_identity
from .auth import role_required
from .models import Role
from .errors import make_error

ml_bp = Blueprint("ml", __name__, url_prefix="/api/ml")


@ml_bp.get("/status")
@jwt_required()
def ml_status():
    """Return training status / metrics for all ML models."""
    from .ml_engine import store
    return jsonify(store.status_dict())


@ml_bp.post("/train")
@role_required(Role.ADMIN.value)
def ml_train():
    """Retrain all ML models (admin only)."""
    from .ml_engine import train_all_models
    result = train_all_models()
    return jsonify(result), 200


@ml_bp.get("/predict-duration")
@jwt_required()
def ml_predict_duration():
    """Predict parking duration for current user at a given lot.

    Query params:
        lot_id (int, required)
    """
    lot_id = request.args.get("lot_id", type=int)
    if not lot_id:
        return make_error("lot_id query param required", 400)

    user_id = int(get_jwt_identity())

    from .ml_engine import predict_duration
    result = predict_duration(user_id, lot_id)
    if result is None:
        return jsonify({
            "available": False,
            "message": "Model not trained yet. Admin needs to train the ML models first.",
        })

    return jsonify({
        "available": True,
        **result,
    })


@ml_bp.get("/recommend")
@jwt_required()
def ml_recommend():
    """Get smart lot recommendations for the current user.

    Query params:
        top_n (int, optional, default=3)
    """
    user_id = int(get_jwt_identity())
    top_n = request.args.get("top_n", default=3, type=int)

    from .ml_engine import recommend_lots
    recommendations = recommend_lots(user_id, top_n)

    if not recommendations:
        return jsonify({
            "available": False,
            "message": "Recommendations not available. Model may not be trained yet.",
            "recommendations": [],
        })

    return jsonify({
        "available": True,
        "recommendations": recommendations,
    })


@ml_bp.get("/pricing")
@role_required(Role.ADMIN.value)
def ml_pricing():
    """Get dynamic pricing suggestions for all lots.

    Returns suggested price multipliers for the next 6 hours per lot.
    """
    from .ml_engine import get_pricing_suggestions
    result = get_pricing_suggestions()
    return jsonify(result)
