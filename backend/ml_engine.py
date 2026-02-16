"""
VeloPark – AI/ML Engine
========================
Three ML/DL models:
1.  ParkingDurationPredictor   – Random-Forest regressor that predicts how long
    a user will park (in minutes) based on time-of-day, day-of-week,
    lot price, user history, etc.
2.  SmartLotRecommender        – Gradient-Boosting classifier / scorer that
    ranks available lots for a given user, producing a personal
    recommendation.
3.  DynamicPricingEngine       – Neural-network pricing model that suggests
    optimal price multipliers per lot per hour (0.5x–2.5x) based on
    demand, occupancy, and time features.

Models are trained on historical (completed) reservations and stored in
memory as singleton objects.  An admin endpoint can trigger retraining.
"""

import os
import math
import logging
from datetime import datetime
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)

# ── Lazy sklearn import (module-level so import errors surface early) ──
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not installed – ML features will be unavailable.")


# ══════════════════════════════════════════════════════════════════════
#  Singleton model store
# ══════════════════════════════════════════════════════════════════════
class _ModelStore:
    """Thread-safe-ish in-memory store for trained models & metadata."""
    def __init__(self):
        self.duration_model = None
        self.duration_mae = None           # Mean Absolute Error (minutes)
        self.duration_samples = 0
        self.duration_trained_at = None

        self.recommender_model = None
        self.recommender_accuracy = None
        self.recommender_samples = 0
        self.recommender_trained_at = None
        self.recommender_le = None         # LabelEncoder for lot names

        self.lot_features_cache = {}       # lot_id -> features dict

        # Dynamic Pricing (neural network from dl_engine)
        self.pricing_engine = None         # DynamicPricingEngine instance
        self.pricing_samples = 0
        self.pricing_trained_at = None

    def status_dict(self) -> dict:
        pe = self.pricing_engine
        return {
            "sklearn_available": SKLEARN_AVAILABLE,
            "duration": {
                "trained": self.duration_model is not None,
                "samples": self.duration_samples,
                "mae_minutes": round(self.duration_mae, 1) if self.duration_mae else None,
                "trained_at": self.duration_trained_at.isoformat() if self.duration_trained_at else None,
            },
            "recommender": {
                "trained": self.recommender_model is not None,
                "samples": self.recommender_samples,
                "accuracy": round(self.recommender_accuracy * 100, 1) if self.recommender_accuracy else None,
                "trained_at": self.recommender_trained_at.isoformat() if self.recommender_trained_at else None,
            },
            "pricing": {
                "trained": pe is not None and pe.trained,
                "samples": self.pricing_samples,
                "trained_at": self.pricing_trained_at.isoformat() if self.pricing_trained_at else None,
                "architecture": "Neural Network (10→64→32→1) + Sigmoid",
            },
        }

store = _ModelStore()


# ══════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING HELPERS
# ══════════════════════════════════════════════════════════════════════

def _time_features(dt: datetime) -> dict:
    """Extract cyclic & categorical time features."""
    hour = dt.hour + dt.minute / 60.0
    dow = dt.weekday()               # 0=Mon … 6=Sun
    is_weekend = 1 if dow >= 5 else 0
    # Cyclic encoding of hour and day-of-week
    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)
    dow_sin = math.sin(2 * math.pi * dow / 7)
    dow_cos = math.cos(2 * math.pi * dow / 7)
    return {
        "hour": hour,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
    }


def _user_history_features(user_id: int, reservations_by_user: dict) -> dict:
    """Compute aggregate features from a user's past reservations."""
    past = reservations_by_user.get(user_id, [])
    if not past:
        return {
            "user_avg_duration": 60.0,          # default 1 hr
            "user_booking_count": 0,
            "user_avg_cost": 0.0,
        }
    durations = []
    costs = []
    for r in past:
        if r["duration_min"] is not None:
            durations.append(r["duration_min"])
        if r["cost"] is not None:
            costs.append(r["cost"])
    return {
        "user_avg_duration": np.mean(durations) if durations else 60.0,
        "user_booking_count": len(past),
        "user_avg_cost": np.mean(costs) if costs else 0.0,
    }


# ══════════════════════════════════════════════════════════════════════
#  DATA PREPARATION  (called inside an app context)
# ══════════════════════════════════════════════════════════════════════

def _load_training_data():
    """Pull completed reservations from DB, return list of dicts."""
    from .models import Reservation, ParkingSpot, ParkingLot

    completed = (
        Reservation.query
        .filter(Reservation.end_time.isnot(None))
        .order_by(Reservation.start_time)
        .all()
    )
    lot_cache = {}
    rows = []
    for r in completed:
        spot = r.spot
        lot_id = spot.lot_id
        if lot_id not in lot_cache:
            lot = ParkingLot.query.get(lot_id)
            lot_cache[lot_id] = {
                "lot_id": lot_id,
                "lot_name": lot.prime_location_name,
                "price_per_hour": lot.price_per_hour,
                "total_spots": lot.number_of_spots,
            }
        lot_info = lot_cache[lot_id]
        delta = r.end_time - r.start_time
        duration_min = delta.total_seconds() / 60.0
        rows.append({
            "reservation_id": r.id,
            "user_id": r.user_id,
            "lot_id": lot_id,
            "lot_name": lot_info["lot_name"],
            "price_per_hour": lot_info["price_per_hour"],
            "total_spots": lot_info["total_spots"],
            "start_time": r.start_time,
            "duration_min": duration_min,
            "cost": r.parking_cost,
        })
    return rows, lot_cache


def _group_by_user(rows: list) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["user_id"]].append(r)
    return dict(groups)


# ══════════════════════════════════════════════════════════════════════
#  1) PARKING DURATION PREDICTOR
# ══════════════════════════════════════════════════════════════════════

DURATION_FEATURE_NAMES = [
    "hour", "is_weekend", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "price_per_hour", "total_spots",
    "user_avg_duration", "user_booking_count", "user_avg_cost",
]


def train_duration_model():
    """Train the Random-Forest duration predictor.  Returns status dict."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed"}

    rows, _ = _load_training_data()
    if len(rows) < 3:
        return {"error": f"Need at least 3 completed reservations to train; found {len(rows)}"}

    by_user = _group_by_user(rows)

    X, y = [], []
    for r in rows:
        tf = _time_features(r["start_time"])
        uf = _user_history_features(r["user_id"], by_user)
        feature_vec = [
            tf["hour"], tf["is_weekend"], tf["hour_sin"], tf["hour_cos"],
            tf["dow_sin"], tf["dow_cos"],
            r["price_per_hour"], r["total_spots"],
            uf["user_avg_duration"], uf["user_booking_count"], uf["user_avg_cost"],
        ]
        X.append(feature_vec)
        y.append(r["duration_min"])

    X = np.array(X)
    y = np.array(y)

    # Split
    if len(X) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    store.duration_model = model
    store.duration_mae = mae
    store.duration_samples = len(rows)
    store.duration_trained_at = datetime.utcnow()

    logger.info(f"Duration model trained on {len(rows)} samples – MAE={mae:.1f} min")
    return {
        "message": "Duration model trained",
        "samples": len(rows),
        "mae_minutes": round(mae, 1),
    }


def predict_duration(user_id: int, lot_id: int) -> dict | None:
    """Predict parking duration (minutes) for a user at a given lot."""
    if store.duration_model is None:
        return None

    from .models import ParkingLot, Reservation, ParkingSpot

    lot = ParkingLot.query.get(lot_id)
    if not lot:
        return None

    # Get user history from DB
    completed = (
        Reservation.query
        .filter(Reservation.user_id == user_id, Reservation.end_time.isnot(None))
        .all()
    )
    user_durations = []
    user_costs = []
    for r in completed:
        delta = r.end_time - r.start_time
        user_durations.append(delta.total_seconds() / 60.0)
        if r.parking_cost is not None:
            user_costs.append(r.parking_cost)

    now = datetime.now()
    tf = _time_features(now)
    uf = {
        "user_avg_duration": np.mean(user_durations) if user_durations else 60.0,
        "user_booking_count": len(completed),
        "user_avg_cost": np.mean(user_costs) if user_costs else 0.0,
    }

    feature_vec = np.array([[
        tf["hour"], tf["is_weekend"], tf["hour_sin"], tf["hour_cos"],
        tf["dow_sin"], tf["dow_cos"],
        lot.price_per_hour, lot.number_of_spots,
        uf["user_avg_duration"], uf["user_booking_count"], uf["user_avg_cost"],
    ]])

    predicted_minutes = float(store.duration_model.predict(feature_vec)[0])
    predicted_minutes = max(5.0, predicted_minutes)  # min 5 minutes
    estimated_cost = round((predicted_minutes / 60.0) * lot.price_per_hour, 2)

    return {
        "predicted_duration_minutes": round(predicted_minutes, 0),
        "estimated_cost": estimated_cost,
        "confidence": "high" if store.duration_samples >= 20 else "medium" if store.duration_samples >= 8 else "low",
        "lot_name": lot.prime_location_name,
        "price_per_hour": lot.price_per_hour,
    }


# ══════════════════════════════════════════════════════════════════════
#  2) SMART LOT RECOMMENDER
# ══════════════════════════════════════════════════════════════════════

RECOMMENDER_FEATURE_NAMES = [
    "hour", "is_weekend", "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "price_per_hour", "total_spots",
    "user_lot_visit_ratio",    # fraction of user's bookings at this lot
    "user_avg_duration", "user_booking_count",
]


def train_recommender():
    """Train the lot-recommender on which lot a user actually chose."""
    if not SKLEARN_AVAILABLE:
        return {"error": "scikit-learn not installed"}

    rows, lot_cache = _load_training_data()
    if len(rows) < 3:
        return {"error": f"Need at least 3 completed reservations; found {len(rows)}"}

    by_user = _group_by_user(rows)

    # Label = lot_name (which lot the user chose)
    le = LabelEncoder()
    lot_names = [r["lot_name"] for r in rows]
    y_encoded = le.fit_transform(lot_names)

    X = []
    for r in rows:
        tf = _time_features(r["start_time"])
        uf = _user_history_features(r["user_id"], by_user)

        # How often does this user visit THIS lot?
        user_hist = by_user.get(r["user_id"], [])
        lot_visits = sum(1 for h in user_hist if h["lot_id"] == r["lot_id"])
        visit_ratio = lot_visits / max(len(user_hist), 1)

        feature_vec = [
            tf["hour"], tf["is_weekend"], tf["hour_sin"], tf["hour_cos"],
            tf["dow_sin"], tf["dow_cos"],
            r["price_per_hour"], r["total_spots"],
            visit_ratio,
            uf["user_avg_duration"], uf["user_booking_count"],
        ]
        X.append(feature_vec)

    X = np.array(X)
    y = np.array(y_encoded)

    if len(X) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    else:
        X_train, X_test, y_train, y_test = X, X, y, y

    model = GradientBoostingClassifier(
        n_estimators=80,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, model.predict(X_test))

    store.recommender_model = model
    store.recommender_le = le
    store.recommender_accuracy = accuracy
    store.recommender_samples = len(rows)
    store.recommender_trained_at = datetime.utcnow()
    store.lot_features_cache = lot_cache

    logger.info(f"Recommender trained on {len(rows)} samples – accuracy={accuracy:.2%}")
    return {
        "message": "Recommender model trained",
        "samples": len(rows),
        "accuracy_pct": round(accuracy * 100, 1),
    }


def recommend_lots(user_id: int, top_n: int = 3) -> list:
    """Return ranked list of recommended lots for a user."""
    if store.recommender_model is None or store.recommender_le is None:
        return []

    from .models import ParkingLot, ParkingSpot, SpotStatus, Reservation

    lots = ParkingLot.query.all()
    if not lots:
        return []

    # Get user history
    completed = (
        Reservation.query
        .filter(Reservation.user_id == user_id, Reservation.end_time.isnot(None))
        .all()
    )
    user_durations = []
    user_costs = []
    user_lot_counts = defaultdict(int)
    for r in completed:
        delta = r.end_time - r.start_time
        user_durations.append(delta.total_seconds() / 60.0)
        if r.parking_cost is not None:
            user_costs.append(r.parking_cost)
        user_lot_counts[r.spot.lot_id] += 1

    total_user_bookings = len(completed)
    now = datetime.now()
    tf = _time_features(now)

    uf = {
        "user_avg_duration": np.mean(user_durations) if user_durations else 60.0,
        "user_booking_count": total_user_bookings,
    }

    scored = []
    for lot in lots:
        total = len(lot.spots)
        available = sum(1 for s in lot.spots if s.status == SpotStatus.AVAILABLE.value)
        if available == 0:
            continue  # skip full lots

        visit_ratio = user_lot_counts.get(lot.id, 0) / max(total_user_bookings, 1)

        feature_vec = np.array([[
            tf["hour"], tf["is_weekend"], tf["hour_sin"], tf["hour_cos"],
            tf["dow_sin"], tf["dow_cos"],
            lot.price_per_hour, lot.number_of_spots,
            visit_ratio,
            uf["user_avg_duration"], uf["user_booking_count"],
        ]])

        # Get probability for each lot class
        probas = store.recommender_model.predict_proba(feature_vec)[0]

        # Find the index for this lot's label (if it exists in training data)
        try:
            lot_idx = list(store.recommender_le.classes_).index(lot.prime_location_name)
            ml_score = float(probas[lot_idx])
        except (ValueError, IndexError):
            ml_score = 0.0

        # Blend ML score with heuristic availability bonus
        availability_bonus = available / max(total, 1) * 0.3
        final_score = ml_score * 0.7 + availability_bonus

        scored.append({
            "lot_id": lot.id,
            "lot_name": lot.prime_location_name,
            "score": round(final_score, 3),
            "ml_score": round(ml_score, 3),
            "price_per_hour": lot.price_per_hour,
            "available": available,
            "total": total,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]


# ══════════════════════════════════════════════════════════════════════
#  TRAIN ALL
# ══════════════════════════════════════════════════════════════════════

def train_pricing_model():
    """Train the neural-network dynamic pricing engine."""
    from .dl_engine import DynamicPricingEngine

    rows, lot_cache = _load_training_data()
    if len(rows) < 5:
        return {"error": f"Need at least 5 completed reservations ({len(rows)} found)"}

    # Build lot_cache with occupancy percentages
    from .models import ParkingLot, ParkingSpot, SpotStatus
    enriched_cache = {}
    for lid, info in lot_cache.items():
        lot = ParkingLot.query.get(lid)
        if lot:
            total = len(lot.spots)
            occupied = sum(1 for s in lot.spots if s.status != SpotStatus.AVAILABLE.value)
            enriched_cache[lid] = {
                **info,
                'occupancy_pct': occupied / max(total, 1),
                'total_spots': total,
            }
        else:
            enriched_cache[lid] = {**info, 'occupancy_pct': 0.5}

    pe = DynamicPricingEngine()
    result = pe.train(rows, enriched_cache)

    if 'error' not in result:
        store.pricing_engine = pe
        store.pricing_samples = pe.train_samples
        store.pricing_trained_at = pe.trained_at

    return result


def get_pricing_suggestions():
    """Get dynamic pricing suggestions from the trained pricing engine."""
    pe = store.pricing_engine
    if pe is None or not pe.trained:
        return {"available": False, "suggestions": []}

    from .models import ParkingLot, ParkingSpot, SpotStatus
    lot_cache = {}
    for lot in ParkingLot.query.all():
        total = len(lot.spots)
        occupied = sum(1 for s in lot.spots if s.status != SpotStatus.AVAILABLE.value)
        lot_cache[lot.id] = {
            'lot_name': lot.prime_location_name,
            'price_per_hour': lot.price_per_hour,
            'total_spots': total,
            'occupancy_pct': occupied / max(total, 1),
        }

    suggestions = pe.suggest_prices(lot_cache)
    return {"available": True, "suggestions": suggestions}


def train_all_models() -> dict:
    """Train all three models.  Returns combined status."""
    d = train_duration_model()
    r = train_recommender()
    p = train_pricing_model()
    return {
        "duration": d,
        "recommender": r,
        "pricing": p,
        "status": store.status_dict(),
    }
