from typing import Tuple, Dict, Any

def _validate_positive_int(value, field: str, errors: Dict[str, str]) -> int | None:
    try:
        iv = int(value)
        if iv <= 0:
            errors[field] = f"{field} must be > 0"
            return None
        return iv
    except Exception:
        errors[field] = f"{field} must be an integer"
        return None

def validate_lot_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Validate payload for creating/updating a ParkingLot.
    Returns (cleaned_data, errors)."""
    errors = {}
    cleaned = {}

    name = (data.get("prime_location_name") or "").strip()
    if not name:
        errors["prime_location_name"] = "Location name is required"
    else:
        cleaned["prime_location_name"] = name

    try:
        price = float(data.get("price_per_hour") or 0)
        if price <= 0:
            errors["price_per_hour"] = "price_per_hour must be a positive number"
        else:
            cleaned["price_per_hour"] = price
    except Exception:
        errors["price_per_hour"] = "price_per_hour must be a number"

    pincode = (data.get("pincode") or "").strip()
    cleaned["pincode"] = pincode

    address = (data.get("address") or "").strip()
    cleaned["address"] = address

    try:
        count = int(data.get("number_of_spots") or 0)
        if count <= 0:
            errors["number_of_spots"] = "number_of_spots must be > 0"
        else:
            cleaned["number_of_spots"] = count
    except Exception:
        errors["number_of_spots"] = "number_of_spots must be an integer"

    return cleaned, errors


def validate_reservation_payload(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    errors = {}
    cleaned = {}
    try:
        lot_id = int(data.get("lot_id") or 0)
        if lot_id <= 0:
            errors["lot_id"] = "lot_id required"
        else:
            cleaned["lot_id"] = lot_id
    except Exception:
        errors["lot_id"] = "lot_id must be an integer"

    vehicle_number = (data.get("vehicle_number") or "").strip() or None
    vehicle_model = (data.get("vehicle_model") or "").strip() or None
    cleaned["vehicle_number"] = vehicle_number
    cleaned["vehicle_model"] = vehicle_model

    return cleaned, errors


def validate_lot_update_payload(current: Dict[str, Any], incoming: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Validate partial update for a lot. current is existing values; incoming is payload.
    Returns cleaned (merged) and errors."""
    merged = {
        "prime_location_name": (incoming.get("prime_location_name", current.get("prime_location_name")) or "").strip(),
        "price_per_hour": incoming.get("price_per_hour", current.get("price_per_hour")),
        "address": incoming.get("address", current.get("address")) or None,
        "pincode": incoming.get("pincode", current.get("pincode")) or None,
        "number_of_spots": incoming.get("number_of_spots", current.get("number_of_spots")),
    }
    errors: Dict[str, str] = {}
    cleaned: Dict[str, Any] = {}
    if not merged["prime_location_name"].strip():
        errors["prime_location_name"] = "Location name is required"
    else:
        cleaned["prime_location_name"] = merged["prime_location_name"].strip()
    # price
    try:
        price = float(merged["price_per_hour"])
        if price <= 0:
            errors["price_per_hour"] = "price_per_hour must be a positive number"
        else:
            cleaned["price_per_hour"] = price
    except Exception:
        errors["price_per_hour"] = "price_per_hour must be a number"
    # number_of_spots (allow same value)
    _spots = merged["number_of_spots"]
    try:
        iv = int(_spots)
        if iv <= 0:
            errors["number_of_spots"] = "number_of_spots must be > 0"
        else:
            cleaned["number_of_spots"] = iv
    except Exception:
        errors["number_of_spots"] = "number_of_spots must be an integer"
    cleaned["address"] = merged["address"]
    cleaned["pincode"] = merged["pincode"]
    return cleaned, errors


def validate_vehicle_fields(vehicle_number: Any, vehicle_model: Any) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Basic validation for optional vehicle fields."""
    errors: Dict[str, str] = {}
    cleaned: Dict[str, Any] = {}
    vn = (vehicle_number or "").strip() or None
    vm = (vehicle_model or "").strip() or None
    if vn and len(vn) > 32:
        errors["vehicle_number"] = "vehicle_number too long (max 32)"
    if vm and len(vm) > 100:
        errors["vehicle_model"] = "vehicle_model too long (max 100)"
    cleaned["vehicle_number"] = vn
    cleaned["vehicle_model"] = vm
    return cleaned, errors
