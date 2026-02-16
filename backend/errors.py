from flask import jsonify


def make_error(message: str, status: int = 400, errors: dict | None = None):
    payload = {"message": message}
    if errors:
        payload["errors"] = errors
    return jsonify(payload), status
