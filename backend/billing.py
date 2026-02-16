from datetime import datetime, timedelta
import math

def compute_parking_cost(start: datetime, end: datetime, price_per_hour: float, policy: str = 'proportional', minutes_unit: int = 15) -> float:
    """Compute parking cost according to billing policy.

    Policies:
    - proportional: exact fractional hours * price
    - per_15_min: round up to nearest 15 minutes
    - per_hour: round up to next hour
    - minimum_1_hour: min 1 hour charged, rounded up
    """
    if end < start:
        return 0.0
    delta = end - start
    hours = delta.total_seconds() / 3600.0
    if policy == 'proportional':
        cost = hours * price_per_hour
    elif policy == 'per_15_min':
        unit_hours = minutes_unit / 60.0
        charged_units = math.ceil(hours / unit_hours)
        cost = charged_units * unit_hours * price_per_hour
    elif policy == 'per_hour':
        charged = math.ceil(hours)
        cost = charged * price_per_hour
    elif policy == 'minimum_1_hour':
        charged = max(1, math.ceil(hours))
        cost = charged * price_per_hour
    else:
        # default fallback
        cost = hours * price_per_hour
    return round(cost, 2)
