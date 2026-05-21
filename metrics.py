"""Shared metric functions and constants used across the NAVIG pipeline.

All pipeline and analysis scripts import from here so the GeoScore formula,
haversine distance, coordinate parsing, and scoring logic have exactly one
definition.
"""

import math
import re

import numpy as np

# GeoScore exponential decay constant (matches the GeoGuessr formula).
GEOSCORE_SCALE: float = 1492.7

THRESHOLDS: list[int] = [1, 25, 200, 750, 2500]
THRESHOLD_NAMES: list[str] = [
    "Street(1km)", "City(25km)", "Region(200km)", "Country(750km)", "Continent(2500km)"
]


def geoscore(distance: float) -> float:
    """GeoGuessr scoring formula.  Returns 5000 at distance=0, ~0 at ≥10 000 km."""
    return 5000.0 * math.exp(-distance / GEOSCORE_SCALE)


def haversine_distance(point1, point2) -> float:
    """Great-circle distance in km between two [lat, lon] pairs."""
    lat1, lon1 = point1
    lat2, lon2 = point2
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def parse_coord(value) -> float:
    """Parse a coordinate that may be a plain float, degree-notation string
    ('8.6836° N'), or hedged string ('Approximately 75° W').

    Raises ValueError for unparseable values (None, 'Unknown', empty, etc.).
    """
    if value is None:
        raise ValueError("Unparseable coordinate: None")
    s = str(value).strip()
    if s.lower() in ("unknown", "n/a", "", "nan", "null", "none", "infinity", "undefined"):
        raise ValueError(f"Unparseable coordinate: {s!r}")
    s = re.sub(r"(?i)approximately\s*", "", s).strip()
    m = re.match(r"^(-?\d+(?:\.\d+)?)\s*°?\s*([NSEWnsew])?$", s)
    if m:
        val = float(m.group(1))
        if (m.group(2) or "").upper() in ("S", "W"):
            val = -val
        return val
    return float(s)


# Distance assigned to parse failures so they sink to the bottom of rankings.
PARSE_FAIL_DISTANCE: float = 10_000.0


def get_row_distance(row: dict) -> tuple[float, bool]:
    """Return (distance_km, is_parse_failure) for one result row."""
    correct = [float(row["LAT"]), float(row["LON"])]
    try:
        pred = [
            parse_coord(row["answer"]["latitude"]),
            parse_coord(row["answer"]["longitude"]),
        ]
        return haversine_distance(pred, correct), False
    except Exception:
        return PARSE_FAIL_DISTANCE, True


def score_results(rows: list[dict]) -> dict:
    """Compute aggregate metrics from a list of result rows.

    Returns a dict with keys:
        n, geoscore, avg_dist, median_dist, failures, accs (list per threshold)
    """
    distances: list[float] = []
    failures = 0
    for row in rows:
        dist, fail = get_row_distance(row)
        distances.append(dist)
        failures += fail

    n = len(distances)
    if n == 0:
        return {"n": 0, "geoscore": 0.0, "avg_dist": 0.0, "median_dist": 0.0,
                "failures": 0, "accs": [0.0] * len(THRESHOLDS)}

    accs = [sum(d <= t for d in distances) / n for t in THRESHOLDS]
    return {
        "n": n,
        "geoscore": float(np.mean([geoscore(d) for d in distances])),
        "avg_dist": float(np.mean(distances)),
        "median_dist": float(np.median(distances)),
        "failures": failures,
        "accs": accs,
    }


def print_score_summary(stats: dict, label: str = "") -> None:
    """Print a formatted score summary to stdout."""
    prefix = f"\n=== {label} ===" if label else "\n=== Results ==="
    print(prefix)
    n = stats["n"]
    print(f"N                : {n}")
    print(f"Avg GeoScore     : {stats['geoscore']:.2f}")
    print(f"Avg Distance     : {stats['avg_dist']:.2f} km")
    print(f"Median Distance  : {stats['median_dist']:.2f} km")
    print(f"Parse failures   : {stats['failures']} ({100 * stats['failures'] / max(n, 1):.1f}%)")
    for acc, name in zip(stats["accs"], THRESHOLD_NAMES):
        print(f"  Acc@{name:<20}: {acc:.4f}")
