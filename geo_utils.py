"""Shared geographic utilities for the NAVIG pipeline and figure scripts.

Provides :func:`lat_to_continent` — accurate continent assignment via
Nominatim reverse geocoding backed by a persistent on-disk cache.  Falls
back to a bounding-box heuristic when the network is unavailable.
"""

import hashlib
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent
_RGEO_CACHE_PATH = _ROOT / ".cache" / "reverse_geocode.json"

# ── ISO 3166-1 α-2 → NAVIG continent label ────────────────────────────────────
# Covers all 249 UN-listed territories.
_CC_TO_CONTINENT: dict[str, str] = {
    # Africa
    **dict.fromkeys([
        "AO","BF","BI","BJ","BW","CD","CF","CG","CI","CM","CV","DJ","DZ","EG",
        "EH","ER","ET","GA","GH","GM","GN","GQ","GW","KE","KM","LR","LS","LY",
        "MA","MG","ML","MR","MU","MW","MZ","NA","NE","NG","RE","RW","SC","SD",
        "SL","SN","SO","SS","ST","SZ","TD","TG","TN","TZ","UG","YT","ZA","ZM","ZW",
    ], "Africa"),
    # Asia (North & Central)
    **dict.fromkeys([
        "AM","AZ","CN","GE","JP","KG","KP","KR","KZ","MN","RU","TJ","TM","UZ",
    ], "Asia (N)"),
    # Asia (South & Southeast) — includes Middle East
    **dict.fromkeys([
        "AE","AF","BD","BH","BN","BT","CY","ID","IN","IQ","IR","JO","KH","KW",
        "LA","LB","LK","MM","MV","MY","NP","OM","PH","PK","PS","QA","SA","SG",
        "SY","TH","TL","TR","VN","YE",
    ], "Asia (S/SE)"),
    # Europe
    **dict.fromkeys([
        "AD","AL","AT","BA","BE","BG","BY","CH","CZ","DE","DK","EE","ES","FI",
        "FO","FR","GB","GG","GI","GR","HR","HU","IE","IM","IS","IT","JE","LI",
        "LT","LU","LV","MC","MD","ME","MK","MT","NL","NO","PL","PT","RO","RS",
        "SE","SI","SK","SM","UA","VA","XK",
    ], "Europe"),
    # North America (including Caribbean and Central America)
    **dict.fromkeys([
        "AG","AI","AN","AW","BB","BL","BM","BS","BZ","CA","CR","CU","DM","DO",
        "GD","GL","GP","GT","HN","HT","JM","KN","KY","LC","MF","MQ","MS","MX",
        "NI","PA","PM","PR","SV","TC","TT","US","VC","VG","VI",
    ], "N. America"),
    # South America
    **dict.fromkeys([
        "AR","BO","BR","CL","CO","EC","FK","GF","GY","PE","PY","SR","UY","VE",
    ], "S. America"),
    # Oceania
    **dict.fromkeys([
        "AS","AU","CK","FJ","FM","GU","KI","MH","MP","NC","NF","NR","NU","NZ",
        "PF","PG","PW","SB","TK","TO","TV","UM","VU","WF","WS",
    ], "Oceania"),
}

# ── In-process reverse-geocode cache ─────────────────────────────────────────
_rgeo_cache: dict[str, str] | None = None


def _load_rgeo_cache() -> dict[str, str]:
    global _rgeo_cache
    if _rgeo_cache is not None:
        return _rgeo_cache
    try:
        _rgeo_cache = json.loads(_RGEO_CACHE_PATH.read_text())
    except (FileNotFoundError, Exception):
        _rgeo_cache = {}
    return _rgeo_cache


def _save_rgeo_cache() -> None:
    try:
        _RGEO_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = _RGEO_CACHE_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(_rgeo_cache))
        tmp.replace(_RGEO_CACHE_PATH)
    except Exception:
        pass


# ── Bounding-box fallback ─────────────────────────────────────────────────────

def _bbox_continent(lat: float, lon: float) -> str:
    """Fast bounding-box heuristic — used when Nominatim is unavailable."""
    if lat > 66.5:
        return "Arctic"
    if lat < -60:
        return "Antarctica"
    if -35 < lat < 37 and -20 < lon < 55:
        return "Africa"
    if lat > 35 and -30 < lon < 60:
        return "Europe"
    if lat > 0 and 60 < lon < 180:
        return "Asia (N)"
    if -10 < lat < 35 and 60 < lon < 180:
        return "Asia (S/SE)"
    if -55 < lat < -10 and 110 < lon < 180:
        return "Oceania"
    if 15 < lat < 75 and -170 < lon < -50:
        return "N. America"
    if -60 < lat < 15 and -90 < lon < -30:
        return "S. America"
    return "Other"


def _nominatim_reverse(lat: float, lon: float) -> str | None:
    """Return a continent label via Nominatim reverse geocoding, or None on failure."""
    try:
        import requests as _req
        resp = _req.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json", "zoom": 3},
            headers={"User-Agent": "UMIACS/NAVIG_analysis (contact: kinsey.long@berkeley.edu)"},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        cc = resp.json().get("address", {}).get("country_code", "").upper()
        if cc:
            return _CC_TO_CONTINENT.get(cc, "Other")
    except Exception:
        pass
    return None


# ── Public API ────────────────────────────────────────────────────────────────

def lat_to_continent(lat: float, lon: float, use_api: bool = True) -> str:
    """Return a NAVIG continent label for *(lat, lon)*.

    Uses Nominatim reverse geocoding (1 req/s, shared cache at
    ``.cache/reverse_geocode.json``) for accurate country-level assignment.
    Results are cached at 0.1° precision so re-runs are instant.
    Falls back to a bounding-box heuristic when the API is unavailable or
    *use_api* is ``False``.

    Continent labels: ``"Europe"``, ``"N. America"``, ``"S. America"``,
    ``"Africa"``, ``"Asia (N)"``, ``"Asia (S/SE)"``, ``"Oceania"``,
    ``"Arctic"``, ``"Antarctica"``, ``"Other"``.
    """
    if lat is None or lon is None:
        return "Other"
    lat, lon = float(lat), float(lon)
    # Normalise longitude to [-180, 180]
    lon = ((lon + 180) % 360) - 180

    cache = _load_rgeo_cache()
    key = f"{round(lat, 1)},{round(lon, 1)}"
    if key in cache:
        return cache[key]

    continent: str | None = None
    if use_api:
        continent = _nominatim_reverse(lat, lon)

    if continent is None:
        continent = _bbox_continent(lat, lon)

    cache[key] = continent
    _save_rgeo_cache()
    return continent
