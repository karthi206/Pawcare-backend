import json
import urllib.request
from datetime import datetime, timedelta

GEOCODE_CACHE = {}
GEOCODE_CACHE_TTL = timedelta(hours=6)
MAX_GEOCODE_CACHE_ENTRIES = 500


def resolve_coordinates_to_address(lat: float, lng: float) -> str | None:
    """Reverse geocodes (lat, lng) to a clean human-readable wording address with caching and fallbacks."""
    cache_key = (round(lat, 4), round(lng, 4))
    now = datetime.utcnow()
    if cache_key in GEOCODE_CACHE:
        cached_time, cached_address = GEOCODE_CACHE[cache_key]
        if now - cached_time < GEOCODE_CACHE_TTL and cached_address:
            return cached_address

    address = None
    # 1. OpenStreetMap Nominatim with structured address formatting
    try:
        url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lng}&zoom=18&addressdetails=1"
        req = urllib.request.Request(url, headers={"User-Agent": "PawCareAI/1.0 (animal-welfare-locator)"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            addr = data.get("address", {})
            parts = []
            road = addr.get("road") or addr.get("pedestrian")
            neighborhood = addr.get("neighbourhood") or addr.get("suburb") or addr.get("residential")
            if road:
                parts.append(road)
            if neighborhood and neighborhood not in parts:
                parts.append(neighborhood)
            city = addr.get("city") or addr.get("town") or addr.get("village") or addr.get("county") or addr.get("city_district")
            if city:
                city_clean = city.replace(" Corporation", "").replace(" District", "")
                if city_clean not in parts:
                    parts.append(city_clean)
            state = addr.get("state")
            if state and state not in parts:
                parts.append(state)

            if parts:
                address = ", ".join(parts)
            elif data.get("display_name"):
                address = data.get("display_name")
    except Exception as exc:
        print(f"[resolve_coordinates_to_address] Nominatim notice: {exc}")

    # 2. BigDataCloud fallback if Nominatim is unavailable or rate-limited
    if not address:
        try:
            url = f"https://api.bigdatacloud.net/data/reverse-geocode-client?latitude={lat}&longitude={lng}&localityLanguage=en"
            req = urllib.request.Request(url, headers={"User-Agent": "PawCareAI/1.0 (animal-welfare-locator)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                bdc_data = json.loads(resp.read().decode("utf-8"))
                bdc_parts = []
                locality = bdc_data.get("locality")
                city = bdc_data.get("city")
                subdivision = bdc_data.get("principalSubdivision")
                if locality:
                    bdc_parts.append(locality)
                if city and city not in bdc_parts:
                    bdc_parts.append(city)
                if subdivision and subdivision not in bdc_parts:
                    bdc_parts.append(subdivision)
                if bdc_parts:
                    address = ", ".join(bdc_parts)
        except Exception as bdc_exc:
            print(f"[resolve_coordinates_to_address] BigDataCloud notice: {bdc_exc}")

    if address:
        if len(GEOCODE_CACHE) >= MAX_GEOCODE_CACHE_ENTRIES:
            for k in list(GEOCODE_CACHE.keys())[:100]:
                GEOCODE_CACHE.pop(k, None)
        GEOCODE_CACHE[cache_key] = (now, address)

    return address
