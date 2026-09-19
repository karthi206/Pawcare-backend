import urllib.request
import urllib.parse
import json
from datetime import datetime, timedelta

from utils.geo import haversine_distance

OVERPASS_CACHE = {}
OVERPASS_CACHE_TTL = timedelta(minutes=10)
MAX_OVERPASS_CACHE_ENTRIES = 200

ANIMAL_KEYWORDS = (
    'animal', 'animals', 'dog', 'dogs', 'cat', 'cats', 'pet', 'pets',
    'puppy', 'puppies', 'kitten', 'kittens', 'rescue', 'shelter',
    'veterinary', 'vet', 'wildlife', 'stray', 'strays', 'welfare',
    'spca', 'blue cross', 'paws', 'paw', 'canine', 'feline',
    'sanctuary', 'fauna', 'creature', 'creatures', 'humane',
    'gaushala', 'goshala', 'gau', 'jivdaya', 'jeevdaya', 'prani', 'ahimsa'
)


def get_live_nearby_places(lat: float, lng: float, radius_km: float) -> list:
    """Query Overpass for animal shelters / vets / animal-welfare NGOs near (lat, lng),
    with a 10-minute in-memory cache keyed on rounded coordinates (~1.1km resolution)."""
    # Cap radius to 100km to avoid excessively large Overpass queries
    radius_km = min(radius_km, 100.0)

    cache_key = (round(lat, 2), round(lng, 2), round(radius_km, 1))
    now = datetime.utcnow()

    if cache_key in OVERPASS_CACHE:
        cached_time, cached_data = OVERPASS_CACHE[cache_key]
        if now - cached_time < OVERPASS_CACHE_TTL:
            return cached_data

    radius_meters = int(radius_km * 1000)
    overpass_query = f"""
    [out:json][timeout:15];
    (
      node["amenity"="animal_shelter"](around:{radius_meters},{lat},{lng});
      node["amenity"="veterinary"](around:{radius_meters},{lat},{lng});
      node["office"="ngo"](around:{radius_meters},{lat},{lng});
      way["amenity"="animal_shelter"](around:{radius_meters},{lat},{lng});
      way["amenity"="veterinary"](around:{radius_meters},{lat},{lng});
      way["office"="ngo"](around:{radius_meters},{lat},{lng});
    );
    out center;
    """

    results = []
    try:
        post_data = urllib.parse.urlencode({"data": overpass_query}).encode("utf-8")
        req = urllib.request.Request(
            "https://overpass-api.de/api/interpreter",
            data=post_data,
            headers={
                "User-Agent": "PawCareAI/1.0 (animal-welfare-locator)",
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
            }
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw_body = resp.read().decode("utf-8")
            osm_data = json.loads(raw_body)
            elements = osm_data.get("elements", [])
            for el in elements:
                el_lat = el.get("lat") or el.get("center", {}).get("lat")
                el_lng = el.get("lon") or el.get("center", {}).get("lon")
                if el_lat is None or el_lng is None:
                    continue

                tags = el.get("tags", {})
                amenity = tags.get("amenity")
                office = tags.get("office")
                raw_name = (tags.get("name") or "").strip()

                # Reliable tags: keep animal_shelter and veterinary as-is
                if amenity == "animal_shelter":
                    place_type = "animal_shelter"
                    type_label = "Animal Shelter"
                    street = tags.get("addr:street")
                    name = raw_name or (f"{type_label} ({street})" if street else f"{type_label} (Unlisted Name)")
                elif amenity == "veterinary":
                    place_type = "veterinary"
                    type_label = "Veterinary Clinic"
                    street = tags.get("addr:street")
                    name = raw_name or (f"{type_label} ({street})" if street else f"{type_label} (Unlisted Name)")
                elif office == "ngo":
                    # office=ngo is too generic (driving schools, unrelated trusts,
                    # generic placeholders). Drop unnamed ones, and only keep if
                    # name contains an animal welfare keyword.
                    if not raw_name:
                        continue

                    name_lower = raw_name.lower()
                    if not any(kw in name_lower for kw in ANIMAL_KEYWORDS):
                        continue

                    place_type = "ngo"
                    type_label = "NGO / Rescue"
                    name = raw_name
                else:
                    continue

                addr_parts = [
                    tags.get("addr:housenumber"),
                    tags.get("addr:street"),
                    tags.get("addr:city"),
                    tags.get("addr:postcode")
                ]
                address = ", ".join([p for p in addr_parts if p]) or tags.get("address") or "Address not listed in OSM"
                phone = tags.get("phone") or tags.get("contact:phone") or None

                dist = haversine_distance(lat, lng, el_lat, el_lng)
                if dist <= radius_km:
                    results.append({
                        "id": f"osm_{el.get('type', 'node')}_{el.get('id')}",
                        "name": name,
                        "address": address,
                        "phone": phone,
                        "lat": el_lat,
                        "lng": el_lng,
                        "type": place_type,
                        "source": "osm",
                        "distance_km": round(dist, 2)
                    })

            results.sort(key=lambda x: x["distance_km"])
    except Exception as exc:
        print(f"[live-nearby] Overpass API request notice: {exc}")

    # Cache successful results or empty results up to cache size limit
    if len(OVERPASS_CACHE) >= MAX_OVERPASS_CACHE_ENTRIES:
        keys_to_delete = list(OVERPASS_CACHE.keys())[:50]
        for k in keys_to_delete:
            OVERPASS_CACHE.pop(k, None)

    OVERPASS_CACHE[cache_key] = (now, results)
    return results
