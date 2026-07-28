from math import radians, sin, cos, sqrt, atan2

CLUSTER_RADIUS_KM = 1.0   # cases within 1km of each other count as "nearby"
MIN_CLUSTER_SIZE = 2      # need at least this many nearby same-disease cases to call it a cluster


def haversine_distance(lat1, lon1, lat2, lon2):
    """Returns distance in kilometers between two lat/lon points."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def parse_location(location_str):
    """Converts '12.9716, 77.5946' into (12.9716, 77.5946). Returns None if invalid."""
    if not location_str:
        return None
    try:
        lat, lon = map(float, location_str.split(","))
        return (lat, lon)
    except (ValueError, AttributeError):
        return None


def detect_clusters(cases):
    """
    Groups cases by disease, then finds tight geographic clusters within each disease group.
    `cases` should be a list of dicts with at least: id, prediction, location.
    Returns a list of cluster summaries.
    """
    # Step 1: parse and filter to only cases with valid locations
    located_cases = []
    for case in cases:
        coords = parse_location(case.get("location"))
        if coords:
            located_cases.append({**case, "lat": coords[0], "lon": coords[1]})

    # Step 2: group by disease type
    by_disease = {}
    for case in located_cases:
        disease = case["prediction"]
        by_disease.setdefault(disease, []).append(case)

    # Step 3: within each disease group, find clusters
    clusters = []
    for disease, disease_cases in by_disease.items():
        visited = set()

        for i, case in enumerate(disease_cases):
            if case["id"] in visited:
                continue

            # Find all other cases of this disease within CLUSTER_RADIUS_KM
            nearby = [case]
            for other in disease_cases:
                if other["id"] == case["id"] or other["id"] in visited:
                    continue
                dist = haversine_distance(case["lat"], case["lon"], other["lat"], other["lon"])
                if dist <= CLUSTER_RADIUS_KM:
                    nearby.append(other)

            if len(nearby) >= MIN_CLUSTER_SIZE:
                for c in nearby:
                    visited.add(c["id"])

                avg_lat = sum(c["lat"] for c in nearby) / len(nearby)
                avg_lon = sum(c["lon"] for c in nearby) / len(nearby)

                clusters.append({
                    "disease": disease,
                    "case_count": len(nearby),
                    "case_ids": [c["id"] for c in nearby],
                    "center_lat": round(avg_lat, 4),
                    "center_lon": round(avg_lon, 4),
                })

    return clusters