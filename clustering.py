"""
Outbreak cluster detection.

CHANGES vs. the original version:
  1. Time windows — previously every case ever reported was clustered
     together with no regard for when it happened, so a handful of old
     resolved cases from months ago could still show as an "active"
     outbreak today. Clusters are now built only from cases inside a
     rolling time window (default 14 days).
  2. Vet-confirmation weighting — previously a raw AI label counted
     exactly the same as a vet-confirmed diagnosis. A cluster of five
     low-confidence AI guesses is much weaker evidence than two
     vet-confirmed cases. Each case now contributes a weight based on
     how trustworthy its label is, and a cluster only fires once the
     total weight crosses a threshold — not just a raw case count.

Weighting scheme (tunable via WEIGHT_* constants below):
  - vet_confirmed case                     -> 1.0
  - AI prediction, NOT flagged uncertain   -> 0.6
  - AI prediction, flagged uncertain       -> 0.25

"Healthy" is never counted toward an outbreak, whether it's the AI
prediction or the vet-confirmed label.
"""

import os
import math
from datetime import datetime, timedelta

RADIUS_KM = float(os.environ.get("OUTBREAK_RADIUS_KM", "1.0"))
TIME_WINDOW_DAYS = int(os.environ.get("OUTBREAK_TIME_WINDOW_DAYS", "14"))
MIN_WEIGHT = float(os.environ.get("OUTBREAK_MIN_WEIGHT", "2.0"))          # weighted score required to call it a cluster
MIN_CASE_COUNT = int(os.environ.get("OUTBREAK_MIN_CASES", "2"))          # require at least 2 distinct cases

WEIGHT_VET_CONFIRMED = float(os.environ.get("WEIGHT_VET_CONFIRMED", "1.0"))
WEIGHT_AI_CONFIDENT = float(os.environ.get("WEIGHT_AI_CONFIDENT", "0.6"))
WEIGHT_AI_UNCERTAIN = float(os.environ.get("WEIGHT_AI_UNCERTAIN", "0.25"))

EXCLUDED_LABELS = {'Healthy'}


def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _parse_location(location_str):
    """Location is stored as 'lat,lon'. Returns (lat, lon) or None if
    missing/unparseable (e.g. a free-text address instead of coordinates)."""
    if not location_str:
        return None
    try:
        lat_str, lon_str = location_str.split(',', 1)
        return float(lat_str.strip()), float(lon_str.strip())
    except (ValueError, AttributeError):
        return None


def _parse_created_at(created_at_str):
    try:
        return datetime.fromisoformat(created_at_str)
    except (ValueError, TypeError):
        return None


def _effective_label_and_weight(case):
    """Decide what disease label this case counts as, and how much it
    should weigh. Vet-confirmed labels always win over the AI prediction —
    they're a stronger signal and may correct the AI entirely."""
    vet_label = case.get('vet_confirmed_label')
    if vet_label:
        return vet_label, WEIGHT_VET_CONFIRMED

    prediction = case.get('prediction')
    if case.get('is_uncertain'):
        return prediction, WEIGHT_AI_UNCERTAIN
    return prediction, WEIGHT_AI_CONFIDENT


def detect_clusters(cases, radius_km=RADIUS_KM, time_window_days=TIME_WINDOW_DAYS,
                     min_weight=MIN_WEIGHT, now=None):
    """
    cases: list of dicts, as produced by Case.to_dict()
    Returns a list of cluster dicts:
      {
        "disease": str,
        "case_count": int,
        "case_ids": [int, ...],
        "center_lat": float,
        "center_lon": float,
        "weighted_score": float,
        "vet_confirmed_count": int,
      }
    """
    now = now or datetime.utcnow()
    cutoff = now - timedelta(days=time_window_days)

    # Build enriched, filtered records: valid location, recent, real disease label
    points = []
    for case in cases:
        coords = _parse_location(case.get('location'))
        if coords is None:
            continue

        created_at = _parse_created_at(case.get('created_at'))
        if created_at is None or created_at < cutoff:
            continue

        label, weight = _effective_label_and_weight(case)
        if not label or label in EXCLUDED_LABELS:
            continue

        points.append({
            'id': case.get('id'),
            'lat': coords[0],
            'lon': coords[1],
            'label': label,
            'weight': weight,
            'is_vet_confirmed': bool(case.get('vet_confirmed_label')),
        })

    clusters = []
    # Group by disease label first — an outbreak cluster is always single-disease
    labels = {p['label'] for p in points}
    for label in labels:
        label_points = [p for p in points if p['label'] == label]

        # Simple single-linkage grouping: start every point in its own group,
        # then merge any two groups that have at least one point pair within
        # radius_km of each other. Small case counts (dozens, not millions)
        # so an O(n^2) pass per merge round is plenty fast here.
        groups = [[p] for p in label_points]
        merged = True
        while merged:
            merged = False
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    if _groups_within_radius(groups[i], groups[j], radius_km):
                        groups[i].extend(groups[j])
                        del groups[j]
                        merged = True
                        break
                if merged:
                    break

        for group in groups:
            if len(group) < MIN_CASE_COUNT:
                continue
            weighted_score = sum(p['weight'] for p in group)
            if weighted_score < min_weight:
                continue

            center_lat = sum(p['lat'] for p in group) / len(group)
            center_lon = sum(p['lon'] for p in group) / len(group)
            clusters.append({
                "disease": label,
                "case_count": len(group),
                "case_ids": [p['id'] for p in group],
                "center_lat": center_lat,
                "center_lon": center_lon,
                "weighted_score": round(weighted_score, 2),
                "vet_confirmed_count": sum(1 for p in group if p['is_vet_confirmed']),
            })

    # Strongest evidence first
    clusters.sort(key=lambda c: c['weighted_score'], reverse=True)
    return clusters


def _groups_within_radius(group_a, group_b, radius_km):
    for a in group_a:
        for b in group_b:
            if _haversine_km(a['lat'], a['lon'], b['lat'], b['lon']) <= radius_km:
                return True
    return False