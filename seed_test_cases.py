"""
Seed script - creates fake test cases to verify cluster detection works.
Run this once, check the results, then feel free to delete these test rows later.
NOT part of the real app - just a debugging/testing tool.
"""
from app import app, db
from models import Case

# Cluster 1: 4 ringworm cases close together (simulating an outbreak area)
cluster_1_base_lat, cluster_1_base_lon = 12.9716, 77.5946  # example coordinates

# Cluster 2: 2 demodicosis cases in a different area
cluster_2_base_lat, cluster_2_base_lon = 13.0827, 80.2707  # example coordinates, different city

test_cases = [
    # Cluster 1 - ringworm outbreak
    {"filename": "test1.jpg", "prediction": "ringworm", "confidence": 0.85, "location": f"{cluster_1_base_lat:.4f}, {cluster_1_base_lon:.4f}"},
    {"filename": "test2.jpg", "prediction": "ringworm", "confidence": 0.79, "location": f"{cluster_1_base_lat + 0.002:.4f}, {cluster_1_base_lon + 0.001:.4f}"},
    {"filename": "test3.jpg", "prediction": "ringworm", "confidence": 0.91, "location": f"{cluster_1_base_lat - 0.001:.4f}, {cluster_1_base_lon + 0.003:.4f}"},
    {"filename": "test4.jpg", "prediction": "ringworm", "confidence": 0.68, "location": f"{cluster_1_base_lat + 0.001:.4f}, {cluster_1_base_lon - 0.002:.4f}"},
    # Cluster 2 - smaller demodicosis cluster, different city
    {"filename": "test5.jpg", "prediction": "demodicosis", "confidence": 0.88, "location": f"{cluster_2_base_lat:.4f}, {cluster_2_base_lon:.4f}"},
    {"filename": "test6.jpg", "prediction": "demodicosis", "confidence": 0.72, "location": f"{cluster_2_base_lat + 0.001:.4f}, {cluster_2_base_lon - 0.001:.4f}"},
    # One isolated case, shouldn't cluster with anything
    {"filename": "test7.jpg", "prediction": "Healthy", "confidence": 0.95, "location": "17.3850, 78.4867"},
]

with app.app_context():
    for tc in test_cases:
        case = Case(
            filename=tc["filename"],
            prediction=tc["prediction"],
            confidence=tc["confidence"],
            is_uncertain=tc["confidence"] < 0.60,
            location=tc["location"]
        )
        db.session.add(case)
    db.session.commit()
    print(f"Added {len(test_cases)} test cases.")