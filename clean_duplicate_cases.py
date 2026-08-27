"""
PawCare AI - Duplicate Case Cleanup Utility
Sentinel (Security & QA) Runbook Script

Safely identifies and removes duplicate Case rows in the database (SQLite or PostgreSQL)
caused by double-dispatch upload race conditions. Preserves the canonical lowest-ID
record for each duplicate cluster.

Usage:
  Dry run (inspection only):
    python clean_duplicate_cases.py

  Execute removal:
    python clean_duplicate_cases.py --execute
"""
import sys
import os
from collections import defaultdict
from datetime import datetime

# Add parent backend directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import app, db
from models import Case


def find_and_clean_duplicates(execute=False):
    with app.app_context():
        cases = Case.query.order_by(Case.id.asc()).all()
        print(f"[*] Total cases in database: {len(cases)}")

        # Group by image filename, prediction, and approx timestamp (within 60s) or location
        clusters = defaultdict(list)
        for c in cases:
            # Key based on filename, prediction, confidence, and location
            ts_minute = c.created_at.strftime("%Y-%m-%d %H:%M") if c.created_at else "no-date"
            key = (c.filename, c.prediction, round(c.confidence, 4) if c.confidence else None, c.location, ts_minute)
            clusters[key].append(c)

        total_duplicates = 0
        canonical_to_keep = []
        ids_to_delete = []

        for key, group in clusters.items():
            if len(group) > 1:
                canonical = group[0]  # Lowest ID
                dupes = group[1:]
                canonical_to_keep.append(canonical.id)
                dupe_ids = [d.id for d in dupes]
                ids_to_delete.extend(dupe_ids)
                total_duplicates += len(dupes)

                print(f"\n[!] Duplicate cluster found:")
                print(f"    - Key: {key}")
                print(f"    - Canonical Record to Keep: ID {canonical.id} ({canonical.filename}, pred: {canonical.prediction})")
                print(f"    - Duplicate Records to Delete: IDs {dupe_ids}")

        print(f"\n" + "=" * 50)
        print(f"Summary: Found {total_duplicates} duplicate rows across {len(canonical_to_keep)} clusters.")

        if total_duplicates == 0:
            print("[+] Database is clean. No duplicate cases found.")
            return

        if execute:
            print(f"[*] Executing removal of {len(ids_to_delete)} duplicate rows: {ids_to_delete}...")
            Case.query.filter(Case.id.in_(ids_to_delete)).delete(synchronize_session=False)
            db.session.commit()
            print("[+] Successfully deleted duplicate rows from database.")
        else:
            print("[?] Dry-run mode completed. No records were deleted.")
            print("    To perform actual deletion, run: python clean_duplicate_cases.py --execute")


if __name__ == "__main__":
    should_execute = "--execute" in sys.argv
    find_and_clean_duplicates(execute=should_execute)
