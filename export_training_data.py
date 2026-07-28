"""
Exports vet-confirmed/corrected cases into a folder structure matching your
original training data format: export/DiseaseName/image.jpg

Run this periodically (e.g. monthly, or whenever enough new reviews pile up)
to gather real-world-verified images for your next Colab retraining cycle.
"""
import os
import shutil
from app import app, db
from models import Case

UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "export"

with app.app_context():
    # Only cases a vet has actually reviewed and labeled
    reviewed_cases = Case.query.filter(Case.vet_confirmed_label.isnot(None)).all()

    if not reviewed_cases:
        print("No vet-reviewed cases found yet. Nothing to export.")
        exit()

    corrections = 0
    confirmations = 0
    skipped_missing_file = 0

    for case in reviewed_cases:
        source_path = os.path.join(UPLOAD_FOLDER, case.filename)

        if not os.path.exists(source_path):
            skipped_missing_file += 1
            continue

        # The vet's label is the ground truth now - this is what we train on,
        # regardless of what the AI originally predicted
        label = case.vet_confirmed_label
        dest_dir = os.path.join(EXPORT_FOLDER, label)
        os.makedirs(dest_dir, exist_ok=True)

        dest_path = os.path.join(dest_dir, f"case{case.id}_{case.filename}")
        shutil.copy2(source_path, dest_path)

        if case.vet_confirmed_label == case.prediction:
            confirmations += 1
        else:
            corrections += 1

    print(f"Export complete.")
    print(f"  AI was correct (confirmations): {confirmations}")
    print(f"  AI was wrong (corrections):     {corrections}")
    print(f"  Skipped (image file missing):   {skipped_missing_file}")
    print(f"  Exported to: ./{EXPORT_FOLDER}/")