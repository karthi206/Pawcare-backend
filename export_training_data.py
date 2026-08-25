"""
Exports vet-confirmed/corrected cases into a folder structure matching your
training data format: export/DiseaseName/case_<id>_<name>.jpg

Supports both modern Cloudinary image URLs and legacy local upload files.
Also writes an export manifest (manifest.json) for auditing and reproducible retraining.
"""
import os
import json
import shutil
import urllib.request
import urllib.parse
from app import app, db
from models import Case

UPLOAD_FOLDER = "uploads"
EXPORT_FOLDER = "export"

def export_vet_data():
    with app.app_context():
        # Only export cases that have been reviewed and validated by a veterinarian
        reviewed_cases = Case.query.filter(Case.vet_confirmed_label.isnot(None)).all()

        if not reviewed_cases:
            print("No vet-reviewed cases found yet. Nothing to export.")
            return {"exported": 0, "status": "no_data"}

        os.makedirs(EXPORT_FOLDER, exist_ok=True)

    corrections = 0
    confirmations = 0
    downloaded_cloudinary = 0
    copied_local = 0
    failed_downloads = 0

    manifest_records = []

    for case in reviewed_cases:
        label = case.vet_confirmed_label.strip()
        dest_dir = os.path.join(EXPORT_FOLDER, label)
        os.makedirs(dest_dir, exist_ok=True)

        is_correct = (case.prediction == case.vet_confirmed_label)
        if is_correct:
            confirmations += 1
        else:
            corrections += 1

        # Determine if filename is a remote Cloudinary URL or local file
        is_remote = case.filename.startswith("http://") or case.filename.startswith("https://")
        
        if is_remote:
            parsed_path = urllib.parse.urlparse(case.filename).path
            base_name = os.path.basename(parsed_path) or f"case_{case.id}.jpg"
            if not base_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                base_name += ".jpg"
            dest_filename = f"case_{case.id}_{base_name}"
            dest_path = os.path.join(dest_dir, dest_filename)

            try:
                # Download remote Cloudinary image
                req = urllib.request.Request(
                    case.filename,
                    headers={'User-Agent': 'PawCare-Data-Exporter/1.0'}
                )
                with urllib.request.urlopen(req, timeout=15) as response, open(dest_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
                downloaded_cloudinary += 1
            except Exception as e:
                print(f"Error downloading case {case.id} from {case.filename}: {e}")
                failed_downloads += 1
                continue
        else:
            source_path = os.path.join(UPLOAD_FOLDER, case.filename)
            if not os.path.exists(source_path):
                print(f"Skipping case {case.id}: local file {source_path} not found.")
                failed_downloads += 1
                continue

            dest_filename = f"case_{case.id}_{case.filename}"
            dest_path = os.path.join(dest_dir, dest_filename)
            try:
                shutil.copy2(source_path, dest_path)
                copied_local += 1
            except Exception as e:
                print(f"Error copying local case {case.id}: {e}")
                failed_downloads += 1
                continue

        manifest_records.append({
            "case_id": case.id,
            "image_path": os.path.relpath(dest_path, EXPORT_FOLDER),
            "source_url": case.filename,
            "ai_prediction": case.prediction,
            "ai_confidence": case.confidence,
            "vet_confirmed_label": case.vet_confirmed_label,
            "ai_was_correct": is_correct,
            "location": case.location,
            "reviewed_by_id": case.reviewed_by_id,
            "created_at": case.created_at.isoformat() if case.created_at else None,
        })

    # Save manifest for dataset provenance and ML training pipelines
    manifest_path = os.path.join(EXPORT_FOLDER, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "total_cases": len(manifest_records),
            "confirmations": confirmations,
            "corrections": corrections,
            "agreement_rate": round(confirmations / len(reviewed_cases), 4) if reviewed_cases else 0.0,
            "cases": manifest_records
        }, f, indent=2)

    total_exported = downloaded_cloudinary + copied_local
    print(f"\n Export Complete!")
    print(f"  Total Vet-Reviewed Cases: {len(reviewed_cases)}")
    print(f"  Successfully Exported:   {total_exported}")
    print(f"    - Cloudinary downloads: {downloaded_cloudinary}")
    print(f"    - Local file copies:    {copied_local}")
    print(f"  Failed / Missing Files:   {failed_downloads}")
    print(f"  AI Confirmations:         {confirmations}")
    print(f"  AI Corrections:           {corrections}")
    print(f"  Manifest written to:      {manifest_path}")
    print(f"  Images saved under:       ./{EXPORT_FOLDER}/")