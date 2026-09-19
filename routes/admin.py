from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required

from extensions import db
from models import User, Case
from email_service import send_vet_decision_email
from helpers import require_admin

admin_bp = Blueprint('admin', __name__)


@admin_bp.route('/admin/pending-vets', methods=['GET'])
@admin_bp.route('/api/admin/pending-vets', methods=['GET'])
@jwt_required()
def get_pending_vets():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    pending_vets = User.query.filter_by(role='vet', is_verified=False).all()
    return jsonify([v.to_dict() for v in pending_vets])


@admin_bp.route('/admin/vets/<int:vet_id>/approve', methods=['POST'])
@admin_bp.route('/api/admin/vets/<int:vet_id>/approve', methods=['POST'])
@jwt_required()
def approve_vet(vet_id):
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    vet = db.session.get(User, vet_id)
    if not vet or vet.role != 'vet':
        return jsonify({"error": "Vet not found"}), 404

    vet.is_verified = True
    db.session.commit()
    # BUGFIX: this used to come after `return`, so the email never sent.
    send_vet_decision_email(vet, approved=True)
    return jsonify({"message": f"{vet.username} approved", "user": vet.to_dict()})


@admin_bp.route('/admin/vets/<int:vet_id>/reject', methods=['POST'])
@admin_bp.route('/api/admin/vets/<int:vet_id>/reject', methods=['POST'])
@jwt_required()
def reject_vet(vet_id):
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    vet = db.session.get(User, vet_id)
    if not vet or vet.role != 'vet':
        return jsonify({"error": "Vet not found"}), 404

    vet.role = 'user'
    vet.is_verified = False
    db.session.commit()
    # BUGFIX: this used to come after `return`, so the email never sent.
    send_vet_decision_email(vet, approved=False)
    return jsonify({"message": f"{vet.username}'s vet application was rejected. Their account remains active as a regular user."})


@admin_bp.route('/admin/export-corrections', methods=['GET'])
@admin_bp.route('/api/admin/export-corrections', methods=['GET'])
@jwt_required()
def export_corrections():
    admin = require_admin()
    if not admin:
        return jsonify({"error": "Admin access required"}), 403

    corrected_cases = Case.query.filter(Case.vet_confirmed_label.isnot(None)).all()

    export_data = []
    for case in corrected_cases:
        export_data.append({
            "case_id": case.id,
            "image_url": case.filename if case.filename.startswith('http')
                         else f"/uploads/{case.filename}",
            "ai_prediction": case.prediction,
            "ai_confidence": case.confidence,
            "vet_confirmed_label": case.vet_confirmed_label,
            "ai_was_correct": case.prediction == case.vet_confirmed_label,
            "reviewed_by_id": case.reviewed_by_id,
        })

    agreement_count = sum(1 for c in export_data if c["ai_was_correct"])
    total = len(export_data)

    return jsonify({
        "total_vet_reviewed_cases": total,
        "ai_agreement_rate": round(agreement_count / total, 3) if total > 0 else None,
        "cases": export_data
    })
